"""
Put motion from extract_pose.py onto a VRM in Blender.

1. Import your .vrm with the VRM Add-on for Blender and click the armature.
2. Scripting tab -> open this file -> set JSON_PATH below -> Run Script.
3. The motion is baked into normal keyframes on the armature (an action named "Mocap_<file>").
"""
import json
import os
import re

import bpy
from mathutils import Vector

JSON_PATH = "/Users/you/Desktop/pole_2_pose.json"   # <- change this
USE_ROOT_MOTION = True          # move the whole body around like in the video
DELETE_TARGETS_AFTER_BAKE = True

# Fallback names for VRoid-made models, used if the VRM add-on mapping isn't found
VROID_NAMES = {
    "hips": "J_Bip_C_Hips", "spine": "J_Bip_C_Spine", "chest": "J_Bip_C_Chest",
    "upperChest": "J_Bip_C_UpperChest", "neck": "J_Bip_C_Neck", "head": "J_Bip_C_Head",
    "leftUpperArm": "J_Bip_L_UpperArm", "leftLowerArm": "J_Bip_L_LowerArm", "leftHand": "J_Bip_L_Hand",
    "rightUpperArm": "J_Bip_R_UpperArm", "rightLowerArm": "J_Bip_R_LowerArm", "rightHand": "J_Bip_R_Hand",
    "leftUpperLeg": "J_Bip_L_UpperLeg", "leftLowerLeg": "J_Bip_L_LowerLeg", "leftFoot": "J_Bip_L_Foot",
    "rightUpperLeg": "J_Bip_R_UpperLeg", "rightLowerLeg": "J_Bip_R_LowerLeg", "rightFoot": "J_Bip_R_Foot",
}


def find_armature():
    obj = bpy.context.active_object
    if obj and obj.type == "ARMATURE":
        return obj
    for obj in bpy.context.scene.objects:
        if obj.type == "ARMATURE":
            return obj
    raise RuntimeError("No armature found. Import your VRM and select it first.")


def bone_name(arm, human):
    """Find the real bone name for a VRM humanoid bone like 'leftUpperArm'."""
    bones = arm.data.bones
    ext = getattr(arm.data, "vrm_addon_extension", None)
    if ext is not None:
        try:  # VRM 1.0
            snake = re.sub(r"([A-Z])", r"_\1", human).lower()
            name = getattr(ext.vrm1.humanoid.human_bones, snake).node.bone_name
            if name in bones:
                return name
        except Exception:
            pass
        try:  # VRM 0.x
            for hb in ext.vrm0.humanoid.human_bones:
                if hb.bone == human and hb.node.bone_name in bones:
                    return hb.node.bone_name
        except Exception:
            pass
    name = VROID_NAMES.get(human)
    return name if name in bones else None


def main():
    with open(JSON_PATH) as f:
        data = json.load(f)
    fps = data["fps"]
    names = data["names"]
    frames = [[Vector(p) for p in fr] for fr in data["frames"]]
    roots = [Vector(r) for r in data["root"]]
    idx = {n: i for i, n in enumerate(names)}
    n_frames = len(frames)

    arm = find_armature()
    bones = {h: bone_name(arm, h) for h in VROID_NAMES}
    missing = [h for h in ("hips", "spine", "head", "leftUpperArm", "leftUpperLeg") if not bones[h]]
    if missing:
        raise RuntimeError(f"Couldn't find bones for {missing}. Edit VROID_NAMES to match your rig.")

    def rest_head(human):
        return arm.matrix_world @ arm.data.bones[bones[human]].head_local

    # scale the performer to the avatar using leg length
    avatar_leg = ((rest_head("leftUpperLeg") - rest_head("leftLowerLeg")).length +
                  (rest_head("leftLowerLeg") - rest_head("leftFoot")).length)
    scale = avatar_leg / data["leg_length"]
    base = rest_head("hips")

    def p(f, name):
        return frames[f][idx[name]]

    def mid(f, a, b):
        return (p(f, a) + p(f, b)) / 2

    # target points the bones will aim at
    targets = {
        "hip_mid": lambda f: mid(f, "left_hip", "right_hip"),
        "shoulder_mid": lambda f: mid(f, "left_shoulder", "right_shoulder"),
        "ear_mid": lambda f: mid(f, "left_ear", "right_ear"),
        "nose": lambda f: p(f, "nose"),
    }
    for side in ("left", "right"):
        s = side
        targets[f"{s}_hip"] = lambda f, s=s: p(f, f"{s}_hip")
        targets[f"{s}_shoulder"] = lambda f, s=s: p(f, f"{s}_shoulder")
        targets[f"{s}_elbow"] = lambda f, s=s: p(f, f"{s}_elbow")
        targets[f"{s}_wrist"] = lambda f, s=s: p(f, f"{s}_wrist")
        targets[f"{s}_hand"] = lambda f, s=s: mid(f, f"{s}_index", f"{s}_pinky")
        targets[f"{s}_knee"] = lambda f, s=s: p(f, f"{s}_knee")
        targets[f"{s}_ankle"] = lambda f, s=s: p(f, f"{s}_ankle")
        targets[f"{s}_toe"] = lambda f, s=s: p(f, f"{s}_foot_index")

    scene = bpy.context.scene
    scene.render.fps = int(round(fps))
    scene.frame_start = 1
    scene.frame_end = n_frames

    coll = bpy.data.collections.new("MocapTargets")
    scene.collection.children.link(coll)
    empties = {}
    for tname in targets:
        e = bpy.data.objects.new("T_" + tname, None)
        e.empty_display_size = 0.03
        coll.objects.link(e)
        empties[tname] = e

    print("Keyframing targets...")
    for f in range(n_frames):
        hip = targets["hip_mid"](f)
        root = roots[f] if USE_ROOT_MOTION else Vector((0, 0, 0))
        for tname, fn in targets.items():
            e = empties[tname]
            e.location = base + (root + fn(f) - hip) * scale
            e.keyframe_insert("location", frame=f + 1)

    # rest-pose directions, used to pick the right twist axis per bone
    left_dir = (rest_head("leftUpperLeg") - rest_head("rightUpperLeg")).normalized()
    up_dir = (rest_head("head") - rest_head("hips")).normalized()
    forward_dir = left_dir.cross(up_dir).normalized()

    def pick_axis(human, world_dir):
        m = arm.matrix_world @ arm.data.bones[bones[human]].matrix_local
        x = m.col[0].xyz.normalized()
        z = m.col[2].xyz.normalized()
        options = [("TRACK_X", x), ("TRACK_NEGATIVE_X", -x), ("TRACK_Z", z), ("TRACK_NEGATIVE_Z", -z)]
        return max(options, key=lambda o: o[1].dot(world_dir))[0]

    def pose_bone(human):
        return arm.pose.bones[bones[human]] if bones.get(human) else None

    def aim(human, target):
        pb = pose_bone(human)
        if pb:
            c = pb.constraints.new("DAMPED_TRACK")
            c.target = empties[target]
            c.track_axis = "TRACK_Y"

    def twist(human, target, world_dir):
        pb = pose_bone(human)
        if pb:
            c = pb.constraints.new("LOCKED_TRACK")
            c.target = empties[target]
            c.lock_axis = "LOCK_Y"
            c.track_axis = pick_axis(human, world_dir)

    # hips: position + facing
    if USE_ROOT_MOTION:
        c = pose_bone("hips").constraints.new("COPY_LOCATION")
        c.target = empties["hip_mid"]
    aim("hips", "shoulder_mid")
    twist("hips", "left_hip", left_dir)
    aim("spine", "shoulder_mid")
    chest = "upperChest" if bones.get("upperChest") else "chest"
    twist(chest, "left_shoulder", left_dir)
    aim("neck", "ear_mid")
    twist("head", "nose", forward_dir)
    for s, S in (("left", "left"), ("right", "right")):
        aim(f"{S}UpperArm", f"{s}_elbow")
        aim(f"{S}LowerArm", f"{s}_wrist")
        aim(f"{S}Hand", f"{s}_hand")
        aim(f"{S}UpperLeg", f"{s}_knee")
        aim(f"{S}LowerLeg", f"{s}_ankle")
        aim(f"{S}Foot", f"{s}_toe")

    # bake constraints into regular keyframes
    print("Baking...")
    bpy.context.view_layer.objects.active = arm
    arm.select_set(True)
    bpy.ops.object.mode_set(mode="POSE")
    for pb in arm.pose.bones:
        if hasattr(pb, "select"):
            pb.select = True
        else:
            pb.bone.select = True
    bpy.ops.nla.bake(frame_start=1, frame_end=n_frames, only_selected=True, visual_keying=True,
                     clear_constraints=True, use_current_action=False, bake_types={"POSE"})
    bpy.ops.object.mode_set(mode="OBJECT")
    if arm.animation_data and arm.animation_data.action:
        arm.animation_data.action.name = "Mocap_" + os.path.splitext(os.path.basename(JSON_PATH))[0]

    if DELETE_TARGETS_AFTER_BAKE:
        for e in empties.values():
            bpy.data.objects.remove(e, do_unlink=True)
        bpy.data.collections.remove(coll)

    print(f"Done: {n_frames} frames at {fps} fps on {arm.name}")


main()