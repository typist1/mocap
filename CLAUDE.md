# CLAUDE.md — 3D Motion Capture

## 1. Project Objective

Build a professional-grade monocular 3D human motion capture system using Python.

Input: A single prerecorded RGB video.
Output: Accurate, temporally consistent 3D human reconstruction and smooth, animation-ready output.

Priorities:
1. 3D reconstruction accuracy.
2. Temporal consistency and natural movement.
3. Skeletal and anatomical correctness.
4. Reliability and numerical stability.
5. Performance and maintainability.

Never sacrifice actual reconstruction accuracy for visually smooth results.

## 2. Engineering Standards

- Write clean, modular, production-quality Python.
- Use type hints, meaningful names, and clear interfaces.
- Keep components independently testable.
- Follow established design patterns where appropriate.
- Avoid duplicated code, unnecessary abstractions, and excessive dependencies.
- Use configuration files instead of hardcoded parameters.
- Handle errors explicitly; never silently suppress failures.
- Never use placeholder implementations for critical functionality.
- Prefer simple, technically justified solutions over unnecessary complexity.

## 3. Understand Before Modifying

- Inspect existing files and understand the architecture before changing code.
- Never assume APIs, dependencies, or functionality exist without verifying.
- Preserve working functionality.
- Avoid unnecessary rewrites and unrelated modifications.
- Investigate root causes instead of applying arbitrary fixes.
- Check the Git state before substantial changes and preserve unrelated user modifications.

## 4. Motion Capture Requirements

This is a single-camera monocular reconstruction system. Account for inherent depth and scale ambiguity.

- Maintain consistent 2D keypoint definitions and confidence scores.
- Handle occlusion, missing landmarks, and temporary tracking failures.
- Use temporal information for 3D pose estimation.
- Maintain consistent coordinate systems, units, and skeletal hierarchy.
- Prevent depth jumps, skeletal flipping, and bone-length instability.
- Maintain consistent body proportions across frames.
- Use camera and body models appropriately.
- Validate reconstruction using 2D reprojection consistency.
- Distinguish camera-relative coordinates from world-space coordinates.
- Never claim physically accurate absolute 3D reconstruction without sufficient evidence.

## 5. Animation Quality

- Minimize jitter, flickering, and discontinuous joint rotations.
- Preserve natural acceleration and rapid movements.
- Avoid excessive smoothing and artificial movement.
- Use appropriate temporal filtering, such as One Euro or Kalman filters, when justified.
- Use quaternion-based rotation handling and SLERP where appropriate.
- Maintain bone-length consistency and valid skeletal transformations.
- Minimize foot sliding where observations permit correction.
- Validate animation output rather than relying on visual assumptions.

## 6. Algorithms and Dependencies

Evaluate established solutions before implementing custom algorithms.

Potential technologies include:
- RTMPose / ViTPose for 2D pose estimation.
- MotionBERT / VideoPose3D for temporal 3D estimation.
- SMPL / SMPL-X for parametric body reconstruction.
- PyTorch, OpenCV, and NumPy for core processing.

Do not combine models indiscriminately. Verify compatibility, licensing, maintenance status, checkpoints, input formats, and skeleton conventions before integration.

## 7. Architecture

Keep video ingestion, detection, pose estimation, reconstruction, filtering, kinematics, animation export, and evaluation separate.

- Separate model initialization from inference.
- Avoid loading models repeatedly.
- Keep intermediate data formats consistent.
- Keep visualization separate from reconstruction logic.
- Use structured logging and explicit configuration.
- Keep the codebase modular without unnecessary abstraction.

## 8. Local and Google Colab Compatibility

Maintain one codebase that works locally and in Google Colab.

- Avoid machine-specific hardcoded paths.
- Support configurable input, output, and checkpoint locations.
- Support CPU and CUDA GPU execution.
- Detect available hardware automatically.
- Manage GPU memory and long video sequences carefully.
- Keep model weights and large videos outside source control.
- Maintain reproducible dependencies.
- Provide a Colab notebook that imports actual project modules rather than duplicating implementation.
- Verify dependency and CUDA compatibility rather than assuming it.

## 9. Testing and Validation

Test every significant change.

Prioritize:
- Unit tests for geometry, transformations, rotations, filtering, and kinematics.
- Integration tests from video input to animation export.
- Regression tests for existing functionality.
- Performance tests for processing speed and memory usage.

Measure relevant metrics when reference data exists:
- MPJPE / PA-MPJPE.
- 2D reprojection error.
- Temporal jitter.
- Bone-length variance.
- Foot-sliding error.
- Processing time and memory consumption.

Never fabricate test results or benchmark improvements.

## 10. Development Workflow

For substantial changes:

1. Inspect the relevant code and dependencies.
2. Identify the root problem and affected components.
3. Explain the proposed approach and significant tradeoffs.
4. Implement focused changes.
5. Run relevant tests.
6. Check for regressions.
7. Report changed files, actual test results, and remaining limitations.

Do not rewrite major architectural components without explaining the reason and obtaining approval.

## FINAL STANDARD

Act as a senior computer vision engineer responsible for a production-quality system.

Think before coding. Inspect before modifying. Test before claiming success.

Every change must have a clear technical justification and improve accuracy, reliability, maintainability, or performance without unnecessarily compromising the others.