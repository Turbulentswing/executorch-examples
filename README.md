# LungCAD iPad: research starter kit

**Status: source scaffold and CI tools, NOT a functioning lung-nodule detector or an installable iPad app.**

This kit targets an iPad-only developer workflow and provides:

1. A GitHub Actions macOS workflow to build the upstream ExecuTorch **image classification demo for the iOS simulator**.
2. A GitHub Actions workflow to inspect the published MONAI lung-nodule model repository files.
3. A minimal SwiftUI *source file* showing DICOM file selection and explicit disabled inference. It is not wired into an Xcode project.

## Using your existing fork

Your fork of `meta-pytorch/executorch-examples` contains `mv3/apple/ExecuTorchDemo/ExecuTorchDemo.xcodeproj` with scheme `App` (as confirmed by your earlier GitHub Actions inspection).

Upload `.github/workflows/build-demo.yml` into your fork and trigger **Build ExecuTorch Demo** under Actions. The result, if it compiles, is a **simulator-only `.app`**. It cannot be installed on your physical iPad.

To run the model-inspection workflow, upload both `.github/workflows/inspect-model.yml` and `scripts/inspect_bundle.py` to your fork, then run **Inspect MONAI Nodule Model**. This lists files and metadata only, not an ONNX or ExecuTorch export.

## Required engineering work before a functional iPad CAD app

- Download and run the MONAI pretrained model in a compatible PyTorch/MONAI environment; document expected CT preprocessing and output format.
- Validate an export to `torch.export`/ExecuTorch `.pte` or ONNX. This has **not** been demonstrated for this specific model. If unsupported operators are encountered, a different detector or a rewritten architecture may be needed.
- Build an iOS app project and integrate a supported runtime and converted model.
- Implement DICOM series sorting, orientation, resampling, Hounsfield unit reconstruction, inference tiling, bounding-box decoding, NMS and measurements.
- Profile on iPad Air M3 (8 GB), optimize peak RAM and inference latency.
- Sign the device build using an Apple-authorized method. A simulator artifact cannot be sideloaded.
- Test on de-identified CT scans, including difficult subsolid/juxtavascular nodules. Do not use for clinical diagnosis without appropriate validation.

## Constraints

- GitHub Actions may incur usage charges; check your plan.
- Apple code signing and app distribution require valid credentials; do not upload private signing certificates or patient CT scans into public repositories.
- Cloud development does not mean cloud inference. Once a real app is built and model packaged, inference can be local.

Official resources:
- https://github.com/pytorch/executorch
- https://github.com/meta-pytorch/executorch-examples
- https://huggingface.co/MONAI/lung_nodule_ct_detection
- https://docs.pytorch.org/executorch/stable/using-executorch-ios.html
