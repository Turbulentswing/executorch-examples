# LungCAD iPad research prototype

Native SwiftUI iPadOS 17+ application with local DICOM import, axial CT viewing,
HU reconstruction, RAS resampling, experimental tiled MONAI RetinaNet inference
through ONNX Runtime CPU, and candidate boxes linked to slices.

**Not clinically validated. Physical iPad inference is not yet verified.**
The app does not contain a mock detector or generate substitute candidates when
the model is absent. Import a model export whose synthetic parity check passed.
No networking API, telemetry or scan upload is implemented in the app.

## Build and run

On a Mac with Xcode and XcodeGen:

```sh
cd lungcad
swift test
brew install xcodegen
xcodegen generate
open LungCAD.xcodeproj
```

Select the LungCAD scheme. For a physical iPad, set your own unique bundle ID and
development team in Signing & Capabilities, connect the device, enable Developer
Mode when required and let Xcode manage the development provisioning profile.
Simulator builds and unsigned device compilation do not produce an installable,
signed iPad app. TestFlight/App Store distribution requires Apple Developer
Program membership, distribution signing and an App Store Connect application.
No signing credentials are included or requested in the repository.

GitHub Actions:

- **LungCAD build and synthetic tests**: runs Swift package tests and compiles
  simulator and unsigned device targets. The artifact is a simulator app.
- **LungCAD synthetic model export validation**: runs on exporter changes or manually, downloading only
  pinned public pretrained weights and using generated tensors. Produces
  `lungcad.onnx`, `lungcad.json` and `validation.json`.

Unzip the model artifact into a local folder accessible through Files. Import
that folder in the app, then import a folder containing one supported CT series
or select its individual files. Models and studies must be imported again after
relaunch; the app does not persist a scan library. A cloud-backed Files provider
may download files on selection; use **On My iPad** for offline operation.
The model remains local during detection.

## Supported DICOM subset

Part 10, classic single-frame CT, Explicit or Implicit VR Little Endian,
uncompressed 16-bit MONOCHROME2 pixels. Signed and unsigned stored values,
Bits Stored, High Bit and HU rescale are checked. A study must contain one series,
unique instances, consistent dimensions/spacing and uniformly spaced axial slices
with standard LPS orientation. Slice position determines sorting.

Compressed, enhanced/multiframe, oblique, gantry-tilted, flipped-axis,
undefined-length sequence and irregular/missing-slice inputs are rejected.
Folders must contain only slices of the selected series. This is a restricted
importer, not a general DICOM toolkit. Initial memory limits are 2048 files,
64 MiB per file and 160 million source voxels. Limits are engineering safeguards,
not measurements of the iPad's available memory.

## Inference implementation and limitations

- Fixed FP32 `[1,1,96,96,64]` tensor, axes `N,C,X,Y,Z`, Z contiguous.
- RAS orientation; target spacing 0.703125 × 0.703125 × 1.25 mm.
- Trilinear sampling with border clamping; HU clipped from −1024…300 to 0…1.
- Fixed patches, 25% overlap, air padding, per-patch decoded boxes and scores.
- Per-level threshold 0.02 and top 1000, clipping, then global 3D NMS at 0.22,
  at most 300 retained candidates.
- Model checksum and output contract checked before/while running.
- Candidate scores are detector scores, not malignancy probabilities.
- Box extent is displayed, not a validated clinical nodule diameter.
- Cancellation is checked between patches and during preprocessing. An active
  ONNX Runtime call finishes before cancellation can be handled.

**This smaller-window, per-patch detection pipeline differs from MONAI's default
512 × 512 × 192 sliding-window feature-map pipeline.** Boundary candidates and
scores can differ. Clinical performance cannot be inherited from the MONAI model
card. Remaining work includes a reference preprocessing comparison, full
pipeline candidate comparison, broader DICOM support, physical-device peak
memory/latency tests and independent accuracy evaluation.

## Why ONNX Runtime first

MONAI provides an ONNX-based export route for this network. A fixed tensor graph
and FP32 CPU runtime allow direct comparison against PyTorch. The exporter
includes anchor-based regression decoding, with a check against MONAI BoxCoder.
Swift handles DICOM, sampling, tile placement and NMS.

ExecuTorch requires a new `torch.export` capture/lowering and operator validation;
the published MONAI checkpoint is not a `.pte` program. Core ML conversion or
ONNX Runtime's Core ML provider is a possible optimization, but conversion and
3D operator coverage, fallback behavior, numerical differences and performance
must be measured before adoption. Neither alternative has been tested here.

## Evidence

See `docs/local-validation.json` for the recorded local synthetic export result.
That run used the available Windows PyTorch/MONAI environment; the CI workflow
uses the older versions declared by the bundle for a separate reproducibility
check. A model-export success is not an app build or device-execution success.

## Sources and model license

- [MONAI model bundle](https://huggingface.co/MONAI/lung_nodule_ct_detection/tree/9d6622fda0e52be0a4155fed2e0afb9c17bcd80b)
- [MONAI model configuration](https://github.com/Project-MONAI/model-zoo/blob/dev/models/lung_nodule_ct_detection/configs/inference.json)
- [ONNX Runtime iOS](https://onnxruntime.ai/docs/get-started/with-obj-c.html)
- [Core ML provider restrictions](https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html)
- [Core ML conversion](https://apple.github.io/coremltools/docs-guides/source/convert-pytorch-workflow.html)
- [ExecuTorch Core ML backend](https://docs.pytorch.org/executorch/stable/backends/coreml/coreml-overview.html)

MONAI is Apache-2.0 licensed and identifies the model as an example not for
diagnostic use. Redistribute its model license with any model artifact.
No patient data is required by tests or build workflows. Do not add patient files
to this repository, CI input, logs or artifacts.
