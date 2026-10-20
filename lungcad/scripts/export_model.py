"""Export pretrained network + box decoding; validate with synthetic inputs only.

Never supply scans to this script. Outputs are a model and synthetic parity evidence,
not evidence of clinical performance or execution on an iPad.
"""
import argparse
import hashlib
import json
import platform
import urllib.request
from pathlib import Path

import monai
import numpy as np
import onnx
import onnxruntime as ort
import torch
from monai.apps.detection.networks.retinanet_network import RetinaNet, resnet_fpn_feature_extractor
from monai.apps.detection.utils.anchor_utils import AnchorGeneratorWithAnchorShape
from monai.apps.detection.utils.box_coder import BoxCoder
from monai.networks.nets.resnet import resnet50

REVISION = "9d6622fda0e52be0a4155fed2e0afb9c17bcd80b"
CHECKPOINT_SHA256 = "b5e79231466adae93a6fe8e8594029e9add142914e223b879aa0343bb2402d01"
PATCH = (96, 96, 64)


class ExportDetector(torch.nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network
        with torch.no_grad():
            maps = network(torch.zeros(1, 1, *PATCH))
        generator = AnchorGeneratorWithAnchorShape(
            feature_map_scales=(1, 2, 4),
            base_anchor_shapes=((6, 8, 4), (8, 6, 5), (10, 10, 6)),
        )
        anchors = generator(torch.zeros(1, 1, *PATCH), maps["classification"])[0]
        self.register_buffer("anchor_size", anchors[:, 3:] - anchors[:, :3])
        self.register_buffer("anchor_center", (anchors[:, 3:] + anchors[:, :3]) * 0.5)
        self.level_counts = [int(x.numel()) for x in maps["classification"]]

    def forward(self, image):
        maps = self.network(image)
        logits = torch.cat([x.permute(0, 2, 3, 4, 1).reshape(-1) for x in maps["classification"]])
        deltas = torch.cat([
            x.reshape(1, 3, 6, *x.shape[2:]).permute(0, 3, 4, 5, 1, 2).reshape(-1, 6)
            for x in maps["box_regression"]
        ])
        center = deltas[:, :3] * self.anchor_size + self.anchor_center
        extent = torch.exp(torch.clamp(deltas[:, 3:], max=np.log(1000.0 / 16))) * self.anchor_size
        return torch.cat([center - extent / 2, center + extent / 2], dim=1), torch.sigmoid(logits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(
        f"https://huggingface.co/MONAI/lung_nodule_ct_detection/resolve/{REVISION}/LICENSE",
        args.output / "LICENSE",
    )
    torch.set_num_threads(4)
    torch.manual_seed(1729)
    checkpoint = args.output / "model.pt"
    if not checkpoint.exists():
        urllib.request.urlretrieve(
            f"https://huggingface.co/MONAI/lung_nodule_ct_detection/resolve/{REVISION}/models/model.pt",
            checkpoint,
        )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == CHECKPOINT_SHA256, "Checkpoint checksum mismatch"
    backbone = resnet50(spatial_dims=3, n_input_channels=1,
                        conv1_t_stride=[2, 2, 1], conv1_t_size=[7, 7, 7])
    features = resnet_fpn_feature_extractor(backbone, 3, False, [1, 2], None)
    network = RetinaNet(spatial_dims=3, num_classes=1, num_anchors=3,
                        feature_extractor=features, size_divisible=[16, 16, 8])
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    network.load_state_dict(state, strict=True)
    network.eval()
    model = ExportDetector(network).eval()
    image = torch.rand(1, 1, *PATCH)
    destination = args.output / "lungcad.onnx"
    with torch.no_grad():
        torch.onnx.export(model, (image,), destination, input_names=["image"],
                          output_names=["boxes", "scores"], opset_version=17,
                          dynamo=False)
    onnx.checker.check_model(str(destination))
    options = ort.SessionOptions()
    options.intra_op_num_threads = 4
    session = ort.InferenceSession(str(destination), sess_options=options,
                                   providers=["CPUExecutionProvider"])
    evidence = []
    for name, tensor in [
        ("air", torch.zeros_like(image)),
        ("uniform", torch.full_like(image, 0.5)),
        ("seeded_noise", image),
    ]:
        with torch.no_grad():
            reference = [x.numpy() for x in model(tensor)]
            # Independent MONAI decoder check, including anchor/channel ordering.
            maps = network(tensor)
            generator = AnchorGeneratorWithAnchorShape(
                feature_map_scales=(1, 2, 4),
                base_anchor_shapes=((6, 8, 4), (8, 6, 5), (10, 10, 6)),
            )
            anchors = generator(tensor, maps["classification"])[0]
            deltas = torch.cat([
                x.reshape(1, -1, 6, *x.shape[2:]).permute(0, 3, 4, 5, 1, 2).reshape(-1, 6)
                for x in maps["box_regression"]
            ])
            decoded = BoxCoder(weights=(1, 1, 1, 1, 1, 1)).decode_single(deltas, anchors)
            np.testing.assert_allclose(reference[0], decoded.numpy(), rtol=1e-5, atol=1e-4)
        actual = session.run(None, {"image": tensor.numpy()})
        for ref, result in zip(reference, actual):
            assert np.isfinite(result).all(), name
            np.testing.assert_allclose(result, ref, rtol=2e-3, atol=2e-3)
        evidence.append({"input": name, "max_absolute_error": [
            float(np.max(np.abs(a - b))) for a, b in zip(reference, actual)
        ]})
    manifest = {
        "schema": 1, "revision": REVISION, "patch": PATCH,
        "spacing": [0.703125, 0.703125, 1.25], "levelCounts": model.level_counts,
        "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
        "checkpointSHA256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "syntheticParityPassed": True,
        "deviceValidated": False, "clinicalValidated": False,
    }
    (args.output / "lungcad.json").write_text(json.dumps(manifest, indent=2))
    report = {"platform": platform.platform(), "torch": torch.__version__,
              "monai": monai.__version__, "onnxruntime": ort.__version__,
              "syntheticNetworkParity": evidence,
              "monaiBoxDecoderParityPassed": True,
              "limitations": [
                  "Small-patch inference differs from MONAI's default sliding-window pipeline.",
                  "Native preprocessing and candidate parity require additional tests.",
                  "No clinical or physical iPad validation.",
              ]}
    (args.output / "validation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
