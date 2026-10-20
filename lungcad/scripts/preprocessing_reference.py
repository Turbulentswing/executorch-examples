"""Generate a small synthetic MONAI reference; never reads patient data."""
import json
from pathlib import Path
import sys

import monai
import numpy as np
import torch
from monai.data import MetaTensor
from monai.transforms import Orientation, ScaleIntensityRange, Spacing

size = [4, 5, 3]
spacing = [0.8, 0.9, 1.5]
origin = [10.0, 20.0, 30.0]
values = np.array([
    -1100 + x * 100 + y * 150 + z * 200
    for z in range(size[2]) for y in range(size[1]) for x in range(size[0])
], dtype=np.float32)
affine = np.diag([-spacing[0], -spacing[1], spacing[2], 1.0])
affine[:3, 3] = [-origin[0], -origin[1], origin[2]]
image = MetaTensor(torch.from_numpy(values.reshape(3, 5, 4).transpose(2, 1, 0).copy()[None]), affine=affine)
image = Orientation(axcodes="RAS")(image)
image = Spacing(pixdim=(0.703125, 0.703125, 1.25), mode="bilinear", padding_mode="border")(image)
image = ScaleIntensityRange(a_min=-1024, a_max=300, b_min=0, b_max=1, clip=True)(image)
result = {
    "monai": monai.__version__, "sourceSize": size, "sourceSpacing": spacing,
    "sourceOriginLPS": origin, "sourceHU": values.tolist(),
    "expectedSize": list(image.shape[1:]), "expectedRASOrigin": image.affine[:3, 3].tolist(),
    "expectedTensor": image.as_tensor().flatten().tolist(),
}
Path(sys.argv[1]).write_text(json.dumps(result, indent=2))
