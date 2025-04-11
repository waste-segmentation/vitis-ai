import os
import numpy as np

os.makedirs("calib_data", exist_ok=True)

for i in range(10):
    np.savez(
        f"calib_data/input_{i}.npz",
        **{
            "onnx::Concat_0": np.random.rand(1, 3, 256, 256).astype(np.float32),
            "onnx::Concat_1": np.random.rand(1, 3, 256, 256).astype(np.float32),
        }
    )

