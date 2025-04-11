import numpy as np
import os
from vai_q_onnx import quantize
from onnxruntime.quantization.calibrate import CalibrationDataReader

print("[*] Starting quantization...")

# Dummy data reader for 2-input model (Concat_0 and Concat_1)
class DummyDataReader(CalibrationDataReader):
    def __init__(self):
        self.data = []
        for _ in range(10):  # number of calibration samples
            self.data.append({
                "onnx::Concat_0": np.random.rand(1, 3, 256, 256).astype(np.float32),
                "onnx::Concat_1": np.random.rand(1, 3, 256, 256).astype(np.float32),
            })
        self.index = 0

    def get_next(self):
        if self.index < len(self.data):
            input_data = self.data[self.index]
            self.index += 1
            return input_data
        return None

# Create output directory
os.makedirs("quant_output", exist_ok=True)

# Perform quantization
try:
    quantize.quantize_static(
        model_input="YGAN_generator.onnx",
        model_output="quant_output/YGAN_generator_quant.onnx",
        calibration_data_reader=DummyDataReader(),
        quant_format=quantize.VitisQuantFormat.FixNeuron,  # or VitisQuantFormat.QDQ
        input_nodes=[],
        output_nodes=[],
        op_types_to_quantize=[],
        per_channel=False,
        reduce_range=False,
        activation_type=quantize.QuantType.QInt8,
        weight_type=quantize.QuantType.QInt8,
        nodes_to_quantize=[],
        nodes_to_exclude=[],
        optimize_model=True,
        use_external_data_format=False,
        calibrate_method=quantize.PowerOfTwoMethod.MinMSE,
        execution_providers=["CPUExecutionProvider"],
        use_dpu=False,
        extra_options={}
    )
    print("[+] Quantization complete.")
except Exception as e:
    print(f"[!] Quantization failed: {e}")
