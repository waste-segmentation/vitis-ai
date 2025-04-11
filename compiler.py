import os
import subprocess

# === CONFIGURATION ===
onnx_model_path = "/workspace/YGAN_generator_quant.onnx"  # already quantized ONNX
xmodel_output_dir = "./compiled_output"
arch_config_file = "/opt/vitis_ai/compiler/arch/DPUCZDX8G/ZCU104/arch.json"  # for ZCU104 board

# === Step: Compile to XMODEL ===
print("[*] Compiling ONNX model (assumed quantized)...")

compile_cmd = [
    "vai_c_xir",
    "--xmodel", onnx_model_path,
    "--arch", arch_config_file,
    "--output_dir", xmodel_output_dir,
    "--net_name", "ygan_compiled"
]

ret = subprocess.run(compile_cmd)
if ret.returncode != 0:
    raise RuntimeError("Compilation failed.")
print(f"[+] Compilation complete. XModel is located at: {xmodel_output_dir}")
