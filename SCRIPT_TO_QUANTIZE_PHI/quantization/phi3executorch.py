"""
ExecuTorch export for Phi-3 Mini with INT4 quantization
Optimized for on-device inference (Android, embedded systems)
Based on EdgeMind architecture: on-device LM + zero cloud egress
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.granularity import PerGroup
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import torch.export

# ============================================================================
# Step 1: Load Phi-3 Mini Model
# ============================================================================
print("Loading Phi-3 Mini model...")
model_name = "microsoft/phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # use float32 for quantization
).eval()
model.config.use_cache = False  # avoid returning a DynamicCache, which torch.export can't trace

tokenizer = AutoTokenizer.from_pretrained(model_name)

# ============================================================================
# Step 2: Prepare Sample Inputs (Token IDs)
# ============================================================================
print("Preparing sample inputs...")
sample_prompt = "Hello, how are you?"
inputs = tokenizer(
    sample_prompt,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=128
)
sample_inputs = (inputs["input_ids"],)  # Just token IDs
seq_length = inputs["input_ids"].shape[1]

print(f"Input shape: {inputs['input_ids'].shape}")

# ============================================================================
# Step 3: Apply INT4 weight / INT8 dynamic activation quantization
# ============================================================================
print("Applying quantization (INT4 weights + INT8 dynamic activations)...")

config = Int8DynamicActivationIntxWeightConfig(
    weight_dtype=torch.int4,
    weight_granularity=PerGroup(32),  # group size: balance compression vs accuracy
)
quantize_(model, config)

# ============================================================================
# Step 4: Export with torch.export
# ============================================================================
print("Exporting model with torch.export...")
exported_model = torch.export.export(
    model,
    sample_inputs,
    strict=False,  # needed for HF model control flow
)

# ============================================================================
# Step 5: Lower to ExecuTorch with the XNNPACK backend
# ============================================================================
print("Lowering to ExecuTorch (XNNPACK backend)...")

et_program = to_edge_transform_and_lower(
    exported_model,
    partitioner=[XnnpackPartitioner()],
).to_executorch()

# ============================================================================
# Step 6: Save Exported Model
# ============================================================================
output_path = "phi3_mini_int4.pte"
print(f"Saving to {output_path}...")

with open(output_path, "wb") as f:
    f.write(et_program.buffer)

print(f"✓ Model exported successfully: {output_path}")
print(f"  File size: {len(et_program.buffer) / (1024**2):.2f} MB")

# ============================================================================
# Step 7 (Optional): Validate Quantized Model
# ============================================================================
print("\nValidating quantized model...")
model.eval()
with torch.no_grad():
    original_output = model(**inputs)
    print(f"Model output shape: {original_output.logits.shape}")
    print(f"Sample logits (first 5): {original_output.logits[0, 0, :5]}")

print("\n✓ Export complete! Ready for on-device deployment.")
print(f"  Next steps:")
print(f"  1. Convert to Android using ExecuTorch Android delegate")
print(f"  2. Deploy with zero cloud egress (DPDP/GDPR compliant)")
print(f"  3. Use with MCP governance layer for policy enforcement")