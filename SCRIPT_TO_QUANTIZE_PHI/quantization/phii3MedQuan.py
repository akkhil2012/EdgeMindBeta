"""
ExecuTorch export for MediPhi with INT4 quantization
Optimized for on-device inference (Android, embedded systems)

Fixed-shape export (128 tokens) for Android deployment.
The exported .pte expects an input_ids tensor of EXACTLY shape [1, 128].
torch.export specializes every dimension to the example input's size unless
`dynamic_shapes` is passed, so by feeding a [1, 128] sample and omitting
dynamic_shapes, the static shape is baked into the program -- the Android
runtime never needs dynamic-shape support.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from torchao.quantization.quant_api import Int8DynamicActivationIntxWeightConfig, quantize_
from torchao.quantization.granularity import PerGroup
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
import torch.export

# Fixed sequence length the .pte will be locked to. The Android side must
# always feed exactly this many token IDs (right-pad shorter prompts,
# truncate longer ones).
FIXED_SEQ_LEN = 128

# ============================================================================
# Step 1: Load MediPhi Model
# ============================================================================
print("Loading MediPhi model...")
model_name = "microsoft/MediPhi"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float32,  # use float32 for quantization
).eval()
model.config.use_cache = False  # avoid returning a DynamicCache, which torch.export can't trace

# MediPhi uses LongRoPE, which recomputes inv_freq on every forward pass via an
# `if seq_len > original_max_position_embeddings` check on a tensor value --
# torch.export can't trace that data-dependent branch. Our prompts never
# exceed original_max_position_embeddings (4096), so the "long" factors branch
# is never actually needed; forcing rope_type to "default" makes the rotary
# embedding skip that dynamic check entirely.
model.model.rotary_emb.rope_type = "default"

tokenizer = AutoTokenizer.from_pretrained(model_name)

# A pad token is required for padding="max_length". Phi/MediPhi tokenizers
# often ship without one, so fall back to EOS as the pad token.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================================================================
# Step 2: Prepare Sample Inputs (fixed [1, FIXED_SEQ_LEN] token IDs)
# ============================================================================
print(f"Preparing fixed-shape sample inputs ({FIXED_SEQ_LEN} tokens)...")
sample_prompt = "What are the symptoms of type 2 diabetes?"
inputs = tokenizer(
    sample_prompt,
    return_tensors="pt",
    padding="max_length",      # always pad up to max_length (not just longest-in-batch)
    truncation=True,
    max_length=FIXED_SEQ_LEN,  # -> guarantees shape [1, 128]
)
sample_inputs = (inputs["input_ids"],)  # Just token IDs
seq_length = inputs["input_ids"].shape[1]

assert seq_length == FIXED_SEQ_LEN, (
    f"Expected fixed length {FIXED_SEQ_LEN}, got {seq_length}. "
    "Check that padding='max_length' is set."
)
print(f"Input shape: {tuple(inputs['input_ids'].shape)}  (fixed)")

# Note on right-padding + a causal decoder: the real prefix tokens (positions
# 0..N-1) only attend to positions <= themselves, so the trailing pad tokens
# never influence the prefix's hidden states. On-device, read the logits at the
# index of the LAST REAL token (not position 127, which is a pad slot).

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
# Step 4: Export with torch.export (static shape -- no dynamic_shapes)
# ============================================================================
print("Exporting model with torch.export (static [1, 128] shape)...")
exported_model = torch.export.export(
    model,
    sample_inputs,
    strict=False,  # needed for HF model control flow
    # dynamic_shapes intentionally omitted -> [1, 128] is baked in.
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
output_path = "medi_phi_int4_seq128.pte"
print(f"Saving to {output_path}...")

with open(output_path, "wb") as f:
    f.write(et_program.buffer)

print(f"✓ Model exported successfully: {output_path}")
print(f"  Fixed input shape: [1, {FIXED_SEQ_LEN}]")
print(f"  File size: {len(et_program.buffer) / (1024**2):.2f} MB")

# ============================================================================
# Step 7 (Optional): Validate Quantized Model
# ============================================================================
print("\nValidating quantized model...")
model.eval()
with torch.no_grad():
    original_output = model(**inputs)
    print(f"Model output shape: {original_output.logits.shape}")  # [1, 128, vocab]
    print(f"Sample logits (first 5): {original_output.logits[0, 0, :5]}")

print("\n✓ Export complete! Ready for on-device deployment.")
