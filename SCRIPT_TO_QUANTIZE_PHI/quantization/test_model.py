import torch
import torchvision.models as models
from torchvision.models.mobilenetv2 import MobileNet_V2_Weights
from executorch.runtime import Runtime

MODEL_PATH = "model.pte"

torch.manual_seed(0)
sample_inputs = (torch.randn(1, 3, 224, 224),)

reference_model = models.mobilenetv2.mobilenet_v2(
    weights=MobileNet_V2_Weights.DEFAULT
).eval()
with torch.no_grad():
    expected_output = reference_model(*sample_inputs)

runtime = Runtime.get()
program = runtime.load_program(MODEL_PATH)
method = program.load_method("forward")
actual_output = method.execute(sample_inputs)[0]

max_abs_diff = (actual_output - expected_output).abs().max().item()
top1_match = actual_output.argmax(dim=1).equal(expected_output.argmax(dim=1))

print(f"max abs diff: {max_abs_diff:.6f}")
print(f"top-1 prediction match: {top1_match}")

assert top1_match, "top-1 prediction mismatch between model.pte and eager model"
assert max_abs_diff < 1e-2, f"output diverges too much: {max_abs_diff}"
print("PASS")
