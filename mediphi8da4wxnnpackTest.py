"""
Host-side test harness for the MediPhi 8da4w .pte (XNNPACK).

Runs on any CPU box (laptop / Colab / cloud VM) -- the prebuilt `executorch`
wheel links the XNNPACK backend, so the .pte runs in-process. No phone, no GPU.

It does three things, in increasing cost:
  1. SMOKE   -- load the .pte, run one forward on a [1,128] tensor.
  2. PARITY  -- compare .pte logits vs eager MediPhi at the last REAL token.
                (needs ~15GB RAM for the fp32 eager model; gate with --parity)
  3. DECODE  -- full static-window greedy loop; mirrors the Android runtime.

IMPORTANT: host x86 tokens/sec is NOT device latency. Use adb-on-phone for that.

Usage:
    python test_mediphi_pte.py --pte medi_phi_8da4w_gs128_seq128.pte
    python test_mediphi_pte.py --pte ... --parity        # also run eager parity
    python test_mediphi_pte.py --pte ... --max-new 40 --prompt "..."
"""

import argparse
import torch
from executorch.runtime import Runtime
from transformers import AutoModelForCausalLM, AutoTokenizer

FIXED_SEQ_LEN = 128
MODEL_NAME = "microsoft/MediPhi"


def load_tokenizer():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def pad_to_fixed(ids: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Right-pad (or truncate) a [1, n] LongTensor to [1, FIXED_SEQ_LEN]."""
    n = ids.shape[1]
    if n > FIXED_SEQ_LEN:
        return ids[:, :FIXED_SEQ_LEN]
    if n < FIXED_SEQ_LEN:
        pad = torch.full((1, FIXED_SEQ_LEN - n), pad_id, dtype=ids.dtype)
        return torch.cat([ids, pad], dim=1)
    return ids


def load_pte(path: str):
    runtime = Runtime.get()
    # Sanity: confirm XNNPACK actually linked, else you're on portable kernels.
    backends = runtime.backend_registry.registered_backend_names
    print(f"Registered backends: {backends}")
    assert any("Xnnpack" in b for b in backends), "XNNPACK backend not linked!"
    program = runtime.load_program(path)
    print(f"Program methods: {program.method_names}")
    return program.load_method("forward")


def run_pte(method, input_ids: torch.Tensor) -> torch.Tensor:
    """input_ids: [1, 128] Long -> logits [1, 128, vocab]."""
    outputs = method.execute([input_ids])
    return outputs[0]


def smoke(method, tok):
    print("\n=== SMOKE ===")
    ids = pad_to_fixed(torch.tensor([[tok.bos_token_id or 1]]), tok.pad_token_id)
    logits = run_pte(method, ids)
    print(f"forward OK. logits shape: {tuple(logits.shape)}")
    assert logits.shape[:2] == (1, FIXED_SEQ_LEN), "static shape mismatch!"


def parity(method, tok, prompt):
    print("\n=== PARITY (vs eager MediPhi) ===")
    enc = tok(prompt, return_tensors="pt")
    real_len = enc["input_ids"].shape[1]
    ids = pad_to_fixed(enc["input_ids"], tok.pad_token_id)

    pte_logits = run_pte(method, ids)[0, real_len - 1, :]  # last REAL token

    print("Loading eager fp32 reference (~15GB RAM)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float32).eval()
    model.config.use_cache = False
    model.model.rotary_emb.rope_type = "default"
    with torch.no_grad():
        eager_logits = model(ids).logits[0, real_len - 1, :]

    cos = torch.nn.functional.cosine_similarity(
        pte_logits.float(), eager_logits.float(), dim=0
    ).item()
    max_abs = (pte_logits.float() - eager_logits.float()).abs().max().item()
    same_argmax = pte_logits.argmax().item() == eager_logits.argmax().item()

    print(f"cosine similarity : {cos:.5f}   (want > 0.99)")
    print(f"max abs diff      : {max_abs:.4f}")
    print(f"top-1 token match : {same_argmax}   "
          f"(pte={pte_logits.argmax().item()} eager={eager_logits.argmax().item()})")
    if cos < 0.99 or not same_argmax:
        print("  ! Quantization drift looks high -- check group_size / which "
              "layers got quantized, or try gs32 on this corpus.")


def decode(method, tok, prompt, max_new):
    """Static-window greedy decode -- the reference for the Android loop.

    Each step: right-pad current tokens to 128, run forward, take argmax at the
    last REAL position, append, repeat. Re-runs the full [1,128] forward per
    token (no KV cache), matching the on-device static-shape path.
    """
    print("\n=== DECODE (static-window greedy) ===")
    eos_ids = set(tok.all_special_ids) | {tok.eos_token_id}
    ids = tok(prompt, return_tensors="pt")["input_ids"]
    generated = []

    for _ in range(max_new):
        real_len = ids.shape[1]
        if real_len > FIXED_SEQ_LEN:
            print("  (context full at 128 tokens, stopping)")
            break
        windowed = pad_to_fixed(ids, tok.pad_token_id)
        logits = run_pte(method, windowed)
        next_id = int(logits[0, real_len - 1, :].argmax().item())
        if next_id in eos_ids:
            break
        generated.append(next_id)
        ids = torch.cat([ids, torch.tensor([[next_id]])], dim=1)

    print(f"Prompt    : {prompt}")
    print(f"Generated : {tok.decode(generated, skip_special_tokens=True)}")
    print(f"({len(generated)} tokens)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pte", required=True)
    ap.add_argument("--prompt", default="What are the symptoms of type 2 diabetes?")
    ap.add_argument("--max-new", type=int, default=40)
    ap.add_argument("--parity", action="store_true",
                    help="also run eager parity (needs ~15GB RAM)")
    args = ap.parse_args()

    tok = load_tokenizer()
    method = load_pte(args.pte)

    smoke(method, tok)
    if args.parity:
        parity(method, tok, args.prompt)
    decode(method, tok, args.prompt, args.max_new)

    print("\n\u2713 Done.")


if __name__ == "__main__":
    main()