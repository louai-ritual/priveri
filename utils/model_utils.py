import crypten
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from models.modeling_gemma2 import Gemma2ForCausalLM
from models.modeling_llama import LlamaForCausalLM
from models.modeling_qwen2 import Qwen2ForCausalLM
from models.modeling_mistral import MistralForCausalLM
from utils.sentinel_cache import TrapTokenAugmenter
from utils.verification import load_array, ensure_min_dim, find_best_with_numpy_chunked


def get_model_class(model_name):
    if "gemma" in model_name:
        return Gemma2ForCausalLM
    if "Llama" in model_name:
        return LlamaForCausalLM
    if "Mistral" in model_name:
        return MistralForCausalLM
    if "Qwen" in model_name:
        return Qwen2ForCausalLM
    raise Exception(f"Unknown model {model_name}.")


def transform_state_dict(state_dict):
    state_dict["model.embed_tokens.wpe.weight"] = state_dict.pop(
        "model.embed_tokens.weight"
    ).T
    keys = list(state_dict.keys())
    for k in keys:
        if "norm" in k:
            state_dict[k + ".data"] = state_dict.pop(k)
    return state_dict


def get_state_dict_from_path(model_path):
    pytorch_model = AutoModelForCausalLM.from_pretrained(model_path)
    return transform_state_dict(pytorch_model.state_dict())


def load_model(model_name, check_weights=False):
    model_path = model_name
    config = AutoConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model_cls = get_model_class(model_path)
    model = model_cls(config)
    model.config._attn_implementation = 'eager'
    state_dict = get_state_dict_from_path(model_path)
    model.load_state_dict(state_dict, strict=False)

    if check_weights:
        compare_weights(state_dict, model.state_dict())

    return model, tokenizer


def compare_weights(hf_state_dict, crypten_state_dict, atol=1e-5):
    mismatch_found = False

    for name, hf_param in hf_state_dict.items():
        if name not in crypten_state_dict:
            print(f"[Missing in CrypTen] {name}")
            mismatch_found = True
            continue

        crypten_param = crypten_state_dict[name]

        if not torch.allclose(hf_param, crypten_param, atol=atol):
            print(f"[Mismatch] {name}")
            print(
                f"  HF mean: {hf_param.float().mean().item():.6f}, std: {hf_param.float().std().item():.6f}"
            )
            print(
                f"  CrypTen mean: {crypten_param.float().mean().item():.6f}, std: {crypten_param.float().std().item():.6f}"
            )
            mismatch_found = True

    for name in crypten_state_dict:
        if name not in hf_state_dict:
            print(f"[Extra in CrypTen] {name}")
            mismatch_found = True

    if not mismatch_found:
        print("✅ All weights match correctly!")


def get_module_by_name(model, module_name):
    parts = module_name.split(".")
    module = model
    for part in parts:
        if part.isdigit():
            module = module[int(part)]
        else:
            module = getattr(module, part)
    return module


def set_module_by_name(model, module_name, new_module):
    parts = module_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]
        else:
            parent = getattr(parent, part)
    final_part = parts[-1]
    if final_part.isdigit():
        parent[int(final_part)] = new_module
    else:
        setattr(parent, final_part, new_module)


def generate(model, tokenizer, prompt, max_new_tokens=10, device="cuda"):
    """
    Plain CrypTen-based greedy generation *without* sentinel traps / verification.

    This is kept as a simple baseline. For per-token, cache-based verification,
    use `generate_with_sentinels_and_verification`.
    """
    input_ids = tokenizer.encode(prompt)
    prompt_length = len(input_ids)  # in tokens

    with crypten.no_grad():
        for _ in range(max_new_tokens):
            one_hot_input_ids = torch.nn.functional.one_hot(
                torch.tensor([input_ids]),
                num_classes=len(tokenizer),
            ).float()
            if device == "cuda":
                one_hot_input_ids = one_hot_input_ids.cuda()
            one_hot_input_ids = crypten.cryptensor(one_hot_input_ids)
            encrypted_logits = model(one_hot_input_ids).logits
            logits: torch.Tensor = encrypted_logits.get_plain_text()[0][-1]
            generated_token = int(torch.argmax(logits).item())
            input_ids.append(generated_token)

            if generated_token == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[prompt_length:])


def _compute_position_ids(is_sentinel):
    """
    Compute position_ids under the sentinel scheme:
      - Base and generated non-sentinel tokens get 0..N-1 in order.
      - Sentinel tokens always get position_id == 0.
    """
    pos_ids = []
    base_pos = 0
    for flag in is_sentinel:
        if flag:
            pos_ids.append(0)
        else:
            pos_ids.append(base_pos)
            base_pos += 1
    return pos_ids


def generate_with_sentinels_and_verification(
    model,
    tokenizer,
    prompt: str,
    cache_path: str,
    max_new_tokens: int = 10,
    device: str = "cuda",
    metric: str = "cosine",
    threshold: float = 0.99,
):
    """
    CrypTen-based greedy generation with sentinel traps and per-token verification.

    Flow:
      1) Load a sentinel cache (built by the verifier) from `cache_path`.
      2) Reuse its sentinel token IDs to augment the prover's prompt.
      3) For each generated token, run the CrypTen model with a sentinel
         attention mask, extract the sentinel-block logits, and check these
         against the cache.
      4) If any step cannot be verified, raise a RuntimeError.
    """
    # Load cache as a NumPy array for fast vector comparisons.
    cache_array = load_array(cache_path)

    # Build an augmenter around the CrypTen model so that we can:
    #   - Reuse sentinel IDs from the cache.
    #   - Rebuild square masks for the current sequence length.
    augmenter = TrapTokenAugmenter(model, tokenizer, device=device)
    vocab_size = int(getattr(model.config, "vocab_size", len(tokenizer)))

    try:
        cache_obj = augmenter.load_cache(cache_path, attach=True)
        meta = cache_obj.get("meta", {}) if isinstance(cache_obj, dict) else {}
        group_k = meta.get("group_k", None)
        if group_k is None and augmenter.augmentation_blocks:
            group_k = int(augmenter.augmentation_blocks[0]["group_k"])
        if group_k is None:
            raise RuntimeError("Could not infer group_k from cache; make sure it was saved correctly.")

        # Build one sentinel-augmented prompt using the cached sentinel token IDs.
        aug = augmenter.augment(prompt, group_k=group_k, force_random=False)
        augmented_ids = list(map(int, aug["augmented_ids"]))  # base + sentinel tokens
        sentinel_positions = list(map(int, aug["trap_info"][0]["positions"]))

        # Track which positions are sentinel tokens.
        is_sentinel = [False] * len(augmented_ids)
        for idx in sentinel_positions:
            if 0 <= idx < len(is_sentinel):
                is_sentinel[idx] = True

        generated_ids = []

        with crypten.no_grad():
            for _ in range(max_new_tokens):
                full_ids = augmented_ids + generated_ids
                is_sentinel_full = is_sentinel + [False] * len(generated_ids)
                seq_len = len(full_ids)

                # One-hot encode full sequence.
                one_hot = torch.nn.functional.one_hot(
                    torch.tensor([full_ids], device=device),
                    num_classes=vocab_size,
                ).float()
                one_hot_enc = crypten.cryptensor(one_hot)

                # Build position ids and sentinel square mask.
                pos_ids_list = _compute_position_ids(is_sentinel_full)
                position_ids = torch.tensor([pos_ids_list], dtype=torch.long, device=device)

                sq_mask = augmenter._build_mask_for_sentinels(seq_len, sentinel_positions)
                additive = (1.0 - sq_mask).to(dtype=torch.float32) * -255
                additive = additive.unsqueeze(0).unsqueeze(1)
                for m in augmenter.attn_modules:
                    m.custom_attn_mask = additive

                # Forward pass through CrypTen model.
                outputs = model(one_hot_enc, position_ids=position_ids)
                encrypted_logits = outputs.logits
                logits: torch.Tensor = encrypted_logits.get_plain_text()  # (1, S, V)

                # Clear masks to avoid leaking across steps.
                for m in augmenter.attn_modules:
                    if hasattr(m, "custom_attn_mask"):
                        delattr(m, "custom_attn_mask")

                # Extract sentinel-block logits (K,V) in fixed group order.
                block = logits[0, sentinel_positions, :]  # (K,V)
                candidate = block.detach().cpu().numpy().reshape(1, -1)

                # Verify this step against the cache.
                A, B = ensure_min_dim(cache_array, candidate)
                (_, _), score = find_best_with_numpy_chunked(
                    A, B, metric=metric, a_chunk=8, b_chunk=1024
                )

                if metric == "cosine":
                    matched = score >= threshold
                else:
                    matched = score <= threshold

                if not matched:
                    raise RuntimeError(
                        f"Sentinel verification failed for current step "
                        f"(metric={metric}, score={score}, threshold={threshold})."
                    )

                # Greedy next-token selection from the last (most recent) position.
                next_logits = logits[0, -1, :]
                next_id = int(torch.argmax(next_logits).item())
                generated_ids.append(next_id)

                if next_id == tokenizer.eos_token_id:
                    break

        # Only return the newly generated tokens (excluding the base prompt).
        return tokenizer.decode(generated_ids)

    finally:
        augmenter.restore()

def get_logits(model, input_ids):
    with crypten.no_grad():
        one_hot_input_ids = torch.nn.functional.one_hot(torch.tensor(input_ids), num_classes=model.config.vocab_size).float().cuda()
        one_hot_input_ids = crypten.cryptensor(one_hot_input_ids)
        encrypted_logits = model(one_hot_input_ids).logits
        logits: torch.tensor = encrypted_logits.get_plain_text()

    return logits