import random
import copy
import types
from typing import Any, Dict, List, Optional

import torch


class SentinelAugmenter:
    """
    Sentinel-mode augmenter.

    One sentinel group of size K. The K sentinel tokens can appear anywhere in the prompt
    (not necessarily consecutive).

    Attention rules:
      - No token (including base) may attend TO any sentinel token.
      - A sentinel token i can attend ONLY to earlier sentinels {0..i} (autoregressive inside the group).
      - Sentinel tokens cannot attend to base tokens.

    Position IDs:
      - Base tokens keep 0..base_len-1 (in their original order).
      - All sentinel tokens use position_id == 0.

    Public API
    ----------
    augment(prompt, group_k, force_random=False, rng=None)
        -> builds one augmentation with a sentinel group of size K.

    generate_and_record(prompt, group_k, num_augmentations=..., verbose=True)
        -> builds many random augmentations, runs inference with patched masks, caches logits.

    save_cache(path, dtype='float16', extra_meta=None) / load_cache(path, attach=True)
        -> portable cache on disk (Torch .pt file with augmentation_blocks).
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.attn_modules = []
        self._patched = False
        self._patch_attention_modules()

        # One entry per augmentation (single sentinel block per augmentation)
        self.logits_cache: List[Dict[str, Any]] = []
        self.augmentation_blocks: List[Dict[str, Any]] = []

    # -------------------- attention monkey-patch --------------------
    def _patch_attention_modules(self):
        if self._patched:
            return

        for module in self.model.modules():
            name = module.__class__.__name__
            if "Attention" in name or "attention" in name.lower():
                if not hasattr(module, "_orig_forward"):
                    # Bypass crypten's Module.__getattribute__ so we capture the
                    # true underlying forward once and avoid recursion.
                    orig_forward = object.__getattribute__(module, "forward")
                    module._orig_forward = orig_forward  # for potential debugging

                    def make_wrapper(orig_fwd):
                        def wrapper(self_module, hidden_states, *args, **kwargs):
                            mask = getattr(self_module, "custom_attn_mask", None)
                            if mask is None:
                                return orig_fwd(hidden_states, *args, **kwargs)
                            m = mask
                            # Accept (S,S), (B,S,S), (B,1,S,S), (B,H,S,S)
                            if m.dim() == 2:
                                b = hidden_states.shape[0]
                                m = (
                                    m.unsqueeze(0)
                                    .unsqueeze(1)
                                    .expand(b, 1, m.size(0), m.size(1))
                                    .contiguous()
                                )
                            elif m.dim() == 3:
                                m = m.unsqueeze(1)
                            if not torch.is_floating_point(m):
                                # convert 0/1 -> additive mask
                                m = (1.0 - m.float()) * -1e9
                            # For both PyTorch and CrypTen runs we pass a plain
                            # float32 tensor mask; device/dtype alignment is handled
                            # by model internals.
                            kwargs["attention_mask"] = m
                            return orig_fwd(hidden_states, *args, **kwargs)

                        return wrapper

                    module.forward = types.MethodType(make_wrapper(orig_forward), module)  # type: ignore[arg-type]
                    self.attn_modules.append(module)
        self._patched = True

    def restore(self):
        """Undo all monkey-patching and clear custom masks."""
        for m in self.attn_modules:
            if hasattr(m, "_orig_forward"):
                m.forward = m._orig_forward  # type: ignore[attr-defined]
                delattr(m, "_orig_forward")
            if hasattr(m, "custom_attn_mask"):
                delattr(m, "custom_attn_mask")
        self.attn_modules.clear()
        self._patched = False

    # -------------------- utils --------------------
    def _rand_token_id(self) -> int:
        vs = getattr(self.tokenizer, "vocab_size", None)
        if vs is None:
            vs = len(self.tokenizer.get_vocab())
        return random.randint(0, vs - 1)

    def _build_mask_for_sentinels(
        self, seq_len: int, sentinel_positions_in_order: List[int]
    ) -> torch.Tensor:
        """
        Start from standard causal mask and enforce sentinel rules.
        Returns (seq, seq) float {0/1}.
        """
        sq = torch.tril(
            torch.ones(seq_len, seq_len, dtype=torch.float32, device=self.device)
        )

        S = sentinel_positions_in_order  # len K; S[i] is augmented index of sentinel #i
        # 1) No one may attend to any sentinel -> zero columns at S[*]
        for p in S:
            sq[:, p] = 0.0

        # 2) For each sentinel i: it can attend to S[0..i] only (including itself)
        for i, p_i in enumerate(S):
            sq[p_i, :] = 0.0  # clear entire row
            for j in range(i + 1):  # allow earlier sentinels (and self)
                sq[p_i, S[j]] = 1.0

        return sq

    def _interleave_with_sentinels(
        self,
        base_ids: List[int],
        sentinel_token_ids_in_order: List[int],
        rng: Optional[random.Random] = None,
    ):
        """
        Place each sentinel token at a random 'gap' among base tokens.
        Gaps are indices 0..base_len. Multiple sentinels may land in the same gap.
        Sentinel group *order* is fixed by their index in sentinel_token_ids_in_order; placement is arbitrary.
        """
        if rng is None:
            rng = random

        base_len = len(base_ids)
        gaps: List[List[int]] = [[] for _ in range(base_len + 1)]  # per-gap list of sentinel indices

        for i in range(len(sentinel_token_ids_in_order)):
            g = rng.randint(0, base_len)
            gaps[g].append(i)

        augmented_ids: List[int] = []
        is_sentinel: List[bool] = []
        sentinel_positions_in_order: List[int] = [-1] * len(sentinel_token_ids_in_order)

        # Build augmented: for each base idx i, dump any sentinels assigned to gap i (in group order), then base token i
        for i in range(base_len):
            for si in gaps[i]:
                pos = len(augmented_ids)
                augmented_ids.append(int(sentinel_token_ids_in_order[si]))
                is_sentinel.append(True)
                sentinel_positions_in_order[si] = pos
            augmented_ids.append(int(base_ids[i]))
            is_sentinel.append(False)
        # trailing gap
        for si in gaps[base_len]:
            pos = len(augmented_ids)
            augmented_ids.append(int(sentinel_token_ids_in_order[si]))
            is_sentinel.append(True)
            sentinel_positions_in_order[si] = pos

        # position_ids: base keep 0..base_len-1, sentinels -> 0
        pos_ids: List[int] = []
        next_base_pos = 0
        for flag in is_sentinel:
            if flag:
                pos_ids.append(0)
            else:
                pos_ids.append(next_base_pos)
                next_base_pos += 1

        return augmented_ids, sentinel_positions_in_order, pos_ids

    # -------------------- public API --------------------
    def augment(
        self,
        prompt: str,
        group_k: int,
        force_random: bool = False,
        rng: Optional[random.Random] = None,
    ) -> Dict[str, Any]:
        """
        Create one augmentation with a sentinel group of size K.

        If force_random=False and a cached augmentation exists with same K, reuse the *sentinel token IDs*
        (re-sample positions every time; works for any base length).
        """
        if rng is None:
            rng = random

        enc = self.tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        base_ids = enc["input_ids"][0].tolist()
        base_len = len(base_ids)

        # Try reusing sentinel token IDs from cache with same group_k
        sentinel_ids: List[int] = []
        chosen_aug: Optional[Dict[str, Any]] = None
        if not force_random and self.augmentation_blocks:
            candidates = [a for a in self.augmentation_blocks if a["group_k"] == group_k]
            if candidates:
                chosen_aug = rng.choice(candidates)
                sentinel_ids = list(map(int, chosen_aug["trap_block"]["token_ids"]))

        if not sentinel_ids:
            sentinel_ids = [self._rand_token_id() for _ in range(group_k)]

        augmented_ids, sentinel_positions_in_order, pos_ids = self._interleave_with_sentinels(
            base_ids, sentinel_ids, rng=rng
        )
        seq_len = len(augmented_ids)
        sq = self._build_mask_for_sentinels(seq_len, sentinel_positions_in_order)

        input_ids = torch.tensor([augmented_ids], dtype=torch.long, device=self.device)
        position_ids = torch.tensor([pos_ids], dtype=torch.long, device=self.device)

        out: Dict[str, Any] = {
            "input_ids": input_ids,
            "position_ids": position_ids,
            "square_mask": sq,
            "trap_info": [
                {
                    "type": "sentinel_group",
                    "positions": list(map(int, sentinel_positions_in_order)),  # in group order
                    "token_ids": list(map(int, sentinel_ids)),
                }
            ],
            "augmented_ids": augmented_ids,
            "base_len": base_len,
        }
        if chosen_aug is not None:
            out["source_aug_idx"] = chosen_aug["aug_idx"]
        return out

    def generate_and_record(
        self,
        prompt: str,
        group_k: int,
        num_augmentations: int = 1000,
        verbose: bool = True,
    ):
        """
        Build many random augmentations of a single sentinel group and record logits.

        augmentation_blocks entry:
          {
            aug_idx,
            group_k,
            trap_block: {type, token_ids[K], positions[K], logits[K,V]}
          }
        """
        self.logits_cache.clear()
        self.augmentation_blocks.clear()

        iterator = range(num_augmentations)
        if verbose:
            try:
                from tqdm import trange  # type: ignore[import]
            except Exception:  # pragma: no cover - optional dependency
                trange = range  # type: ignore[assignment]
            iterator = trange(num_augmentations, desc="Augment+Infer (sentinels)")

        for aug_idx in iterator:
            aug = self.augment(prompt, group_k, force_random=True)
            input_ids = aug["input_ids"]
            position_ids = aug["position_ids"]
            sq_mask = aug["square_mask"]
            trap = aug["trap_info"][0]

            # inject additive mask
            additive = (1.0 - sq_mask).to(dtype=torch.float32) * -1e9
            additive = additive.unsqueeze(0).unsqueeze(1)
            for m in self.attn_modules:
                m.custom_attn_mask = additive

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, position_ids=position_ids)
            logits = outputs.logits.detach().cpu()  # (1,S,V)

            pos = trap["positions"]
            block_logits = logits[0, pos, :].contiguous()  # (K,V) in sentinel order

            block = {
                "type": "sentinel_group",
                "token_ids": list(map(int, trap["token_ids"])),
                "positions": list(map(int, pos)),
                "logits": block_logits,
            }
            self.augmentation_blocks.append(
                {
                    "aug_idx": int(aug_idx),
                    "group_k": int(group_k),
                    "trap_block": block,
                }
            )
            self.logits_cache.append(
                {
                    "aug_idx": int(aug_idx),
                    "k": int(group_k),
                    "token_ids": block["token_ids"],
                    "logits": block_logits,
                }
            )

            # cleanup
            for m in self.attn_modules:
                if hasattr(m, "custom_attn_mask"):
                    delattr(m, "custom_attn_mask")

        if verbose:
            print(f"[Cache] Recorded {len(self.augmentation_blocks)} augmentations (K={group_k}).")

    # -------------------- portable cache --------------------
    def save_cache(
        self,
        path: str,
        dtype: str = "float16",
        extra_meta: Optional[Dict[str, Any]] = None,
    ):
        """
        Save the current augmentation_blocks to disk in a simple Torch format.
        """
        dt = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
        }.get(dtype, torch.float16)

        if self.augmentation_blocks:
            cached_vocab_size = int(self.augmentation_blocks[0]["trap_block"]["logits"].shape[1])
            group_k = int(self.augmentation_blocks[0]["trap_block"]["logits"].shape[0])
        else:
            cached_vocab_size = int(getattr(self.tokenizer, "vocab_size", 0) or 0)
            group_k = None

        pack: Dict[str, Any] = {
            "meta": {
                "cached_vocab_size": cached_vocab_size,
                "group_k": group_k,
                "model_name": getattr(
                    getattr(self.model, "config", None), "_name_or_path", None
                ),
                "tokenizer_name": getattr(self.tokenizer, "name_or_path", None),
                "logits_dtype": dtype,
                "mode": "sentinel_group",
            },
            "augmentation_blocks": [],
        }
        if extra_meta:
            pack["meta"].update(copy.deepcopy(extra_meta))

        for aug in self.augmentation_blocks:
            tb = aug["trap_block"]
            pack["augmentation_blocks"].append(
                {
                    "aug_idx": int(aug["aug_idx"]),
                    "group_k": int(aug["group_k"]),
                    "trap_block": {
                        "type": tb["type"],
                        "token_ids": list(map(int, tb["token_ids"])),
                        "positions": list(map(int, tb["positions"])),
                        "logits": tb["logits"].to(dt).cpu(),
                    },
                }
            )
        torch.save(pack, path)

    def load_cache(self, path: str, attach: bool = True) -> Dict[str, Any]:
        """
        Load a cache previously produced by `save_cache`.

        If attach=True, also populate this instance's in-memory augmentation_blocks and logits_cache.
        """
        obj = torch.load(path, map_location="cpu")
        aug_blocks = obj.get("augmentation_blocks", [])
        if attach:
            # normalize in-memory layout
            self.augmentation_blocks = [
                {
                    "aug_idx": ab["aug_idx"],
                    "group_k": ab["group_k"],
                    "trap_block": ab["trap_block"],
                }
                for ab in aug_blocks
            ]
            self.logits_cache = []
            for ab in self.augmentation_blocks:
                tb = ab["trap_block"]
                self.logits_cache.append(
                    {
                        "aug_idx": ab["aug_idx"],
                        "k": len(tb["positions"]),
                        "token_ids": tb["token_ids"],
                        "logits": tb["logits"],
                    }
                )
        return obj


def build_sentinel_cache(
    model_name_or_path: str,
    prompt: str,
    group_k: int,
    num_augmentations: int,
    cache_path: str,
    device: str = "cuda",
):
    """
    High-level helper for the *verifier* (user) to build a sentinel cache.

    This uses a standard Hugging Face causal LM as the reference model and stores a portable
    .pt cache (augmentation_blocks with sentinel logits) to `cache_path`.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[Cache] Loading verifier model from '{model_name_or_path}' on {device}...")
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False)

    augmenter = SentinelAugmenter(model, tokenizer, device=device)
    try:
        print(
            f"[Cache] Generating {num_augmentations} augmentations (K={group_k}) "
            f"for prompt: {prompt!r}"
        )
        augmenter.generate_and_record(
            prompt=prompt,
            group_k=group_k,
            num_augmentations=num_augmentations,
            verbose=True,
        )
        extra_meta = {"prompt": prompt, "group_k": group_k}
        augmenter.save_cache(cache_path, dtype="float16", extra_meta=extra_meta)
        print(f"[Cache] Saved sentinel cache to: {cache_path}")
    finally:
        augmenter.restore()


