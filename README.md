## Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs

This repository contains the reference implementation for the paper
**“Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs”**.

The code provides a minimal pipeline that lets a *verifier* (user) check that a
remote *prover* (inference provider) ran a requested model on a given prompt.
The core idea is to embed small “sentinel” structures into the prompt, build a
cache of the corresponding activations/logits, and then enforce that every
step of the prover’s generation is consistent with that cache.

The entry point is `main.py`. All logic specific to trap construction,
encrypted inference, and verification lives under `utils/` and `models/`.

---

## High-level actors and roles

- **User / Verifier**
  - Chooses a reference model (e.g. Hugging Face checkpoint).
  - Builds a *sentinel cache* for a chosen prompt using that model.
  - Receives only the prover’s outputs (and optionally logits) and checks them
    against the cache.

- **Inference providers / Provers (parties 0 and 1)**
  - Run the user’s prompt through a CrypTen-based version of the model.
  - Participate in multi-party computation for encrypted inference.
  - Do not see each other’s plaintext inputs or weights.

The verifier is assumed to be computationally cheap (single GPU / CPU), while
the provers bear the cost of encrypted forward passes.

---

## Sentinel tokens in a sentence

Given a base prompt, the system injects a small group of *sentinel tokens* at
random positions. The attention graph is rewritten so that:

- No other token can attend to any sentinel.
- Sentinel tokens see only themselves and earlier sentinels in the group.
- All sentinels share a “collapsed” position (position id 0), while base tokens
  keep the usual 0..N–1 positions.

This creates a tiny, self-contained subgraph whose logits are predictable and
easy to cache, but hard for a misbehaving prover to fake without matching the
same computation.

The implementation lives in `utils/sentinel_cache.py` as the
`SentinelAugmenter` class.

---

## Repository layout (practical view)

- `main.py`  
  Single CLI entry point with three subcommands:
  - `construct-cache`: build a sentinel cache for a prompt.
  - `generate`: run encrypted generation (CrypTen) for a prompt, optionally
    with per-token verification against a cache.
  - `verify`: compare previously saved logits against a cache using fast
    array-level search.

- `models/`  
  Crypten-compatible reimplementations of several common decoder-only models:
  - `modeling_gemma2.py`
  - `modeling_llama.py`
  - `modeling_mistral.py`
  - `modeling_qwen2.py`

  These mirror the Hugging Face architectures but replace layers with
  CrypTen modules and MPC-friendly operations.

- `utils/`
  - `model_utils.py`: model loading from Hugging Face checkpoints into the
    Crypten-compatible architectures; CrypTen generation utilities (with and
    without sentinels).
  - `crypten_modifications.py`: small patches to CrypTen for public weights,
    numerical stability, and memory behaviour.
  - `sentinel_cache.py`: sentinel construction, attention-mask monkey-patching,
    cache building and I/O.
  - `verification.py`: vector-space comparison routines and a small API for
    checking candidate logits against a cache.

---

## Installing dependencies

This code assumes a working PyTorch + CUDA stack and CrypTen, alongside
Transformers for loading reference checkpoints.

In pseudocode (exact versions depend on your environment):

```bash
conda create -n priveri python=3.9
conda activate priveri

pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cu121
pip install crypten
pip install transformers tqdm numpy faiss-cpu
```

You will also need access to at least one Hugging Face causal LM checkpoint
compatible with the provided model wrappers (Gemma 2, Llama, Qwen2, Mistral).

---

## 1. Constructing a sentinel cache (verifier side)

The verifier starts by fixing:

- A base model checkpoint (Hugging Face path or name).
- A *cache prompt* for which sentinels will be created.
  - Sentinel group size `K`.
  - Number of randomly augmented variants to record.

Example:

```bash
python main.py construct-cache \
  --model google/gemma-2b-it \
  --prompt "Provide a concise overview of recent advances in privacy-preserving large language models." \
  --group-k 3 \
  --num-augmentations 10000 \
  --cache-path gemma_sentinel_cache.pt \
  --device cuda
```

What this does:

- Loads the reference model in standard (unencrypted) PyTorch form.
- Wraps it with `SentinelAugmenter`.
- For each augmentation:
  - Randomly interleaves the sentinel tokens into the prompt.
  - Applies the sentinel attention rules via a custom mask.
  - Runs a single forward pass and collects the logits at the sentinel
    positions (shape `(K, vocab_size)`).
- Writes everything to `gemma_sentinel_cache.pt` as:
  - `meta`: basic information (vocab size, `group_k`, model name, etc.).
  - `augmentation_blocks`: list of sentinel blocks, each storing token ids,
    positions and logits.

This cache file is all the verifier needs to later check the prover’s runs.

---

## 2. Encrypted generation with verification (prover side)

The prover runs a CrypTen-backed version of the same model. The CLI supports
both plain generation and generation with traps.

### Plain encrypted generation

```bash
python main.py generate \
  --model /path/to/model \
  --prompt "User prompt here" \
  --max-new-tokens 64 \
  --device cuda \
  --num-parties 1 \
  --rank 0
```

Under the hood:

- The checkpoint is loaded into the Crypten-compatible architecture defined in
  `models/`.
- We one-hot encode token ids and feed encrypted tensors into the model.
- The model returns encrypted logits; the code decrypts just enough to pick the
  next token greedily.

No traps, no verification – this is the baseline behaviour.

### Encrypted generation with sentinels and per-token checking

To enable verification, the prover is told to use a previously constructed cache:

```bash
python main.py generate \
  --model /path/to/model \
  --prompt "Draft a 150-word abstract summarizing the main contributions of the paper 'Privacy-Preserving Mechanisms Enable Cheap Verifiable Inference of LLMs'." \
  --max-new-tokens 64 \
  --device cuda \
  --num-parties 1 \
  --rank 0 \
  --cache-path gemma_sentinel_cache.pt \
  --verify-metric cosine \
  --verify-threshold 0.99
```

In this mode:

- The prover wraps its Crypten model in a `SentinelAugmenter` that shares the
  same sentinel token ids as the cache.
- For *every* decoding step:
  - The current prompt + generated tokens are re-encoded with the same sentinel
    attention mask and position id scheme used during cache construction.
  - The model runs one encrypted forward pass.
  - The logits at the sentinel positions are extracted and flattened.
  - These “live” trap logits are compared, in NumPy space, against all cached
    trap logits using either cosine similarity or a distance metric.
  - If no cache entry is sufficiently close (e.g. cosine < 0.99), the run is
    rejected immediately.

If all steps pass, the prover returns the generated text. The verifier can
either trust this or further inspect the logits, depending on the protocol you
want to implement.

---

## 3. Standalone verification of logits

While the typical flow is “verify during generation”, the repository also
includes a small offline tool for comparing any set of logits against a cache.

```bash
python main.py verify \
  --cache-path gemma_sentinel_cache.pt \
  --candidate-path prover_logits.pt \
  --metric cosine \
  --threshold 0.99
```

Here:

- `candidate-path` may be:
  - A `.pt` file with a tensor of shape `(N, D)`.
  - A `.pt` file with the same `augmentation_blocks` format as the cache.
  - A `.npy` / `.npz` array.
- The tool loads both sides, aligns feature dimensions, and searches for the
  best-matching pair using chunked NumPy routines.

This is useful for debugging, ablation experiments, or when you want to inspect
the verification behaviour without running full MPC.

---

## Notes and caveats

- The codebase is deliberately minimal and aims to illustrate the mechanisms
  used in the paper rather than provide a production-ready serving stack.
- Numerical details matter: changing CrypTen’s approximations, precision bits,
  or softmax ranges can affect how strict the verification thresholds need to
  be.
- The `context/` folder contains a more exploratory prototype, including
  additional visualisations and CLI experiments. It is not required for the
  main path (`main.py` + `utils/` + `models/`), but can be helpful if you want
  to dig into design choices.

---

## Reproducing experiments

The exact commands and checkpoints used in the paper will depend on the
experimental section. At a high level, each experiment consists of:

1. Selecting a base model and prompt distribution.
2. Running `construct-cache` to build sentinel caches.
3. Running the Crypten-backed `generate` command with `--cache-path` supplied.
4. Recording rates of verification success / failure, timings, and overheads.

The repository is organised so these steps can be scripted directly around the
CLI, without modifying the library code. Further configuration (e.g. different
group sizes, thresholds, number of augmentations) can be passed via flags. 


