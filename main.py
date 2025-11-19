import os
import argparse

import crypten

from utils.crypten_modifications import (
    enable_public_weights,
    replace_approx,
    fix_memory_blowup,
)
from utils.model_utils import (
    generate as crypten_generate,
    generate_with_sentinels_and_verification,
    load_model,
)
from utils.sentinel_cache import build_sentinel_cache
from utils.verification import verify_with_cache


def build_parser() -> argparse.ArgumentParser:
    """
    Top-level CLI for the Priveri pipeline.

    Exposes three main functionalities:
      - construct-cache: build a sentinel cache as the verifier (user).
      - generate: run CrypTen-based generation as a prover.
      - verify: check prover logits against a saved sentinel cache.
    """
    parser = argparse.ArgumentParser(description="Priveri Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # -------------------- construct-cache --------------------
    cache_p = subparsers.add_parser(
        "construct-cache",
        help="Construct a sentinel cache for a given prompt using a reference model.",
    )
    cache_p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Hugging Face model name or path used by the verifier.",
    )
    cache_p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Plain-text prompt for which to build the sentinel cache.",
    )
    cache_p.add_argument(
        "--group-k",
        type=int,
        default=3,
        help="Number of sentinel tokens in the trap group.",
    )
    cache_p.add_argument(
        "--num-augmentations",
        type=int,
        default=1000,
        help="Number of random augmentations to record in the cache.",
    )
    cache_p.add_argument(
        "--cache-path",
        type=str,
        required=True,
        help="Output path for the sentinel cache (.pt).",
    )
    cache_p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the verifier model on.",
    )

    # -------------------- generate --------------------
    gen_p = subparsers.add_parser(
        "generate",
        help="Generate a response using CrypTen as a prover.",
    )
    gen_p.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the model checkpoint to use for CrypTen generation.",
    )
    gen_p.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="User prompt to answer.",
    )
    gen_p.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
        help="Maximum number of tokens to generate.",
    )
    gen_p.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the prover model on.",
    )
    gen_p.add_argument(
        "--num-parties",
        type=int,
        default=1,
        help="Number of CrypTen parties (WORLD_SIZE).",
    )
    gen_p.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Rank of this process in the CrypTen world.",
    )
    gen_p.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help=(
            "Optional path to a sentinel cache (.pt). If provided, generation will "
            "use sentinel tokens and perform per-token verification against this cache."
        ),
    )
    gen_p.add_argument(
        "--verify-metric",
        choices=["euclidean", "l1", "cosine"],
        default="cosine",
        help=(
            "Metric to use for per-token verification when --cache-path is given."
        ),
    )
    gen_p.add_argument(
        "--verify-threshold",
        type=float,
        default=0.99,
        help=(
            "Threshold for per-token verification. For cosine: minimum similarity; "
            "for Euclidean/L1: maximum allowed distance."
        ),
    )

    # -------------------- verify --------------------
    ver_p = subparsers.add_parser(
        "verify",
        help="Verify prover logits against a previously constructed sentinel cache.",
    )
    ver_p.add_argument(
        "--cache-path",
        type=str,
        required=True,
        help="Path to the sentinel cache (.pt) created by construct-cache.",
    )
    ver_p.add_argument(
        "--candidate-path",
        type=str,
        required=True,
        help=(
            "Path to prover logits (.pt/.npy/.npz). "
            "For .pt, either a tensor or an object with 'augmentation_blocks[*].trap_block.logits'."
        ),
    )
    ver_p.add_argument(
        "--metric",
        choices=["euclidean", "l1", "cosine"],
        default="cosine",
        help="Similarity / distance metric for verification.",
    )
    ver_p.add_argument(
        "--threshold",
        type=float,
        default=0.99,
        help=(
            "Acceptance threshold. For cosine, minimum similarity; "
            "for Euclidean/L1, maximum distance."
        ),
    )

    return parser


def run_construct_cache(args: argparse.Namespace) -> None:
    build_sentinel_cache(
        model_name_or_path=args.model,
        prompt=args.prompt,
        group_k=args.group_k,
        num_augmentations=args.num_augmentations,
        cache_path=args.cache_path,
        device=args.device,
    )


def _init_crypten_world(num_parties: int, rank: int) -> None:
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(num_parties)
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    os.environ.setdefault("RENDEZVOUS", "env://")

    enable_public_weights()
    replace_approx()
    fix_memory_blowup()

    # Numerical settings tuned for stability in this project.
    crypten.cfg.functions.sqrt_nr_iters = 9
    crypten.cfg.encoder.precision_bits = 20
    crypten.cfg.functions.exp_iterations = 13
    crypten.init()


def run_generate(args: argparse.Namespace) -> None:
    _init_crypten_world(num_parties=args.num_parties, rank=args.rank)

    model, tokenizer = load_model(args.model, check_weights=True)
    if args.device == "cuda":
        model = model.cuda()

    if args.cache_path:
        response = generate_with_sentinels_and_verification(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            cache_path=args.cache_path,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            metric=args.verify_metric,
            threshold=args.verify_threshold,
        )
    else:
        response = crypten_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
        )
    print(response)


def run_verify(args: argparse.Namespace) -> None:
    matched, details = verify_with_cache(
        cache_path=args.cache_path,
        candidate_path=args.candidate_path,
        metric=args.metric,
        threshold=args.threshold,
    )
    print(f"matched={matched}")
    print(f"details={details}")


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "construct-cache":
        run_construct_cache(args)
    elif args.command == "generate":
        run_generate(args)
    elif args.command == "verify":
        run_verify(args)
    else:  # pragma: no cover - argparse should prevent this
        raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
