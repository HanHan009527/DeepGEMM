import argparse
import os
import random
import sys

import torch
import torch.distributed as dist

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import deep_gemm
from deep_gemm.utils.dist import init_dist


def _make_fp8_weights(num_experts_per_rank: int, hidden: int,
                      intermediate_hidden: int):
    l1 = torch.zeros(
        (num_experts_per_rank, intermediate_hidden * 2, hidden),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    l2 = torch.zeros(
        (num_experts_per_rank, hidden, intermediate_hidden),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    l1_sf = torch.ones(
        (num_experts_per_rank, (intermediate_hidden * 2) // 128,
         hidden // 128),
        dtype=torch.float32,
        device="cuda",
    )
    l2_sf = torch.ones(
        (num_experts_per_rank, hidden // 128, intermediate_hidden // 128),
        dtype=torch.float32,
        device="cuda",
    )
    return (l1.contiguous(), l1_sf.contiguous()), (l2.contiguous(),
                                                   l2_sf.contiguous())


def _run(local_rank: int, num_local_ranks: int, args: argparse.Namespace):
    rank, world_size, group = init_dist(local_rank, num_local_ranks)
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    assert args.num_experts % world_size == 0
    num_experts_per_rank = args.num_experts // world_size
    actual_tokens = args.actual_tokens
    if args.tokens_per_rank:
        tokens_per_rank = [int(x) for x in args.tokens_per_rank.split(",")]
        assert len(tokens_per_rank) == world_size
        actual_tokens = tokens_per_rank[rank]
    assert 0 < actual_tokens <= args.num_max_tokens_per_rank

    buffer = deep_gemm.get_symm_buffer_for_mega_moe(
        group,
        args.num_experts,
        args.num_max_tokens_per_rank,
        args.num_topk,
        args.hidden,
        args.intermediate_hidden,
    )
    buffer_rows = buffer.topk_idx.shape[0]

    x_fp8 = torch.zeros(
        (actual_tokens, args.hidden),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    x_sf = torch.ones(
        (actual_tokens, args.hidden // 128),
        dtype=torch.float32,
        device="cuda",
    )

    scores = torch.randn(
        (actual_tokens, args.num_experts),
        dtype=torch.float32,
        device="cuda",
    )
    topk_weights, topk_idx = torch.topk(
        scores, args.num_topk, dim=-1, largest=True, sorted=False)
    topk_weights = torch.softmax(topk_weights, dim=-1).to(torch.float32)

    buffer.x.zero_()
    buffer.x_sf.zero_()
    buffer.topk_idx.fill_(-1)
    buffer.topk_weights.zero_()
    buffer.x[:actual_tokens].copy_(x_fp8)
    buffer.x_sf[:actual_tokens].copy_(x_sf)
    buffer.topk_idx[:actual_tokens].copy_(topk_idx)
    buffer.topk_weights[:actual_tokens].copy_(topk_weights)

    l1_weights, l2_weights = _make_fp8_weights(
        num_experts_per_rank, args.hidden, args.intermediate_hidden)
    transformed_l1, transformed_l2 = deep_gemm.transform_weights_for_mega_moe_sm90(
        l1_weights, l2_weights)

    y_rows = buffer_rows if args.full_y else actual_tokens
    y = torch.empty((y_rows, args.hidden),
                    dtype=torch.bfloat16,
                    device="cuda")
    cum_stats = torch.zeros((num_experts_per_rank, ),
                            dtype=torch.int32,
                            device="cuda")

    if rank == 0:
        print(
            f"case actual_tokens={actual_tokens} full_y={args.full_y} "
            f"y_rows={y_rows} buffer_rows={buffer_rows} world_size={world_size} "
            f"experts={args.num_experts} topk={args.num_topk} "
            f"hidden={args.hidden} ih={args.intermediate_hidden}",
            flush=True,
        )

    deep_gemm.fp8_mega_moe(
        y,
        transformed_l1,
        transformed_l2,
        buffer,
        cumulative_local_expert_recv_stats=cum_stats,
        recipe=(128, 128, 128),
        activation="swiglu",
        activation_clamp=args.activation_clamp,
        fast_math=bool(args.fast_math),
    )
    torch.cuda.synchronize()
    if rank == 0:
        print("done", flush=True)

    buffer.destroy()
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-processes", type=int, default=1)
    parser.add_argument("--actual-tokens", type=int, required=True)
    parser.add_argument(
        "--tokens-per-rank",
        type=str,
        default="",
        help="Comma-separated actual token count per rank; overrides --actual-tokens.",
    )
    parser.add_argument("--num-max-tokens-per-rank", type=int, default=768)
    parser.add_argument("--hidden", type=int, default=4096)
    parser.add_argument("--intermediate-hidden", type=int, default=2048)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--num-topk", type=int, default=6)
    parser.add_argument("--full-y", action="store_true")
    parser.add_argument("--activation-clamp", type=float, default=10.0)
    parser.add_argument("--fast-math", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    torch.multiprocessing.spawn(
        _run, args=(args.num_processes, args), nprocs=args.num_processes)


if __name__ == "__main__":
    main()
