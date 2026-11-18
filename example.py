"""Single-version NPU accuracy example for the unchanged 50fca6d operator."""
import argparse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--q', type=int, default=128)
    parser.add_argument('--kv', type=int, default=512)
    parser.add_argument('--seed', type=int, default=7)
    args = parser.parse_args()
    if args.q <= 0 or args.kv <= 0 or args.kv % 128:
        parser.error('Q must be positive; KV must be positive and divisible by 128')

    import torch
    import torch_npu
    from operators.mha.kernels.mha_paged_decode import build_kernel

    torch.set_num_threads(4)
    torch.npu.set_device(0)
    generator = torch.Generator().manual_seed(args.seed)
    pages = args.kv // 128
    q = torch.randn(1, args.q, 1, 128, generator=generator).bfloat16()
    k = torch.randn((pages+1)*128, 1, 128, generator=generator).bfloat16()
    v = torch.randn((pages+1)*128, 1, 128, generator=generator).bfloat16()
    table = torch.randperm(pages+1, generator=generator)[:pages].reshape(1, pages).int()
    lengths = torch.tensor([args.kv], dtype=torch.int32)

    print('Computing CPU reference...', flush=True)
    rows = table[0].long()[:, None]*128 + torch.arange(128)
    kk, vv = k[rows.flatten(), 0].float(), v[rows.flatten(), 0].float()
    expected = torch.empty_like(q)
    for start in range(0, args.q, 128):
        stop = min(start+128, args.q)
        scores = (q[0, start:stop, 0].float() @ kk.T) * 128**-0.5
        expected[0, start:stop, 0] = (scores.softmax(-1) @ vv).bfloat16()

    inputs = [x.to('npu') for x in (q, k, v, table, lengths)]
    print('Compiling 50fca6d MHA...', flush=True)
    program = build_kernel(args.q, args.kv, k.shape[0], plan_policy='cost')
    print('Plan:', program.plan.describe(), flush=True)
    program.validate_inputs(*inputs)
    actual = program(*inputs).cpu()
    torch.testing.assert_close(actual, expected, rtol=.01, atol=.004)
    error = (actual.float()-expected.float()).abs().max().item()
    print(f'PASS Q={args.q} KV={args.kv} seed={args.seed} max_abs_error={error:.6g}', flush=True)


if __name__ == '__main__':
    main()
