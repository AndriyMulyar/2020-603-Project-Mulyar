import torch
from fast_transformers.builders import TransformerEncoderBuilder


def run_seq_len_benchmark(model: torch.nn.Module, builder=None, start_n=128, geometric_step=2):
    model.cuda()
    model.eval()
    torch.cuda.synchronize()
    # Warmup GPU
    with torch.no_grad():
        model(torch.rand(1, 128, 8 * 64).cuda())
    torch.cuda.synchronize()

    N = start_n

    with open(f'runs/attention_benchmarks_{builder.attention_type}.csv', 'w') as file:
        while N <= (geometric_step**6)*start_n:
            X = torch.rand(1, N, 8 * 64, device='cuda')

            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                y = model(X)
                end.record()
            torch.cuda.synchronize()
            file.write(f"{N}, {start.elapsed_time(end)}\n")
            N *= geometric_step

# Create the builder for our transformers
builder = TransformerEncoderBuilder.from_kwargs(
    n_layers=12,
    n_heads=8,
    attention_type = "improved-cluster",
    # clusters = 128,
    local_context=3,
    query_dimensions=64,
    value_dimensions=64,
    feed_forward_dimensions=1024
)

model = builder.get()

run_seq_len_benchmark(model, builder=builder)
