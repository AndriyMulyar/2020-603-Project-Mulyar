import torch, os
from fast_transformers.builders import TransformerEncoderBuilder


def run_seq_len_benchmark(model: torch.nn.Module, builder=None, start_n=128, geometric_step=2):
    if not os.path.exists('runs'):
        os.makedirs('runs')
    model.cuda()
    model.eval()
    torch.cuda.synchronize()
    # Warmup GPU
    with torch.no_grad():
        model(torch.rand(1, 512, 8 * 64).cuda())
    torch.cuda.synchronize()

    N = start_n

    with open(f'runs/attention_benchmarks_{builder.attention_type}.csv', 'w') as file:

        while N <= (geometric_step**6)*start_n:
            trials = []
            for trial in range(0,5):
                print(f"Trials {trial} of {builder.attention_type} attention benchmark.")
                X = torch.rand(1, N, 8 * 64, device='cuda')

                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                with torch.no_grad():
                    start.record()
                    y = model(X)
                    end.record()
                torch.cuda.synchronize()
                trials.append(start.elapsed_time(end))
            file.write(f"{N}, {round(sum(trials)/len(trials), 2)}\n")
            N *= geometric_step






for attention_type in ('linear', 'local', 'full'):
    args = {
        'n_layers': 12,
        'n_heads': 8,
        'attention_type': attention_type,
        'query_dimensions': 64,
        'value_dimensions': 64,
        'feed_forward_dimensions': 1024
    }
    if attention_type == 'local':
        args['local_context'] = 3

    builder = TransformerEncoderBuilder.from_kwargs(
        **args
    )

    model = builder.get()
    run_seq_len_benchmark(model, builder=builder)
