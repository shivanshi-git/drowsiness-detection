import argparse
from evaluation.benchmark import run_benchmark

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd.yaml")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    run_benchmark(args.config, args.checkpoint)
