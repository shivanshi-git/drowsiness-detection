import argparse
from inference.realtime_stream import run_realtime_inference

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="0", help="Camera index or video filepath")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint path")
    parser.add_argument("--no-xai", action="store_true", help="Disable XAI dashboard view")
    args = parser.parse_args()

    src = int(args.source) if args.source.isdigit() else args.source
    run_realtime_inference(
        video_source=src,
        checkpoint_path=args.checkpoint,
        show_xai_dashboard=not args.no_xai
    )
