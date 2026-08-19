import argparse
import os

from inference.realtime_stream import run_realtime_inference
from inference.demo_image_runner import run_image_inference


def main():
    parser = argparse.ArgumentParser(description="Driver Drowsiness Detection: SOTA Transformer & XAI Demo")
    parser.add_argument("--image", default=None, help="Path to single image file for image dataset demonstration")
    parser.add_argument("--source", default="0", help="Webcam device index (0, 1) or video filepath")
    parser.add_argument("--checkpoint", default=None, help="Trained model checkpoint path")
    parser.add_argument("--no-xai", action="store_true", help="Disable XAI dashboard view")
    args = parser.parse_args()

    if args.image:
        print(f"[DEMO MODE] Running Single-Image / Image Dataset Demonstration...")
        run_image_inference(
            image_path=args.image,
            checkpoint_path=args.checkpoint
        )
    else:
        print(f"[DEMO MODE] Running Live Camera / Video Stream Demonstration...")
        src = int(args.source) if args.source.isdigit() else args.source
        run_realtime_inference(
            video_source=src,
            checkpoint_path=args.checkpoint,
            show_xai_dashboard=not args.no_xai
        )


if __name__ == "__main__":
    main()
