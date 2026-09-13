import os
import sys
import time
import webbrowser
import argparse
import uvicorn
import torch


def main():
    parser = argparse.ArgumentParser(description="Driver Guardian: SOTA Low-Light Drowsiness Detection & XAI Web App")
    parser.add_argument("--host", default="127.0.0.1", help="Host IP address (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically")
    parser.add_argument("--test-only", action="store_true", help="Test initialization and exit without blocking")
    args = parser.parse_args()

    device_str = f"NVIDIA GPU ({torch.cuda.get_device_name(0)})" if torch.cuda.is_available() else "CPU"

    print("======================================================================")
    print(" [*] DRIVER GUARDIAN: SOTA Drowsiness Detection & Explainable AI Suite")
    print("======================================================================")
    print(f" [DEVICE]    : {device_str}")
    print(f" [FASTAPI]   : Running on http://{args.host}:{args.port}")
    print(f" [DASHBOARD] : http://{args.host}:{args.port}/")
    print(f" [DOCS]      : http://{args.host}:{args.port}/docs")
    print("======================================================================")

    if args.test_only:
        print("[TEST MODE] Verifying server import and StreamManager initialization...")
        from web_app.server import app, manager
        print(f"[TEST SUCCESS] Active model: {manager.active_model_id}, device: {manager.device}")
        sys.exit(0)

    # Open default browser after brief delay
    if not args.no_browser:
        def open_tab():
            time.sleep(1.2)
            webbrowser.open(f"http://{args.host}:{args.port}/")

        import threading
        threading.Thread(target=open_tab, daemon=True).start()

    # Start Uvicorn
    uvicorn.run("web_app.server:app", host=args.host, port=args.port, reload=False, log_level="info")


if __name__ == "__main__":
    main()
