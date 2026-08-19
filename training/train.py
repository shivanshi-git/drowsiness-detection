import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from data.datasets.nthu_ddd import NTHUDDDDataset
from models.drowsiness_pipeline import LowLightDrowsinessPipeline
from training.trainer import PipelineTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/nthu_ddd.yaml", help="Path to config yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Initializing Training using {args.config} on {device}")

    # Dataset & Loader
    data_cfg = cfg["dataset"]
    train_ds = NTHUDDDDataset(root_dir=data_cfg.get("raw_dir", ""), is_train=True)
    val_ds = NTHUDDDDataset(root_dir=data_cfg.get("raw_dir", ""), is_train=False)

    batch_size = cfg["training"].get("batch_size", 8)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Model
    model = LowLightDrowsinessPipeline(
        num_classes=data_cfg.get("num_classes", 5),
        embed_dim=cfg["model"].get("embed_dim", 256)
    )

    trainer = PipelineTrainer(model, train_loader, val_loader, cfg["training"], device=device)
    trainer.fit()


if __name__ == "__main__":
    main()
