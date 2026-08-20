import time
import torch
from torch.cuda.amp import GradScaler, autocast
from training.losses import MultimodalDrowsinessLoss
from training.checkpoint import CheckpointManager
from evaluation.metrics import compute_comprehensive_metrics


class PipelineTrainer:
    """
    Standard Trainer for Low-Light Drowsiness Transformers and Backbones.
    """
    def __init__(self, model: torch.nn.Module, train_loader, val_loader, cfg: dict, device: str = "cpu"):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.criterion = MultimodalDrowsinessLoss()
        
        lr = float(cfg.get("lr", 1e-4))
        wd = float(cfg.get("weight_decay", 1e-4))
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        self.epochs = int(cfg.get("epochs", 30))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=self.epochs)
        self.scaler = GradScaler(enabled=(device == "cuda"))
        self.ckpt_manager = CheckpointManager(cfg.get("checkpoint_dir", "saved_models"))

    def fit(self):
        best_f1 = 0.0
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            preds, targets = [], []

            for batch in self.train_loader:
                video = batch["video"].to(self.device)
                flow = batch["flow"].to(self.device)
                labels = batch["label"].to(self.device)

                self.optimizer.zero_grad()
                with autocast(enabled=(self.device == "cuda")):
                    out = self.model(video, flow)
                    loss = self.criterion(out["logits"], labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                total_loss += loss.item() * video.size(0)
                preds.extend(torch.argmax(out["logits"], dim=1).cpu().tolist())
                targets.extend(labels.cpu().tolist())

            self.scheduler.step()
            train_acc = sum(p == t for p, t in zip(preds, targets)) / max(1, len(targets))

            # Val
            val_metrics = self.evaluate()
            print(f"Epoch [{epoch:02d}/{self.epochs:02d}] Train Acc: {train_acc*100:.1f}% | Val Acc: {val_metrics['accuracy']*100:.1f}% | Val F1: {val_metrics['macro_f1']*100:.1f}%")

            if val_metrics["macro_f1"] > best_f1:
                best_f1 = val_metrics["macro_f1"]
                self.ckpt_manager.save_best(self.model, epoch, best_f1)

        return best_f1

    def evaluate(self):
        self.model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                video = batch["video"].to(self.device)
                flow = batch["flow"].to(self.device)
                labels = batch["label"].to(self.device)

                out = self.model(video, flow)
                prob = torch.softmax(out["logits"], dim=1).cpu().numpy()
                pred = torch.argmax(out["logits"], dim=1).cpu().numpy()

                preds.extend(pred.tolist())
                targets.extend(labels.cpu().tolist())
                probs.extend(prob.tolist())

        return compute_comprehensive_metrics(targets, preds, probs)
