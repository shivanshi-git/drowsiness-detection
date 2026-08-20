import time
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from training.losses import MultimodalDrowsinessLoss
from training.checkpoint import CheckpointManager
from evaluation.metrics import compute_comprehensive_metrics


class PipelineTrainer:
    """
    High-performance Trainer for Low-Light Drowsiness Transformers.

    Key upgrades for 90%+ accuracy:
      1. Warmup LR schedule (linear) + CosineAnnealing
      2. Gradient clipping (prevents exploding gradients in Transformers)
      3. Separate LR groups: lower LR for LLFormer, higher for heads
      4. Class-weighted loss computed from training set distribution
      5. MixUp augmentation at batch level (temporal clip MixUp)
      6. EMA (Exponential Moving Average) of model weights for stable val
      7. Early stopping with patience
    """

    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        val_loader,
        cfg: dict,
        device: str = "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.device = device
        self.epochs = int(cfg.get("epochs", 50))
        self.warmup_epochs = int(cfg.get("warmup_epochs", 5))
        self.grad_clip = float(cfg.get("gradient_clip_norm", 1.0))
        self.use_amp = cfg.get("mixed_precision", True) and (device == "cuda")
        self.use_mixup = cfg.get("use_mixup", True)
        self.mixup_alpha = float(cfg.get("mixup_alpha", 0.4))
        self.ema_decay = float(cfg.get("ema_decay", 0.9995))
        self.early_stop_patience = int(cfg.get("early_stop_patience", 12))

        # Compute class weights from training set for imbalance handling
        class_weights = self._compute_class_weights(train_loader)
        self.criterion = MultimodalDrowsinessLoss(
            num_classes=cfg.get("num_classes", 5),
            focal_gamma=2.0,
            label_smoothing=0.1,
            focal_weight=0.5,
            class_weights=class_weights.to(device) if class_weights is not None else None
        )

        # Param groups: lower LR for backbone (LLFormer), normal for heads
        base_lr = float(cfg.get("lr", 1e-4))
        wd = float(cfg.get("weight_decay", 1e-4))
        param_groups = self._build_param_groups(model, base_lr, wd)
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=wd)
        self.base_lr = base_lr

        # Scheduler: linear warmup + cosine annealing
        total_steps = self.epochs * len(train_loader)
        warmup_steps = self.warmup_epochs * len(train_loader)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / max(1, warmup_steps)
            progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
            return max(0.01, 0.5 * (1.0 + __import__('math').cos(__import__('math').pi * progress)))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = GradScaler(enabled=self.use_amp)

        # EMA model for stable validation
        self.ema_model = None
        if self.ema_decay > 0:
            import copy
            self.ema_model = copy.deepcopy(model).to(device)
            for p in self.ema_model.parameters():
                p.requires_grad_(False)

        self.ckpt_manager = CheckpointManager(cfg.get("checkpoint_dir", "saved_models"))

    @staticmethod
    def _build_param_groups(model: nn.Module, base_lr: float, wd: float):
        """Lower LR for LLFormer (pre-trained style), normal for everything else."""
        llformer_params, other_params = [], []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "llformer" in name:
                llformer_params.append(param)
            else:
                other_params.append(param)
        return [
            {"params": llformer_params, "lr": base_lr * 0.1, "weight_decay": wd},
            {"params": other_params,   "lr": base_lr,        "weight_decay": wd},
        ]

    @staticmethod
    def _compute_class_weights(loader) -> torch.Tensor:
        """Compute inverse-frequency class weights from the training loader dataset."""
        try:
            labels = [s["label"] for s in loader.dataset.samples]
            labels_t = torch.tensor(labels)
            counts = torch.bincount(labels_t, minlength=5).float()
            weights = 1.0 / counts.clamp(min=1)
            weights = weights / weights.sum() * len(counts)
            print(f"[TRAINER] Class weights: {weights.tolist()}")
            return weights
        except Exception as e:
            print(f"[TRAINER] Could not compute class weights: {e}. Using uniform weights.")
            return None

    def _update_ema(self):
        """Update EMA model with exponential moving average of current model weights."""
        if self.ema_model is None:
            return
        for ema_p, model_p in zip(self.ema_model.parameters(), self.model.parameters()):
            ema_p.data.mul_(self.ema_decay).add_(model_p.data, alpha=1.0 - self.ema_decay)

    def _mixup_batch(self, video, flow, labels, num_classes=5):
        """
        Temporal clip-level MixUp: linearly interpolate two clips and blend labels.
        """
        import numpy as np
        lam = np.random.beta(self.mixup_alpha, self.mixup_alpha)
        lam = max(lam, 1 - lam)  # always use the dominant clip
        idx = torch.randperm(video.size(0), device=video.device)
        mixed_video = lam * video + (1 - lam) * video[idx]
        mixed_flow  = lam * flow  + (1 - lam) * flow[idx]
        labels_a, labels_b = labels, labels[idx]
        return mixed_video, mixed_flow, labels_a, labels_b, lam

    def _mixup_loss(self, logits, labels_a, labels_b, lam, fatigue_score=None):
        loss_a = self.criterion(logits, labels_a, fatigue_score)
        loss_b = self.criterion(logits, labels_b, fatigue_score)
        return lam * loss_a + (1 - lam) * loss_b

    def fit(self):
        best_acc = 0.0
        best_f1  = 0.0
        no_improve_count = 0
        step = 0

        for epoch in range(1, self.epochs + 1):
            self.model.train()
            total_loss = 0.0
            preds, targets = [], []

            for batch in self.train_loader:
                video  = batch["video"].to(self.device)
                flow   = batch["flow"].to(self.device)
                labels = batch["label"].to(self.device)

                # MixUp
                do_mix = self.use_mixup and (torch.rand(1).item() < 0.5)
                if do_mix:
                    video, flow, labels_a, labels_b, lam = self._mixup_batch(video, flow, labels)

                self.optimizer.zero_grad()
                with autocast(enabled=self.use_amp):
                    out = self.model(video, flow)
                    logits = out["logits"]
                    fatigue = out.get("fatigue_score", None)

                    if do_mix:
                        loss = self._mixup_loss(logits, labels_a, labels_b, lam, fatigue)
                    else:
                        loss = self.criterion(logits, labels, fatigue)

                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.scheduler.step()
                self._update_ema()
                step += 1

                total_loss += loss.item() * video.size(0)
                with torch.no_grad():
                    preds.extend(torch.argmax(logits, dim=1).cpu().tolist())
                    if do_mix:
                        targets.extend(labels_a.cpu().tolist())
                    else:
                        targets.extend(labels.cpu().tolist())

            train_acc = sum(p == t for p, t in zip(preds, targets)) / max(1, len(targets))
            avg_loss  = total_loss / max(1, len(self.train_loader.dataset))
            current_lr = self.optimizer.param_groups[1]["lr"]

            # Validate using EMA model if available, else main model
            eval_model = self.ema_model if self.ema_model is not None else self.model
            val_metrics = self.evaluate(model=eval_model)
            val_acc = val_metrics["accuracy"]
            val_f1  = val_metrics["macro_f1"]

            print(
                f"Epoch [{epoch:02d}/{self.epochs:02d}] LR={current_lr:.2e} | "
                f"Loss={avg_loss:.4f} | Train={train_acc*100:.1f}% | "
                f"Val Acc={val_acc*100:.1f}% | Val F1={val_f1*100:.1f}%"
            )

            # Save best by accuracy (target metric)
            if val_acc > best_acc:
                best_acc = val_acc
                best_f1  = val_f1
                no_improve_count = 0
                self.ckpt_manager.save_best(
                    eval_model, epoch, val_acc, filename="best_model.pth"
                )
                print(f"  ✅ New best! Val Acc={best_acc*100:.2f}% | Val F1={best_f1*100:.2f}%")
            else:
                no_improve_count += 1
                if no_improve_count >= self.early_stop_patience:
                    print(f"\n[EARLY STOP] No improvement for {self.early_stop_patience} epochs. Stopping.")
                    break

        print(f"\n[DONE] Best Val Accuracy: {best_acc*100:.2f}%  |  Best Val F1: {best_f1*100:.2f}%")
        return best_acc

    def evaluate(self, model=None):
        eval_model = model if model is not None else self.model
        eval_model.eval()
        preds, targets, probs = [], [], []
        with torch.no_grad():
            for batch in self.val_loader:
                video  = batch["video"].to(self.device)
                flow   = batch["flow"].to(self.device)
                labels = batch["label"].to(self.device)

                out   = eval_model(video, flow)
                prob  = torch.softmax(out["logits"], dim=1).cpu().numpy()
                pred  = torch.argmax(out["logits"], dim=1).cpu().numpy()

                preds.extend(pred.tolist())
                targets.extend(labels.cpu().tolist())
                probs.extend(prob.tolist())

        eval_model.train()
        return compute_comprehensive_metrics(targets, preds, probs)
