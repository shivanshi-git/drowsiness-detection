import torch
import torch.nn.functional as F
import numpy as np

class XAIEvaluator:
    """
    Quantitative XAI Benchmark Suite for Driver Drowsiness Detection.
    Eliminates confirmation bias by calculating empirical quantitative attribution metrics:
    1. Pointing Game Accuracy (Hit Rate)
    2. Deletion AUC & Insertion AUC
    3. Landmark Mask IoU (Intersection over Union)
    """

    @staticmethod
    def pointing_game_hit(heatmap: np.ndarray, landmark_mask: np.ndarray) -> bool:
        """
        Returns True if the maximum attribution peak in the heatmap falls inside the landmark mask.
        """
        if heatmap is None or landmark_mask is None:
            return False
        y_max, x_max = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        return bool(landmark_mask[y_max, x_max] > 0)

    @staticmethod
    def compute_mask_iou(heatmap: np.ndarray, landmark_mask: np.ndarray, threshold: float = 0.5) -> float:
        """
        Computes Intersection over Union (IoU) between binarized heatmap (> threshold) and ground-truth landmark mask.
        """
        if heatmap is None or landmark_mask is None:
            return 0.0

        binarized_h = (heatmap >= threshold).astype(np.uint8)
        binarized_m = (landmark_mask > 0).astype(np.uint8)

        intersection = np.logical_and(binarized_h, binarized_m).sum()
        union = np.logical_or(binarized_h, binarized_m).sum()

        if union == 0:
            return 0.0
        return float(intersection / union)

    @staticmethod
    def deletion_auc(model, image_tensor: torch.Tensor, heatmap: np.ndarray, target_class: int, steps: int = 10, device: str = 'cpu') -> float:
        """
        Progressively removes top-attributed pixels (in 10% steps) and measures class probability decay.
        A steeper drop (lower AUC) indicates a high-fidelity explanation heatmap.
        """
        model.eval()
        h, w = heatmap.shape
        flat_indices = np.argsort(heatmap.flatten())[::-1]

        probabilities = []
        img_copy = image_tensor.clone().to(device)
        step_size = max(1, len(flat_indices) // steps)

        with torch.no_grad():
            for s in range(steps + 1):
                out = F.softmax(model(img_copy), dim=1)
                prob = out[0, target_class].item()
                probabilities.append(prob)

                if s < steps:
                    idx_to_mask = flat_indices[s * step_size : min((s + 1) * step_size, len(flat_indices))]
                    for idx in idx_to_mask:
                        r, c = divmod(idx, w)
                        img_copy[0, :, r, c] = 0.0

        auc = float(np.trapz(probabilities, dx=1.0 / steps))
        return auc

    @staticmethod
    def insertion_auc(model, image_tensor: torch.Tensor, heatmap: np.ndarray, target_class: int, steps: int = 10, device: str = 'cpu') -> float:
        """
        Progressively inserts top-attributed pixels into a blank canvas and measures probability recovery.
        A faster recovery (higher AUC) indicates high explanation fidelity.
        """
        model.eval()
        h, w = heatmap.shape
        flat_indices = np.argsort(heatmap.flatten())[::-1]

        probabilities = []
        blank_canvas = torch.zeros_like(image_tensor).to(device)
        step_size = max(1, len(flat_indices) // steps)

        with torch.no_grad():
            for s in range(steps + 1):
                out = F.softmax(model(blank_canvas), dim=1)
                prob = out[0, target_class].item()
                probabilities.append(prob)

                if s < steps:
                    idx_to_insert = flat_indices[s * step_size : min((s + 1) * step_size, len(flat_indices))]
                    for idx in idx_to_insert:
                        r, c = divmod(idx, w)
                        blank_canvas[0, :, r, c] = image_tensor[0, :, r, c]

        auc = float(np.trapz(probabilities, dx=1.0 / steps))
        return auc

def run_quantitative_xai_audit(model, dataloader, grad_cam_obj, device='cpu', max_samples=50, landmark_mask_provider=None):
    """
    Runs a quantitative XAI audit across validation samples and computes deletion and insertion AUC.
    Pointing Game Accuracy is computed only when landmark_mask_provider supplies real annotation masks.
    """
    model.eval()
    pointing_hits = []
    deletion_aucs = []
    insertion_aucs = []

    count = 0
    for inputs, targets in dataloader:
        if count >= max_samples:
            break

        inputs = inputs.to(device)
        for i in range(inputs.size(0)):
            if count >= max_samples:
                break

            img_tensor = inputs[i:i+1]
            heatmap, pred_idx, conf = grad_cam_obj.generate_heatmap(img_tensor)

            landmark_mask = None
            if landmark_mask_provider is not None:
                landmark_mask = landmark_mask_provider(img_tensor, i, targets[i].item(), heatmap.shape)

            if landmark_mask is not None:
                pointing_hits.append(XAIEvaluator.pointing_game_hit(heatmap, landmark_mask))
            del_auc = XAIEvaluator.deletion_auc(model, img_tensor, heatmap, pred_idx, device=device)
            ins_auc = XAIEvaluator.insertion_auc(model, img_tensor, heatmap, pred_idx, device=device)

            deletion_aucs.append(del_auc)
            insertion_aucs.append(ins_auc)

            count += 1

    mean_pointing_acc = float(np.mean(pointing_hits)) * 100 if pointing_hits else None
    mean_del_auc = float(np.mean(deletion_aucs)) if deletion_aucs else 0.0
    mean_ins_auc = float(np.mean(insertion_aucs)) if insertion_aucs else 0.0

    print(f"\n======================================================================")
    print(f" 🧠 QUANTITATIVE XAI EVALUATION METRICS REPORT")
    print(f"======================================================================")
    pointing_text = f"{mean_pointing_acc:.2f}%" if mean_pointing_acc is not None else "NOT EVALUATED (no landmark masks)"
    print(f"  • Pointing Game Hit Rate (Landmark Peak Accuracy): {pointing_text}")
    print(f"  • Deletion AUC (Lower = Better Attribution):       {mean_del_auc:.4f}")
    print(f"  • Insertion AUC (Higher = Better Attribution):     {mean_ins_auc:.4f}")
    print(f"======================================================================\n")

    return {
        "pointing_game_acc": mean_pointing_acc,
        "deletion_auc": mean_del_auc,
        "insertion_auc": mean_ins_auc
    }
