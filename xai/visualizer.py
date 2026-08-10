import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_heatmap(image_bgr, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """
    Blends a 2D 0..1 float heatmap over an input BGR image.
    Returns: Blended BGR image.
    """
    # Convert heatmap float (0..1) to uint8 (0..255)
    heatmap_uint8 = np.uint8(255 * heatmap)
    
    # Apply JET or TURBO colormap
    color_heatmap = cv2.applyColorMap(heatmap_uint8, colormap)
    
    # Resize color_heatmap if necessary
    if color_heatmap.shape[:2] != image_bgr.shape[:2]:
        color_heatmap = cv2.resize(color_heatmap, (image_bgr.shape[1], image_bgr.shape[0]))

    # Perform weighted overlay
    blended = cv2.addWeighted(image_bgr, 1.0 - alpha, color_heatmap, alpha, 0)
    return blended, color_heatmap

def plot_xai_comparison(original_rgb, heatmap, blended_rgb, class_name, confidence, model_name="VGG16", save_path=None):
    """
    Plots a 3-panel figure showing Original Image, Raw Grad-CAM Heatmap, and Blended Overlay.
    """
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    axes[0].imshow(original_rgb)
    axes[0].set_title("Input Image")
    axes[0].axis('off')

    im1 = axes[1].imshow(heatmap, cmap='jet')
    axes[1].set_title(f"Grad-CAM Heatmap ({model_name})")
    axes[1].axis('off')
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(blended_rgb)
    title_color = 'red' if 'drowsy' in class_name.lower() else 'green'
    axes[2].set_title(f"Prediction: {class_name}\n({confidence*100:.1f}% Confidence)", color=title_color, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        return fig
