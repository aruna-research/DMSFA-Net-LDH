
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def show_gradcam_with_lesion(dmsfa, val_ds, class_names,
                            num_samples=4,
                            results_dir="results/gradcam_samples"):
    """
    Grad-CAM + Lesion Map visualization (stable version)

    ✔ No model change required
    ✔ Fixes OpenCV dtype error
    ✔ Safe normalization
    ✔ Works with 2-output model
    """

    os.makedirs(results_dir, exist_ok=True)

    print(f"\n Generating Grad-CAM + Lesion Map visualizations...")

    for x_batch, (y_batch, _) in val_ds.take(1):

        preds, lesion_maps = dmsfa.model_core(x_batch, training=False)
        cams = dmsfa.compute_gradcam(x_batch)

        for i in range(min(num_samples, x_batch.shape[0])):

            # ---------------------------
            # Extract data
            # ---------------------------
            img = x_batch[i].numpy()
            lesion = lesion_maps[i, ..., 0].numpy()
            cam = cams[i, ..., 0].numpy()

            # ---------------------------
            # Labels
            # ---------------------------
            true_cls = class_names[np.argmax(y_batch[i])]
            pred_cls = class_names[np.argmax(preds[i])]

            # ---------------------------
            # Normalize image
            # ---------------------------
            img = img.astype(np.float32)
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)

            if img.ndim == 2:
                img = np.stack([img]*3, axis=-1)

            H, W = img.shape[:2]

            # ---------------------------
            # Normalize maps
            # ---------------------------
            lesion_norm = (lesion - lesion.min()) / (lesion.max() - lesion.min() + 1e-8)
            cam_norm = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

            lesion_resized = cv2.resize(lesion_norm, (W, H), interpolation=cv2.INTER_LINEAR)
            cam_resized = cv2.resize(cam_norm, (W, H), interpolation=cv2.INTER_LINEAR)

            # ---------------------------
            # Apply colormaps
            # ---------------------------
            cam_color = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            cam_color = cv2.cvtColor(cam_color, cv2.COLOR_BGR2RGB)

            lesion_color = cv2.applyColorMap(np.uint8(255 * lesion_resized), cv2.COLORMAP_JET)
            lesion_color = cv2.cvtColor(lesion_color, cv2.COLOR_BGR2RGB)

            # CRITICAL FIX: convert ALL to float32 SAME TYPE
            cam_color = cam_color.astype(np.float32) / 255.0
            lesion_color = lesion_color.astype(np.float32) / 255.0
            img = img.astype(np.float32)

            # ---------------------------
            # Overlay
            # ---------------------------
            overlay = cv2.addWeighted(cam_color, 0.5, lesion_color, 0.5, 0)
            overlay = cv2.addWeighted(overlay, 0.6, img, 0.4, 0)

            # ---------------------------
            # Plot
            # ---------------------------
            fig, axs = plt.subplots(1, 4, figsize=(14, 4))

            axs[0].imshow(img)
            axs[0].set_title("Original")

            axs[1].imshow(cam_resized, cmap='jet')
            axs[1].set_title("Grad-CAM")

            axs[2].imshow(lesion_resized, cmap='jet')
            axs[2].set_title("Lesion Map")

            axs[3].imshow(overlay)
            axs[3].set_title("Overlay")

            for ax in axs:
                ax.axis('off')

            plt.suptitle(f"True: {true_cls} | Pred: {pred_cls}", fontsize=13)
            plt.tight_layout()

            save_path = os.path.join(results_dir, f"sample_{i+1}_{pred_cls}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            plt.show()

            print(f" Saved: {save_path}")

