
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def evaluate_model(dmsfa, val_ds, class_names):
    print("\n Evaluating model...")
    y_true_all, y_pred_all, y_prob_all = [], [], []

    # -------------------------
    # COLLECT PREDICTIONS
    # -------------------------
    for x_batch, (y_batch, _) in val_ds:
        preds, _ = dmsfa.model_core(x_batch, training=False)

        y_true_all.extend(np.argmax(y_batch.numpy(), axis=1))
        y_pred_all.extend(np.argmax(preds.numpy(), axis=1))
        y_prob_all.extend(preds.numpy())

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)
    y_prob_all = np.array(y_prob_all)

    num_classes = len(class_names)

    # -------------------------
    # CLASSIFICATION REPORT
    # -------------------------
    print("\n Classification Report:")
    print(classification_report(
        y_true_all, y_pred_all,
        target_names=class_names,
        digits=4
    ))

    # -------------------------
    # CONFUSION MATRIX (OVERALL)
    # -------------------------
    cm = confusion_matrix(y_true_all, y_pred_all)
    cm_norm = confusion_matrix(y_true_all, y_pred_all, normalize='true')

    labels_numeric = list(range(num_classes))

    plt.figure(figsize=(12,5))

    # Raw CM
    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=labels_numeric,
                yticklabels=labels_numeric)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    # Normalized CM
    plt.subplot(1,2,2)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=labels_numeric,
                yticklabels=labels_numeric)
    plt.title("Normalized Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.show()

    # -------------------------
    # PER-CLASS CONFUSION MATRIX (ONE-vs-REST)
    # -------------------------
    print("\n Generating per-class confusion matrices...")

    for i in range(num_classes):

        # Convert to binary (one-vs-rest)
        y_true_bin = (y_true_all == i).astype(int)
        y_pred_bin = (y_pred_all == i).astype(int)

        cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
        cm_bin_norm = confusion_matrix(y_true_bin, y_pred_bin, normalize='true')

        plt.figure(figsize=(8,4))

        # Raw
        plt.subplot(1,2,1)
        sns.heatmap(cm_bin, annot=True, fmt="d", cmap="Blues",
                    xticklabels=[0,1], yticklabels=[0,1])
        plt.title(f"Class {i} (One-vs-Rest)")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        # Normalized
        plt.subplot(1,2,2)
        sns.heatmap(cm_bin_norm, annot=True, fmt=".2f", cmap="Greens",
                    xticklabels=[0,1], yticklabels=[0,1])
        plt.title(f"Class {i} (Normalized)")
        plt.xlabel("Predicted")
        plt.ylabel("True")

        plt.tight_layout()
        plt.show()

    # -------------------------
    # METRICS
    # -------------------------
    TP = np.diag(cm)
    FN = np.sum(cm, axis=1) - TP
    FP = np.sum(cm, axis=0) - TP
    TN = np.sum(cm) - (TP + FP + FN)

    sensitivity = TP / (TP + FN + 1e-9)
    specificity = TN / (TN + FP + 1e-9)
    precision   = TP / (TP + FP + 1e-9)
    f1_score    = 2 * precision * sensitivity / (precision + sensitivity + 1e-9)

    metrics_df = pd.DataFrame({
        "Class": labels_numeric,
        "Sensitivity": sensitivity,
        "Specificity": specificity,
        "Precision": precision,
        "F1-score": f1_score
    })

    print("\n📄 Per-Class Metrics:")
    print(metrics_df)

    acc = np.mean(y_true_all == y_pred_all)
    print(f"\n Overall Accuracy: {acc*100:.2f}%")

    return y_true_all, y_pred_all, y_prob_all, metrics_df

def plot_kfold_curves(histories):
    acc = [h.history['accuracy'] for h in histories]
    val_acc = [h.history['val_accuracy'] for h in histories]

    loss = [h.history['loss'] for h in histories]
    val_loss = [h.history['val_loss'] for h in histories]

    acc = np.array(acc)
    val_acc = np.array(val_acc)
    loss = np.array(loss)
    val_loss = np.array(val_loss)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(acc.mean(axis=0), label="Train")
    plt.plot(val_acc.mean(axis=0), label="Validation")
    plt.fill_between(range(len(acc[0])),
                     acc.mean(0)-acc.std(0),
                     acc.mean(0)+acc.std(0),
                     alpha=0.2)
    plt.title("K-Fold Accuracy")

    # Loss
    plt.subplot(1,2,2)
    plt.plot(loss.mean(axis=0), label="Train")
    plt.plot(val_loss.mean(axis=0), label="Validation")
    plt.fill_between(range(len(loss[0])),
                     loss.mean(0)-loss.std(0),
                     loss.mean(0)+loss.std(0),
                     alpha=0.2)
    plt.title("K-Fold Loss")

    plt.legend()
    plt.show()

def plot_final_confusion(all_y_true, all_y_pred, class_names):
    cm = confusion_matrix(all_y_true, all_y_pred)
    cm_norm = confusion_matrix(all_y_true, all_y_pred, normalize='true')

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Final Confusion Matrix")

    plt.subplot(1,2,2)
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens",
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title("Normalized Confusion Matrix")

    plt.tight_layout()
    plt.show()

def plot_fold_curves(history, fold, save_dir="results/fold_curves"):

    os.makedirs(save_dir, exist_ok=True)

    plt.figure(figsize=(12,5))

    # Accuracy
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], 'o-', label='Train')
    plt.plot(history.history['val_accuracy'], 'o-', label='Validation')
    plt.title(f"Fold {fold} - Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], 'o-', label='Train')
    plt.plot(history.history['val_loss'], 'o-', label='Validation')
    plt.title(f"Fold {fold} - Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"fold_{fold}_curves.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f" Saved training curves → {save_path}")

def plot_fold_roc_binary(y_true, y_prob, fold, save_dir="results/fold_roc"):
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    import numpy as np

    os.makedirs(save_dir, exist_ok=True)

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    if y_prob.shape[1] == 2:
        prob = y_prob[:,1]
    else:
        prob = y_prob.squeeze()

    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Fold {fold} ROC (Binary)")
    plt.legend()

    save_path = os.path.join(save_dir, f"fold_{fold}_roc.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

def plot_fold_roc_multiclass(y_true, y_prob, fold, class_names,
                            save_dir="results/fold_roc"):

    os.makedirs(save_dir, exist_ok=True)

    # Binarize labels
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

    plt.figure()

    for i in range(len(class_names)):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    # Diagonal
    plt.plot([0,1], [0,1], 'k--')

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Fold {fold} - Multi-class ROC")
    plt.legend()

    save_path = os.path.join(save_dir, f"fold_{fold}_roc.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f" Saved ROC curve -> {save_path}")

