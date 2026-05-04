
import gc
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedKFold

from config import *
from data_loader import load_dataset
from dataset import make_datasets
from model import build_model_multi_output, DMSFAModel
from evaluation import *
from gradcam import show_gradcam_with_lesion

def run_experiment():

    images, labels, class_names = load_dataset(DATA_DIRECTORY, INPUT_SHAPE)

    skf = StratifiedKFold(n_splits=FOLDS, shuffle=True, random_state=SEED)
    y_labels = np.argmax(labels, axis=1)

    all_histories, all_y_true, all_y_pred, all_y_prob = [], [], [], []

    for fold, (train_idx, val_idx) in enumerate(skf.split(images, y_labels), 1):

        print(f"\n===== Fold {fold}/{FOLDS} =====")

        x_train, x_val = images[train_idx], images[val_idx]
        y_train, y_val = labels[train_idx], labels[val_idx]

        train_ds, val_ds = make_datasets(x_train, y_train, x_val, y_val, BATCH_SIZE)
        last_val_ds = val_ds

        model_core, base = build_model_multi_output(INPUT_SHAPE, len(class_names))
        dmsfa = DMSFAModel(model_core, base)

        dmsfa.compile(optimizer=tf.keras.optimizers.Adam(LR))

        class LambdaRampCallback(tf.keras.callbacks.Callback):
            def on_epoch_begin(self, epoch, logs=None):
                self.model.current_epoch.assign(float(epoch))

        history = dmsfa.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCH,
            callbacks=[
                LambdaRampCallback(),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=10, restore_best_weights=True
                )
            ]
        )

        all_histories.append(history)
        last_model = dmsfa

        y_true, y_pred, y_prob, _ = evaluate_model(dmsfa, val_ds, class_names)

        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)
        all_y_prob.extend(y_prob)

        plot_fold_curves(history, fold)

        if len(class_names) == 2:
            plot_fold_roc_binary(y_true, y_prob, fold)
        else:
            plot_fold_roc_multiclass(y_true, y_prob, fold, class_names)

        del dmsfa, model_core, base, train_ds, val_ds
        tf.keras.backend.clear_session()
        gc.collect()

    plot_kfold_curves(all_histories)
    plot_final_confusion(all_y_true, all_y_pred, class_names)

    show_gradcam_with_lesion(last_model, last_val_ds, class_names, num_samples=8)

    return last_model

