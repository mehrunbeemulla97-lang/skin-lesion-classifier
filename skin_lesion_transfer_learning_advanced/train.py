import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
LABELS = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

def main():
    parser = argparse.ArgumentParser(description="Train a HAM10000 transfer-learning classifier.")
    parser.add_argument("--metadata", default="data/HAM10000_metadata.csv")
    parser.add_argument("--images", default="data/images")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0, help="Optional small subset for a quick pipeline test; 0 uses all available images.")
    args = parser.parse_args()

    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    out = Path("artifacts")
    out.mkdir(exist_ok=True)

    metadata = pd.read_csv(args.metadata)
    required = {"image_id", "dx"}
    if not required.issubset(metadata.columns):
        raise ValueError(f"Metadata must include columns {sorted(required)}")
    metadata["path"] = metadata["image_id"].astype(str).map(lambda x: str(Path(args.images) / f"{x}.jpg"))
    metadata = metadata[metadata["dx"].isin(LABELS)]
    metadata = metadata[metadata["path"].map(os.path.isfile)].copy()
    if metadata.empty:
        raise ValueError("No matching images found. Check --metadata and --images paths.")
    if args.limit > 0:
        metadata = metadata.groupby("dx", group_keys=False).apply(
            lambda x: x.sample(min(len(x), max(1, args.limit // len(LABELS))), random_state=SEED)
        ).reset_index(drop=True)

    # Group by lesion_id where possible to reduce image-level leakage.
    if "lesion_id" in metadata.columns and metadata["lesion_id"].notna().all():
        groups = metadata["lesion_id"].astype(str)
        gss1 = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SEED)
        trainval_idx, test_idx = next(gss1.split(metadata, metadata["dx"], groups))
        trainval = metadata.iloc[trainval_idx].copy()
        test = metadata.iloc[test_idx].copy()
        gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1765, random_state=SEED + 1)
        tr_idx, val_idx = next(gss2.split(trainval, trainval["dx"], trainval["lesion_id"].astype(str)))
        train = trainval.iloc[tr_idx].copy()
        val = trainval.iloc[val_idx].copy()
        split_note = "Grouped by lesion_id (approximately 70/15/15; exact class proportions may vary)."
    else:
        trainval, test = train_test_split(metadata, test_size=0.15, stratify=metadata["dx"], random_state=SEED)
        train, val = train_test_split(trainval, test_size=0.1765, stratify=trainval["dx"], random_state=SEED + 1)
        split_note = "Stratified image-level split; lesion_id was unavailable. Potential same-lesion leakage cannot be ruled out."

    class_names = LABELS
    class_to_index = {name: i for i, name in enumerate(class_names)}
    def make_ds(frame, training=False):
        paths = frame["path"].astype(str).values
        labels = frame["dx"].map(class_to_index).astype("int32").values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(min(len(frame), 4096), seed=SEED, reshuffle_each_iteration=True)
        def load(path, label):
            image = tf.io.read_file(path)
            image = tf.image.decode_jpeg(image, channels=3)
            image = tf.image.resize(image, IMG_SIZE)
            image = tf.cast(image, tf.float32)
            return image, tf.one_hot(label, depth=len(class_names))
        ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
        return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    train_ds = make_ds(train, True)
    val_ds = make_ds(val)
    test_ds = make_ds(test)
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.08),
        tf.keras.layers.RandomZoom(0.08),
    ], name="augmentation")

    inputs = tf.keras.Input(shape=(*IMG_SIZE, 3))
    x = aug(inputs)
    x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base = tf.keras.applications.MobileNetV2(include_top=False, weights="imagenet", input_shape=(*IMG_SIZE, 3))
    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.35)(x)
    outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])

    y_train = train["dx"].map(class_to_index).values
    weights = compute_class_weight(class_weight="balanced", classes=np.arange(len(class_names)), y=y_train)
    class_weights = {i: float(w) for i, w in enumerate(weights)}
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        tf.keras.callbacks.ModelCheckpoint(str(out / "best_model.keras"), monitor="val_loss", save_best_only=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2),
    ]
    history1 = model.fit(train_ds, validation_data=val_ds, epochs=max(1, args.epochs),
                         class_weight=class_weights, callbacks=callbacks)

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:-25]:
            layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                      loss="categorical_crossentropy", metrics=["accuracy"])
        history2 = model.fit(train_ds, validation_data=val_ds, epochs=args.fine_tune_epochs,
                             class_weight=class_weights, callbacks=callbacks)
    else:
        history2 = None

    best_path = out / "best_model.keras"
    if best_path.exists():
        model = tf.keras.models.load_model(best_path)
    model.save(out / "skin_lesion_model.keras")
    with open(out / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)

    y_true = test["dx"].map(class_to_index).values
    probs = model.predict(test_ds, verbose=1)
    y_pred = probs.argmax(axis=1)
    report = classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                   target_names=class_names, output_dict=True, zero_division=0)
    report_text = classification_report(y_true, y_pred, labels=np.arange(len(class_names)),
                                        target_names=class_names, zero_division=0)
    (out / "classification_report.txt").write_text(report_text, encoding="utf-8")
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Held-out test confusion matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    ax.set_yticks(range(len(class_names)), class_names)
    fig.colorbar(im, ax=ax)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    hist = {}
    for h in [history1, history2]:
        if h:
            for k, vals in h.history.items():
                hist.setdefault(k, []).extend([float(v) for v in vals])
    with open(out / "evaluation.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "split": split_note,
            "samples": {"train": len(train), "validation": len(val), "test": len(test)},
            "classification_report": report,
            "labels": class_names
        }, f, indent=2)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(hist.get("accuracy", []), label="train")
    axes[0].plot(hist.get("val_accuracy", []), label="validation")
    axes[0].set_title("Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(hist.get("loss", []), label="train")
    axes[1].plot(hist.get("val_loss", []), label="validation")
    axes[1].set_title("Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "training_curves.png", dpi=160); plt.close(fig)
    print("\nSplit:", split_note)
    print("Samples:", {"train": len(train), "validation": len(val), "test": len(test)})
    print("Held-out test accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Saved artifacts to", out.resolve())
    print("\nIMPORTANT: these are dataset benchmark results, not clinical validation.")

if __name__ == "__main__":
    main()
