"""
Optional skin-type model trainer.

Expected layout:
data/skin_types/
  dry/       labeled images
  oily/
  normal/
  combination/  (only if you have a legitimate, correctly labeled dataset)

This model learns the dataset's labels, not a medically validated skin type.
Do not use arbitrary internet images or your own selfies as training data.
Check dataset licence, consent, label definitions and demographic coverage first.
"""
import argparse, json
from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

IMG_SIZE=(224,224)
SEED=42
EXTS={".jpg",".jpeg",".png",".webp"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--data", default="data/skin_types")
    p.add_argument("--epochs", type=int, default=8)
    args=p.parse_args()
    root=Path(args.data)
    classes=["dry","oily","normal","combination"]
    paths=[]; labels=[]
    missing=[]
    for ci, cls in enumerate(classes):
        folder=root/cls
        imgs=[x for x in folder.rglob("*") if x.is_file() and x.suffix.lower() in EXTS] if folder.exists() else []
        if not imgs: missing.append(cls)
        for x in imgs: paths.append(str(x)); labels.append(ci)
    if missing:
        raise SystemExit(
            "Training stopped: missing labelled image folders/classes: "+", ".join(missing)+
            f". Add a properly licensed dataset with all four labels under {root.resolve()}. "
            "Do not create fake labels or infer labels from appearance. The project cannot honestly train a four-class model without these data."
        )
    labels=np.asarray(labels, dtype=np.int32)
    counts=np.bincount(labels, minlength=4)
    if np.any(counts < 3):
        raise SystemExit("Each class needs at least 3 labeled images even for a basic split; more independent samples are strongly recommended.")
    tr_paths, tmp_paths, ytr, ytmp=train_test_split(paths, labels, test_size=.30, random_state=SEED, stratify=labels)
    va_paths, te_paths, yva, yte=train_test_split(tmp_paths, ytmp, test_size=.50, random_state=SEED, stratify=ytmp)
    def ds(ps, ys, training=False):
        d=tf.data.Dataset.from_tensor_slices((ps,ys))
        if training: d=d.shuffle(len(ps), seed=SEED)
        def load(path,y):
            raw=tf.io.read_file(path)
            im=tf.image.decode_image(raw, channels=3, expand_animations=False)
            im=tf.image.resize(im, IMG_SIZE)
            return tf.cast(im,tf.float32), tf.one_hot(y,4)
        return d.map(load,num_parallel_calls=tf.data.AUTOTUNE).batch(16).prefetch(tf.data.AUTOTUNE)
    aug=tf.keras.Sequential([tf.keras.layers.RandomFlip("horizontal"),tf.keras.layers.RandomRotation(.04),tf.keras.layers.RandomZoom(.05)])
    inp=tf.keras.Input((*IMG_SIZE,3))
    x=aug(inp)
    x=tf.keras.applications.mobilenet_v2.preprocess_input(x)
    base=tf.keras.applications.MobileNetV2(include_top=False,weights="imagenet",input_shape=(*IMG_SIZE,3))
    base.trainable=False
    x=base(x,training=False)
    x=tf.keras.layers.GlobalAveragePooling2D()(x)
    x=tf.keras.layers.Dropout(.35)(x)
    out=tf.keras.layers.Dense(4,activation="softmax")(x)
    model=tf.keras.Model(inp,out)
    model.compile(optimizer="adam",loss="categorical_crossentropy",metrics=["accuracy"])
    art=Path("artifacts"); art.mkdir(exist_ok=True)
    hist=model.fit(ds(tr_paths,ytr,True),validation_data=ds(va_paths,yva),epochs=max(1,args.epochs),
                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,restore_best_weights=True)])
    probs=model.predict(ds(te_paths,yte),verbose=0); pred=probs.argmax(1)
    model.save(art/"skin_type_model.keras")
    (art/"skin_type_class_names.json").write_text(json.dumps(classes,indent=2),encoding="utf-8")
    (art/"skin_type_report.txt").write_text(classification_report(yte,pred,labels=[0,1,2,3],target_names=classes,zero_division=0),encoding="utf-8")
    cm=confusion_matrix(yte,pred,labels=[0,1,2,3])
    fig,ax=plt.subplots(figsize=(7,6)); ax.imshow(cm)
    ax.set_xticks(range(4),classes,rotation=30); ax.set_yticks(range(4),classes)
    ax.set_xlabel("Predicted dataset label"); ax.set_ylabel("True dataset label"); ax.set_title("Skin-type dataset test confusion matrix")
    for i in range(4):
        for j in range(4): ax.text(j,i,str(cm[i,j]),ha="center",va="center")
    fig.tight_layout(); fig.savefig(art/"skin_type_confusion_matrix.png",dpi=150); plt.close(fig)
    (art/"skin_type_evaluation.json").write_text(json.dumps({"accuracy":float(accuracy_score(yte,pred)),"classes":classes,
      "samples":{"train":len(tr_paths),"validation":len(va_paths),"test":len(te_paths)},
      "warning":"Dataset-only evaluation; not a clinically validated or reliable personal skin-type assessment."},indent=2),encoding="utf-8")
    print("Saved optional skin-type model and held-out evaluation in artifacts/.")
    print("Dataset test accuracy:",round(accuracy_score(yte,pred),4))
    print("This score is dataset-specific and is not a clinical validation.")

if __name__=="__main__": main()
