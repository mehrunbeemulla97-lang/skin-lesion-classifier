import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from PIL import Image, UnidentifiedImageError
from werkzeug.utils import secure_filename

BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "artifacts"
UPLOADS = BASE / "uploads"
UPLOADS.mkdir(exist_ok=True)
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = (224, 224)
LABEL_DESCRIPTIONS = {
    "akiec": "Actinic keratoses / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis-like lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevi",
    "vasc": "Vascular lesions",
}
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024
DB_PATH = BASE / "artifacts" / "skin_history.sqlite3"
DB_PATH.parent.mkdir(exist_ok=True)

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS skin_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                image_filename TEXT NOT NULL,
                symptoms TEXT,
                itch_score INTEGER,
                redness_score INTEGER,
                dryness_score INTEGER,
                notes TEXT,
                predicted_label TEXT,
                model_score REAL
            )
        """)

init_db()
model = None
class_names = []
model_path = ARTIFACTS / "skin_lesion_model.keras"
labels_path = ARTIFACTS / "class_names.json"
if model_path.exists() and labels_path.exists():
    try:
        model = tf.keras.models.load_model(model_path)
        class_names = json.loads(labels_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Model loading failed:", exc)

# Optional separate skin-type model. It is disabled until trained on a suitable labelled dataset.
skin_type_model = None
skin_type_names = []
skin_type_model_path = ARTIFACTS / "skin_type_model.keras"
skin_type_labels_path = ARTIFACTS / "skin_type_class_names.json"
if skin_type_model_path.exists() and skin_type_labels_path.exists():
    try:
        skin_type_model = tf.keras.models.load_model(skin_type_model_path)
        skin_type_names = json.loads(skin_type_labels_path.read_text(encoding="utf-8"))
    except Exception as exc:
        print("Optional skin-type model loading failed:", exc)

def allowed(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None
    image_url = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Please choose an image file."
        elif not allowed(file.filename):
            error = "Unsupported file type. Upload JPG, JPEG, or PNG."
        elif model is None:
            error = "The trained model is not available yet. Follow README.md and run train.py first."
        else:
            try:
                image = Image.open(file.stream).convert("RGB")
                image.thumbnail((1200, 1200))
                safe_name = secure_filename(file.filename)
                save_name = f"upload_{os.urandom(6).hex()}_{safe_name}"
                save_path = UPLOADS / save_name
                image.save(save_path, format="JPEG", quality=92)
                resized = image.resize(IMG_SIZE)
                arr = np.asarray(resized, dtype=np.float32)
                probs = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                order = np.argsort(probs)[::-1][:len(class_names)]
                scores = [{
                    "label": class_names[int(i)],
                    "description": LABEL_DESCRIPTIONS.get(class_names[int(i)], class_names[int(i)]),
                    "score": float(probs[int(i)]) * 100
                } for i in order]
                result = {"top": scores[0], "scores": scores}
                image_url = "/" + str(save_path.relative_to(BASE)).replace("\\", "/")
                symptoms = (request.form.get("symptoms") or "").strip()[:500]
                notes = (request.form.get("notes") or "").strip()[:2000]
                def safe_score(field):
                    try:
                        return max(0, min(5, int(request.form.get(field, "0"))))
                    except (TypeError, ValueError):
                        return 0
                with get_db() as conn:
                    conn.execute(
                        """INSERT INTO skin_history
                        (created_at, image_filename, symptoms, itch_score, redness_score,
                         dryness_score, notes, predicted_label, model_score)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (datetime.now().astimezone().isoformat(timespec="minutes"),
                         save_name, symptoms, safe_score("itch_score"),
                         safe_score("redness_score"), safe_score("dryness_score"),
                         notes, scores[0]["label"], scores[0]["score"])
                    )
            except (UnidentifiedImageError, OSError, ValueError):
                error = "That image could not be read. Try a valid JPG or PNG."
    return render_template("index.html", result=result, error=error,
                           image_url=image_url, model_ready=(model is not None))

@app.route("/uploads/<path:filename>")
def uploaded_file(filename):
    from flask import send_from_directory
    return send_from_directory(UPLOADS, filename)

@app.route("/skin-type", methods=["GET", "POST"])
def skin_type():
    result = None
    error = None
    if request.method == "POST":
        file = request.files.get("image")
        if not file or not file.filename:
            error = "Please choose an image file."
        elif not allowed(file.filename):
            error = "Unsupported file type. Upload JPG, JPEG, or PNG."
        elif skin_type_model is None:
            error = "The optional skin-type model is not trained. See README.md and train_skin_type.py."
        else:
            try:
                image = Image.open(file.stream).convert("RGB").resize(IMG_SIZE)
                arr = np.asarray(image, dtype=np.float32)
                probs = skin_type_model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
                order = np.argsort(probs)[::-1]
                result = [{"label": skin_type_names[int(i)], "score": float(probs[int(i)]) * 100} for i in order]
            except (UnidentifiedImageError, OSError, ValueError):
                error = "That image could not be read. Try a valid JPG or PNG."
    return render_template("skin_type.html", result=result, error=error, model_ready=(skin_type_model is not None))

@app.route("/rash-guide")
def rash_guide():
    return render_template("rash_guide.html")


@app.route("/history")
def history():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM skin_history ORDER BY id DESC LIMIT 200").fetchall()
    return render_template("history.html", rows=rows)

@app.route("/dashboard")
def dashboard():
    with get_db() as conn:
        rows = conn.execute("SELECT * FROM skin_history ORDER BY id ASC LIMIT 500").fetchall()
    dates = []
    itch, redness, dryness = [], [], []
    counts = {}
    for row in rows:
        date = (row["created_at"] or "")[:10]
        if date:
            if date not in dates:
                dates.append(date)
                itch.append(row["itch_score"] or 0)
                redness.append(row["redness_score"] or 0)
                dryness.append(row["dryness_score"] or 0)
            else:
                idx = dates.index(date)
                # Keep the daily maximum score so repeat logs are visible without implying clinical severity.
                itch[idx] = max(itch[idx], row["itch_score"] or 0)
                redness[idx] = max(redness[idx], row["redness_score"] or 0)
                dryness[idx] = max(dryness[idx], row["dryness_score"] or 0)
        label = row["predicted_label"] or "No prediction"
        counts[label] = counts.get(label, 0) + 1
    return render_template("dashboard.html", dates=dates, itch=itch, redness=redness,
                           dryness=dryness, counts=counts, total=len(rows))

@app.route("/care-guide")
def care_guide():
    return render_template("care_guide.html")

@app.route("/history/delete/<int:entry_id>", methods=["POST"])
def delete_history_entry(entry_id):
    with get_db() as conn:
        row = conn.execute("SELECT image_filename FROM skin_history WHERE id=?", (entry_id,)).fetchone()
        if row:
            conn.execute("DELETE FROM skin_history WHERE id=?", (entry_id,))
            try:
                (UPLOADS / row["image_filename"]).unlink(missing_ok=True)
            except OSError:
                pass
    return __import__("flask").redirect("/history")

@app.route("/history/export")
def export_history():
    import csv
    from io import StringIO
    from flask import Response
    with get_db() as conn:
        rows = conn.execute("""SELECT created_at, symptoms, itch_score, redness_score,
            dryness_score, notes, predicted_label, model_score FROM skin_history ORDER BY id DESC""").fetchall()
    buffer = StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["created_at", "symptoms", "itch_score_0_5", "redness_score_0_5",
                     "dryness_score_0_5", "notes", "model_dataset_label", "model_score_percent"])
    for row in rows:
        writer.writerow(list(row))
    return Response(buffer.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=skin_history_export.csv"})

@app.route("/health")
def health():
    return {"status": "ok", "lesion_model_loaded": model is not None,
            "skin_type_model_loaded": skin_type_model is not None,
            "lesion_classes": class_names, "skin_type_classes": skin_type_names}

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
