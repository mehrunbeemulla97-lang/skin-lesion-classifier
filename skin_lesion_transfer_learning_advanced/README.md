# Skin Lesion Classification Using Transfer Learning
**Student:** MEHRUNBEE DASTGEER MULLA  
**Register No.:** 8080120

An educational research prototype using TensorFlow/Keras transfer learning and Flask. It is **not a medical device and must not be used for diagnosis or treatment decisions**.

## Features
- MobileNetV2 transfer learning with optional fine-tuning
- HAM10000 seven-category classification pipeline
- Image preprocessing and training-time augmentation
- Patient/lesion-aware grouped train/validation/test split where metadata permits
- Class-wise precision, recall, F1-score, classification report and confusion matrix
- Training curves and saved model
- Flask image-upload interface with class scores and an always-visible disclaimer
- Clear handling of the untrained-model state
- Optional separate skin-type model endpoint (requires a suitable labelled dataset)
- Rash/allergy educational guide; deliberately no allergy diagnosis from photos

## 1. Requirements
Recommended: Python 3.10 or 3.11, 8 GB RAM or more. TensorFlow installation and training can take time. A compatible GPU is optional; CPU training is slower.

Open PowerShell in this folder:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation, use:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## 2. Obtain the dataset
This project does **not** bundle the dataset because it is large and has separate licence/attribution conditions.

Dataset: **HAM10000**, 10,015 dermatoscopic images across seven categories. Read the terms before downloading:
- ISIC dataset page: https://challenge.isic-archive.com/data/
- Harvard Dataverse record: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

The ISIC 2018 aggregate HAM10000 data is marked **CC BY-NC** (non-commercial). Use it only in line with the dataset's current terms, include attribution in reports, and do not use it commercially without permission. Dataset citation:
Tschandl, P., Rosendahl, C. & Kittler, H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. *Scientific Data* 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161

Download the image archives and metadata CSV from the official source. Extract the images into:
```
data/
  images/
    <image_id>.jpg
    ...
  HAM10000_metadata.csv
```
Some downloads contain two image folders. Copy all JPG files into `data/images/`. The CSV should contain `image_id`, `dx`, and ideally `lesion_id` columns. Do not rename image IDs.

## 3. Train the model
```powershell
python train.py --metadata data/HAM10000_metadata.csv --images data/images --epochs 8 --fine-tune-epochs 3
```
Training saves:
- `artifacts/skin_lesion_model.keras`
- `artifacts/class_names.json`
- `artifacts/evaluation.json`
- `artifacts/classification_report.txt`
- `artifacts/confusion_matrix.png`
- `artifacts/training_curves.png`

A grouped split uses `lesion_id` when available to reduce the chance that images of the same lesion leak across partitions. The held-out test set is used for final evaluation only. Results depend on data setup, random seed, hardware, and training settings; do not claim accuracy until you have run the evaluation.

## 4. Optional skin-type module (requires separate data)
The `/skin-type` page is intentionally disabled until a separate dataset is supplied and trained. HAM10000 does **not** label oily, dry, normal or combination skin.

To train this optional module, obtain a dataset whose licence permits your intended academic use and whose labels genuinely represent all four skin-type classes. Review its label definitions, image collection conditions, consent, demographics, and split strategy. Do not invent labels or treat an image-based model as a reliable personal skin-type test.

Place the dataset in this structure:
```
data/skin_types/
  dry/          # labelled images
  oily/
  normal/
  combination/
```
Then run:
```powershell
python train_skin_type.py --data data/skin_types --epochs 8
```
The script requires all four class folders and stops if any are missing. It creates a separate model and held-out dataset report. This is a research demonstration, not a clinically validated assessment. Dataset splits are image-level; if multiple images per person are present, use a person-grouped split before relying on results.

## 5. Rash and allergy educational guide
Open `/rash-guide` from the website. This module provides general safety information and intentionally does not predict allergy. The current HAM10000 labels are lesion categories, not allergy labels. A rash can have many causes, and a photo alone cannot establish the cause. This page must not be used to choose treatment.

## 6. Run the website
After training completes:
```powershell
python app.py
```
Open http://127.0.0.1:5000 in your browser. Upload a JPG, JPEG or PNG image. The app displays the model's top dataset category and scores, not a diagnosis.

## 7. Seven lesion dataset labels
- `akiec` — actinic keratoses and intraepithelial carcinoma
- `bcc` — basal cell carcinoma
- `bkl` — benign keratosis-like lesions
- `df` — dermatofibroma
- `mel` — melanoma
- `nv` — melanocytic nevi
- `vasc` — vascular lesions

These are dataset label descriptions only. The model may be wrong, and confidence scores are not calibrated probabilities of disease.

## Project structure
```
skin_lesion_transfer_learning_advanced/
├── app.py
├── train.py
├── train_skin_type.py
├── evaluate.py
├── requirements.txt
├── README.md
├── DATASET_ATTRIBUTION.md
├── .gitignore
├── templates/
│   ├── index.html
│   ├── skin_type.html
│   └── rash_guide.html
├── static/
│   └── style.css
├── data/
│   └── images/             # add downloaded dataset here
└── artifacts/              # generated after training
```

## Troubleshooting
- **Model not found:** run `train.py` successfully before using predictions.
- **Missing images:** ensure the CSV's `image_id` values match JPG filenames.
- **Out of memory:** lower `BATCH_SIZE` in `train.py` from 16 to 8, or reduce `IMG_SIZE` from 224 to 160.
- **TensorFlow install failure:** use Python 3.10/3.11 in a fresh virtual environment.
- **Slow training:** start with fewer epochs; training on CPU may take a long time.

## Limitations
This is a student research prototype. HAM10000 is a curated dermatoscopic dataset and does not represent every camera, skin tone, clinical setting, or population. Dataset imbalance and possible domain shift affect results. Do not upload personal or identifiable medical photos. Seek a qualified healthcare professional for real concerns.


## New in v3: photo journal, analytics, and care education
- `/history`: chronological image and symptom log, entry deletion, and CSV export.
- `/dashboard`: graphs of self-reported itch/redness/dryness scores and counts of saved model labels.
- `/care-guide`: general skincare safety information; no personalized cream/tablet prescriptions.
- Main upload form includes optional symptom notes and 0–5 self-reported scores.

The SQLite history database is created at `artifacts/skin_history.sqlite3`; images are stored in `uploads/`. These are local demo files, not encrypted health-record storage. The demo has no authentication, so do not expose it publicly or store identifiable/private medical images. Keep the project on a trusted local computer. Deleting a history entry removes its image, but exported CSV files and backups must be managed separately.

Graphs describe user-entered logs and model outputs only. They do not show disease progression, medical severity, or treatment efficacy. A before/after photo comparison is for personal observation only, not proof of improvement.
