"""
Speech Emotion Recognition using RAVDESS
Complete standalone notebook code:
- Loads saved SVM predictions if available; otherwise trains the SVM from the KaggleHub dataset.
- Generates confusion matrix and classification graphs.
- Saves all outputs in the SVP2/outputs folder.

Run in a Jupyter notebook cell. If predictions CSV is missing, this code will use the
KaggleHub dataset already downloaded in the user's cache.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
    accuracy_score,
    precision_recall_fscore_support,
)

# -------------------------------------------------------------------
# 1. Project paths
# -------------------------------------------------------------------
PROJECT_DIR = Path.cwd()
OUTPUT_DIR = PROJECT_DIR / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PREDICTIONS_FILE = OUTPUT_DIR / "svm_predictions.csv"
DATASET_PATH = Path(
    r"C:\Users\KIIT\.cache\kagglehub\datasets\uwrfkaggler\ravdess-emotional-speech-audio\versions\1"
)

EMOTION_LABELS = [
    "angry", "calm", "disgust", "fearful",
    "happy", "neutral", "sad", "surprised"
]

# -------------------------------------------------------------------
# 2. Obtain actual and predicted labels
# -------------------------------------------------------------------
if PREDICTIONS_FILE.exists():
    print("Loading actual model predictions from:", PREDICTIONS_FILE)
    results = pd.read_csv(PREDICTIONS_FILE)
    if not {"actual_emotion", "predicted_emotion"}.issubset(results.columns):
        raise ValueError(
            f"{PREDICTIONS_FILE} must contain 'actual_emotion' and 'predicted_emotion' columns."
        )
    y_true = results["actual_emotion"].astype(str)
    y_pred = results["predicted_emotion"].astype(str)

else:
    print("Prediction CSV not found. Training SVM from the downloaded RAVDESS dataset...")
    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}\n"
            "Check the KaggleHub download path or update DATASET_PATH."
        )

    import librosa
    from sklearn.model_selection import GroupShuffleSplit
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC

    emotion_map = {
        "01": "neutral", "02": "calm", "03": "happy", "04": "sad",
        "05": "angry", "06": "fearful", "07": "disgust", "08": "surprised"
    }

    wav_files = sorted(DATASET_PATH.glob("Actor_*/*.wav"))
    if not wav_files:
        wav_files = sorted(DATASET_PATH.rglob("*.wav"))

    if not wav_files:
        raise FileNotFoundError(f"No WAV files found under {DATASET_PATH}")

    feature_rows = []
    feature_vectors = []
    skipped = 0

    print(f"Found {len(wav_files)} WAV files. Extracting speech MFCC features...")

    for index, audio_path in enumerate(wav_files, start=1):
        parts = audio_path.stem.split("-")
        if len(parts) != 7:
            skipped += 1
            continue

        modality, channel, emotion_code, intensity, statement, repetition, actor = parts

        # RAVDESS code 03 = audio-only; 01 = speech channel.
        if modality != "03" or channel != "01" or emotion_code not in emotion_map:
            continue

        try:
            audio, sr = librosa.load(audio_path, sr=22050, mono=True)
            if audio.size == 0:
                skipped += 1
                continue
            mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
            features = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
            feature_vectors.append(features)
            feature_rows.append({
                "file": str(audio_path),
                "actual_emotion": emotion_map[emotion_code],
                "actor": actor
            })
        except Exception as exc:
            skipped += 1
            print(f"Skipping {audio_path.name}: {exc}")

        if index % 200 == 0 or index == len(wav_files):
            print(f"Processed {index}/{len(wav_files)} WAV files")

    if not feature_rows:
        raise RuntimeError("No usable speech recordings were found.")

    X = np.asarray(feature_vectors, dtype=np.float32)
    metadata = pd.DataFrame(feature_rows)
    y = metadata["actual_emotion"].to_numpy()
    groups = metadata["actor"].to_numpy()

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    model = make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced")
    )
    model.fit(X[train_idx], y[train_idx])
    y_true = y[test_idx]
    y_pred = model.predict(X[test_idx])

    results = pd.DataFrame({
        "actual_emotion": y_true,
        "predicted_emotion": y_pred
    })
    results.to_csv(PREDICTIONS_FILE, index=False)
    print("Saved model predictions to:", PREDICTIONS_FILE)

# -------------------------------------------------------------------
# 3. Draw and save confusion matrix
# -------------------------------------------------------------------
labels_present = [label for label in EMOTION_LABELS if label in set(y_true) | set(y_pred)]
cm = confusion_matrix(y_true, y_pred, labels=labels_present)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels_present)
disp.plot(ax=ax, cmap="Blues", values_format="d", xticks_rotation=45, colorbar=True)
ax.set_title("SVM Confusion Matrix - RAVDESS Speech Emotion Recognition")
ax.set_xlabel("Predicted Emotion")
ax.set_ylabel("Actual Emotion")
plt.tight_layout()
cm_path = OUTPUT_DIR / "confusion_matrix.png"
plt.savefig(cm_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# -------------------------------------------------------------------
# 4. Print and save classification report
# -------------------------------------------------------------------
report = classification_report(
    y_true, y_pred, labels=labels_present, zero_division=0, digits=4
)
accuracy = accuracy_score(y_true, y_pred)

print(f"\nAccuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
print("\nClassification report:\n")
print(report)

(OUTPUT_DIR / "classification_report_from_notebook.txt").write_text(
    f"Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n{report}",
    encoding="utf-8"
)

# -------------------------------------------------------------------
# 5. Draw precision, recall and F1-score graph
# -------------------------------------------------------------------
precision, recall, f1, support = precision_recall_fscore_support(
    y_true, y_pred, labels=labels_present, zero_division=0
)

metrics_df = pd.DataFrame({
    "emotion": labels_present,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "support": support
})
metrics_df.to_csv(OUTPUT_DIR / "per_class_metrics_from_notebook.csv", index=False)

positions = np.arange(len(labels_present))
bar_width = 0.25

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(positions - bar_width, precision, bar_width, label="Precision")
ax.bar(positions, recall, bar_width, label="Recall")
ax.bar(positions + bar_width, f1, bar_width, label="F1-score")
ax.set_xticks(positions)
ax.set_xticklabels(labels_present, rotation=25, ha="right")
ax.set_ylim(0, 1)
ax.set_ylabel("Score")
ax.set_title("SVM Per-Class Evaluation Metrics")
ax.legend()
plt.tight_layout()
metrics_path = OUTPUT_DIR / "classification_metrics_from_notebook.png"
plt.savefig(metrics_path, dpi=300, bbox_inches="tight")
plt.show()
plt.close(fig)

# -------------------------------------------------------------------
# 6. Print file locations
# -------------------------------------------------------------------
print("\nCompleted successfully. Files saved:")
print("Confusion matrix:", cm_path.resolve())
print("Metrics graph:", metrics_path.resolve())
print("Classification report:", (OUTPUT_DIR / "classification_report_from_notebook.txt").resolve())
print("Metrics CSV:", (OUTPUT_DIR / "per_class_metrics_from_notebook.csv").resolve())
