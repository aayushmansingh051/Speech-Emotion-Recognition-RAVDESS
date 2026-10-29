# Speech Emotion Recognition Using RAVDESS

## Overview

This project implements a Speech Emotion Recognition (SER) system using machine learning and the RAVDESS emotional speech dataset.

The system extracts acoustic features from speech recordings and uses a Support Vector Machine (SVM) to classify emotions.

## Objectives

* Extract MFCC features from audio recordings.
* Train an SVM classifier for emotion recognition.
* Evaluate the model using accuracy, precision, recall, and F1-score.
* Visualize results using a confusion matrix and performance graphs.
* Explore unsupervised learning using K-Means clustering.

## Dataset

**RAVDESS — Emotional Speech Audio**

Dataset source: https://www.kaggle.com/datasets/uwrfkaggler/ravdess-emotional-speech-audio

This project uses 1,440 audio-only speech recordings across eight emotion classes:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

## Technologies

* Python
* NumPy
* Pandas
* Librosa
* Matplotlib
* Scikit-learn
* Jupyter Notebook

## Methodology

1. Load the RAVDESS speech recordings.
2. Extract 13 MFCC coefficients from each recording.
3. Calculate the mean and standard deviation of each coefficient, producing 26 features.
4. Split the dataset by actor into training and testing sets.
5. Train an SVM classifier.
6. Evaluate the model using classification metrics and a confusion matrix.
7. Apply K-Means clustering to explore acoustic feature similarities.

## Results

The initial SVM experiment achieved **39.17% test accuracy** on 360 held-out recordings.

This is a baseline result. Further experiments can investigate additional acoustic features and model tuning.

## Visualizations

The project generates:

* Confusion matrix
* Classification metrics graph
* Emotion distribution graph
* Sample audio waveform
* K-Means clustering visualization

These images are saved in the `outputs` directory when the Python script runs.

## How to Run

Install the dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn librosa soundfile
```

Download the dataset and update the dataset path in the Python script if necessary.

Run the project:

```bash
python ravdess_speech_emotion_analysis.py
```

## Future Improvements

* Tune SVM hyperparameters.
* Test additional acoustic features.
* Compare SVM with Random Forest, KNN, and neural networks.
* Investigate misclassifications using the confusion matrix.
* Evaluate improvements using a consistent actor-independent test split.

## Author

**Aayushman Singh**

Speech and Video Processing Assignment
