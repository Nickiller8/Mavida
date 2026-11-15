import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from google.colab import files

uploaded = files.upload()

df = pd.read_csv("dataset.csv")

print(df.head())
print("\nDataset size:", df.shape)
print("\nSentences per language:")
print(df["Language"].value_counts())

df = df.dropna(subset=["Text", "Language"])

df["Text"] = df["Text"].astype(str).str.strip()
df = df[df["Text"] != ""]

df = df.drop_duplicates(subset=["Text", "Language"])

X = df["Text"]
y = df["Language"]

print("Total sentences:", len(X))
print("Languages:", y.unique())

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training sentences:", len(X_train))
print("Testing sentences:", len(X_test))

pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3)
    )),

    ("classifier", LogisticRegression(
        max_iter=1000
    ))
])

pipeline.fit(X_train, y_train)

print("Model training complete!")

predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model accuracy:", round(accuracy * 100, 2), "%")

print("\nClassification Report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

cm = confusion_matrix(
    y_test,
    predictions,
    labels=pipeline.classes_
)

plt.figure(figsize=(10, 7))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=pipeline.classes_,
    yticklabels=pipeline.classes_
)

plt.xlabel("Predicted Language")
plt.ylabel("Actual Language")
plt.title("Language Identification Confusion Matrix")

plt.show()

!pip install langdetect

from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42

code_mapping = {
    "en": "English",
    "fr": "French",
    "pt": "Portuguese",
    "sn": "Shona",
    "es": "Spanish",
    "zh-cn": "Mandarin Chinese",
    "zh-tw": "Mandarin Chinese"
}

langdetect_predictions = []

for sentence in X_test:

    try:
        detected = detect(sentence)
        detected = code_mapping.get(detected, detected)

        langdetect_predictions.append(detected)

    except:
        langdetect_predictions.append("Unknown")


langdetect_accuracy = accuracy_score(
    y_test,
    langdetect_predictions
)

print("Our model:", round(accuracy * 100, 2), "%")
print("Langdetect:", round(langdetect_accuracy * 100, 2), "%")

comparison = pd.DataFrame({
    "Model": ["Our Model", "Langdetect"],
    "Accuracy": [
        accuracy * 100,
        langdetect_accuracy * 100
    ]
})

print(comparison)

sentence = input("Enter a sentence: ")

prediction = pipeline.predict([sentence])

print("Detected language:", prediction[0])

joblib.dump(pipeline, "language_model.pkl")

print("Model saved successfully!")
