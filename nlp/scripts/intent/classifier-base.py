import pickle

from pathlib import Path


# =========================
# LOAD MODEL
# =========================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "intent"
    / "base_intent_model(w+char).pkl"
)

with open(MODEL_PATH, "rb") as file:
    model = pickle.load(file)


# =========================
# CLASSIFY
# =========================

def classify(text):

    prediction = model.predict([text])[0]

    probabilities = model.predict_proba([text])[0]

    confidence = max(probabilities)

    return prediction, confidence


# =========================
# TEST
# =========================

if __name__ == "__main__":

    while True:

        text = input("Command: ")

        if text.lower() == "exit":
            break

        intent, confidence = classify(text)

        print(f"Intent: {intent}")
        print(f"Confidence: {confidence:.2f}")
