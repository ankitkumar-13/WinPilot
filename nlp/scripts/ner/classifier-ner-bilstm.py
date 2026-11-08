import torch
import torch.nn as nn
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "ner"
    / "ner_bilstm"
    / "ner_model.pt"
)


# ============================================================
# MODEL
# ============================================================

class NERModel(nn.Module):

    def __init__(self, voc, tag):
        super().__init__()

        self.emb = nn.Embedding(
            voc,
            100,
            padding_idx=0
        )

        self.lstm = nn.LSTM(
            input_size=100,
            hidden_size=128,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(
            256,
            tag
        )

    def forward(self, x):

        x = self.emb(x)
        x, _ = self.lstm(x)

        return self.fc(x)


# ============================================================
# LOAD MODEL
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location="cpu"
)

word2id = checkpoint["word2id"]
tag2id = checkpoint["tag2id"]

id2tag = {
    v: k
    for k, v in tag2id.items()
}

model = NERModel(
    len(word2id),
    len(tag2id)
)

model.load_state_dict(
    checkpoint["model"]
)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

def predict(text):

    words = text.split()

    ids = [
        word2id.get(
            word.lower(),
            word2id["<UNK>"]
        )
        for word in words
    ]

    x = torch.tensor(
        ids,
        dtype=torch.long
    ).unsqueeze(0)

    with torch.no_grad():

        output = model(x)

        pred = output.argmax(
            dim=2
        )[0].tolist()

    tags = [
        id2tag[tag]
        for tag in pred
    ]

    return words, tags


# ============================================================
# ENTITY EXTRACTION
# ============================================================

def extract_entities(words, tags):

    entities = []

    current = []
    current_type = None

    for word, tag in zip(words, tags):

        if tag.startswith("B-"):

            if current:
                entities.append(
                    (
                        current_type,
                        " ".join(current)
                    )
                )

            current_type = tag[2:]
            current = [word]

        elif tag.startswith("I-"):

            typ = tag[2:]

            if current_type == typ:
                current.append(word)

            else:

                if current:
                    entities.append(
                        (
                            current_type,
                            " ".join(current)
                        )
                    )

                current_type = typ
                current = [word]

        else:

            if current:

                entities.append(
                    (
                        current_type,
                        " ".join(current)
                    )
                )

                current = []
                current_type = None

    if current:

        entities.append(
            (
                current_type,
                " ".join(current)
            )
        )

    return entities


# ============================================================
# DISPLAY
# ============================================================

def show_result(words, tags, entities):

    print()
    print("Tokens:")
    print("-" * 35)

    for word, tag in zip(words, tags):

        print(
            f"{word:<20} {tag}"
        )

    print()
    print("Entities:")
    print("-" * 35)

    if entities:

        for typ, value in entities:

            print(
                f"{typ}: {value}"
            )

    else:

        print("None")

    print()


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("WinPilot NER Classifier")
    print("Type 'exit' to quit.")
    print()

    while True:

        text = input("Command: ").strip()

        if not text:
            continue

        if text.lower() == "exit":
            break

        words, tags = predict(text)

        entities = extract_entities(
            words,
            tags
        )

        show_result(
            words,
            tags,
            entities
        )
