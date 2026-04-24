#!/usr/bin/env python3
"""
Conversational Bot — Vanilla RNN (NumPy) + NLTK preprocessing.
No TensorFlow/PyTorch required.

Usage:
    python chatbot.py           # train on first run, then chat
    python chatbot.py --retrain # force retrain, then chat
"""

import json
import os
import pickle
import random
import sys
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder

for resource in ("punkt", "punkt_tab", "wordnet"):
    nltk.download(resource, quiet=True)


# ── Configuration ─────────────────────────────────────────────────────────────
HIDDEN_SIZE          = 64
LEARNING_RATE        = 0.005
EPOCHS               = 600
CONFIDENCE_THRESHOLD = 0.60      # BUG 1 FIX: was 0.95 — impossibly high threshold
                                  # meant the bot always fell back to "I don't understand"
INTENTS_FILE         = "intents.json"
MODEL_DIR            = "model_artifacts"
# ─────────────────────────────────────────────────────────────────────────────

lemmatizer = WordNetLemmatizer()


# ── Text Preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str) -> list:
    """Tokenise → lowercase → lemmatise; drop non-alpha tokens."""
    tokens = nltk.word_tokenize(text.lower())
    return [lemmatizer.lemmatize(tok) for tok in tokens if tok.isalpha()]
    # BUG 2 FIX: was tok.isdigit() — kept only digit tokens (numbers), dropping
    # every alphabetic word. Vocabulary was always empty → crash on first run.


def build_vocabulary(intents: dict) -> dict:
    """Build word → index mapping from all training patterns."""
    vocab = set()
    for intent in intents["intents"]:
        for pattern in intent["patterns"]:
            vocab.update(preprocess(pattern))
    return {word: idx for idx, word in enumerate(sorted(vocab))}


def tokens_to_one_hot(tokens: list, vocab: dict) -> list:
    """Convert token list into a list of one-hot column vectors."""
    size = len(vocab)
    vectors = [
        _one_hot(vocab[tok], size)
        for tok in tokens if tok in vocab
    ]
    return vectors if vectors else [np.zeros((size, 1))]


def _one_hot(idx: int, size: int) -> np.ndarray:
    vec = np.zeros((size, 1))
    vec[idx] = 1.0               # BUG 3 FIX: was 0.0 — set the "hot" position to
                                  # zero, making every one-hot vector all-zeros.
                                  # The model received no input signal at all.
    return vec


# ── Vanilla RNN ───────────────────────────────────────────────────────────────

class VanillaRNN:
    """
    Single-layer RNN with tanh activation.
    Forward pass:  h_t = tanh(Wxh·x_t + Whh·h_{t-1} + bh)
    Output:        y   = softmax(Why·h_T + by)
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int, lr: float):
        self.lr = lr
        self.hidden_size = hidden_size

        # Xavier initialisation
        self.Wxh = np.random.randn(hidden_size, input_size)  * np.sqrt(2.0 / input_size)
        self.Whh = np.random.randn(hidden_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.Why = np.random.randn(output_size, hidden_size) * np.sqrt(2.0 / hidden_size)
        self.bh  = np.zeros((hidden_size, 1))
        self.by  = np.zeros((output_size, 1))

    def forward(self, inputs: list) -> tuple:
        h = np.zeros((self.hidden_size, 1))
        self._inputs = inputs
        self._hs = {0: h.copy()}

        for t, x in enumerate(inputs):
            h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
            self._hs[t + 1] = h.copy()

        logits = self.Why @ h + self.by
        probs  = _softmax(logits)
        return probs

    def backward(self, probs: np.ndarray, target_idx: int) -> float:
        n = len(self._inputs)

        d_logits = probs.copy()
        d_logits[target_idx] -= 1.0                  # cross-entropy gradient

        d_Why = d_logits @ self._hs[n].T
        d_by  = d_logits.copy()
        d_Whh = np.zeros_like(self.Whh)
        d_Wxh = np.zeros_like(self.Wxh)
        d_bh  = np.zeros_like(self.bh)

        d_h = self.Why.T @ d_logits

        for t in reversed(range(n)):
            dtanh  = (1.0 - self._hs[t + 1] ** 2) * d_h   # BUG 4 FIX: was (1.0 + h²)
                                                             # Correct tanh derivative is
                                                             # (1 - tanh²), not (1 + tanh²).
                                                             # Wrong sign caused gradients to
                                                             # explode/diverge; loss never fell.
            d_bh  += dtanh
            d_Wxh += dtanh @ self._inputs[t].T
            d_Whh += dtanh @ self._hs[t].T
            d_h    = self.Whh.T @ dtanh

        # Gradient clipping — prevents exploding gradients on small datasets
        for grad in (d_Wxh, d_Whh, d_Why, d_bh, d_by):
            np.clip(grad, -5, 5, out=grad)

        self.Wxh -= self.lr * d_Wxh
        self.Whh -= self.lr * d_Whh
        self.Why -= self.lr * d_Why
        self.bh  -= self.lr * d_bh
        self.by  -= self.lr * d_by

        return float(-np.log(probs[target_idx, 0] + 1e-8))

    def predict(self, inputs: list) -> np.ndarray:
        return self.forward(inputs)

    def save(self, path: str) -> None:
        np.savez(
            path,
            Wxh=self.Wxh, Whh=self.Whh, Why=self.Why,
            bh=self.bh,   by=self.by,
            meta=np.array([self.hidden_size, self.lr]),
        )

    @classmethod
    def load(cls, path: str) -> "VanillaRNN":
        d      = np.load(path)
        hs     = int(d["meta"][0])
        lr     = float(d["meta"][1])
        rnn    = cls(d["Wxh"].shape[1], hs, d["Why"].shape[0], lr)
        rnn.Wxh, rnn.Whh, rnn.Why = d["Wxh"], d["Whh"], d["Why"]
        rnn.bh,  rnn.by            = d["bh"],  d["by"]
        return rnn


def _softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - x.max())   # BUG 5 FIX: subtract max for numerical stability.
                                  # Without this, large logits cause np.exp() to
                                  # overflow to inf → nan probabilities → silent
                                  # training failure / crashes at inference.
    return e_x / e_x.sum()


# ── Training ──────────────────────────────────────────────────────────────────

def prepare_training_data(intents: dict, vocab: dict, encoder: LabelEncoder) -> list:
    data = []
    classes = list(encoder.classes_)
    for intent in intents["intents"]:
        label_idx = classes.index(intent["tag"])
        for pattern in intent["patterns"]:
            tokens  = preprocess(pattern)
            vectors = tokens_to_one_hot(tokens, vocab)
            data.append((vectors, label_idx))
    return data


def train_and_save(intents: dict) -> tuple:
    print("\n  Building vocabulary …")
    vocab   = build_vocabulary(intents)
    tags    = [intent["tag"] for intent in intents["intents"]]
    encoder = LabelEncoder()
    encoder.fit(tags)

    training_data = prepare_training_data(intents, vocab, encoder)

    vocab_size  = len(vocab)
    num_classes = len(encoder.classes_)

    print(f"  Vocabulary size : {vocab_size}")
    print(f"  Intent classes  : {num_classes}")
    print(f"  Training samples: {len(training_data)}")
    print(f"\n  Training RNN for up to {EPOCHS} epochs …\n")

    rnn = VanillaRNN(vocab_size, HIDDEN_SIZE, num_classes, LEARNING_RATE)

    for epoch in range(1, EPOCHS + 1):
        random.shuffle(training_data)
        total_loss, correct = 0.0, 0

        for vectors, label_idx in training_data:
            probs      = rnn.forward(vectors)
            loss       = rnn.backward(probs, label_idx)
            total_loss += loss
            if int(np.argmax(probs)) == label_idx:
                correct += 1

        avg_loss = total_loss / len(training_data)
        accuracy = correct / len(training_data)

        if epoch % 100 == 0:
            bar = "█" * int(accuracy * 20) + "░" * (20 - int(accuracy * 20))
            print(f"  Epoch {epoch:4d}/{EPOCHS}  [{bar}]  loss={avg_loss:.4f}  acc={accuracy:.2%}")

        if avg_loss < 0.05:
            print(f"\n  Converged at epoch {epoch}  loss={avg_loss:.4f}  acc={accuracy:.2%}")
            break

    os.makedirs(MODEL_DIR, exist_ok=True)
    rnn.save(os.path.join(MODEL_DIR, "rnn_weights.npz"))
    with open(os.path.join(MODEL_DIR, "vocab.pkl"),   "wb") as f:
        pickle.dump(vocab, f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "wb") as f:
        pickle.dump(encoder, f)

    print(f"\n  Model artifacts saved to: {MODEL_DIR}/")
    return rnn, vocab, encoder


def load_artifacts() -> tuple:
    print("\n  Loading pre-trained model …")
    rnn = VanillaRNN.load(os.path.join(MODEL_DIR, "rnn_weights.npz"))
    with open(os.path.join(MODEL_DIR, "vocab.pkl"),   "rb") as f:
        vocab = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "encoder.pkl"), "rb") as f:
        encoder = pickle.load(f)
    print("  Model loaded successfully.")
    return rnn, vocab, encoder


def artifacts_exist() -> bool:
    return all(
        os.path.exists(os.path.join(MODEL_DIR, name))
        for name in ("rnn_weights.npz", "vocab.pkl", "encoder.pkl")
    )


# ── Inference ─────────────────────────────────────────────────────────────────

def predict_intent(text: str, rnn: VanillaRNN, vocab: dict, encoder: LabelEncoder) -> tuple:
    tokens  = preprocess(text)
    vectors = tokens_to_one_hot(tokens, vocab)
    probs   = rnn.predict(vectors)
    idx     = int(np.argmax(probs))
    return encoder.classes_[idx], float(probs[idx, 0])


def get_response(tag: str, intents: dict) -> str:
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return random.choice(intent["responses"])
    return "I'm not sure I understand. Could you rephrase that?"


# ── Chat Loop ─────────────────────────────────────────────────────────────────

def start_chat(intents: dict, rnn: VanillaRNN, vocab: dict, encoder: LabelEncoder) -> None:
    divider = "─" * 52
    print(f"\n{divider}")
    print("  PyBot is ready!  Type 'quit' or 'exit' to stop.")
    print(f"{divider}\n")

    while True:
        try:
            user_input = input("  You   : ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n  PyBot : Goodbye! Have a great day!\n")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "bye", "goodbye"}:
            print("  PyBot : Goodbye! It was great chatting with you!\n")
            break

        tag, confidence = predict_intent(user_input, rnn, vocab, encoder)

        if confidence < CONFIDENCE_THRESHOLD:
            response = "I'm not quite sure I understand. Could you rephrase that?"
        else:
            response = get_response(tag, intents)

        print(f"  PyBot : {response}\n")


# ── Entry Point ───────────────────────────────────────────────────────────────

def main() -> None:
    force_retrain = "--retrain" in sys.argv

    if not os.path.exists(INTENTS_FILE):
        print(f"  Error: '{INTENTS_FILE}' not found.")
        sys.exit(1)

    with open(INTENTS_FILE) as f:
        intents = json.load(f)

    if force_retrain or not artifacts_exist():
        rnn, vocab, encoder = train_and_save(intents)
    else:
        rnn, vocab, encoder = load_artifacts()

    start_chat(intents, rnn, vocab, encoder)


if __name__ == "__main__":
    main()
