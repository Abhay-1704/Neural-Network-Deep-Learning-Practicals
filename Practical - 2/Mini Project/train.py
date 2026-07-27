"""
Trains a perceptron on a logic gate truth table and saves the
resulting weights so they can be reloaded later by the Streamlit app.

Run this once before starting the app:
    python train.py
"""

import torch
import torch.nn as nn
import torch.optim as optim

from model import Perceptron

# Which gate to learn. Only AND and OR are used here since both are
# linearly separable and a single perceptron can solve them. XOR is
# NOT linearly separable and would need a second layer, so it is left
# out on purpose.
GATE = "AND"  # change to "OR" to train the other gate

GATES = {
    "AND": [
        ([0, 0], 0),
        ([0, 1], 0),
        ([1, 0], 0),
        ([1, 1], 1),
    ],
    "OR": [
        ([0, 0], 0),
        ([0, 1], 1),
        ([1, 0], 1),
        ([1, 1], 1),
    ],
}

EPOCHS = 2000
LEARNING_RATE = 0.1
MODEL_PATH = "perceptron_model.pth"


def build_dataset(gate_name):
    rows = GATES[gate_name]
    inputs = torch.tensor([r[0] for r in rows], dtype=torch.float32)
    targets = torch.tensor([[r[1]] for r in rows], dtype=torch.float32)
    return inputs, targets


def train():
    inputs, targets = build_dataset(GATE)

    model = Perceptron(in_features=2)
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 500 == 0:
            print(f"epoch {epoch + 1}/{EPOCHS} - loss: {loss.item():.4f}")

    # quick sanity check on the four training rows
    with torch.no_grad():
        final_preds = model(inputs)
        print("\nFinal predictions on training data:")
        for x, y, p in zip(inputs, targets, final_preds):
            print(f"  input={x.tolist()} target={int(y.item())} "
                  f"predicted_prob={p.item():.4f}")

    weight = model.linear.weight.detach().numpy().flatten()
    bias = model.linear.bias.detach().item()
    print(f"\nLearned weights: w1={weight[0]:.4f}, w2={weight[1]:.4f}")
    print(f"Learned bias: b={bias:.4f}")

    torch.save({
        "state_dict": model.state_dict(),
        "gate": GATE,
    }, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")


if __name__ == "__main__":
    train()
