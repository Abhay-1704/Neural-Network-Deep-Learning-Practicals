"""
Streamlit front end for the trained perceptron.

Loads the weights saved by train.py and lets the user pick two binary
inputs to see what the model predicts, along with the raw sigmoid
output and the weights/bias that were learned during training.
"""

import os

import streamlit as st
import torch

from model import Perceptron

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "perceptron_model.pth")

st.set_page_config(page_title="Binary Pattern Classifier", page_icon="🔲")

st.title("Binary Pattern Classifier")
st.write(
    "A single-layer perceptron trained on a logic gate truth table. "
    "Pick two binary inputs below and the model will predict the gate's output."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        "No trained model found. Run `python train.py` first, then restart the app."
    )
    st.stop()

checkpoint = torch.load(MODEL_PATH, map_location="cpu")

model = Perceptron(in_features=2)
model.load_state_dict(checkpoint["state_dict"])
model.eval()

gate_name = checkpoint.get("gate", "unknown")
st.caption(f"Model trained on the **{gate_name}** gate.")

col1, col2 = st.columns(2)
with col1:
    x1 = st.selectbox("Input 1", options=[0, 1])
with col2:
    x2 = st.selectbox("Input 2", options=[0, 1])

if st.button("Predict"):
    x = torch.tensor([[float(x1), float(x2)]])

    with torch.no_grad():
        prob = model(x).item()

    predicted_class = 1 if prob >= 0.5 else 0

    st.subheader("Result")
    st.write(f"Predicted class: **{predicted_class}**")
    st.write(f"Raw sigmoid probability: **{prob:.4f}**")

    weight = model.linear.weight.detach().numpy().flatten()
    bias = model.linear.bias.detach().item()

    st.subheader("Learned parameters")
    st.write(f"w1 = {weight[0]:.4f}")
    st.write(f"w2 = {weight[1]:.4f}")
    st.write(f"bias = {bias:.4f}")