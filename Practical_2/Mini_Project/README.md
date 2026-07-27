# Binary Pattern Classifier

A small project that trains a single-layer perceptron on a logic gate
(AND or OR) and serves it through a Streamlit app. You pick two binary
inputs, hit Predict, and it shows you the predicted class along with
the raw sigmoid output and the weights/bias the model learned.

It's a follow-up to the perceptron exercise from the lab notes - same
idea (`Linear` -> `Sigmoid`), just wrapped in a UI and saved to disk so
it doesn't need to be retrained every time.

## How it works

The model is just:

```
Linear(in_features=2, out_features=1) -> Sigmoid
```

It's trained with binary cross-entropy loss on the 4 rows of a truth
table. AND and OR are both linearly separable, so a single perceptron
is enough to learn them - no hidden layers needed. (XOR would not
work here, that needs at least one extra layer.)

## Project structure

```
binary_pattern_classifier/
├── model.py         # Perceptron class (Linear + Sigmoid)
├── train.py         # trains the model on the chosen gate, saves weights
├── app.py            # Streamlit interface
├── requirements.txt
└── README.md
```

## Running it locally (conda + Spyder)

1. Create and activate an environment:

```bash
conda create -n perceptron python=3.10
conda activate perceptron
```

2. Install the dependencies:

```bash
pip install -r requirements.txt
```

3. Train the model. You can do this from Spyder (open `train.py` and
   hit run) or from a terminal:

```bash
python train.py
```

This trains on the AND gate by default and saves the weights to
`perceptron_model.pth` in the same folder. To train on OR instead,
open `train.py` and change:

```python
GATE = "AND"   # change to "OR"
```

then run it again.

4. Launch the app:

```bash
streamlit run app.py
```

It should open in your browser at `http://localhost:8501`. If the app
tells you no model was found, that just means step 3 hasn't been run
yet.

## Deploying to Streamlit Community Cloud

1. Push this folder to a GitHub repo (make sure `perceptron_model.pth`
   is committed too, otherwise the deployed app has nothing to load -
   there's no training step on the cloud).
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with
   GitHub, and click "New app".
3. Point it at your repo, branch, and set the main file path to
   `app.py`.
4. Deploy. Streamlit Cloud will install everything from
   `requirements.txt` automatically.

## Notes

- The model file (`perceptron_model.pth`) stores both the learned
  weights and which gate it was trained on, so the app can display
  that in the UI.
- Since AND/OR are linearly separable, training converges pretty
  fast - loss drops close to zero well before 2000 epochs, that
  number is just there to be safe.
