# MNIST Digit Classification with a Feedforward Neural Network

## Task
Build a feedforward neural network with Keras to classify handwritten MNIST digits.

## Project structure

```text
MNIST_Keras_Feedforward_Neural_Network/
├── train.py
├── requirements.txt
├── README.md
├── .gitignore
├── models/
│   └── (trained model is saved here)
└── outputs/
    ├── training_history.png
    └── sample_predictions.png
```

## Requirements
- Python 3.9–3.11 recommended
- TensorFlow / Keras
- NumPy
- Matplotlib

## Setup (Windows)

Open Command Prompt in this project folder and run:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Run the project

```bash
python train.py
```

The first run downloads the MNIST dataset through Keras. An internet connection is required for this download.

## What the code does
1. Imports TensorFlow/Keras and loads the MNIST dataset.
2. Normalizes image pixel values from 0–255 to 0–1.
3. Builds a Sequential feedforward network using Flatten and Dense layers.
4. Compiles the model with Adam, sparse categorical cross-entropy, and accuracy.
5. Trains for 5 epochs with a validation split.
6. Evaluates the model on the test set.
7. Saves the trained model and plots of training history and sample predictions.

## Generated files
After a successful run:
- `models/mnist_feedforward.keras` — trained Keras model
- `outputs/training_history.png` — accuracy and loss curves
- `outputs/sample_predictions.png` — example predictions

These generated files are excluded from Git by default. If your assignment requires the trained model or plots in the GitHub submission, remove the relevant entries from `.gitignore` before committing.

## GitHub upload

Create a public GitHub repository, then upload this project folder. Include `train.py`, `requirements.txt`, `README.md`, and `.gitignore`.
