"""
Task 3: Build a feedforward neural network with Keras to classify MNIST digits.

Run:
    python -m pip install -r requirements.txt
    python train.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "outputs"
MODEL_DIR = ROOT / "models"
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# Make results more reproducible.
keras.utils.set_random_seed(42)


def load_and_preprocess_data():
    """Load MNIST and normalize pixel values to the range [0, 1]."""
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    return (x_train, y_train), (x_test, y_test)


def build_model():
    """Create a simple feedforward neural network using Dense layers."""
    model = keras.Sequential(
        [
            keras.Input(shape=(28, 28)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(64, activation="relu"),
            layers.Dense(10, activation="softmax"),
        ],
        name="mnist_feedforward_network",
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_training_plot(history):
    """Save training and validation accuracy/loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Training accuracy")
    axes[0].plot(history.history["val_accuracy"], label="Validation accuracy")
    axes[0].set_title("Model Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.history["loss"], label="Training loss")
    axes[1].plot(history.history["val_loss"], label="Validation loss")
    axes[1].set_title("Model Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def save_sample_predictions(model, x_test, y_test, count=10):
    """Plot a small set of test images with predicted and actual labels."""
    predictions = model.predict(x_test[:count], verbose=0)
    predicted_labels = np.argmax(predictions, axis=1)

    fig, axes = plt.subplots(2, 5, figsize=(12, 5))
    for index, ax in enumerate(axes.flat):
        ax.imshow(x_test[index], cmap="gray")
        color = "green" if predicted_labels[index] == y_test[index] else "red"
        ax.set_title(
            f"Pred: {predicted_labels[index]} | True: {y_test[index]}",
            color=color,
        )
        ax.axis("off")

    fig.suptitle("MNIST Sample Predictions")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "sample_predictions.png", dpi=150)
    plt.close(fig)


def main():
    print("Loading MNIST dataset...")
    (x_train, y_train), (x_test, y_test) = load_and_preprocess_data()

    print(f"Training images: {x_train.shape}")
    print(f"Test images: {x_test.shape}")

    model = build_model()
    model.summary()

    print("\nTraining model...")
    history = model.fit(
        x_train,
        y_train,
        epochs=5,
        batch_size=32,
        validation_split=0.1,
        verbose=1,
    )

    print("\nEvaluating model on test data...")
    test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4%}")

    model_path = MODEL_DIR / "mnist_feedforward.keras"
    model.save(model_path)
    print(f"Saved model to: {model_path}")

    save_training_plot(history)
    save_sample_predictions(model, x_test, y_test)

    print(f"Saved plots in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
