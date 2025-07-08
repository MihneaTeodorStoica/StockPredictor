from ai.dataset import load_dataset
from ai.model import build_model
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from pathlib import Path

def main():
    X_train, X_test, y_train, y_test = load_dataset()

    model = build_model(input_shape=X_train.shape[1])
    print("Training model...")

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=100,
        batch_size=32,
        callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
        verbose=1
    )

    # ── Ensure model output directory exists ───────────────
    model_path = Path("data/model/simple_model.keras")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Save model ──────────────────────────────────────────
    model.save(model_path)

    # Plot loss
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training Curve")
    plt.legend()
    plt.tight_layout()
    log_dir = Path("data/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(log_dir / "training_loss.png")
    plt.show()

    print("Model saved to data/model/simple_model.keras")

if __name__ == "__main__":
    main()
