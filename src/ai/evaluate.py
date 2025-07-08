from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split

def evaluate():
    # ── Load dataset ─────────────────────────────
    X = np.load("data/log/X.npy")
    y = np.load("data/log/y.npy")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ── Load model ───────────────────────────────
    model = load_model("data/model/simple_model.keras")
    y_pred = model.predict(X_test).flatten()

    # ── Reconstruct predicted and actual Close prices ───────
    last_closes = X_test[:, -1]  # % change of last known price in each window
    prev_prices = np.ones_like(last_closes)
    for i in range(X_test.shape[1]):
        prev_prices *= (1 + X_test[:, i])  # simulate price chain

    true_prices = prev_prices * (1 + y_test)
    predicted_prices = prev_prices * (1 + y_pred)

    # ── Plot comparison ──────────────────────────
    plt.figure(figsize=(12, 6))
    plt.plot(true_prices, label="Actual Close", linewidth=1.5)
    plt.plot(predicted_prices, label="Predicted Close", linestyle="--", linewidth=1.5)
    plt.title("Predicted vs Actual Close Prices (Test Set)")
    plt.xlabel("Sample Index")
    plt.ylabel("Close Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save to disk
    log_dir = Path("data/log")
    log_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(log_dir / "close_price_comparison.png")
    plt.show()

if __name__ == "__main__":
    evaluate()
