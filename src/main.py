from core.extract_data import extract_archive
from core.analyze_data import select_random_csvs
from core.load_data import load_data
from core.plot_close_price import plot_close_prices
from core.prepare_data import prepare_training_data

from ai.train import main as train_model

def main():
    # ── Step 1: Extract archive ───────────────────────────────
    extract_archive()

    # ── Step 2: Select + load CSVs ─────────────────────────────
    df_path_list = select_random_csvs(1000)
    df_list = load_data(df_path_list)

    # ── Step 3: Plot Close prices ──────────────────────────────
    plot_close_prices(df_list)

    # ── Step 4: Prepare training data ──────────────────────────
    X, y = prepare_training_data(df_list, window_size=10)
    print(f"Prepared training data: X={X.shape}, y={y.shape}")
    if X.size == 0 or y.size == 0:
        print("No valid training data found. Exiting.")
        return

    # ── Step 5: Train model ────────────────────────────────────
    train_model()

if __name__ == "__main__":
    main()
