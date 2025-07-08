# 📈 StockPredictor

**StockPredictor** is an AI-powered stock forecasting and trading simulation system. It uses deep learning to predict the next-day percentage price change from historical time series, and simulates realistic trading strategies based on these predictions. All operations—from data extraction to evaluation—are fully automated.

---

## 🔍 Key Features

- 📂 **Automated Data Pipeline**: Extracts and processes thousands of real stock/ETF CSV files.
- 🧠 **Deep Learning Model**: Trains a neural network on historical price movements.
- 📊 **Trading Simulation**: Evaluates the model using a configurable strategy vs. buy-and-hold.
- 🧪 **Evaluation and Visualization**: Generates comparison plots and performance metrics.

---

## 🧠 AI Model

The model is a simple fully-connected neural network with the following architecture:
- Input: Historical percentage changes (10-day sliding window by default)
- 2× Dense layers with ReLU activations
- Output: Next-day percentage change

Training uses MSE loss with early stopping. Outputs are saved to:
- `data/model/simple_model.keras` (model)
- `data/log/training_loss.png` (loss curve)
- `data/log/close_price_comparison.png` (predicted vs actual)

---

## 🚀 Getting Started

### 1. Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Expected structure:
```
project/
├── ai/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
├── core/
│   └── [data loading utilities]
├── data/
│   ├── archive.zip           # zipped dataset (external)
│   └── extracted/            # contains CSVs for stocks/ETFs
├── main.py                   # full pipeline runner
├── simulate.py               # batch trading simulator
└── requirements.txt
```

---

### 2. Run Full Pipeline

```bash
python main.py
```

This will:
- Extract data from `archive.zip`
- Load and preprocess 1000 random stock/ETF CSVs
- Visualize raw prices
- Prepare training dataset
- Train and save the model

---

### 3. Evaluate the Model

```bash
python -m ai.evaluate
```

Generates plots comparing predicted vs. actual close prices on the test set.

---

### 4. Simulate Trading Strategy

```bash
python simulate.py --random 100
```

Simulates AI vs. buy-and-hold strategy on 100 randomly selected tickers. Outputs:
- `simulation_vs_bh_mean.png`
- `outperformance_hist.png`
- `individual_curves.png`

Or run on specific CSV files:
```bash
python simulate.py data/extracted/Stocks/AAPL.csv data/extracted/ETFs/SPY.csv
```

---

## 📁 Data Format

CSV files must include a `Close` column. The model uses 10-day sliding windows of percentage changes to predict the next-day move.

Training features and labels are saved to:
```
data/log/X.npy
data/log/y.npy
```

---

## ⚙️ Configuration

Adjustable parameters:
- `WINDOW_SIZE` (default: 10)
- `THRESH` for buy/sell decision (default: 0.002 → 0.2%)
- `START_CASH` (default: $1.0 virtual equity)

See `simulate.py` for simulation logic.

---

## 📌 Dependencies

- TensorFlow / Keras
- NumPy, Pandas
- scikit-learn
- Matplotlib

---

## 📈 Future Work

- Integrate LSTM or Transformer architectures
- Add risk management and stop-loss features
- Live data support (via APIs)
- Portfolio-level simulation

---

## 📜 License

MIT – free for personal and commercial use. Attribution appreciated.
