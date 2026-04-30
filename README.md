# AlphaSignal 📈

**ML-driven stock prediction, strategy backtesting, and model comparison platform**

---

## 🚀 Overview

AlphaSignal is an end-to-end machine learning system for predicting stock direction and evaluating trading strategies under realistic conditions.

Rather than focusing only on prediction accuracy, this project emphasizes:
- Strategy performance
- Model robustness
- Real-world trading constraints

---

## 🧠 Core Capabilities

### 🔹 Multi-Model Prediction Engine
- Random Forest (robust baseline)
- XGBoost (captures non-linear relationships)
- Logistic Regression (interpretable baseline)
- Ensemble model (probability averaging for stability)

---

### 🔹 Walk-Forward Backtesting
- Eliminates lookahead bias
- Retrains models over time
- Simulates realistic trading conditions

---

### 🔹 Strategy Evaluation Metrics

Each model is evaluated using:

- Accuracy, Precision, Recall, F1 Score
- Sharpe Ratio
- Win Rate
- Total Return
- Max Drawdown
- Trade Count

Compared against:
- Buy & Hold baseline
- Always-Up baseline

---

### 🔹 Model Selection System

Automatically identifies the best-performing model based on:

- Sharpe Ratio (strategy performance)
- or F1 Score (classification performance)

---

### 🔹 Regime Detection (Market Awareness)

Stocks are classified into:
- Trending
- Mean-Reverting
- Mixed

Model behavior and signal interpretation adapt based on the detected regime.

---

### 🔹 Confidence-Based Trading

Trades are only taken when the model is confident:

```python
P(up) > threshold
```

This filters out low-conviction signals and reduces overtrading.

---

### 🔹 Transaction Cost Modeling

Backtests include trading friction:

```python
returns -= transaction_cost
```

Ensures results are realistic and not artificially inflated.

---

## 🖥️ Interactive Dashboard (Streamlit)

**Features:**
- Multi-ticker analysis
- Adjustable thresholds (adaptive or manual)
- Backtest parameter tuning
- CSV export of results

**Visualizations:**
- Strategy vs Buy & Hold performance
- Feature importance rankings
- Confusion matrix
- Prediction distribution
- Model comparison tables

---

## 📊 Key Insight

Even with multiple machine learning models:
- Accuracy ≈ **50–52%**
- Stock direction prediction is inherently noisy

👉 The real edge comes from:
- Strategy construction
- Risk management
- Confidence filtering

—not raw prediction accuracy alone.

---

## 💡 Why This Project Matters

Most stock prediction projects stop at accuracy — but in real markets, **accuracy alone is not enough**.

AlphaSignal is built around a more realistic perspective:

### 🔹 Markets Are Noisy
- Even strong models struggle to exceed **50–52% accuracy**
- Small predictive edges are difficult to extract and sustain

---

### 🔹 Prediction ≠ Profit

A model can have:
- Good accuracy
- Good F1 score

...and still produce **poor trading results**

👉 This project focuses on:
- Strategy performance (Sharpe Ratio, drawdown, returns)
- Risk-adjusted decision making
- Trade selection based on confidence

---

### 🔹 Real-World Constraints Matter

Unlike simplified ML demos, AlphaSignal includes:
- Walk-forward backtesting (no data leakage)
- Transaction cost modeling
- Confidence thresholds
- Model comparison under identical conditions

---

### 🔹 From ML Project → Quant Thinking

This project demonstrates the shift from:

```
"Can I predict the market?"
```

to:

```
"Can I build a strategy that performs under uncertainty?"
```

---

## 🖥️ How to Run

### 1. Clone the repository

```bash
git clone https://github.com/your-username/AlphaSignal.git
cd AlphaSignal
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
```

Activate it:

- **Windows (PowerShell):**
```bash
venv\Scripts\activate
```

- **Mac/Linux:**
```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

### 5. Open in browser

Go to:

```
http://localhost:8501
```

---

## 🌐 Live Demo

Try the deployed app here:  
https://alphasignal-htqfscs5wjbqgt6el4o4jo.streamlit.app/

## ⚙️ Usage

1. Enter stock tickers (e.g., `AAPL`, `MSFT`, `NVDA`)
2. Adjust:
   - Decision threshold
   - Backtest parameters
3. Click **Run Analysis**
4. Explore:
   - Model comparison
   - Strategy performance
   - Diagnostics

---

## 🛠️ Tech Stack

- Python
- pandas / numpy
- scikit-learn
- XGBoost
- yfinance
- Streamlit
- matplotlib

---

## 📁 Project Structure

```
AlphaSignal/
├── app.py
├── stock_model.py
├── requirements.txt
├── README.md
└── screenshots/
```

---

## 🚀 Future Improvements

- Deep learning models (LSTM, Transformers)
- Sector-specific models
- Portfolio optimization
- Live trading signals
- News + sentiment integration
- Hyperparameter tuning pipeline

---

## ⚠️ Disclaimer

This project is for educational purposes only and does not constitute financial advice.

---

## 👤 Author

Yash Chauhan 