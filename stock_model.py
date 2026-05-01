import pandas as pd
import yfinance as yf

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from xgboost import XGBClassifier


def load_data(ticker: str) -> pd.DataFrame:
    data = yf.Ticker(ticker).history(period="5y")
    data = data.drop(columns=["Dividends", "Stock Splits"], errors="ignore")

    if data.empty:
        raise ValueError(f"No data found for ticker '{ticker}'.")

    return data


def create_features(data: pd.DataFrame):
    data = data.copy()

    data["Tomorrow"] = data["Close"].shift(-1)
    data["Target"] = (data["Tomorrow"] > data["Close"]).astype(int)

    horizons = [2, 5, 20, 60, 250]
    predictors = []

    for h in horizons:
        rolling = data.rolling(h).mean()

        ratio_col = f"Close_Ratio_{h}"
        trend_col = f"Trend_{h}"

        data[ratio_col] = data["Close"] / rolling["Close"]
        data[trend_col] = data.shift(1).rolling(h).sum()["Target"]

        predictors += [ratio_col, trend_col]

    data["Daily_Return"] = data["Close"].pct_change()
    data["Volatility_5"] = data["Daily_Return"].rolling(5).std()
    data["Volatility_20"] = data["Daily_Return"].rolling(20).std()
    data["Momentum_5"] = data["Close"] / data["Close"].shift(5)
    data["Momentum_20"] = data["Close"] / data["Close"].shift(20)

    data["SMA_50"] = data["Close"].rolling(50).mean()
    data["SMA_200"] = data["Close"].rolling(200).mean()
    data["Trend_50_200"] = (data["SMA_50"] > data["SMA_200"]).astype(int)

    delta = data["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = -delta.clip(upper=0).rolling(14).mean()
    rs = gain / loss
    data["RSI_14"] = 100 - (100 / (1 + rs))

    predictors += [
        "Daily_Return",
        "Volatility_5",
        "Volatility_20",
        "Momentum_5",
        "Momentum_20",
        "Trend_50_200",
        "RSI_14",
    ]

    data = data.dropna()

    if len(data) < 600:
        raise ValueError(
            "Not enough usable historical data after feature engineering. "
            "Try an older ticker or lower the backtest start row."
        )

    return data, predictors


def detect_stock_regime(data: pd.DataFrame) -> dict:
    returns = data["Close"].pct_change().dropna()
    recent_returns = returns.tail(252)

    avg_daily_return = recent_returns.mean()
    volatility = recent_returns.std()

    sma_50 = data["Close"].rolling(50).mean()
    sma_200 = data["Close"].rolling(200).mean()
    trend_days = (sma_50 > sma_200).tail(252).mean()

    total_return_1y = data["Close"].iloc[-1] / data["Close"].iloc[-252] - 1

    if volatility > 0.035:
        regime = "Volatile"
        description = "High-volatility stock; conservative thresholds may help."
    elif trend_days > 0.65 and total_return_1y > 0:
        regime = "Trending"
        description = "Strong trend behavior; buy-and-hold may be hard to beat."
    else:
        regime = "Mean-Reverting / Mixed"
        description = "Mixed behavior; short-term model signals may be more useful."

    return {
        "regime": regime,
        "description": description,
        "avg_daily_return": avg_daily_return,
        "volatility": volatility,
        "trend_days": trend_days,
        "total_return_1y": total_return_1y,
    }


def get_adaptive_settings(regime_info: dict, user_threshold=None) -> dict:
    regime = regime_info["regime"]

    if regime == "Trending":
        threshold = 0.55
        min_samples_split = 150
    elif regime == "Volatile":
        threshold = 0.60
        min_samples_split = 200
    else:
        threshold = 0.50
        min_samples_split = 100

    if user_threshold is not None:
        threshold = user_threshold

    return {
        "threshold": threshold,
        "min_samples_split": min_samples_split,
        "n_estimators": 75,
    }


def build_models(settings: dict):
    rf = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        min_samples_split=settings["min_samples_split"],
        random_state=1,
        n_jobs=-1,
    )

    xgb = XGBClassifier(
        n_estimators=75,
        max_depth=3,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=1,
        n_jobs=-1,
    )

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000)
    )

    return {
        "Random Forest": rf,
        "XGBoost": xgb,
        "Logistic Regression": lr,
    }


def predict_with_models(train, test, predictors, models, threshold=0.50):
    X_train = train[predictors]
    y_train = train["Target"]
    X_test = test[predictors]

    output = pd.DataFrame(index=test.index)
    output["Target"] = test["Target"]

    probability_columns = []

    for name, model in models.items():
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]

        prob_col = f"{name}_Probability"
        pred_col = f"{name}_Predictions"

        output[prob_col] = probs
        output[pred_col] = (probs >= threshold).astype(int)

        probability_columns.append(prob_col)

    output["Ensemble_Probability"] = output[probability_columns].mean(axis=1)
    output["Ensemble_Predictions"] = (
        output["Ensemble_Probability"] >= threshold
    ).astype(int)

    # Default columns used by app
    output["Probability"] = output["Ensemble_Probability"]
    output["Predictions"] = output["Ensemble_Predictions"]

    return output


def backtest(data, models, predictors, start=2500, step=250, threshold=0.50):
    all_predictions = []

    for i in range(start, len(data), step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:i + step].copy()

        preds = predict_with_models(
            train=train,
            test=test,
            predictors=predictors,
            models=models,
            threshold=threshold,
        )

        all_predictions.append(preds)

    if not all_predictions:
        raise ValueError("Backtest produced no predictions. Lower the start row.")

    return pd.concat(all_predictions)


def evaluate_model_metrics(predictions, pred_col="Predictions"):
    return {
        "accuracy": accuracy_score(predictions["Target"], predictions[pred_col]),
        "precision": precision_score(
            predictions["Target"], predictions[pred_col], zero_division=0
        ),
        "recall": recall_score(
            predictions["Target"], predictions[pred_col], zero_division=0
        ),
        "f1": f1_score(
            predictions["Target"], predictions[pred_col], zero_division=0
        ),
    }


def evaluate_all_models(predictions):
    model_columns = {
        "Random Forest": "Random Forest_Predictions",
        "XGBoost": "XGBoost_Predictions",
        "Logistic Regression": "Logistic Regression_Predictions",
        "Ensemble": "Ensemble_Predictions",
    }

    rows = []

    for model_name, pred_col in model_columns.items():
        metrics = evaluate_model_metrics(predictions, pred_col)

        rows.append({
            "Model": model_name,
            "Accuracy": metrics["accuracy"],
            "Precision": metrics["precision"],
            "Recall": metrics["recall"],
            "F1": metrics["f1"],
        })

    return pd.DataFrame(rows)


def evaluate_baseline(predictions):
    baseline_preds = pd.Series(1, index=predictions.index)

    return {
        "accuracy": accuracy_score(predictions["Target"], baseline_preds),
        "precision": precision_score(
            predictions["Target"], baseline_preds, zero_division=0
        ),
        "recall": recall_score(
            predictions["Target"], baseline_preds, zero_division=0
        ),
        "f1": f1_score(
            predictions["Target"], baseline_preds, zero_division=0
        ),
    }


def calculate_max_drawdown(curve: pd.Series) -> float:
    running_max = curve.cummax()
    drawdown = (curve - running_max) / running_max
    return drawdown.min()


def evaluate_strategy(data, predictions, pred_col="Predictions", prob_col="Probability"):
    data = data.copy()
    data["Actual_Return"] = data["Close"].pct_change()

    strategy = predictions.join(data[["Actual_Return"]], how="left")
    strategy = strategy.dropna()

    strategy["Strategy_Return"] = strategy[pred_col] * strategy["Actual_Return"]
    strategy["Position_Size_Return"] = strategy[prob_col] * strategy["Actual_Return"]
    strategy["Buy_Hold_Return"] = strategy["Actual_Return"]

    strategy_curve = (1 + strategy["Strategy_Return"]).cumprod()
    position_curve = (1 + strategy["Position_Size_Return"]).cumprod()
    buy_hold_curve = (1 + strategy["Buy_Hold_Return"]).cumprod()

    strategy_std = strategy["Strategy_Return"].std()
    position_std = strategy["Position_Size_Return"].std()
    buy_hold_std = strategy["Buy_Hold_Return"].std()

    strategy_sharpe = (
        strategy["Strategy_Return"].mean() / strategy_std
    ) * (252 ** 0.5) if strategy_std != 0 else 0

    position_size_sharpe = (
        strategy["Position_Size_Return"].mean() / position_std
    ) * (252 ** 0.5) if position_std != 0 else 0

    buy_hold_sharpe = (
        strategy["Buy_Hold_Return"].mean() / buy_hold_std
    ) * (252 ** 0.5) if buy_hold_std != 0 else 0

    trades = strategy[strategy[pred_col] == 1]
    number_of_trades = len(trades)
    win_rate = (trades["Actual_Return"] > 0).mean() if number_of_trades else 0

    return {
        "strategy_table": strategy,
        "strategy_sharpe": strategy_sharpe,
        "position_size_sharpe": position_size_sharpe,
        "buy_hold_sharpe": buy_hold_sharpe,
        "strategy_total_return": strategy_curve.iloc[-1],
        "position_size_total_return": position_curve.iloc[-1],
        "buy_hold_total_return": buy_hold_curve.iloc[-1],
        "strategy_max_drawdown": calculate_max_drawdown(strategy_curve),
        "buy_hold_max_drawdown": calculate_max_drawdown(buy_hold_curve),
        "win_rate": win_rate,
        "number_of_trades": number_of_trades,
    }


def evaluate_strategy_by_model(data, predictions):
    model_map = {
        "Random Forest": ("Random Forest_Predictions", "Random Forest_Probability"),
        "XGBoost": ("XGBoost_Predictions", "XGBoost_Probability"),
        "Logistic Regression": (
            "Logistic Regression_Predictions",
            "Logistic Regression_Probability",
        ),
        "Ensemble": ("Ensemble_Predictions", "Ensemble_Probability"),
    }

    rows = []

    for model_name, (pred_col, prob_col) in model_map.items():
        result = evaluate_strategy(data, predictions, pred_col, prob_col)

        rows.append({
            "Model": model_name,
            "StrategySharpe": result["strategy_sharpe"],
            "PositionSizeSharpe": result["position_size_sharpe"],
            "BuyHoldSharpe": result["buy_hold_sharpe"],
            "StrategyGrowth": result["strategy_total_return"],
            "PositionSizeGrowth": result["position_size_total_return"],
            "BuyHoldGrowth": result["buy_hold_total_return"],
            "WinRate": result["win_rate"],
            "Trades": result["number_of_trades"],
            "MaxDrawdown": result["strategy_max_drawdown"],
        })

    return pd.DataFrame(rows)


def feature_importance_report(data, predictors, settings):
    model = RandomForestClassifier(
        n_estimators=settings["n_estimators"],
        min_samples_split=settings["min_samples_split"],
        random_state=1,
        n_jobs=-1,
    )

    model.fit(data[predictors], data["Target"])

    return pd.Series(
        model.feature_importances_,
        index=predictors
    ).sort_values(ascending=False)


def run_single_ticker(ticker, threshold=None, start=500, step=500):
    data = load_data(ticker)
    data, predictors = create_features(data)

    regime_info = detect_stock_regime(data)
    settings = get_adaptive_settings(regime_info, user_threshold=threshold)

    models = build_models(settings)

    predictions = backtest(
        data=data,
        models=models,
        predictors=predictors,
        start=start,
        step=step,
        threshold=settings["threshold"],
    )

    metrics = evaluate_model_metrics(predictions, "Predictions")
    all_model_metrics = evaluate_all_models(predictions)

    baseline_metrics = evaluate_baseline(predictions)

    strategy_results = evaluate_strategy(
        data=data,
        predictions=predictions,
        pred_col="Predictions",
        prob_col="Probability",
    )

    strategy_by_model = evaluate_strategy_by_model(data, predictions)

    importances = feature_importance_report(data, predictors, settings)

    latest_signal = int(predictions["Predictions"].iloc[-1])
    latest_prob = float(predictions["Probability"].iloc[-1])

    return {
        "ticker": ticker,
        "rows": len(data),
        "data": data,
        "predictors": predictors,
        "regime_info": regime_info,
        "adaptive_settings": settings,
        "predictions": predictions,
        "metrics": metrics,
        "all_model_metrics": all_model_metrics,
        "baseline_metrics": baseline_metrics,
        "strategy_results": strategy_results,
        "strategy_by_model": strategy_by_model,
        "importances": importances,
        "latest_signal": latest_signal,
        "latest_prob": latest_prob,
    }


def main():
    user_input = input(
        "Enter stock ticker(s), separated by commas "
        "(example: AAPL,MSFT,NVDA,^GSPC): "
    ).strip()

    tickers = [t.strip().upper() for t in user_input.split(",") if t.strip()]

    summary_rows = []

    for ticker in tickers:
        try:
            result = run_single_ticker(ticker)

            print(f"\n===== {ticker} =====")
            print("Regime:", result["regime_info"]["regime"])
            print("Settings:", result["adaptive_settings"])
            print("\nModel Metrics:")
            print(result["all_model_metrics"])
            print("\nStrategy by Model:")
            print(result["strategy_by_model"])
            print("\nTop Features:")
            print(result["importances"].head(10))

            summary_rows.append({
                "Ticker": ticker,
                "Regime": result["regime_info"]["regime"],
                "Accuracy": result["metrics"]["accuracy"],
                "Precision": result["metrics"]["precision"],
                "Recall": result["metrics"]["recall"],
                "F1": result["metrics"]["f1"],
                "StrategySharpe": result["strategy_results"]["strategy_sharpe"],
                "BuyHoldSharpe": result["strategy_results"]["buy_hold_sharpe"],
                "LatestSignal": "UP" if result["latest_signal"] == 1 else "DOWN",
                "LatestUpProbability": result["latest_prob"],
            })

        except Exception as e:
            print(f"\n===== {ticker} =====")
            print("Error:", e)

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows).sort_values(
            by="StrategySharpe",
            ascending=False
        )

        print("\n===== SUMMARY =====")
        print(summary_df.to_string(index=False))
        summary_df.to_csv("stock_summary.csv", index=False)


if __name__ == "__main__":
    main()

    