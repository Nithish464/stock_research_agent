import json
import os
from datetime import datetime, timedelta
import yfinance as yf

MEMORY_FILE = "predictions.json"

def load_memory() -> list:
    """Load all past predictions"""
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, "r") as f:
        return json.load(f)

def save_prediction(ticker: str, recommendation: str, conviction: float, price: float):
    """Save a new prediction to memory"""
    predictions = load_memory()
    
    prediction = {
        "id": len(predictions) + 1,
        "ticker": ticker,
        "recommendation": recommendation,
        "conviction_score": conviction,
        "price_at_prediction": price,
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "verified": False,
        "actual_price_7d": None,
        "correct": None,
        "return_pct": None
    }
    
    predictions.append(prediction)
    
    with open(MEMORY_FILE, "w") as f:
        json.dump(predictions, f, indent=2)
    
    print(f"Prediction saved: {ticker} — {recommendation} @ ${price}")
    return prediction

def verify_predictions():
    """Check if past predictions were correct (after 7 days)"""
    predictions = load_memory()
    updated = False
    
    for pred in predictions:
        if pred["verified"]:
            continue
            
        pred_date = datetime.strptime(pred["date"], "%Y-%m-%d")
        days_passed = (datetime.now() - pred_date).days
        
        if days_passed < 7:
            continue
        
        # Get actual price after 7 days
        try:
            stock = yf.Ticker(pred["ticker"])
            hist = stock.history(period="30d")
            
            if len(hist) > 0:
                actual_price = round(hist["Close"].iloc[-1], 2)
                price_at_pred = pred["price_at_prediction"]
                return_pct = round(((actual_price - price_at_pred) / price_at_pred) * 100, 2)
                
                # Was prediction correct?
                if pred["recommendation"] == "BUY" and return_pct > 2:
                    correct = True
                elif pred["recommendation"] == "SELL" and return_pct < -2:
                    correct = True
                elif pred["recommendation"] in ["HOLD", "NEUTRAL"] and abs(return_pct) <= 5:
                    correct = True
                else:
                    correct = False
                
                pred["actual_price_7d"] = actual_price
                pred["return_pct"] = return_pct
                pred["correct"] = correct
                pred["verified"] = True
                updated = True
                
                print(f"Verified: {pred['ticker']} — Predicted {pred['recommendation']}, Return: {return_pct}%, Correct: {correct}")
        except Exception as e:
            print(f"Could not verify {pred['ticker']}: {e}")
    
    if updated:
        with open(MEMORY_FILE, "w") as f:
            json.dump(predictions, f, indent=2)
    
    return predictions

def get_accuracy_stats() -> dict:
    """Calculate agent accuracy and bias detection"""
    predictions = load_memory()
    verified = [p for p in predictions if p["verified"]]
    
    if not verified:
        return {
            "total_predictions": len(predictions),
            "verified": 0,
            "accuracy": 0,
            "pending": len(predictions),
            "bias": "Not enough data"
        }
    
    correct = sum(1 for p in verified if p["correct"])
    accuracy = round((correct / len(verified)) * 100, 1)
    
    # Bias detection
    buy_predictions = [p for p in verified if p["recommendation"] == "BUY"]
    buy_correct = sum(1 for p in buy_predictions if p["correct"])
    buy_accuracy = round((buy_correct / len(buy_predictions)) * 100, 1) if buy_predictions else 0
    
    # High conviction bias check
    high_conv = [p for p in verified if p["conviction_score"] >= 7]
    high_conv_correct = sum(1 for p in high_conv if p["correct"])
    high_conv_accuracy = round((high_conv_correct / len(high_conv)) * 100, 1) if high_conv else 0
    
    bias = []
    if buy_accuracy < 40:
        bias.append("Overconfident on BUY signals")
    if high_conv_accuracy < 50:
        bias.append("High conviction predictions unreliable")
    if not bias:
        bias.append("No significant bias detected")
    
    return {
        "total_predictions": len(predictions),
        "verified": len(verified),
        "correct": correct,
        "accuracy": accuracy,
        "pending": len(predictions) - len(verified),
        "buy_accuracy": buy_accuracy,
        "high_conviction_accuracy": high_conv_accuracy,
        "bias": ", ".join(bias),
        "recent_predictions": predictions[-5:]
    }

def get_learning_insight(ticker: str) -> str:
    """Get learning insight for a specific ticker"""
    predictions = load_memory()
    ticker_preds = [p for p in predictions if p["ticker"] == ticker and p["verified"]]
    
    if not ticker_preds:
        return f"No past predictions for {ticker} — first analysis."
    
    correct = sum(1 for p in ticker_preds if p["correct"])
    accuracy = round((correct / len(ticker_preds)) * 100, 1)
    
    avg_return = round(sum(p["return_pct"] for p in ticker_preds) / len(ticker_preds), 2)
    
    insight = f"Past {ticker} predictions: {len(ticker_preds)} total, {accuracy}% accurate, avg return: {avg_return}%"
    
    if accuracy < 50:
        insight += f" — LOW TRUST: Past predictions on {ticker} were unreliable. Lowering confidence."
    elif accuracy > 70:
        insight += f" — HIGH TRUST: Strong track record on {ticker}."
    
    return insight

if __name__ == "__main__":
    # Test memory
    save_prediction("TSLA", "NEUTRAL", 7.2, 250.0)
    
    stats = get_accuracy_stats()
    print("\nAccuracy Stats:")
    print(f"Total predictions: {stats['total_predictions']}")
    print(f"Accuracy: {stats['accuracy']}%")
    print(f"Bias: {stats['bias']}")
    
    insight = get_learning_insight("TSLA")
    print(f"\nInsight: {insight}")