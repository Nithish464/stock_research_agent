import yfinance as yf
from textblob import TextBlob
from langchain.tools import tool

@tool
def get_stock_price(ticker: str) -> dict:
    """Get current stock price, change, and trend data"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        hist = stock.history(period="30d")
        
        # 30 day trend
        if len(hist) > 1:
            trend = "UP" if hist["Close"].iloc[-1] > hist["Close"].iloc[0] else "DOWN"
            change_30d = ((hist["Close"].iloc[-1] - hist["Close"].iloc[0]) / hist["Close"].iloc[0]) * 100
        else:
            trend = "N/A"
            change_30d = 0
            
        return {
            "ticker": ticker,
            "current_price": round(info.get("currentPrice", 0), 2),
            "previous_close": round(info.get("previousClose", 0), 2),
            "day_change_pct": round(info.get("regularMarketChangePercent", 0), 2),
            "52w_high": round(info.get("fiftyTwoWeekHigh", 0), 2),
            "52w_low": round(info.get("fiftyTwoWeekLow", 0), 2),
            "volume": info.get("volume", 0),
            "trend_30d": trend,
            "change_30d_pct": round(change_30d, 2)
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_financials(ticker: str) -> dict:
    """Get key financial metrics — P/E, EPS, revenue, market cap"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "ticker": ticker,
            "pe_ratio": round(info.get("trailingPE", 0), 2),
            "forward_pe": round(info.get("forwardPE", 0), 2),
            "eps": round(info.get("trailingEps", 0), 2),
            "revenue": info.get("totalRevenue", 0),
            "profit_margin": round(info.get("profitMargins", 0) * 100, 2),
            "market_cap": info.get("marketCap", 0),
            "debt_to_equity": round(info.get("debtToEquity", 0), 2),
            "roe": round(info.get("returnOnEquity", 0) * 100, 2),
            "beta": round(info.get("beta", 0), 2),
            "analyst_rating": info.get("recommendationKey", "N/A")
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_stock_news(ticker: str) -> dict:
    """Get latest news and sentiment score for a stock"""
    try:
        stock = yf.Ticker(ticker)
        news = stock.news[:5]
        
        articles = []
        sentiments = []
        
        for article in news:
            title = article.get("title", "")
            sentiment = TextBlob(title).sentiment.polarity
            sentiments.append(sentiment)
            articles.append({
                "title": title,
                "publisher": article.get("publisher", ""),
                "sentiment": round(sentiment, 2),
                "sentiment_label": "POSITIVE" if sentiment > 0.1 else "NEGATIVE" if sentiment < -0.1 else "NEUTRAL"
            })
        
        avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
        
        return {
            "ticker": ticker,
            "articles": articles,
            "avg_sentiment": round(avg_sentiment, 2),
            "overall_sentiment": "POSITIVE" if avg_sentiment > 0.1 else "NEGATIVE" if avg_sentiment < -0.1 else "MIXED",
            "conflicting_signals": any(s > 0.1 for s in sentiments) and any(s < -0.1 for s in sentiments)
        }
    except Exception as e:
        return {"error": str(e)}

@tool
def get_analyst_ratings(ticker: str) -> dict:
    """Get Wall Street analyst consensus and price targets"""
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        return {
            "ticker": ticker,
            "recommendation": info.get("recommendationKey", "N/A"),
            "target_price": round(info.get("targetMeanPrice", 0), 2),
            "target_high": round(info.get("targetHighPrice", 0), 2),
            "target_low": round(info.get("targetLowPrice", 0), 2),
            "num_analysts": info.get("numberOfAnalystOpinions", 0),
            "current_price": round(info.get("currentPrice", 0), 2),
            "upside_potential": round(
                ((info.get("targetMeanPrice", 0) - info.get("currentPrice", 0)) / 
                 info.get("currentPrice", 1)) * 100, 2
            )
        }
    except Exception as e:
        return {"error": str(e)}