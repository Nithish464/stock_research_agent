import streamlit as st
import yfinance as yf
import plotly.graph_objects as go
from datetime import datetime
from agent import analyze_stock, app as agent_app
from memory import save_prediction, get_accuracy_stats, get_learning_insight, verify_predictions
import json

st.set_page_config(
    page_title="Stock Research Agent",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Research AI Agent")
st.caption("Financial Decision Engine — Conviction + Uncertainty + Learning Memory")

# Sidebar — Accuracy tracker
with st.sidebar:
    st.header("🧠 Agent Memory")
    
    # Verify old predictions
    verify_predictions()
    
    stats = get_accuracy_stats()
    
    st.metric("Total Predictions", stats["total_predictions"])
    st.metric("Verified", stats["verified"])
    
    if stats["verified"] > 0:
        st.metric("Accuracy", f"{stats['accuracy']}%")
        st.metric("BUY Accuracy", f"{stats['buy_accuracy']}%")
        st.metric("High Conviction Accuracy", f"{stats['high_conviction_accuracy']}%")
        st.warning(f"Bias: {stats['bias']}")
    else:
        st.info("No verified predictions yet — check back in 7 days!")
    
    st.divider()
    
    # Recent predictions
    if stats["total_predictions"] > 0:
        st.subheader("Recent Predictions")
        for pred in stats.get("recent_predictions", [])[-3:]:
            status = "✅" if pred.get("correct") else "⏳" if not pred.get("verified") else "❌"
            st.write(f"{status} {pred['ticker']} — {pred['recommendation']} ({pred['conviction_score']}/10)")

# Main area
col1, col2 = st.columns([2, 1])

with col1:
    ticker = st.text_input(
        "Enter stock ticker",
        placeholder="TSLA, AAPL, NVDA, MSFT...",
        help="Enter any valid stock ticker symbol"
    ).upper()

with col2:
    st.write("")
    st.write("")
    analyze_btn = st.button("🔍 Analyze Stock", type="primary")

if analyze_btn and ticker:
    # Learning insight
    insight = get_learning_insight(ticker)
    if "first analysis" not in insight:
        st.info(f"🧠 Learning Insight: {insight}")
    
    # Stock chart
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="3mo")
        
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=hist.index,
            open=hist["Open"],
            high=hist["High"],
            low=hist["Low"],
            close=hist["Close"],
            name=ticker
        ))
        fig.update_layout(
            title=f"{ticker} — 3 Month Price Chart",
            height=300,
            margin=dict(l=0, r=0, t=40, b=0),
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig, use_container_width=True)
        current_price = round(hist["Close"].iloc[-1], 2)
    except:
        current_price = 0.0

    # Run agent
    with st.spinner(f"🤖 Agent analyzing {ticker}... (Planner → Tools → Analyst → Skeptic → Scoring)"):
        report = analyze_stock(ticker)

    # Parse report for metrics
    lines = report.split("\n")
    recommendation = "HOLD"
    conviction = 5.0
    uncertainty = "MEDIUM"

    for line in lines:
        if "RECOMMENDATION:" in line:
            recommendation = line.split(":")[-1].strip()
        if "CONVICTION:" in line:
            try:
                conviction = float(line.split(":")[1].strip().split("/")[0])
            except:
                pass
        if "UNCERTAINTY:" in line:
            uncertainty = line.split(":")[-1].strip()

    # Metrics row
    m1, m2, m3, m4 = st.columns(4)

    with m1:
        color = "green" if "BUY" in recommendation else "red" if "SELL" in recommendation else "orange"
        st.markdown(f"### :{color}[{recommendation}]")
        st.caption("Recommendation")

    with m2:
        st.metric("Conviction Score", f"{conviction}/10")

    with m3:
        badge = "🔴" if uncertainty == "HIGH" else "🟡" if uncertainty == "MEDIUM" else "🟢"
        st.metric("Uncertainty", f"{badge} {uncertainty}")

    with m4:
        st.metric("Current Price", f"${current_price}")

    # Conviction meter
    st.subheader("🎯 Conviction Meter")
    conviction_color = "green" if conviction >= 7 else "orange" if conviction >= 5 else "red"
    st.progress(conviction / 10)
    st.caption(f"Conviction: {conviction}/10 — {'Strong Signal' if conviction >= 7 else 'Moderate Signal' if conviction >= 5 else 'Weak Signal'}")

    # Full report
    st.subheader("📋 Full Analysis Report")
    st.code(report, language=None)

    # Save to memory
    save_prediction(ticker, recommendation, conviction, current_price)
    st.success(f"✅ Prediction saved to memory — will verify in 7 days!")

elif analyze_btn and not ticker:
    st.warning("Please enter a ticker symbol!")