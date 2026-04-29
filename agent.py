import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from typing import TypedDict, Annotated
import operator
import json

load_dotenv()

from tools import get_stock_price, get_financials, get_stock_news, get_analyst_ratings

# State
class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    ticker: str
    raw_data: dict
    recommendation: str
    conviction_score: float
    uncertainty_level: str
    counter_thesis: str
    final_report: str

# Tools + LLM
tools = [get_stock_price, get_financials, get_stock_news, get_analyst_ratings]

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    api_key=os.getenv("GROQ_API_KEY")  # 
)
llm_with_tools = llm.bind_tools(tools)

# ============================================================
# NODE 1 — Planner
# ============================================================
def planner_node(state: AgentState):
    ticker = state["ticker"]
    plan_prompt = f"""You are analyzing {ticker} stock.

Fetch ALL of these in order:
1. get_stock_price for {ticker}
2. get_financials for {ticker}
3. get_stock_news for {ticker}
4. get_analyst_ratings for {ticker}

Call all 4 tools now."""

    messages = [
        SystemMessage(content="You are a financial data collector. Call all required tools."),
        HumanMessage(content=plan_prompt)
    ]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}

# ============================================================
# NODE 2 — Analyst
# ============================================================
def analyst_node(state: AgentState):
    messages = state["messages"]
    ticker = state["ticker"]

    tool_results = []
    for msg in messages:
        if hasattr(msg, "content") and isinstance(msg.content, str):
            tool_results.append(msg.content)

    analyst_prompt = f"""Based on all the data collected for {ticker}:

{chr(10).join(tool_results[-8:])}

Create a detailed investment analysis:

1. RECOMMENDATION: BUY / HOLD / SELL
2. CONVICTION SCORE: X/10 (based on data quality and signal strength)
3. KEY BULL POINTS: 3 reasons to buy
4. KEY BEAR POINTS: 3 risks
5. UNCERTAINTY FACTORS: What data is missing or conflicting?

Be specific with numbers. Use the actual data."""

    response = llm.invoke([
        SystemMessage(content="You are a senior equity analyst at a hedge fund."),
        HumanMessage(content=analyst_prompt)
    ])

    return {"messages": [response], "recommendation": response.content}

# ============================================================
# NODE 3 — Skeptic
# ============================================================
def skeptic_node(state: AgentState):
    recommendation = state.get("recommendation", "")
    ticker = state["ticker"]

    skeptic_prompt = f"""The analyst just recommended the following for {ticker}:

{recommendation}

You are the Devil's Advocate. ATTACK this recommendation.

Find:
1. What did the analyst miss or ignore?
2. What could go wrong with this thesis?
3. What are the hidden risks?
4. Is the analyst being overconfident?
5. Counter-thesis: What is the opposite view?

Be brutal and specific."""

    response = llm.invoke([
        SystemMessage(content="You are a contrarian hedge fund manager who challenges every investment thesis."),
        HumanMessage(content=skeptic_prompt)
    ])

    return {"messages": [response], "counter_thesis": response.content}

# ============================================================
# NODE 4 — Scoring Engine
# ============================================================
def scoring_node(state: AgentState):
    recommendation = state.get("recommendation", "")
    counter_thesis = state.get("counter_thesis", "")
    ticker = state["ticker"]

    scoring_prompt = f"""Based on this analysis for {ticker}:

ANALYST VIEW:
{recommendation}

SKEPTIC VIEW:
{counter_thesis}

Output EXACTLY in this JSON format — no extra text:
{{
  "final_recommendation": "BUY",
  "conviction_score": 7.2,
  "uncertainty_level": "MEDIUM",
  "uncertainty_reason": "specific reason why uncertain",
  "bull_factors": ["factor1", "factor2", "factor3"],
  "bear_factors": ["factor1", "factor2", "factor3"],
  "key_risk": "single biggest risk",
  "data_quality": "GOOD"
}}"""

    response = llm.invoke([
        SystemMessage(content="You are a quantitative risk analyst. Output only valid JSON. No markdown, no explanation."),
        HumanMessage(content=scoring_prompt)
    ])

    try:
        content = response.content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        scores = json.loads(content)
    except:
        scores = {
            "final_recommendation": "HOLD",
            "conviction_score": 5.0,
            "uncertainty_level": "HIGH",
            "uncertainty_reason": "Could not parse scoring",
            "bull_factors": [],
            "bear_factors": [],
            "key_risk": "Unknown",
            "data_quality": "POOR"
        }

    return {
        "messages": [response],
        "conviction_score": scores.get("conviction_score", 5.0),
        "uncertainty_level": scores.get("uncertainty_level", "HIGH"),
        "raw_data": scores
    }

# ============================================================
# NODE 5 — Final Report
# ============================================================
def report_node(state: AgentState):
    scores = state.get("raw_data", {})
    counter_thesis = state.get("counter_thesis", "")
    ticker = state["ticker"]

    bull_factors = "\n".join([f"  + {f}" for f in scores.get("bull_factors", [])])
    bear_factors = "\n".join([f"  - {f}" for f in scores.get("bear_factors", [])])

    report = f"""
╔══════════════════════════════════════════════════════════╗
║        STOCK RESEARCH AGENT — {ticker:<10}                 ║
╚══════════════════════════════════════════════════════════╝

📊 RECOMMENDATION:  {scores.get('final_recommendation', 'HOLD')}
🎯 CONVICTION:      {scores.get('conviction_score', 5.0)}/10
⚠️  UNCERTAINTY:    {scores.get('uncertainty_level', 'MEDIUM')}
📁 DATA QUALITY:    {scores.get('data_quality', 'PARTIAL')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
WHY BUY (Bull Case):
{bull_factors}

WHY NOT (Bear Case):
{bear_factors}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚠️  UNCERTAINTY WARNING:
  {scores.get('uncertainty_reason', 'N/A')}

🔴 BIGGEST RISK:
  {scores.get('key_risk', 'N/A')}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🤔 COUNTER-THESIS (Skeptic View):
{counter_thesis[:600]}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
⚡ AGENT NOTE:
  Conviction score may change with new data.
  Always verify before investing.
╚══════════════════════════════════════════════════════════╝
"""

    return {"messages": [AIMessage(content=report)], "final_report": report}

# ============================================================
# Router
# ============================================================
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "analyst"

# ============================================================
# Build Graph
# ============================================================
tool_node = ToolNode(tools)

graph = StateGraph(AgentState)
graph.add_node("planner", planner_node)
graph.add_node("tools", tool_node)
graph.add_node("analyst", analyst_node)
graph.add_node("skeptic", skeptic_node)
graph.add_node("scoring", scoring_node)
graph.add_node("report", report_node)

graph.set_entry_point("planner")
graph.add_conditional_edges("planner", should_continue)
graph.add_edge("tools", "analyst")
graph.add_edge("analyst", "skeptic")
graph.add_edge("skeptic", "scoring")
graph.add_edge("scoring", "report")
graph.add_edge("report", END)

app = graph.compile()

# ============================================================
# Run Agent
# ============================================================
def analyze_stock(ticker: str) -> str:
    print(f"\nAnalyzing {ticker}...")
    result = app.invoke({
        "messages": [],
        "ticker": ticker.upper(),
        "raw_data": {},
        "recommendation": "",
        "conviction_score": 0.0,
        "uncertainty_level": "",
        "counter_thesis": "",
        "final_report": ""
    })
    return result["final_report"]

if __name__ == "__main__":
    report = analyze_stock("TSLA")
    print(report)