# 📈 Stock Research AI Agent

An AI-powered stock research agent that autonomously researches NSE/BSE stocks using LangChain tools, memory, and real-time data.

## 🤖 What It Does

- Autonomously researches any stock using AI agent reasoning
- Uses LangChain tools to fetch price data, news, and financials
- Maintains conversation memory for multi-turn research sessions
- FastAPI-powered REST API for easy integration

## 🛠️ Tech Stack

- **AI / Agent** — LangChain, OpenAI / Groq LLM
- **Backend** — FastAPI, Python
- **Tools** — yfinance, web search, financial data APIs
- **Memory** — LangChain ConversationBufferMemory

## 🚀 Quick Start

```bash
git clone https://github.com/Nithish464/stock_research_agent
cd stock_research_agent
pip install -r requirements.txt
uvicorn app:app --reload
```

## 📁 Project Structure
├── agent.py      # LangChain agent setup & reasoning logic
├── tools.py      # Custom tools (price fetch, news, analysis)
├── memory.py     # Conversation memory management
├── app.py        # FastAPI endpoints
└── requirements.txt

## 💡 Example Usage
User: "Research Infosys stock and give me a buy/sell recommendation"
Agent: Fetches live price → Analyzes technicals → Checks recent news → Returns recommendation
