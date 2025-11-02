# AgentSeer: Multi-Agent Financial Analysis System

**Track 1: Workplace AI Solution**

AgentSeer is an autonomous analyst swarm that provides comprehensive, multi-perspective financial risk assessments by orchestrating specialist AI agents in parallel, then synthesizing their findings through governed debate.

---

## 🎯 Project Overview

### The Problem
Financial risk assessment is slow, manual, and biased by a single analyst's opinion. Traditional analysis lacks multiple perspectives and often misses critical signals across different data sources.

### The Solution
An autonomous "Analyst Swarm" - a multi-agent system where specialist agents analyze a stock in parallel across different dimensions (fundamentals, news, sentiment, technicals, analyst ratings), then debate to produce a governed, multi-perspective risk report.

---

## 🏗️ Architecture

```
User Input ($TICKER)
        ↓
   ORCHESTRATOR (LangGraph)
        ↓
   ┌────┴────┬────────┬────────┬─────────┐
   ↓         ↓        ↓        ↓         ↓
SEC Agent News   Social  Chart    Analyst
          Agent   Agent   Agent    Agent
   ↓         ↓        ↓        ↓         ↓
   └────┬────┴────────┴────────┴─────────┘
        ↓
  GOVERNOR AGENT (Synthesis)
        ↓
  RISK ASSESSMENT AGENT
        ↓
   FINAL REPORT
```

---

## 🤖 Specialist Agents

### 1. **SEC Agent** 📄
- Navigates SEC EDGAR database
- Analyzes 10-K "Risk Factors" sections
- Extracts regulatory and compliance risks

### 2. **News Agent** 📰
- Scrapes financial news sites
- Summarizes breaking news and developments
- Identifies market-moving events

### 3. **Social Sentiment Agent** 💬
- Analyzes retail investor sentiment
- Sources: Reddit (r/wallstreetbets, r/stocks), Twitter, StockTwits
- Provides crowd psychology insights

### 4. **Chart Agent** 📊
- Technical analysis of price action
- Pattern recognition and momentum analysis
- Price predictions and support/resistance levels

### 5. **Analyst Agent** 📈
- Aggregates professional analyst ratings
- Tracks upgrades, downgrades, price targets
- Consensus recommendations

---

## 🎯 High-Level Agents

### **Governor Agent** 
Synthesizes all specialist reports following Holistic AI principles:
- ✅ Unbiased analysis - presents all perspectives fairly
- ✅ Source attribution - cites which agent provided each insight
- ✅ Conflict resolution - presents disagreements transparently
- ✅ Consensus building - identifies agreements across agents
- ✅ Gap identification - acknowledges missing data

**Output**: Comprehensive investment memo

### **Risk Assessment Agent**
Produces quantitative risk scoring across six dimensions:
- 🏢 Market Risk
- 🏭 Company-Specific Risk
- 📱 Sentiment Risk
- 📈 Technical Risk
- ⚖️ Regulatory/Legal Risk
- 🌍 Macroeconomic Risk

**Output**: Detailed risk report with scores, mitigation strategies, and portfolio implications

---

## 🚀 Installation

### Prerequisites
```bash
python 3.9+
pip install -r requirements.txt
```

### Required Dependencies
```bash
# Core dependencies
pip install langchain langchain-google-genai langchain-google-community
pip install langgraph
pip install finnhub-python
pip install python-dotenv

# For API server (optional)
pip install flask flask-cors
```

### Environment Variables
Create a `.env` file in the project root:

```env
# Required for all agents
GEMINI_API_KEY=your_gemini_api_key

# Social Agent
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CSE_ID_SOCIAL=your_custom_search_engine_id

# Analyst Agent
FINNHUB_API_KEY=your_finnhub_api_key

# TODO: Add when implementing other agents
# SEC_EDGAR_API_KEY=...
# NEWS_API_KEY=...
# CHART_API_KEY=...
```

---

## 📖 Usage

### 1. Single Analysis
```bash
python run_agentseer.py TSLA
```

### 2. Batch Analysis
```bash
python run_agentseer.py --batch TSLA AAPL MSFT GOOGL NVDA
```

### 3. View Analysis History
```bash
python run_agentseer.py --list
```

### 4. Start API Server
```bash
python run_agentseer.py --server
```

Then access:
- API: `http://localhost:8000`
- Health check: `http://localhost:8000/api/health`

### 5. Use Individual Agents
```bash
# Social Sentiment Agent
python social_agent.py TSLA

# Analyst Ratings Agent
python analyst_agent.py AAPL

# Governor Agent (requires agent reports)
python governor_agent.py MSFT

# Risk Assessment Agent (requires investment memo)
python risk_assessment_agent.py GOOGL
```

---

## 📊 AgentSeer Dashboard

The React-based dashboard provides a visual interface to:
- ✅ Submit analysis requests
- ✅ Monitor real-time progress through the agent pipeline
- ✅ View analysis history
- ✅ Visualize the "fork-join" workflow
- ✅ Access detailed reports

### Running the Dashboard
```bash
# The dashboard is a standalone React component
# Copy the code from agentseer_dashboard artifact
# Run in your preferred React environment
```

---

## 📁 Project Structure

```
analyst-swarm/
├── run_agentseer.py          # Main integration script
├── orchestrator.py            # LangGraph workflow manager
├── governor_agent.py          # Synthesis agent
├── risk_assessment_agent.py   # Risk analysis agent
├── social_agent.py            # Social sentiment agent
├── analyst_agent.py           # Professional analyst agent
├── sec_agent.py              # SEC filings agent (TODO)
├── news_agent.py             # News analysis agent (TODO)
├── chart_agent.py            # Technical analysis agent (TODO)
├── requirements.txt           # Python dependencies
├── .env                       # Environment variables (create this)
├── reports/                   # Generated reports
├── workflow_states/           # Saved workflow states
└── logs/                      # Application logs
```

---

## 🔧 API Endpoints

When running the API server:

### `POST /api/analyze`
Analyze a stock ticker
```json
{
  "ticker": "TSLA"
}
```

### `GET /api/analyses`
Get list of all analyses

### `GET /api/analysis/<ticker>`
Get detailed analysis for specific ticker

### `GET /api/health`
Health check endpoint

---

## 📈 Output Files

Each analysis generates:

### 1. Specialist Agent Reports
- `{ticker}_social_sentiment_{timestamp}.txt`
- Individual agent summaries

### 2. Governor Investment Memo
- `{ticker}_investment_memo_{timestamp}.md`
- Comprehensive synthesis of all perspectives

### 3. Risk Assessment Report
- `{ticker}_risk_assessment_{timestamp}.md`
- Detailed risk scoring and portfolio implications

### 4. Workflow State
- `{ticker}_workflow_state_{timestamp}.json`
- Complete execution trace for debugging

---

## 🎨 Key Features

### ✅ Parallel Execution
Five specialist agents run simultaneously using LangGraph's parallel execution

### ✅ Holistic AI Principles
Governor ensures unbiased, multi-perspective analysis

### ✅ Quantitative Risk Scoring
0-100 risk scores across six key dimensions

### ✅ Source Attribution
Every insight is traced back to its originating agent

### ✅ Conflict Resolution
Disagreements between agents are presented transparently

### ✅ Portfolio Implications
Actionable recommendations on position sizing and risk management

### ✅ Stress Testing
What-if scenarios for different market conditions

---

## 🛠️ Extending AgentSeer

### Adding a New Specialist Agent

1. Create `{agent_name}_agent.py` following this template:
```python
def run_{agent_name}_agent(ticker: str) -> str:
    """
    Execute the {Agent Name} analysis.
    
    Returns:
        Formatted markdown report string
    """
    # Your implementation
    return report
```

2. Add to `orchestrator.py`:
```python
def {agent_name}_node(state: AnalystSwarmState) -> AnalystSwarmState:
    report = run_{agent_name}_agent(state['ticker'])
    state['{agent_name}_report'] = report
    return state

# Add to workflow
workflow.add_node("{agent_name}", {agent_name}_node)
```

3. Update the state definition in `orchestrator.py`

---

## 🔒 Security & Privacy

- All API keys stored in `.env` file (never commit)
- No PII or personal data collected
- Analysis reports saved locally
- Optional API server for controlled access

---

## 📊 Performance

- **Parallel Execution**: 5 agents run simultaneously
- **Caching**: Search results cached hourly to reduce API calls
- **Rate Limiting**: Respects API provider limits
- **Error Handling**: Graceful degradation if agents fail

---

## 🐛 Troubleshooting

### "Module not found" errors
```bash
# Ensure all agents are in the same directory
ls *.py

# Check Python path
python -c "import sys; print(sys.path)"
```

### API rate limits
```bash
# Social Agent uses caching - results refresh hourly
# Finnhub free tier: 60 calls/minute
# Implement backoff if needed
```

### Missing environment variables
```bash
# Validate your .env file
python -c "from dotenv import load_dotenv; load_dotenv(); import os; print(os.environ.get('GEMINI_API_KEY'))"
```

---

## 🚧 TODO / Roadmap

- [ ] Implement SEC Agent (EDGAR API integration)
- [ ] Implement News Agent (Financial news aggregation)
- [ ] Implement Chart Agent (Technical analysis with image processing)
- [ ] Add database backend for persistent storage
- [ ] Real-time WebSocket updates for dashboard
- [ ] Email/Slack notifications for completed analyses
- [ ] Comparative analysis (compare multiple stocks)
- [ ] Historical trend tracking
- [ ] Machine learning for risk prediction

---

## 📄 License

MIT License - See LICENSE file for details

---

## 👥 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 🙏 Acknowledgments

Built with:
- **LangChain** - Agent framework
- **LangGraph** - Workflow orchestration
- **Google Gemini** - Language model
- **Finnhub** - Financial data API
- **React** - Dashboard UI

---

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**AgentSeer** - Because financial decisions deserve multiple perspectives.