# AgentSeer API Documentation

## Overview

AgentSeer provides both individual agent endpoints and orchestrated analysis endpoints. This allows you to:

- Call each specialist agent independently
- Use the Governor agent to synthesize multiple agent reports
- Run the full orchestrated analysis workflow

## Starting the Server

```bash
# Activate virtual environment
source venv/bin/activate

# Start the API server (default port 8000)
python run_agentseer.py --server

# Start on custom port
python run_agentseer.py --server --port 8080
```

## API Documentation

Once the server is running, visit:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Individual Agent Endpoints

### 1. SEC Agent

Analyzes 5 years of SEC 10-K filings for financial metrics and risk factors.

```bash
POST /api/agents/sec
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

**Response:**

```json
{
  "status": "success",
  "ticker": "AAPL",
  "agent": "sec",
  "summary": "SEC Agent Summary with financial metrics...",
  "detailed": "Full detailed report...",
  "timestamp": "2025-11-02 10:30:00"
}
```

### 2. News Agent

Analyzes recent news sentiment (last 7 days, up to 30 articles).

```bash
POST /api/agents/news
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

### 3. Social Agent

Analyzes social media sentiment from multiple platforms.

```bash
POST /api/agents/social
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

### 4. Chart Agent

Performs technical analysis (placeholder - implement as needed).

```bash
POST /api/agents/chart
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

### 5. Analyst Agent

Aggregates professional analyst ratings and price targets.

```bash
POST /api/agents/analyst
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

### 6. Governor Agent

Synthesizes reports from all specialist agents into a unified analysis.

**Note:** This endpoint requires summaries from other agents.

```bash
POST /api/agents/governor
Content-Type: application/json

{
  "ticker": "AAPL",
  "sec_summary": "Strong financials with revenue growth...",
  "news_summary": "Positive sentiment, bullish momentum...",
  "social_summary": "High positive sentiment across platforms...",
  "chart_summary": "Strong technical indicators...",
  "analyst_summary": "Majority recommend buy..."
}
```

**Response:**

```json
{
  "status": "success",
  "ticker": "AAPL",
  "agent": "governor",
  "summary": "Executive summary from Governor...",
  "full_memo": "Complete investment memo...",
  "timestamp": "2025-11-02 10:30:00"
}
```

### 7. Risk Assessment Agent

Analyzes risk based on Governor's memo.

```bash
POST /api/agents/risk
Content-Type: application/json

{
  "ticker": "AAPL",
  "sec_summary": "Strong financials...",
  "news_summary": "Positive sentiment...",
  "social_summary": "High engagement...",
  "chart_summary": "Upward trend...",
  "analyst_summary": "Buy ratings..."
}
```

## Orchestrated Analysis Endpoints

### Full Analysis Workflow

Runs all agents in parallel, then Governor and Risk Assessment.

```bash
POST /api/analyze
Content-Type: application/json

{
  "ticker": "AAPL"
}
```

**Response:**

```json
{
  "status": "success",
  "ticker": "AAPL",
  "workflow_status": "completed_successfully",
  "timestamp": "2025-11-02 10:30:00",
  "governor_summary": "Executive summary...",
  "risk_summary": "Risk assessment summary..."
}
```

### Re-run Specific Agent

Re-runs a specific agent and downstream dependencies (Governor, Risk).

```bash
POST /api/rerun
Content-Type: application/json

{
  "ticker": "AAPL",
  "agent": "social"
}
```

## Report & State Endpoints

### Get All Analyses

```bash
GET /api/analyses
```

### Get Analysis State

```bash
GET /api/analysis/{ticker}
```

### Get Agent Summary Report

```bash
GET /api/report/summary/{ticker}/{agent_name}
```

Example: `GET /api/report/summary/AAPL/sec`

### Get Agent Detailed Report

```bash
GET /api/report/detailed/{ticker}/{agent_name}
```

Example: `GET /api/report/detailed/AAPL/governor`

### Health Check

```bash
GET /api/health
```

## Example Workflows

### Workflow 1: Individual Agents → Governor

```python
import requests

BASE_URL = "http://localhost:8000"
ticker = "AAPL"

# Step 1: Call individual agents
sec = requests.post(f"{BASE_URL}/api/agents/sec", json={"ticker": ticker}).json()
news = requests.post(f"{BASE_URL}/api/agents/news", json={"ticker": ticker}).json()
social = requests.post(f"{BASE_URL}/api/agents/social", json={"ticker": ticker}).json()
chart = requests.post(f"{BASE_URL}/api/agents/chart", json={"ticker": ticker}).json()
analyst = requests.post(f"{BASE_URL}/api/agents/analyst", json={"ticker": ticker}).json()

# Step 2: Synthesize with Governor
governor_request = {
    "ticker": ticker,
    "sec_summary": sec["summary"],
    "news_summary": news["summary"],
    "social_summary": social["summary"],
    "chart_summary": chart["summary"],
    "analyst_summary": analyst["summary"]
}
governor = requests.post(f"{BASE_URL}/api/agents/governor", json=governor_request).json()

print(governor["summary"])
```

### Workflow 2: Full Orchestrated Analysis

```python
import requests

response = requests.post(
    "http://localhost:8000/api/analyze",
    json={"ticker": "AAPL"}
)

result = response.json()
print(f"Status: {result['workflow_status']}")
print(f"Governor Summary:\n{result['governor_summary']}")
print(f"Risk Summary:\n{result['risk_summary']}")
```

### Workflow 3: Single Agent Test

```python
import requests

response = requests.post(
    "http://localhost:8000/api/agents/news",
    json={"ticker": "TSLA"}
)

result = response.json()
print(f"News Sentiment Summary:\n{result['summary']}")
```

## Python Client Example

See `example_api_usage.py` for a complete Python client example.

## Shell Script Example

See `test_api_endpoints.sh` for a bash script that tests all endpoints.

## Error Handling

All endpoints return appropriate HTTP status codes:

- `200 OK`: Success
- `400 Bad Request`: Invalid input (e.g., missing ticker)
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Agent execution failed

Error response format:

```json
{
  "detail": "Error message describing what went wrong"
}
```

## Use Cases

### Frontend Integration

Call individual agent endpoints and display results in separate UI panels:

```javascript
// Fetch SEC data
const secData = await fetch("/api/agents/sec", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ ticker: "AAPL" }),
}).then((r) => r.json());

// Display in SEC panel
document.getElementById("sec-summary").innerText = secData.summary;
```

### Parallel Agent Execution

Run agents in parallel for faster results:

```python
import asyncio
import aiohttp

async def fetch_agent(session, agent, ticker):
    async with session.post(
        f"http://localhost:8000/api/agents/{agent}",
        json={"ticker": ticker}
    ) as response:
        return await response.json()

async def parallel_analysis(ticker):
    async with aiohttp.ClientSession() as session:
        tasks = [
            fetch_agent(session, agent, ticker)
            for agent in ["sec", "news", "social", "chart", "analyst"]
        ]
        results = await asyncio.gather(*tasks)
        return results

# Run
results = asyncio.run(parallel_analysis("AAPL"))
```

### Incremental Updates

Update individual agents without re-running the entire analysis:

```python
# Update just the social agent
requests.post(
    "http://localhost:8000/api/rerun",
    json={"ticker": "AAPL", "agent": "social"}
)
```

## Notes

- All endpoints are **asynchronous** (FastAPI)
- Individual agent endpoints create **minimal state** (just ticker)
- Governor endpoint **requires summaries** from other agents
- Full `/api/analyze` runs the **complete workflow** with state persistence
- Reports are **automatically saved** to the `reports/` directory
- Workflow states are saved to `workflow_states/` directory

## Support

For issues or questions:

1. Check the interactive docs at `/docs`
2. Review the example scripts (`example_api_usage.py`, `test_api_endpoints.sh`)
3. Check the logs in the console where the server is running
