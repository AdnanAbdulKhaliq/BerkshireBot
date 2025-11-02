# BerkshireBot Agent API Reference for Frontend Development

## Overview
All agents return a dictionary with common fields plus agent-specific data. All agents follow a consistent pattern with `ticker`, `agent`, `summary_report`, and `detailed_report` fields.

---

## 1. News Agent (`run_news_agent`)

### Function Signature
```python
run_news_agent(ticker: str, save_to_file: bool = True) -> dict
```

### Response Model
```json
{
  "ticker": "AAPL",
  "agent": "News",
  "summary_report": "markdown formatted summary",
  "detailed_report": "full analysis text",
  
  // Core sentiment data
  "company": "Apple Inc.",
  "total_articles_analyzed": 25,
  "weighted_sentiment_score": 0.65,
  "bullish_pressure": 0.72,
  "bearish_pressure": 0.28,
  
  // Sentiment breakdown
  "sentiment_breakdown": {
    "bullish": 15,
    "bearish": 5,
    "neutral": 5
  },
  
  // Source distribution
  "source_distribution": {
    "bloomberg": 8,
    "reuters": 7,
    "wsj": 10
  },
  
  // High impact articles
  "high_impact_count": 8,
  
  // Article arrays (each with title, sentiment, impact, url, source, published_at, reason, weighted_impact, recency_weight)
  "bullish_articles": [...],
  "bearish_articles": [...],
  "neutral_articles": [...],
  "high_impact_articles": [...],
  
  "dashboard_summary": "LLM-generated executive summary"
}
```

---

## 2. Social Agent (`run_social_agent`)

### Function Signature
```python
run_social_agent(ticker: str, save_to_file: bool = True) -> dict
```

### Response Model
```json
{
  "ticker": "AAPL",
  "agent": "Social-Sentimentalist",
  "summary_report": "markdown formatted summary",
  "detailed_report": "full analysis text",
  
  // Sentiment metrics
  "overall_sentiment": "Bullish",
  "sentiment_score": 0.68,
  "data_quality": "Excellent",
  "sources_analyzed": 4,
  
  // Search metadata
  "search_status": "success",
  
  // Data quality stats
  "freshness_stats": {
    "recent_count": 15,
    "stale_count": 2,
    "oldest_date": "2025-10-25"
  },
  
  "quality_stats": {
    "high_quality": 12,
    "medium_quality": 5,
    "low_quality": 0
  },
  
  // Analysis insights
  "executive_summary": "3-sentence overview",
  "consensus_view": "Market consensus description",
  "key_themes": ["theme1", "theme2", "theme3"],
  "bullish_signals": ["signal1", "signal2"],
  "bearish_signals": ["signal1", "signal2"],
  "risk_factors": ["risk1", "risk2"],
  
  // Raw source data
  "raw_sources": {
    "reddit": [...],
    "stocktwits": [...],
    "seeking_alpha": [...],
    "yahoo_finance": [...]
  }
}
```

---

## 3. Analyst Agent (`run_analyst_agent`)

### Function Signature
```python
run_analyst_agent(ticker: str, save_to_file: bool = True) -> dict
```

### Response Model
```json
{
  "ticker": "AAPL",
  "agent": "Analyst",
  "summary_report": "markdown formatted summary",
  "detailed_report": "full analysis text",
  
  // Consensus metrics
  "consensus": "Strong Buy",
  "consensus_score": 4.2,
  "recency_adjusted_score": 4.3,
  "recent_momentum": "Positive",
  "momentum_score": 0.15,
  
  // Price targets
  "current_price": 178.50,
  "average_target": 195.30,
  "target_high": 220.00,
  "target_low": 165.00,
  "upside_percent": 9.42,
  
  // Analyst coverage
  "number_of_analysts": 42,
  
  // Recent activity (array of recent upgrades/downgrades)
  "recent_activity": [
    {
      "date": "2025-10-28",
      "analyst": "Goldman Sachs",
      "action": "upgrade",
      "from": "Neutral",
      "to": "Buy",
      "target": 200.00
    }
  ],
  
  // Computational metrics
  "computational_metrics": {
    "volatility": 0.28,
    "beta": 1.15,
    "sharpe_ratio": 1.8
  },
  
  "analysis_summary": "Overall analyst sentiment",
  "data_quality": "excellent",
  "data_sources_used": ["yfinance", "fmp"]
}
```

---

## 4. Risk Assessment Agent (`run_risk_assessment_agent`)

### Function Signature
```python
run_risk_assessment_agent(
    ticker: str, 
    investment_memo: str, 
    save_to_file: bool = True
) -> dict
```

### Response Model
```json
{
  "ticker": "AAPL",
  "agent": "Risk_Assessment",
  "summary_report": "markdown formatted summary",
  "detailed_report": "full analysis text",
  
  // Overall risk scores
  "overall_risk_score": 6.5,
  "overall_risk_level": "Moderate",
  "baseline_score": 5.8,
  "baseline_level": "Moderate-Low",
  
  // Baseline metrics (volatility, beta, max drawdown, etc.)
  "baseline_metrics": {
    "volatility": 0.28,
    "beta": 1.15,
    "max_drawdown": -0.15,
    "sharpe_ratio": 1.8,
    "var_95": -0.025
  },
  
  // Risk category breakdowns (each has score, level, factors array)
  "risk_categories": {
    "market_risk": {
      "score": 7.0,
      "level": "Moderate-High",
      "factors": ["factor1", "factor2"]
    },
    "company_specific_risk": {
      "score": 6.0,
      "level": "Moderate",
      "factors": [...]
    },
    "sentiment_risk": {...},
    "technical_risk": {...},
    "regulatory_risk": {...},
    "macroeconomic_risk": {...}
  },
  
  // Summary insights
  "overall_assessment": "Overall risk description",
  "key_risk_factors": ["risk1", "risk2", "risk3"],
  "risk_mitigation_suggestions": ["suggestion1", "suggestion2"],
  "baseline_vs_qualitative_note": "Comparison explanation"
}
```

---

## 5. SEC Agent (`run_sec_agent`)

### Function Signature
```python
run_sec_agent(
    ticker: str, 
    company_description: str = None, 
    save_to_file: bool = True
) -> dict
```

### Response Model
```json
{
  "ticker": "AAPL",
  "agent": "SEC",
  "summary_report": "markdown formatted summary",
  "detailed_report": "full multi-year analysis report",
  
  // Filing metadata
  "filings_analyzed": 5,
  "years_covered": "2021 - 2025",
  "overall_risk_level": "Moderate Risk",
  
  // Financial metrics (multi-year data)
  "financial_metrics": {
    "years_list": ["2025", "2024", "2023", "2022", "2021"],
    "years_data": [
      {
        "year": "2025",
        "metrics": {
          "revenue": 307003000000,
          "net_income": 112010000000,
          "capex": 12715000000,
          "gross_profit": 195201000000,
          "operating_income": 133050000000,
          "eps_basic": 7.49,
          "eps_diluted": 7.46,
          "operating_cash_flow": 111482000000,
          "total_assets": 359241000000,
          "total_liabilities": 285508000000,
          "stockholders_equity": 73733000000,
          "cash_and_equivalents": 35934000000,
          "long_term_debt": 78328000000,
          // ... 30+ more financial metrics
        }
      }
      // ... data for other years
    ]
  },
  
  "metrics_documentation": "List of all tracked metrics with XBRL fields"
}
```

---

## 6. Orchestrator (`run_complete_analysis`)

### Function Signature
```python
run_complete_analysis(ticker: str, save_state: bool = True) -> dict
```

### Response Model (AnalystSwarmState)
```json
{
  // Input
  "ticker": "AAPL",
  "timestamp": "2025-11-02T10:30:00",
  
  // Individual agent reports (full text)
  "sec_agent_report": "Full SEC report...",
  "news_agent_report": "Full news report...",
  "social_agent_report": "Full social report...",
  "chart_agent_report": "Full chart report...",
  "analyst_agent_report": "Full analyst report...",
  
  // Agent execution status
  "sec_agent_status": "completed",
  "news_agent_status": "completed",
  "social_agent_status": "completed",
  "chart_agent_status": "completed",
  "analyst_agent_status": "completed",
  
  // Retry tracking
  "sec_agent_attempts": 1,
  "news_agent_attempts": 2,
  "social_agent_attempts": 1,
  "chart_agent_attempts": 1,
  "analyst_agent_attempts": 1,
  
  // Governor Agent (synthesis)
  "governor_summary": "Executive summary of all findings",
  "governor_full_memo": "Complete investment memo",
  "governor_status": "completed",
  "governor_attempts": 1,
  
  // Risk Assessment
  "risk_summary": "Risk assessment summary",
  "risk_full_report": "Detailed risk analysis",
  "risk_status": "completed",
  "risk_attempts": 1,
  
  // Overall workflow status
  "workflow_status": "completed_successfully", // or "completed_with_errors" or "failed"
  "errors": [],
  "warnings": []
}
```

---

## Error Handling

All agents return an error structure when they fail:

```json
{
  "ticker": "AAPL",
  "agent": "AgentName",
  "error": "Error message",
  "summary_report": "Error description",
  "detailed_report": "Full error details with traceback"
}
```

---

## Common Patterns for Frontend

### 1. Display Agent Status
```javascript
const agentStatus = {
  News: response.agent === "News" && !response.error,
  Social: response.agent === "Social-Sentimentalist" && !response.error,
  Analyst: response.agent === "Analyst" && !response.error,
  Risk: response.agent === "Risk_Assessment" && !response.error,
  SEC: response.agent === "SEC" && !response.error
};
```

### 2. Sentiment Display
```javascript
// News Agent sentiment
const newsSentiment = {
  score: response.weighted_sentiment_score, // -1 to 1
  bullish: response.sentiment_breakdown.bullish,
  bearish: response.sentiment_breakdown.bearish,
  neutral: response.sentiment_breakdown.neutral
};

// Social Agent sentiment
const socialSentiment = {
  sentiment: response.overall_sentiment, // "Bullish", "Bearish", "Neutral"
  score: response.sentiment_score, // 0 to 1
  quality: response.data_quality // "Excellent", "Good", "Fair", "Poor"
};
```

### 3. Risk Level Colors
```javascript
const riskColors = {
  "Low": "green",
  "Moderate-Low": "lightgreen",
  "Moderate": "yellow",
  "Moderate-High": "orange",
  "High": "red"
};
```

### 4. Financial Metrics Display (SEC Agent)
```javascript
const latestYear = response.financial_metrics.years_data[0];
const metrics = {
  revenue: latestYear.metrics.revenue,
  netIncome: latestYear.metrics.net_income,
  eps: latestYear.metrics.eps_diluted,
  // ... format and display as needed
};
```

---

## Notes for Frontend Implementation

1. **All monetary values** from SEC Agent are in USD (absolute numbers)
2. **Dates** are in ISO format strings
3. **Sentiment scores** vary by agent:
   - News: -1 (bearish) to +1 (bullish)
   - Social: 0 (bearish) to 1 (bullish)
   - Analyst: 1 (strong sell) to 5 (strong buy)
4. **Reports** include both `summary_report` (concise) and `detailed_report` (comprehensive)
5. **Error handling**: Check for `error` field in response to detect failures
6. **Orchestrator state**: Saved to `workflow_states/` directory as JSON files

---

## Sample Frontend Dashboard Layout

```
┌─────────────────────────────────────────┐
│         Company: AAPL - Apple Inc.      │
├─────────────────────────────────────────┤
│ News Sentiment    │ Social Sentiment    │
│ ● Bullish (0.68)  │ ● Bullish (0.72)   │
├─────────────────────────────────────────┤
│ Analyst Consensus │ Risk Level          │
│ Strong Buy (4.2)  │ Moderate (6.5/10)  │
├─────────────────────────────────────────┤
│ Price Target      │ SEC Filing Status   │
│ $195 (+9.4%)      │ 5 years analyzed   │
├─────────────────────────────────────────┤
│         Governor Recommendation         │
│    [Executive Summary Display]          │
└─────────────────────────────────────────┘
```
