const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

// Types
export interface AgentResponse {
  ticker: string;
  agent: string;
  summary?: string;
  summary_report?: string;
  dashboard_summary?: string; // News agent specific
  detailed?: string;
  detailed_report?: string;
  status?: string;
  timestamp?: string;
  error?: string;
  // News agent specific fields
  company?: string;
  total_articles_analyzed?: number;
  sentiment_breakdown?: {
    bullish: number;
    bearish: number;
    neutral: number;
  };
  weighted_sentiment_score?: number;
  bullish_pressure?: number;
  bearish_pressure?: number;
}

export interface GovernorResponse {
  ticker: string;
  agent: string;
  executive_summary?: string;
  summary_report?: string;
  detailed_report?: string;
  error?: string;
}

export interface RiskResponse {
  ticker: string;
  agent: string;
  overall_risk_score?: number;
  overall_risk_level?: string;
  recommendation?: string;
  summary_report?: string;
  detailed_report?: string;
  error?: string;
}

// Sequential Analysis Flow
export async function runFullSequentialAnalysis(
  ticker: string,
  onAgentComplete?: (agentId: string, result: AgentResponse) => void,
  onGovernorComplete?: (result: GovernorResponse) => void,
  onRiskComplete?: (result: RiskResponse) => void
): Promise<{
  agents: Record<string, AgentResponse>;
  governor: GovernorResponse;
  risk: RiskResponse;
}> {
  const results: Record<string, AgentResponse> = {};

  // Step 1: Run all 5 agents sequentially
  const agents = ["sec", "news", "social", "chart", "analyst"];

  for (const agentId of agents) {
    try {
      const result = await runSingleAgent(ticker, agentId);
      results[agentId] = result;
      if (onAgentComplete) {
        onAgentComplete(agentId, result);
      }
    } catch (error) {
      results[agentId] = {
        ticker,
        agent: agentId,
        error: error instanceof Error ? error.message : "Unknown error",
        summary: `Failed to run ${agentId} agent`,
      };
      if (onAgentComplete) {
        onAgentComplete(agentId, results[agentId]);
      }
    }
  }

  // Step 2: Run Governor with all agent summaries
  let governorResult: GovernorResponse;
  try {
    governorResult = await runGovernorWithAgentResults(ticker, results);
    if (onGovernorComplete) {
      onGovernorComplete(governorResult);
    }
  } catch (error) {
    governorResult = {
      ticker,
      agent: "governor",
      error: error instanceof Error ? error.message : "Unknown error",
      summary_report: "Governor analysis failed",
    };
    if (onGovernorComplete) {
      onGovernorComplete(governorResult);
    }
  }

  // Step 3: Run Risk Assessment with all results
  let riskResult: RiskResponse;
  try {
    riskResult = await runRiskWithAgentResults(ticker, results);
    if (onRiskComplete) {
      onRiskComplete(riskResult);
    }
  } catch (error) {
    riskResult = {
      ticker,
      agent: "risk",
      error: error instanceof Error ? error.message : "Unknown error",
      summary_report: "Risk assessment failed",
    };
    if (onRiskComplete) {
      onRiskComplete(riskResult);
    }
  }

  return {
    agents: results,
    governor: governorResult,
    risk: riskResult,
  };
}

// Individual agent runners
export async function runSingleAgent(
  ticker: string,
  agentId: string
): Promise<AgentResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agents/${agentId}`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: `${agentId} agent failed` }));
    throw new Error(error.detail || `Failed to run ${agentId} agent`);
  }

  return response.json();
}

export async function runGovernorWithAgentResults(
  ticker: string,
  agentResults: Record<string, AgentResponse>
): Promise<GovernorResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agents/governor`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ticker: ticker.toUpperCase(),
      sec_summary:
        agentResults.sec?.summary || agentResults.sec?.summary_report,
      news_summary:
        agentResults.news?.summary || agentResults.news?.summary_report,
      social_summary:
        agentResults.social?.summary || agentResults.social?.summary_report,
      chart_summary:
        agentResults.chart?.summary || agentResults.chart?.summary_report,
      analyst_summary:
        agentResults.analyst?.summary || agentResults.analyst?.summary_report,
    }),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Governor agent failed" }));
    throw new Error(error.detail || "Failed to run governor agent");
  }

  return response.json();
}

export async function runRiskWithAgentResults(
  ticker: string,
  agentResults: Record<string, AgentResponse>
): Promise<RiskResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agents/risk`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ticker: ticker.toUpperCase(),
      sec_summary:
        agentResults.sec?.summary || agentResults.sec?.summary_report,
      news_summary:
        agentResults.news?.summary || agentResults.news?.summary_report,
      social_summary:
        agentResults.social?.summary || agentResults.social?.summary_report,
      chart_summary:
        agentResults.chart?.summary || agentResults.chart?.summary_report,
      analyst_summary:
        agentResults.analyst?.summary || agentResults.analyst?.summary_report,
    }),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Risk agent failed" }));
    throw new Error(error.detail || "Failed to run risk agent");
  }

  return response.json();
}

// Monte Carlo
export interface MCRolloutResponse {
  status: string;
  ticker: string;
  t: number;
  sims: number;
  days: number[];
  forecast: number[];
}

export async function runMCRollout(
  ticker: string,
  t: number = 30,
  sims: number = 1000
): Promise<MCRolloutResponse> {
  const response = await fetch(`${API_BASE_URL}/api/mc_rollout`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      ticker: ticker.toUpperCase(),
      t,
      sims,
    }),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "MC Rollout failed" }));
    throw new Error(error.detail || "Failed to run Monte Carlo simulation");
  }

  return response.json();
}

// News Agent (kept from original for compatibility)
export interface NewsArticle {
  title: string;
  sentiment: string;
  impact: number;
  weighted_impact: number;
  recency_weight: number;
  published_at: string;
  reason: string;
  url: string;
  source: string;
  source_id: string;
}

export interface NewsAgentResponse {
  company: string;
  total_articles_analyzed: number;
  sentiment_breakdown: {
    bullish: number;
    bearish: number;
    neutral: number;
  };
  weighted_sentiment_score: number;
  bullish_articles: NewsArticle[];
  bearish_articles: NewsArticle[];
  neutral_articles: NewsArticle[];
  high_impact_articles: NewsArticle[];
}

export async function getNewsAgentData(
  ticker: string
): Promise<NewsAgentResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agents/news`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });

  if (!response.ok) {
    const error = await response
      .json()
      .catch(() => ({ detail: "Failed to fetch news" }));
    throw new Error(error.detail || "Failed to fetch news data");
  }

  return response.json();
}

// Health check
export async function checkHealth(): Promise<{
  status: string;
  service: string;
}> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error("Health check failed");
  }

  return response.json();
}
