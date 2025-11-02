const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

// Types matching your backend
export interface AnalysisState {
  ticker: string;
  timestamp: string;
  workflow_status: 'completed_successfully' | 'completed_with_errors' | 'failed';
  sec_agent_status?: string;
  news_agent_status?: string;
  social_agent_status?: string;
  chart_agent_status?: string;
  analyst_agent_status?: string;
  governor_status?: string;
  risk_status?: string;
  governor_summary?: string;
  risk_summary?: string;
  sec_summary?: string;
  news_summary?: string;
  social_summary?: string;
  chart_summary?: string;
  analyst_summary?: string;
  errors?: string[];
  warnings?: string[];
}

export interface AnalysisListItem {
  ticker: string;
  timestamp: string;
  workflow_status: string;
  agents_completed: number;
  governor_status: string;
  risk_status: string;
  errors?: string[];
}

export interface AnalysisIndex {
  analyses: AnalysisListItem[];
}

// API Functions
export async function analyzeStock(ticker: string): Promise<AnalysisState> {
  const response = await fetch(`${API_BASE_URL}/api/analyze`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Analysis failed' }));
    throw new Error(error.detail || 'Failed to analyze stock');
  }

  return response.json();
}

export async function rerunAgent(ticker: string, agent: string): Promise<AnalysisState> {
  const response = await fetch(`${API_BASE_URL}/api/rerun`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ 
      ticker: ticker.toUpperCase(),
      agent: agent.toLowerCase()
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Rerun failed' }));
    throw new Error(error.detail || 'Failed to rerun agent');
  }

  return response.json();
}

export async function getAllAnalyses(): Promise<AnalysisIndex> {
  const response = await fetch(`${API_BASE_URL}/api/analyses`);

  if (!response.ok) {
    throw new Error('Failed to fetch analyses');
  }

  return response.json();
}

export async function getAnalysisState(ticker: string): Promise<AnalysisState> {
  const response = await fetch(`${API_BASE_URL}/api/analysis/${ticker.toUpperCase()}`);

  if (!response.ok) {
    throw new Error('Analysis not found');
  }

  return response.json();
}

export async function getSummaryReport(ticker: string, agent: string): Promise<string | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/report/summary/${ticker.toUpperCase()}/${agent.toLowerCase()}`
  );

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return data.content;
}

export async function getDetailedReport(ticker: string, agent: string): Promise<string | null> {
  const response = await fetch(
    `${API_BASE_URL}/api/report/detailed/${ticker.toUpperCase()}/${agent.toLowerCase()}`
  );

  if (!response.ok) {
    return null;
  }

  const data = await response.json();
  return data.content;
}

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

export async function getNewsAgentData(ticker: string): Promise<NewsAgentResponse> {
  const response = await fetch(`${API_BASE_URL}/api/agents/news`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ ticker: ticker.toUpperCase() }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: 'Failed to fetch news' }));
    throw new Error(error.detail || 'Failed to fetch news data');
  }

  return response.json();
}

export async function checkHealth(): Promise<{ status: string; service: string }> {
  const response = await fetch(`${API_BASE_URL}/api/health`);

  if (!response.ok) {
    throw new Error('Health check failed');
  }

  return response.json();
}