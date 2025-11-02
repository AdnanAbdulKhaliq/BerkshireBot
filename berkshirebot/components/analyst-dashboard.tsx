"use client";

import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts";
import {
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from "@/components/ui/chart";
import {
  Activity,
  TrendingUp,
  TrendingDown,
  Minus,
  Newspaper,
  Brain,
  FileText,
  AlertCircle,
  Loader2,
  ExternalLink,
} from "lucide-react";
import { useState, useEffect } from "react";
import {
  analyzeStock,
  rerunAgent,
  getAnalysisState,
  getNewsAgentData,
  runMonteCarloSimulation,
  type AnalysisState,
  type NewsArticle,
  type NewsAgentResponse,
  type MonteCarloResult,
} from "@/lib/api";

// Mock data for stock price chart (you can replace this with real data later)
const stockData = [
  { time: "9:30", price: 245.2 },
  { time: "10:00", price: 247.8 },
  { time: "10:30", price: 246.5 },
  { time: "11:00", price: 249.3 },
  { time: "11:30", price: 251.7 },
  { time: "12:00", price: 250.4 },
  { time: "12:30", price: 252.9 },
  { time: "13:00", price: 254.1 },
  { time: "13:30", price: 253.2 },
  { time: "14:00", price: 255.8 },
  { time: "14:30", price: 257.3 },
  { time: "15:00", price: 256.1 },
  { time: "15:30", price: 258.4 },
  { time: "16:00", price: 259.7 },
];

interface DisplayNewsArticle {
  id: string;
  title: string;
  source: string;
  time: string;
  sentiment: "positive" | "negative" | "neutral";
  summary: string;
  url: string;
}

interface AgentData {
  name: string;
  sentiment: "positive" | "negative" | "neutral";
  score: number;
  status: string;
  summary: string;
  details?: any;
}

function getSentimentIcon(sentiment: string) {
  switch (sentiment) {
    case "positive":
      return <TrendingUp className="h-4 w-4 text-accent" />;
    case "negative":
      return <TrendingDown className="h-4 w-4 text-destructive" />;
    default:
      return <Minus className="h-4 w-4 text-muted-foreground" />;
  }
}

function getSentimentColor(sentiment: string) {
  switch (sentiment) {
    case "positive":
      return "bg-accent/10 text-accent border-accent/20";
    case "negative":
      return "bg-destructive/10 text-destructive border-destructive/20";
    default:
      return "bg-muted text-muted-foreground border-border";
  }
}

function calculateSentimentScore(summary: string | undefined): number {
  if (!summary) return 5.0;

  const text = summary.toLowerCase();
  let score = 5.0;

  // Positive indicators
  const positiveWords = [
    "strong",
    "growth",
    "positive",
    "bullish",
    "upgraded",
    "outperform",
    "buy",
    "excellent",
    "improving",
  ];
  const negativeWords = [
    "weak",
    "decline",
    "negative",
    "bearish",
    "downgraded",
    "underperform",
    "sell",
    "poor",
    "deteriorating",
  ];

  positiveWords.forEach((word) => {
    if (text.includes(word)) score += 0.5;
  });

  negativeWords.forEach((word) => {
    if (text.includes(word)) score -= 0.5;
  });

  return Math.max(1.0, Math.min(10.0, score));
}

function determineSentiment(
  summary: string | undefined
): "positive" | "negative" | "neutral" {
  const score = calculateSentimentScore(summary);
  if (score > 6.5) return "positive";
  if (score < 4.5) return "negative";
  return "neutral";
}

function convertStateToAgentData(state: AnalysisState): AgentData[] {
  const agents: AgentData[] = [
    {
      name: "SEC Agent",
      sentiment: determineSentiment(state.sec_summary),
      score: calculateSentimentScore(state.sec_summary),
      status: state.sec_agent_status || "pending",
      summary: state.sec_summary || "Analysis pending...",
      details: { content: state.sec_summary },
    },
    {
      name: "News Agent",
      sentiment: determineSentiment(state.news_summary),
      score: calculateSentimentScore(state.news_summary),
      status: state.news_agent_status || "pending",
      summary: state.news_summary || "Analysis pending...",
      details: { content: state.news_summary },
    },
    {
      name: "Social Agent",
      sentiment: determineSentiment(state.social_summary),
      score: calculateSentimentScore(state.social_summary),
      status: state.social_agent_status || "pending",
      summary: state.social_summary || "Analysis pending...",
      details: { content: state.social_summary },
    },
    {
      name: "Chart Agent",
      sentiment: determineSentiment(state.chart_summary),
      score: calculateSentimentScore(state.chart_summary),
      status: state.chart_agent_status || "pending",
      summary: state.chart_summary || "Analysis pending...",
      details: { content: state.chart_summary },
    },
    {
      name: "Analyst Agent",
      sentiment: determineSentiment(state.analyst_summary),
      score: calculateSentimentScore(state.analyst_summary),
      status: state.analyst_agent_status || "pending",
      summary: state.analyst_summary || "Analysis pending...",
      details: { content: state.analyst_summary },
    },
  ];

  return agents;
}

function formatTimeAgo(dateString: string): string {
  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffHours / 24);

  if (diffDays > 0) return `${diffDays} day${diffDays > 1 ? "s" : ""} ago`;
  if (diffHours > 0) return `${diffHours} hour${diffHours > 1 ? "s" : ""} ago`;
  return "Just now";
}

function convertNewsToDisplay(
  newsData: NewsAgentResponse
): DisplayNewsArticle[] {
  const allArticles: NewsArticle[] = [
    ...newsData.high_impact_articles.slice(0, 3),
    ...newsData.bullish_articles,
    ...newsData.bearish_articles,
    ...newsData.neutral_articles,
  ];

  // Remove duplicates
  const seen = new Set<string>();
  const uniqueArticles = allArticles.filter((article) => {
    if (seen.has(article.url)) return false;
    seen.add(article.url);
    return true;
  });

  return uniqueArticles.map((article, index) => ({
    id: article.url || `article-${index}`,
    title: article.title,
    source: article.source,
    time: formatTimeAgo(article.published_at),
    sentiment:
      article.sentiment.toLowerCase() === "bullish"
        ? ("positive" as const)
        : article.sentiment.toLowerCase() === "bearish"
        ? ("negative" as const)
        : ("neutral" as const),
    summary: article.reason,
    url: article.url,
  }));
}

export function AnalystDashboard() {
  const [ticker, setTicker] = useState("TSLA");
  const [loading, setLoading] = useState(false);
  const [analysisState, setAnalysisState] = useState<AnalysisState | null>(
    null
  );
  const [agentData, setAgentData] = useState<AgentData[]>([]);
  const [selectedAgent, setSelectedAgent] = useState<AgentData | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [newsArticles, setNewsArticles] = useState<DisplayNewsArticle[]>([]);
  const [selectedNews, setSelectedNews] = useState<DisplayNewsArticle | null>(
    null
  );
  const [mcData, setMcData] = useState<{ day: number; price: number }[]>([]);
  const [mcLoading, setMcLoading] = useState(false);

  const handleAnalyze = async () => {
    if (!ticker.trim()) {
      setError("Please enter a ticker symbol");
      return;
    }

    setLoading(true);
    setMcLoading(true);
    setError(null);

    try {
      // Fetch news agent data and Monte Carlo simulation in parallel
      const [newsData, mcResult] = await Promise.all([
        getNewsAgentData(ticker),
        runMonteCarloSimulation(ticker, 30, 1000),
      ]);

      // Set news articles
      setNewsArticles(convertNewsToDisplay(newsData));

      // Set Monte Carlo data
      if (mcResult.status === "success" && mcResult.forecast) {
        const chartData = mcResult.forecast.map((price, index) => ({
          day: index + 1,
          price: price,
        }));
        setMcData(chartData);
      }

      // Create a simple agent data entry for news agent to show sentiment
      const newsAgentData: AgentData = {
        name: "News Agent",
        sentiment:
          newsData.weighted_sentiment_score > 0.1
            ? "positive"
            : newsData.weighted_sentiment_score < -0.1
            ? "negative"
            : "neutral",
        score: ((newsData.weighted_sentiment_score + 1) / 2) * 10, // Convert -1 to 1 scale to 0-10
        status: "success",
        summary: `Analyzed ${
          newsData.total_articles_analyzed
        } articles. Bullish: ${
          newsData.sentiment_breakdown.bullish
        }, Bearish: ${newsData.sentiment_breakdown.bearish}, Neutral: ${
          newsData.sentiment_breakdown.neutral
        }. Weighted sentiment score: ${newsData.weighted_sentiment_score.toFixed(
          2
        )}`,
        details: { content: JSON.stringify(newsData, null, 2) },
      };

      setAgentData([newsAgentData]);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
      setMcLoading(false);
    }
  };

  const handleRerun = async (agentName: string) => {
    if (!ticker) return;

    setLoading(true);
    setError(null);

    const agentId = agentName.toLowerCase().replace(" agent", "");

    try {
      await rerunAgent(ticker, agentId);
      const fullState = await getAnalysisState(ticker);
      setAnalysisState(fullState);
      setAgentData(convertStateToAgentData(fullState));
    } catch (err) {
      setError(err instanceof Error ? err.message : "Rerun failed");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header with Search */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-balance">
            Analyst Swarm Dashboard
          </h1>
          <p className="text-muted-foreground mt-1">
            Multi-agent financial risk assessment system
          </p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter ticker"
            className="px-3 py-1.5 bg-background border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            onKeyPress={(e) => e.key === "Enter" && handleAnalyze()}
            disabled={loading}
          />
          <button
            onClick={handleAnalyze}
            disabled={loading}
            className="px-4 py-1.5 bg-primary hover:bg-primary/90 disabled:bg-muted text-primary-foreground rounded-md text-sm font-medium transition-colors flex items-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="h-3.5 w-3.5 animate-spin" />
                Analyzing...
              </>
            ) : (
              "Analyze"
            )}
          </button>
        </div>
      </div>

      {error && (
        <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm">
          {error}
        </div>
      )}

      {/* Status Badge */}
      {analysisState && (
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="gap-1.5 px-3 py-1.5">
            <Activity className="h-3.5 w-3.5" />
            <span className="font-mono text-xs">{ticker}</span>
          </Badge>
          <Badge
            className={
              analysisState.workflow_status === "completed_successfully"
                ? "bg-accent text-accent-foreground"
                : "bg-destructive text-destructive-foreground"
            }
          >
            {analysisState.workflow_status}
          </Badge>
          <span className="text-xs text-muted-foreground">
            {analysisState.timestamp}
          </span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Stock Price Chart - Top Left */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Stock Price Movement
            </CardTitle>
            <CardDescription>
              Monte Carlo Simulation - 30 Day Forecast
            </CardDescription>
          </CardHeader>
          <CardContent>
            {mcLoading ? (
              <div className="h-[300px] flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary" />
              </div>
            ) : mcData.length > 0 ? (
              <ChartContainer
                config={{
                  price: {
                    label: "Price",
                    color: "hsl(var(--chart-1))",
                  },
                }}
                className="h-[300px]"
              >
                <AreaChart data={mcData}>
                  <defs>
                    <linearGradient id="fillPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop
                        offset="5%"
                        stopColor="var(--color-price)"
                        stopOpacity={0.8}
                      />
                      <stop
                        offset="95%"
                        stopColor="var(--color-price)"
                        stopOpacity={0.1}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid vertical={false} />
                  <XAxis
                    dataKey="day"
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    tickFormatter={(value) => `Day ${value}`}
                  />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    tickFormatter={(value) => `$${value.toFixed(0)}`}
                  />
                  <ChartTooltip
                    cursor={false}
                    content={<ChartTooltipContent indicator="line" />}
                  />
                  <Area
                    dataKey="price"
                    type="natural"
                    fill="url(#fillPrice)"
                    fillOpacity={0.4}
                    stroke="var(--color-price)"
                    stackId="a"
                  />
                </AreaChart>
              </ChartContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-sm text-muted-foreground">
                Run analysis to see Monte Carlo forecast
              </div>
            )}
          </CardContent>
        </Card>

        {/* News Articles - Top Right */}
        <Card className="lg:row-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Newspaper className="h-5 w-5 text-primary" />
              Top News Articles
            </CardTitle>
            <CardDescription>Latest market-moving news</CardDescription>
          </CardHeader>
          <CardContent className="h-[600px] overflow-y-auto">
            <div className="space-y-4 pr-2">
              {newsArticles.length > 0 ? (
                newsArticles.map((article) => (
                  <button
                    key={article.id}
                    onClick={() => setSelectedNews(article)}
                    className="w-full flex gap-3 rounded-lg border border-border bg-card p-3 transition-all hover:bg-accent/5 hover:border-primary/50 hover:shadow-md cursor-pointer text-left"
                  >
                    <div className="mt-1">
                      {getSentimentIcon(article.sentiment)}
                    </div>
                    <div className="flex-1 space-y-1">
                      <h4 className="text-sm font-medium leading-snug text-pretty">
                        {article.title}
                      </h4>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{article.source}</span>
                        <span>•</span>
                        <span>{article.time}</span>
                      </div>
                    </div>
                  </button>
                ))
              ) : (
                <div className="text-center text-sm text-muted-foreground py-8">
                  No news articles available. Click "Analyze" to fetch latest
                  news.
                </div>
              )}
            </div>
          </CardContent>
        </Card>

        {/* Agent Sentiments - Bottom Left */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Brain className="h-5 w-5 text-primary" />
              Agent Sentiment Analysis
            </CardTitle>
            <CardDescription>
              Multi-perspective risk assessment from specialist agents
            </CardDescription>
          </CardHeader>
          <CardContent>
            {agentData.length > 0 ? (
              <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
                {agentData.map((agent) => (
                  <button
                    key={agent.name}
                    onClick={() => setSelectedAgent(agent)}
                    disabled={loading}
                    className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4 text-left transition-all hover:bg-accent/5 hover:border-primary/50 hover:shadow-md cursor-pointer disabled:opacity-50 disabled:cursor-not-allowed"
                  >
                    <div className="flex items-center justify-between">
                      <h4 className="text-sm font-semibold">{agent.name}</h4>
                      <Badge
                        variant="outline"
                        className={getSentimentColor(agent.sentiment)}
                      >
                        {agent.sentiment}
                      </Badge>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold font-mono">
                        {agent.score.toFixed(1)}
                      </span>
                      <span className="text-sm text-muted-foreground">/10</span>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                      {agent.summary}
                    </p>
                    <div className="mt-1">
                      <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
                        <div
                          className="h-full rounded-full bg-primary transition-all"
                          style={{ width: `${agent.score * 10}%` }}
                        />
                      </div>
                    </div>
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleRerun(agent.name);
                      }}
                      disabled={loading}
                      className="mt-2 w-full px-3 py-1.5 bg-secondary hover:bg-secondary/80 disabled:bg-muted text-secondary-foreground rounded-md text-xs transition-colors"
                    >
                      Rerun Agent
                    </button>
                  </button>
                ))}
              </div>
            ) : (
              <div className="text-center py-12 text-muted-foreground">
                <Brain className="h-12 w-12 mx-auto mb-3 opacity-20" />
                <p>Enter a ticker and click Analyze to begin</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Governor & Risk Summary */}
      {analysisState &&
        (analysisState.governor_summary || analysisState.risk_summary) && (
          <div className="grid gap-6 lg:grid-cols-2">
            {analysisState.governor_summary && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <FileText className="h-5 w-5 text-primary" />
                    Governor Summary
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
                    {analysisState.governor_summary}
                  </p>
                </CardContent>
              </Card>
            )}

            {analysisState.risk_summary && (
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertCircle className="h-5 w-5 text-destructive" />
                    Risk Assessment
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <p className="text-sm text-muted-foreground whitespace-pre-wrap leading-relaxed">
                    {analysisState.risk_summary}
                  </p>
                </CardContent>
              </Card>
            )}
          </div>
        )}

      <Dialog
        open={!!selectedAgent}
        onOpenChange={(open) => !open && setSelectedAgent(null)}
      >
        <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
          {selectedAgent && (
            <>
              <DialogHeader>
                <div className="flex items-center justify-between">
                  <DialogTitle className="text-2xl">
                    {selectedAgent.name}
                  </DialogTitle>
                  <Badge
                    variant="outline"
                    className={getSentimentColor(selectedAgent.sentiment)}
                  >
                    {selectedAgent.sentiment}
                  </Badge>
                </div>
                <DialogDescription className="text-base">
                  {selectedAgent.summary}
                </DialogDescription>
              </DialogHeader>

              <div className="space-y-6 mt-4">
                {/* Score Display */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold font-mono">
                      {selectedAgent.score.toFixed(1)}
                    </span>
                    <span className="text-lg text-muted-foreground">/10</span>
                  </div>
                  <div className="flex-1">
                    <div className="h-3 w-full rounded-full bg-secondary overflow-hidden">
                      <div
                        className="h-full rounded-full bg-primary transition-all"
                        style={{ width: `${selectedAgent.score * 10}%` }}
                      />
                    </div>
                  </div>
                </div>

                {/* Full Content */}
                <div>
                  <h4 className="font-semibold mb-2">Full Analysis</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {selectedAgent.details?.content ||
                      "No detailed analysis available"}
                  </p>
                </div>

                <div className="p-3 rounded-lg bg-muted/50 text-xs text-muted-foreground">
                  Status: {selectedAgent.status}
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* News Article Detail Dialog */}
      <Dialog
        open={!!selectedNews}
        onOpenChange={(open) => !open && setSelectedNews(null)}
      >
        <DialogContent className="max-w-2xl">
          {selectedNews && (
            <>
              <DialogHeader>
                <div className="flex items-start justify-between gap-4">
                  <DialogTitle className="text-xl leading-tight pr-8">
                    {selectedNews.title}
                  </DialogTitle>
                  <Badge
                    variant="outline"
                    className={getSentimentColor(selectedNews.sentiment)}
                  >
                    {selectedNews.sentiment}
                  </Badge>
                </div>
                <div className="flex items-center gap-2 text-sm text-muted-foreground pt-2">
                  <span className="font-medium">{selectedNews.source}</span>
                  <span>•</span>
                  <span>{selectedNews.time}</span>
                </div>
              </DialogHeader>

              <div className="space-y-4 mt-4">
                {/* Summary */}
                <div>
                  <h4 className="font-semibold mb-2 text-sm">AI Summary</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {selectedNews.summary}
                  </p>
                </div>

                {/* Link to Full Article */}
                <div className="pt-4 border-t">
                  <a
                    href={selectedNews.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm font-medium text-primary hover:underline"
                  >
                    Read Full Article
                    <ExternalLink className="h-4 w-4" />
                  </a>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}
