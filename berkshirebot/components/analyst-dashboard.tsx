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
  RefreshCw,
  Play,
  CheckCircle,
  XCircle,
} from "lucide-react";
import { useState } from "react";
import {
  runFullSequentialAnalysis,
  runSingleAgent,
  runMCRollout,
  getNewsAgentData,
  type AgentResponse,
  type GovernorResponse,
  type RiskResponse,
  type MCRolloutResponse,
  type NewsArticle,
  type NewsAgentResponse,
} from "@/lib/api";

const AGENTS = [
  { id: 'sec', name: 'SEC Agent', icon: '📋', color: 'bg-blue-500' },
  { id: 'news', name: 'News Agent', icon: '📰', color: 'bg-green-500' },
  { id: 'social', name: 'Social Agent', icon: '💬', color: 'bg-purple-500' },
  { id: 'chart', name: 'Chart Agent', icon: '📊', color: 'bg-orange-500' },
  { id: 'analyst', name: 'Analyst Agent', icon: '🔍', color: 'bg-red-500' },
]

type AgentStatus = 'idle' | 'running' | 'success' | 'error'

interface AgentState {
  id: string
  name: string
  status: AgentStatus
  data: AgentResponse | null
  loading: boolean
}

type Recommendation = 'STRONG_BUY' | 'BUY' | 'HOLD' | 'SELL' | 'STRONG_SELL' | null

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

function extractKeyPoints(detailedReport: string | undefined): string[] {
  if (!detailedReport) return []
  
  const lines = detailedReport.split('\n')
  const points: string[] = []
  
  for (const line of lines) {
    const trimmed = line.trim()
    // Match markdown bullets, numbered lists, or lines starting with **
    if (trimmed.startsWith('- ') || trimmed.startsWith('• ') || 
        trimmed.startsWith('* ') || /^\d+\./.test(trimmed) ||
        (trimmed.startsWith('**') && trimmed.length > 10)) {
      let point = trimmed
        .replace(/^[-•*]\s*/, '')
        .replace(/^\d+\.\s*/, '')
        .replace(/^\*\*(.*?)\*\*:?/, '$1')
        .trim()
      
      if (point && point.length > 10) {
        points.push(point)
      }
    }
    if (points.length >= 6) break
  }
  
  return points
}

function getRecommendationColor(riskScore: number | undefined): string {
  if (!riskScore) return ''
  if (riskScore < 25) return 'bg-green-600 text-white'
  if (riskScore < 40) return 'bg-green-500 text-white'
  if (riskScore < 60) return 'bg-yellow-500 text-black'
  if (riskScore < 75) return 'bg-red-500 text-white'
  return 'bg-red-600 text-white'
}

function getRecommendation(riskScore: number | undefined): Recommendation {
  if (!riskScore) return null
  if (riskScore < 25) return 'STRONG_BUY'
  if (riskScore < 40) return 'BUY'
  if (riskScore < 60) return 'HOLD'
  if (riskScore < 75) return 'SELL'
  return 'STRONG_SELL'
}

function getSummary(data: AgentResponse | null): string {
  if (!data) return 'Not yet analyzed'
  if (data.error) return data.error
  return data.summary || data.summary_report || 'No summary available'
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
  const [analysisRunning, setAnalysisRunning] = useState(false)
  const [currentStep, setCurrentStep] = useState<string>("")
  const [error, setError] = useState<string | null>(null);
  
  const [agentStates, setAgentStates] = useState<Record<string, AgentState>>(
    Object.fromEntries(
      AGENTS.map(agent => [
        agent.id,
        { id: agent.id, name: agent.name, status: 'idle' as AgentStatus, data: null, loading: false }
      ])
    )
  )
  
  const [governorData, setGovernorData] = useState<GovernorResponse | null>(null)
  const [riskData, setRiskData] = useState<RiskResponse | null>(null)
  const [recommendation, setRecommendation] = useState<Recommendation>(null)
  
  const [selectedAgent, setSelectedAgent] = useState<AgentState | null>(null)
  const [showGovernorDialog, setShowGovernorDialog] = useState(false)
  const [showRiskDialog, setShowRiskDialog] = useState(false)
  
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
    setAnalysisRunning(true);
    setError(null);
    setRecommendation(null);
    
    // Reset agent states
    setAgentStates(prev => 
      Object.fromEntries(
        Object.entries(prev).map(([id, state]) => [
          id,
          { ...state, status: 'idle' as AgentStatus, data: null, loading: false }
        ])
      )
    );
    setGovernorData(null);
    setRiskData(null);

    try {
      // Run Monte Carlo and News in parallel with full sequential analysis
      const mcPromise = runMCRollout(ticker, 30, 1000).then(mcResult => {
        if (mcResult.status === "success" && mcResult.forecast) {
          const chartData = mcResult.forecast.map((price: number, index: number) => ({
            day: index + 1,
            price: price,
          }));
          setMcData(chartData);
        }
        setMcLoading(false);
      });

      const newsPromise = getNewsAgentData(ticker).then(newsData => {
        setNewsArticles(convertNewsToDisplay(newsData));
      });

      // Run full sequential analysis (all 5 agents + governor + risk)
      const analysisPromise = runFullSequentialAnalysis(
        ticker,
        (agentId, result) => {
          setAgentStates(prev => ({
            ...prev,
            [agentId]: {
              ...prev[agentId],
              status: result.error ? 'error' : 'success',
              data: result,
              loading: false
            }
          }));
          setCurrentStep(`Completed: ${agentId} agent`);
        },
        (result) => {
          setGovernorData(result);
          setCurrentStep('Completed: Governor synthesis');
        },
        (result) => {
          setRiskData(result);
          setCurrentStep('Completed: Risk assessment');
          
          const rec = getRecommendation(result.overall_risk_score);
          setRecommendation(rec);
        }
      );

      // Wait for all operations to complete
      await Promise.all([mcPromise, newsPromise, analysisPromise]);
      
      setCurrentStep('Analysis complete!');
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed");
    } finally {
      setLoading(false);
      setAnalysisRunning(false);
    }
  };

  const handleFullAnalysis = async () => {
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol')
      return
    }

    setAnalysisRunning(true)
    setError(null)
    setRecommendation(null)
    
    setAgentStates(prev => 
      Object.fromEntries(
        Object.entries(prev).map(([id, state]) => [
          id,
          { ...state, status: 'idle' as AgentStatus, data: null, loading: false }
        ])
      )
    )
    setGovernorData(null)
    setRiskData(null)

    try {
      await runFullSequentialAnalysis(
        ticker,
        (agentId, result) => {
          setAgentStates(prev => ({
            ...prev,
            [agentId]: {
              ...prev[agentId],
              status: result.error ? 'error' : 'success',
              data: result,
              loading: false
            }
          }))
          setCurrentStep(`Completed: ${agentId} agent`)
        },
        (result) => {
          setGovernorData(result)
          setCurrentStep('Completed: Governor synthesis')
        },
        (result) => {
          setRiskData(result)
          setCurrentStep('Completed: Risk assessment')
          
          const rec = getRecommendation(result.overall_risk_score)
          setRecommendation(rec)
        }
      )
      
      setCurrentStep('Analysis complete!')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setAnalysisRunning(false)
    }
  }

  const handleRunSingleAgent = async (agentId: string) => {
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol')
      return
    }

    setAgentStates(prev => ({
      ...prev,
      [agentId]: { ...prev[agentId], loading: true, status: 'running' }
    }))
    setError(null)

    try {
      const result = await runSingleAgent(ticker, agentId)
      setAgentStates(prev => ({
        ...prev,
        [agentId]: {
          ...prev[agentId],
          status: result.error ? 'error' : 'success',
          data: result,
          loading: false
        }
      }))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Agent run failed')
      setAgentStates(prev => ({
        ...prev,
        [agentId]: { 
          ...prev[agentId], 
          loading: false, 
          status: 'error',
          data: {
            ticker,
            agent: agentId,
            error: err instanceof Error ? err.message : 'Unknown error',
            summary: `Failed to run ${agentId} agent`
          }
        }
      }))
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

      {/* Error Display */}
      {error && (
        <div className="p-3 bg-destructive/10 border border-destructive/20 rounded-lg text-destructive text-sm">
          {error}
        </div>
      )}

      {/* Current Step */}
      {analysisRunning && currentStep && (
        <div className="p-3 bg-primary/10 border border-primary/20 rounded-lg text-primary text-sm flex items-center gap-2">
          <Loader2 className="h-4 w-4 animate-spin" />
          {currentStep}
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
                    label: "Median Forecast",
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
                        stopOpacity={0.3}
                      />
                      <stop
                        offset="95%"
                        stopColor="var(--color-price)"
                        stopOpacity={0.05}
                      />
                    </linearGradient>
                  </defs>
                  <CartesianGrid
                    vertical={false}
                    strokeDasharray="3 3"
                    opacity={0.3}
                  />
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
                    content={<ChartTooltipContent indicator="dot" />}
                  />
                  <Area
                    dataKey="price"
                    type="monotone"
                    fill="url(#fillPrice)"
                    fillOpacity={1}
                    stroke="var(--color-price)"
                    strokeWidth={2.5}
                    dot={false}
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

        {/* Recommendation Card */}
        <Card className={recommendation ? getRecommendationColor(riskData?.overall_risk_score) : ''}>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              {recommendation ? (
                <>
                  {recommendation.includes('BUY') ? <TrendingUp className="h-5 w-5" /> : <TrendingDown className="h-5 w-5" />}
                  Recommendation
                </>
              ) : (
                <>
                  <AlertCircle className="h-5 w-5" />
                  Awaiting Analysis
                </>
              )}
            </CardTitle>
            <CardDescription className={recommendation ? 'opacity-90' : ''}>
              Based on risk assessment
            </CardDescription>
          </CardHeader>
          <CardContent>
            {recommendation ? (
              <div className="space-y-4">
                <div className="text-4xl font-bold text-center">
                  {recommendation.replace('_', ' ')}
                </div>
                {riskData && (
                  <div className="space-y-2 text-sm opacity-90">
                    <div>Risk Score: {riskData.overall_risk_score}/100</div>
                    <div>Risk Level: {riskData.overall_risk_level}</div>
                  </div>
                )}
                <button
                  onClick={() => setShowRiskDialog(true)}
                  className="w-full px-3 py-2 bg-white/20 hover:bg-white/30 rounded-md text-sm transition-colors"
                >
                  View Full Risk Report
                </button>
              </div>
            ) : (
              <div className="text-center py-8 text-muted-foreground">
                <p>Run full analysis to get recommendation</p>
              </div>
            )}
          </CardContent>
        </Card>
      </div>

      {/* Agent Sentiments */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Brain className="h-5 w-5 text-primary" />
            Specialist Agents
          </CardTitle>
          <CardDescription>Individual agent analysis results</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-5">
            {AGENTS.map((agent) => {
              const state = agentStates[agent.id]
              const summary = getSummary(state.data)
              const sentiment = determineSentiment(summary)
              const score = calculateSentimentScore(summary)

              return (
                <div
                  key={agent.id}
                  className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4 transition-all hover:bg-accent/5 hover:border-primary/50 hover:shadow-md"
                >
                  <div className="flex items-center justify-between">
                    <div className={`w-10 h-10 ${agent.color} rounded-lg flex items-center justify-center text-2xl`}>
                      {agent.icon}
                    </div>
                    {state.loading && <Loader2 className="h-4 w-4 animate-spin text-primary" />}
                    {state.status === 'success' && <CheckCircle className="h-4 w-4 text-green-500" />}
                    {state.status === 'error' && <XCircle className="h-4 w-4 text-red-500" />}
                  </div>
                  <h4 className="text-sm font-semibold">{agent.name}</h4>
                  {state.status !== 'idle' && (
                    <>
                      <div className="flex items-baseline gap-2">
                        <span className="text-2xl font-bold font-mono">{score.toFixed(1)}</span>
                        <span className="text-xs text-muted-foreground">/10</span>
                        <Badge variant="outline" className="ml-auto">
                          {sentiment}
                        </Badge>
                      </div>
                      <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">
                        {summary}
                      </p>
                      <div className="h-1.5 w-full rounded-full bg-secondary overflow-hidden">
                        <div
                          className="h-full rounded-full bg-primary transition-all"
                          style={{ width: `${score * 10}%` }}
                        />
                      </div>
                    </>
                  )}
                  {state.status === 'idle' && (
                    <p className="text-xs text-muted-foreground">Not yet analyzed</p>
                  )}
                  <div className="flex gap-2 mt-auto">
                    <button
                      onClick={() => handleRunSingleAgent(agent.id)}
                      disabled={state.loading}
                      className="flex-1 px-3 py-1.5 bg-secondary hover:bg-secondary/80 disabled:bg-muted text-secondary-foreground rounded-md text-xs transition-colors flex items-center justify-center gap-1"
                    >
                      {state.loading ? (
                        <Loader2 className="h-3 w-3 animate-spin" />
                      ) : (
                        <>
                          <RefreshCw className="h-3 w-3" />
                          Re-run
                        </>
                      )}
                    </button>
                    <button
                      onClick={() => setSelectedAgent(state)}
                      disabled={state.status === 'idle'}
                      className="px-3 py-1.5 bg-secondary hover:bg-secondary/80 disabled:bg-muted text-secondary-foreground rounded-md text-xs transition-colors"
                    >
                      View
                    </button>
                  </div>
                </div>
              )
            })}
          </div>
        </CardContent>
      </Card>

      {/* Governor & Risk Summary */}
      {(governorData || riskData) && (
        <div className="grid gap-6 lg:grid-cols-2">
          {governorData && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <FileText className="h-5 w-5 text-primary" />
                      Governor Synthesis
                    </CardTitle>
                    <CardDescription>Comprehensive investment memo</CardDescription>
                  </div>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <p className="text-sm text-muted-foreground leading-relaxed line-clamp-4">
                    {governorData.summary_report || governorData.executive_summary || 'Processing...'}
                  </p>
                  <button
                    onClick={() => setShowGovernorDialog(true)}
                    className="px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm transition-colors"
                  >
                    View Full Investment Memo
                  </button>
                </div>
              </CardContent>
            </Card>
          )}
          
          {riskData && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <AlertCircle className="h-5 w-5 text-destructive" />
                      Risk Assessment
                    </CardTitle>
                    <CardDescription>Risk analysis and scoring</CardDescription>
                  </div>
                  <CheckCircle className="h-5 w-5 text-green-500" />
                </div>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Overall Risk Score</span>
                    <span className="text-2xl font-bold">{riskData.overall_risk_score}/100</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">Risk Level</span>
                    <Badge variant="outline">{riskData.overall_risk_level}</Badge>
                  </div>
                  <p className="text-sm text-muted-foreground leading-relaxed line-clamp-3">
                    {riskData.summary_report || 'Processing...'}
                  </p>
                  <button
                    onClick={() => setShowRiskDialog(true)}
                    className="px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm transition-colors"
                  >
                    View Full Risk Report
                  </button>
                </div>
              </CardContent>
            </Card>
          )}
        </div>
      )}

      {/* Agent Detail Dialog */}
      <Dialog open={!!selectedAgent} onOpenChange={() => setSelectedAgent(null)}>
        <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
          {selectedAgent && selectedAgent.data && (
            <>
              <DialogHeader>
                <DialogTitle className="text-2xl">{selectedAgent.name}</DialogTitle>
                <DialogDescription>Analysis report</DialogDescription>
              </DialogHeader>
              <div className="space-y-6 mt-4">
                <div className="p-4 rounded-lg bg-muted/50">
                  <h4 className="font-semibold mb-2">Summary</h4>
                  <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                    {getSummary(selectedAgent.data)}
                  </p>
                </div>
                
                {extractKeyPoints(selectedAgent.data.detailed_report || selectedAgent.data.detailed).length > 0 && (
                  <div>
                    <h4 className="font-semibold mb-3">Key Points</h4>
                    <ul className="space-y-2">
                      {extractKeyPoints(selectedAgent.data.detailed_report || selectedAgent.data.detailed).map((point, idx) => (
                        <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                          <span className="text-primary mt-1">•</span>
                          <span>{point}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div>
                  <button
                    onClick={() => {
                      const fullReport = selectedAgent.data?.detailed_report || selectedAgent.data?.detailed || 'No detailed report available'
                      const blob = new Blob([fullReport], { type: 'text/plain' })
                      const url = URL.createObjectURL(blob)
                      const a = document.createElement('a')
                      a.href = url
                      a.download = `${selectedAgent.name.replace(' ', '_')}_report.txt`
                      a.click()
                      URL.revokeObjectURL(url)
                    }}
                    className="px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm transition-colors"
                  >
                    Download Full Report
                  </button>
                </div>
              </div>
            </>
          )}
        </DialogContent>
      </Dialog>

      {/* Governor Dialog */}
      <Dialog open={showGovernorDialog} onOpenChange={setShowGovernorDialog}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl">Governor Investment Memo</DialogTitle>
            <DialogDescription>Comprehensive synthesis of all agent analyses</DialogDescription>
          </DialogHeader>
          <div className="space-y-6 mt-4">
            {governorData?.executive_summary && (
              <div className="p-4 rounded-lg bg-muted/50">
                <h4 className="font-semibold mb-2">Executive Summary</h4>
                <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                  {governorData.executive_summary}
                </p>
              </div>
            )}
            
            {governorData?.summary_report && !governorData?.executive_summary && (
              <div className="p-4 rounded-lg bg-muted/50">
                <h4 className="font-semibold mb-2">Summary</h4>
                <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                  {governorData.summary_report}
                </p>
              </div>
            )}
            
            {extractKeyPoints(governorData?.detailed_report).length > 0 && (
              <div>
                <h4 className="font-semibold mb-3">Key Findings</h4>
                <ul className="space-y-2">
                  {extractKeyPoints(governorData?.detailed_report).map((point, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="text-primary mt-1">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            <div>
              <button
                onClick={() => {
                  const fullReport = governorData?.detailed_report || 'No detailed report available'
                  const blob = new Blob([fullReport], { type: 'text/markdown' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `${ticker}_investment_memo.md`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm transition-colors"
              >
                Download Full Memo
              </button>
            </div>
          </div>
        </DialogContent>
      </Dialog>

      {/* Risk Dialog */}
      <Dialog open={showRiskDialog} onOpenChange={setShowRiskDialog}>
        <DialogContent className="max-w-4xl max-h-[85vh] overflow-y-auto">
          <DialogHeader>
            <DialogTitle className="text-2xl">Risk Assessment Report</DialogTitle>
            <DialogDescription>Comprehensive risk analysis and recommendation</DialogDescription>
          </DialogHeader>
          <div className="space-y-6 mt-4">
            {recommendation && riskData && (
              <div className={`p-6 rounded-lg ${getRecommendationColor(riskData.overall_risk_score)}`}>
                <div className="text-center space-y-3">
                  <div className="text-3xl font-bold">
                    {recommendation.replace('_', ' ')}
                  </div>
                  <div className="grid grid-cols-2 gap-4 text-sm opacity-90">
                    <div>
                      <div className="font-medium">Risk Score</div>
                      <div className="text-2xl font-bold">{riskData.overall_risk_score}/100</div>
                    </div>
                    <div>
                      <div className="font-medium">Risk Level</div>
                      <div className="text-2xl font-bold">{riskData.overall_risk_level}</div>
                    </div>
                  </div>
                </div>
              </div>
            )}
            
            {riskData?.summary_report && (
              <div className="p-4 rounded-lg bg-muted/50">
                <h4 className="font-semibold mb-2">Summary</h4>
                <p className="text-sm text-muted-foreground leading-relaxed whitespace-pre-wrap">
                  {riskData.summary_report}
                </p>
              </div>
            )}
            
            {extractKeyPoints(riskData?.detailed_report).length > 0 && (
              <div>
                <h4 className="font-semibold mb-3">Key Risk Factors</h4>
                <ul className="space-y-2">
                  {extractKeyPoints(riskData?.detailed_report).map((point, idx) => (
                    <li key={idx} className="flex items-start gap-2 text-sm text-muted-foreground">
                      <span className="text-destructive mt-1">•</span>
                      <span>{point}</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            
            <div>
              <button
                onClick={() => {
                  const fullReport = riskData?.detailed_report || 'No detailed report available'
                  const blob = new Blob([fullReport], { type: 'text/markdown' })
                  const url = URL.createObjectURL(blob)
                  const a = document.createElement('a')
                  a.href = url
                  a.download = `${ticker}_risk_assessment.md`
                  a.click()
                  URL.revokeObjectURL(url)
                }}
                className="px-4 py-2 bg-secondary hover:bg-secondary/80 text-secondary-foreground rounded-md text-sm transition-colors"
              >
                Download Full Risk Report
              </button>
            </div>
          </div>
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
