"use client"

import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Area, AreaChart, CartesianGrid, XAxis, YAxis } from "recharts"
import { ChartContainer, ChartTooltip, ChartTooltipContent } from "@/components/ui/chart"
import { Activity, TrendingUp, TrendingDown, Minus, Newspaper, Brain, FileText, AlertCircle, Loader2 } from "lucide-react"
import { useState, useEffect } from "react"
import { analyzeStock, rerunAgent, getAnalysisState, runMonteCarloSimulation, type AnalysisState, type MonteCarloResult } from "@/lib/api"

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
]

// Mock news articles (you can integrate real news data later)
const newsArticles = [
  {
    id: 1,
    title: "Tech Giant Announces Record Q4 Earnings",
    source: "Financial Times",
    time: "2 hours ago",
    sentiment: "positive",
  },
  {
    id: 2,
    title: "New Product Launch Expected to Drive Growth",
    source: "Bloomberg",
    time: "4 hours ago",
    sentiment: "positive",
  },
  {
    id: 3,
    title: "Regulatory Concerns Emerge in European Markets",
    source: "Reuters",
    time: "6 hours ago",
    sentiment: "negative",
  },
  {
    id: 4,
    title: "Analyst Upgrades Price Target to $300",
    source: "MarketWatch",
    time: "8 hours ago",
    sentiment: "positive",
  },
  {
    id: 5,
    title: "Supply Chain Improvements Show Progress",
    source: "WSJ",
    time: "10 hours ago",
    sentiment: "neutral",
  },
]

interface AgentData {
  name: string
  sentiment: "positive" | "negative" | "neutral"
  score: number
  status: string
  summary: string
  details?: any
}

function getSentimentIcon(sentiment: string) {
  switch (sentiment) {
    case "positive":
      return <TrendingUp className="h-4 w-4 text-accent" />
    case "negative":
      return <TrendingDown className="h-4 w-4 text-destructive" />
    default:
      return <Minus className="h-4 w-4 text-muted-foreground" />
  }
}

function getSentimentColor(sentiment: string) {
  switch (sentiment) {
    case "positive":
      return "bg-accent/10 text-accent border-accent/20"
    case "negative":
      return "bg-destructive/10 text-destructive border-destructive/20"
    default:
      return "bg-muted text-muted-foreground border-border"
  }
}

function calculateSentimentScore(summary: string | undefined): number {
  if (!summary) return 5.0
  
  const text = summary.toLowerCase()
  let score = 5.0
  
  // Positive indicators
  const positiveWords = ['strong', 'growth', 'positive', 'bullish', 'upgraded', 'outperform', 'buy', 'excellent', 'improving']
  const negativeWords = ['weak', 'decline', 'negative', 'bearish', 'downgraded', 'underperform', 'sell', 'poor', 'deteriorating']
  
  positiveWords.forEach(word => {
    if (text.includes(word)) score += 0.5
  })
  
  negativeWords.forEach(word => {
    if (text.includes(word)) score -= 0.5
  })
  
  return Math.max(1.0, Math.min(10.0, score))
}

function determineSentiment(summary: string | undefined): "positive" | "negative" | "neutral" {
  const score = calculateSentimentScore(summary)
  if (score > 6.5) return "positive"
  if (score < 4.5) return "negative"
  return "neutral"
}

function convertStateToAgentData(state: AnalysisState): AgentData[] {
  const agents: AgentData[] = [
    {
      name: "SEC Agent",
      sentiment: determineSentiment(state.sec_summary),
      score: calculateSentimentScore(state.sec_summary),
      status: state.sec_agent_status || 'pending',
      summary: state.sec_summary || 'Analysis pending...',
      details: { content: state.sec_summary }
    },
    {
      name: "News Agent",
      sentiment: determineSentiment(state.news_summary),
      score: calculateSentimentScore(state.news_summary),
      status: state.news_agent_status || 'pending',
      summary: state.news_summary || 'Analysis pending...',
      details: { content: state.news_summary }
    },
    {
      name: "Social Agent",
      sentiment: determineSentiment(state.social_summary),
      score: calculateSentimentScore(state.social_summary),
      status: state.social_agent_status || 'pending',
      summary: state.social_summary || 'Analysis pending...',
      details: { content: state.social_summary }
    },
    {
      name: "Chart Agent",
      sentiment: determineSentiment(state.chart_summary),
      score: calculateSentimentScore(state.chart_summary),
      status: state.chart_agent_status || 'pending',
      summary: state.chart_summary || 'Analysis pending...',
      details: { content: state.chart_summary }
    },
    {
      name: "Analyst Agent",
      sentiment: determineSentiment(state.analyst_summary),
      score: calculateSentimentScore(state.analyst_summary),
      status: state.analyst_agent_status || 'pending',
      summary: state.analyst_summary || 'Analysis pending...',
      details: { content: state.analyst_summary }
    },
  ]
  
  return agents
}

export function AnalystDashboard() {
  const [ticker, setTicker] = useState("TSLA")
  const [loading, setLoading] = useState(false)
  const [analysisState, setAnalysisState] = useState<AnalysisState | null>(null)
  const [agentData, setAgentData] = useState<AgentData[]>([])
  const [selectedAgent, setSelectedAgent] = useState<AgentData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [mcData, setMcData] = useState<{ day: number; price: number }[]>([])
  const [mcLoading, setMcLoading] = useState(false)

  const handleAnalyze = async () => {
    if (!ticker.trim()) {
      setError('Please enter a ticker symbol')
      return
    }

    setLoading(true)
    setMcLoading(true)
    setError(null)

    try {
      const [mcResult] = await Promise.all([
        runMonteCarloSimulation(ticker, 30, 1000),
        analyzeStock(ticker)
      ])

      const chartData = mcResult.days.map((day, index) => ({
        day: day,
        price: mcResult.forecast[index]
      }))
      setMcData(chartData)

      const fullState = await getAnalysisState(ticker)
      setAnalysisState(fullState)
      setAgentData(convertStateToAgentData(fullState))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Analysis failed')
    } finally {
      setLoading(false)
      setMcLoading(false)
    }
  }

  const handleRerun = async (agentName: string) => {
    if (!ticker) return

    setLoading(true)
    setError(null)

    const agentId = agentName.toLowerCase().replace(' agent', '')

    try {
      await rerunAgent(ticker, agentId)
      const fullState = await getAnalysisState(ticker)
      setAnalysisState(fullState)
      setAgentData(convertStateToAgentData(fullState))
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Rerun failed')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="flex flex-col gap-6 p-6">
      {/* Header with Search */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-semibold tracking-tight text-balance">Analyst Swarm Dashboard</h1>
          <p className="text-muted-foreground mt-1">Multi-agent financial risk assessment system</p>
        </div>
        <div className="flex items-center gap-3">
          <input
            type="text"
            value={ticker}
            onChange={(e) => setTicker(e.target.value.toUpperCase())}
            placeholder="Enter ticker"
            className="px-3 py-1.5 bg-background border border-border rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-primary"
            onKeyPress={(e) => e.key === 'Enter' && handleAnalyze()}
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
              'Analyze'
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
          <Badge className={analysisState.workflow_status === 'completed_successfully' ? 'bg-accent text-accent-foreground' : 'bg-destructive text-destructive-foreground'}>
            {analysisState.workflow_status}
          </Badge>
          <span className="text-xs text-muted-foreground">{analysisState.timestamp}</span>
        </div>
      )}

      {/* Main Grid */}
      <div className="grid gap-6 lg:grid-cols-3">
        {/* Monte Carlo Simulation Chart - Top Left */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Activity className="h-5 w-5 text-primary" />
              Monte Carlo Simulation
            </CardTitle>
            <CardDescription>30-day price forecast based on 1,000 simulations</CardDescription>
          </CardHeader>
          <CardContent>
            {mcLoading ? (
              <div className="h-[300px] flex items-center justify-center">
                <Loader2 className="h-8 w-8 animate-spin text-muted-foreground" />
              </div>
            ) : mcData.length > 0 ? (
              <ChartContainer
                config={{
                  price: {
                    label: "Price",
                    color: "hsl(var(--primary))",
                  },
                }}
                className="h-[300px]"
              >
                <AreaChart data={mcData}>
                  <defs>
                    <linearGradient id="fillPrice" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="5%" stopColor="hsl(var(--primary))" stopOpacity={0.3} />
                      <stop offset="95%" stopColor="hsl(var(--primary))" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                  <XAxis 
                    dataKey="day" 
                    tickLine={false} 
                    axisLine={false} 
                    tickMargin={8} 
                    className="text-xs"
                    label={{ value: 'Days Ahead', position: 'insideBottom', offset: -5 }}
                  />
                  <YAxis
                    tickLine={false}
                    axisLine={false}
                    tickMargin={8}
                    className="text-xs"
                    domain={["dataMin - 5", "dataMax + 5"]}
                    label={{ value: 'Price ($)', angle: -90, position: 'insideLeft' }}
                  />
                  <ChartTooltip content={<ChartTooltipContent />} />
                  <Area
                    type="monotone"
                    dataKey="price"
                    stroke="hsl(var(--primary))"
                    strokeWidth={2}
                    fill="url(#fillPrice)"
                  />
                </AreaChart>
              </ChartContainer>
            ) : (
              <div className="h-[300px] flex items-center justify-center text-muted-foreground">
                <div className="text-center">
                  <Activity className="h-12 w-12 mx-auto mb-3 opacity-20" />
                  <p>Enter a ticker and click Analyze to generate forecast</p>
                </div>
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
          <CardContent>
            <div className="space-y-4">
              {newsArticles.map((article) => (
                <div
                  key={article.id}
                  className="flex gap-3 rounded-lg border border-border bg-card p-3 transition-colors hover:bg-accent/5"
                >
                  <div className="mt-1">{getSentimentIcon(article.sentiment)}</div>
                  <div className="flex-1 space-y-1">
                    <h4 className="text-sm font-medium leading-snug text-pretty">{article.title}</h4>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground">
                      <span>{article.source}</span>
                      <span>•</span>
                      <span>{article.time}</span>
                    </div>
                  </div>
                </div>
              ))}
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
            <CardDescription>Multi-perspective risk assessment from specialist agents</CardDescription>
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
                      <Badge variant="outline" className={getSentimentColor(agent.sentiment)}>
                        {agent.sentiment}
                      </Badge>
                    </div>
                    <div className="flex items-baseline gap-2">
                      <span className="text-3xl font-bold font-mono">{agent.score.toFixed(1)}</span>
                      <span className="text-sm text-muted-foreground">/10</span>
                    </div>
                    <p className="text-xs text-muted-foreground leading-relaxed line-clamp-2">{agent.summary}</p>
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
                        e.stopPropagation()
                        handleRerun(agent.name)
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
      {analysisState && (analysisState.governor_summary || analysisState.risk_summary) && (
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

      <Dialog open={!!selectedAgent} onOpenChange={(open) => !open && setSelectedAgent(null)}>
        <DialogContent className="max-w-3xl max-h-[85vh] overflow-y-auto">
          {selectedAgent && (
            <>
              <DialogHeader>
                <div className="flex items-center justify-between">
                  <DialogTitle className="text-2xl">{selectedAgent.name}</DialogTitle>
                  <Badge variant="outline" className={getSentimentColor(selectedAgent.sentiment)}>
                    {selectedAgent.sentiment}
                  </Badge>
                </div>
                <DialogDescription className="text-base">{selectedAgent.summary}</DialogDescription>
              </DialogHeader>

              <div className="space-y-6 mt-4">
                {/* Score Display */}
                <div className="flex items-center gap-4 p-4 rounded-lg bg-muted/50">
                  <div className="flex items-baseline gap-2">
                    <span className="text-5xl font-bold font-mono">{selectedAgent.score.toFixed(1)}</span>
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
                    {selectedAgent.details?.content || 'No detailed analysis available'}
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
    </div>
  )
}