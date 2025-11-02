"""
AgentSeer - Complete Integration Script

This script provides a unified interface to run the entire Analyst Swarm system.
It includes CLI, web API, and batch processing capabilities.
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

# Import the orchestrator and *all its components*
try:
    from orchestrator import (
        run_analyst_swarm,
        save_workflow_state,
        AnalystSwarmState,
        sec_agent_node,
        news_agent_node,
        social_agent_node,
        chart_agent_node,
        analyst_agent_node,
        governor_agent_node,
        risk_assessment_node,
    )

    # Import the actual agent functions to get their raw return values
    from sec_agent import run_sec_agent
    from news_agent import analyze_company_sentiment
    from social_agent import run_social_agent
    from analyst_agent import run_analyst_agent
    from governor_agent import run_governor_agent
    from risk_assessment_agent import run_risk_assessment_agent
except ImportError as e:
    print(f"❌ Error: Could not import from orchestrator.py: {e}")
    print(
        "Make sure orchestrator.py is in the same directory and has all dependencies."
    )
    sys.exit(1)

from mc_rollout import MC_sims

# --- UTILITIES ---


def setup_directories():
    """Ensure all required directories exist."""
    directories = ["reports", "workflow_states", "logs"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def save_analysis_summary(ticker: str, state: dict):
    """Save a summary of the analysis to a central index file."""
    index_file = "workflow_states/analysis_index.json"

    if os.path.exists(index_file):
        try:
            with open(index_file, "r") as f:
                index = json.load(f)
        except json.JSONDecodeError:
            index = {"analyses": []}
    else:
        index = {"analyses": []}

    summary = {
        "ticker": ticker,
        "timestamp": state.get("timestamp"),
        "workflow_status": state.get("workflow_status"),
        "agents_completed": sum(
            [
                1
                for status in [
                    state.get("sec_agent_status"),
                    state.get("news_agent_status"),
                    state.get("social_agent_status"),
                    state.get("chart_agent_status"),
                    state.get("analyst_agent_status"),
                ]
                if status == "success"
            ]
        ),
        "governor_status": state.get("governor_status"),
        "risk_status": state.get("risk_status"),
        "errors": state.get("errors", []),
    }

    # Add to index, prevent duplicates
    index["analyses"] = [
        s
        for s in index["analyses"]
        if s["ticker"] != ticker or s["timestamp"] != summary["timestamp"]
    ]
    index["analyses"].insert(0, summary)
    index["analyses"] = index["analyses"][:100]

    with open(index_file, "w") as f:
        json.dump(index, f, indent=2)


def load_latest_state(ticker: str) -> AnalystSwarmState | None:
    """Find and load the most recent workflow state file for a ticker."""
    state_dir = Path("workflow_states")
    state_files = list(state_dir.glob(f"{ticker.upper()}_workflow_state_*.json"))

    if not state_files:
        return None

    try:
        latest_file = max(state_files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, "r") as f:
            state = json.load(f)
        return state
    except Exception as e:
        print(f"❌ Error loading state for {ticker}: {e}")
        return None


# --- SINGLE ANALYSIS ---


def run_single_analysis(ticker: str):
    """Run analysis for a single ticker."""
    print(f"\n{'='*70}")
    print(f"🎯 Running AgentSeer Analysis: ${ticker}")
    print(f"{'='*70}\n")

    setup_directories()

    try:
        final_state = run_analyst_swarm(ticker, save_state=True)
        save_analysis_summary(ticker, final_state)

        print("\n" + "=" * 70)
        print("📊 ANALYSIS COMPLETE")
        print("=" * 70)
        print(f"\n✅ Status: {final_state.get('workflow_status', 'unknown')}")

        if final_state.get("errors"):
            print(f"⚠️ Errors: {len(final_state['errors'])} issue(s) encountered")
            for error in final_state["errors"]:
                print(f"  - {error}")

        print(f"\n📄 Reports saved to 'reports/' directory")
        print(f"💾 State saved to 'workflow_states/' directory")

        return final_state

    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        import traceback

        traceback.print_exc()
        return None


# --- BATCH ANALYSIS ---


def run_batch_analysis(tickers: list):
    """Run analysis for multiple tickers."""
    print(f"\n{'='*70}")
    print(f"🚀 AgentSeer Batch Analysis")
    print(f"{'='*70}")
    print(f"📊 Analyzing {len(tickers)} tickers: {', '.join(tickers)}")
    print(f"{'='*70}\n")

    setup_directories()

    results = []
    successful = 0
    failed = 0

    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing ${ticker}...")

        try:
            state = run_analyst_swarm(ticker, save_state=True)
            save_analysis_summary(ticker, state)
            results.append((ticker, "success", state))
            successful += 1
        except Exception as e:
            print(f"❌ Failed to analyze ${ticker}: {e}")
            results.append((ticker, "failed", str(e)))
            failed += 1

    print("\n" + "=" * 70)
    print("📊 BATCH ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\n✅ Successful: {successful}/{len(tickers)}")
    print(f"❌ Failed: {failed}/{len(tickers)}")

    if failed > 0:
        print("\n⚠️ Failed tickers:")
        for ticker, status, error in results:
            if status == "failed":
                print(f"  - ${ticker}: {error}")

    return results


# --- LIST ANALYSES ---


def list_analyses():
    """List all previous analyses."""
    index_file = "workflow_states/analysis_index.json"

    if not os.path.exists(index_file):
        print("📭 No analyses found. Run your first analysis to get started!")
        return

    with open(index_file, "r") as f:
        index = json.load(f)

    analyses = index.get("analyses", [])

    if not analyses:
        print("📭 No analyses found.")
        return

    print(f"\n{'='*70}")
    print(f"📊 AgentSeer Analysis History ({len(analyses)} total)")
    print(f"{'='*70}\n")

    for i, analysis in enumerate(analyses[:20], 1):  # Show last 20
        status_emoji = {
            "completed_successfully": "✅",
            "completed_with_errors": "⚠️",
            "failed": "❌",
        }.get(analysis.get("workflow_status"), "❓")

        print(f"{i}. {status_emoji} ${analysis['ticker']} - {analysis['timestamp']}")
        print(
            f"   Agents: {analysis.get('agents_completed', 0)}/5 | "
            f"Governor: {analysis.get('governor_status', 'N/A')} | "
            f"Risk: {analysis.get('risk_status', 'N/A')}"
        )

        if analysis.get("errors"):
            print(f"   ⚠️ {len(analysis['errors'])} error(s)")
        print()


# --- NEW: Re-run Logic ---


def rerun_agent_and_downstream(ticker: str, agent_name: str) -> Dict[str, Any]:
    """
    Loads the latest state for a ticker, re-runs a specific agent,
    and then runs the downstream agents (Governor, Risk Assessment).
    """
    print(f"\n🔄 Re-running agent '{agent_name}' for ${ticker}...")

    # 1. Map agent name to node function
    agent_node_map = {
        "sec": sec_agent_node,
        "news": news_agent_node,
        "social": social_agent_node,
        "chart": chart_agent_node,
        "analyst": analyst_agent_node,
    }

    if agent_name not in agent_node_map:
        raise ValueError(
            f"Unknown agent: {agent_name}. Must be one of {list(agent_node_map.keys())}"
        )

    node_to_run = agent_node_map[agent_name]

    # 2. Load the latest state
    state = load_latest_state(ticker)
    if not state:
        raise FileNotFoundError(f"No previous state found for {ticker}. Cannot re-run.")

    # 3. Clear old errors/warnings related to this agent and downstream
    state["errors"] = [
        e for e in state.get("errors", []) if not e.startswith(agent_name.capitalize())
    ]
    state["warnings"] = [
        w
        for w in state.get("warnings", [])
        if not w.startswith(agent_name.capitalize())
    ]

    # 4. Run the specified agent node
    print(f"--- Running {agent_name}_agent_node ---")
    state = node_to_run(state)

    # 5. Run the downstream nodes (Governor and Risk Assessment)
    print(f"--- Running governor_agent_node ---")
    state = governor_agent_node(state)

    print(f"--- Running risk_assessment_node ---")
    state = risk_assessment_node(state)

    # 6. Update final status and save
    state["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if state.get("errors"):
        state["workflow_status"] = "completed_with_errors"
    else:
        state["workflow_status"] = "completed_successfully"

    save_workflow_state(ticker, state)
    save_analysis_summary(ticker, state)

    print(f"✅ Re-run complete for ${ticker}. New state saved.")
    return state


# --- WEB API SERVER (FastAPI) ---


def start_api_server(host="0.0.0.0", port=8000):
    """Start a FastAPI server for AgentSeer."""
    try:
        from fastapi import FastAPI, HTTPException, BackgroundTasks
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        import uvicorn
        import asyncio
        from concurrent.futures import ThreadPoolExecutor
    except ImportError:
        print(
            "❌ FastAPI not installed. Install with: pip install fastapi uvicorn[standard]"
        )
        return

    # Create a thread pool executor for running blocking I/O operations
    executor = ThreadPoolExecutor(max_workers=10)

    app = FastAPI(
        title="AgentSeer API",
        description="Multi-Agent Financial Analysis System",
        version="2.0.0",
    )

    # Enable CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    setup_directories()

    # Request/Response Models
    class TickerRequest(BaseModel):
        ticker: str

    class AgentResponse(BaseModel):
        status: str
        ticker: str
        agent: str
        summary: str | None = None
        detailed: str | None = None
        timestamp: str

    class GovernorRequest(BaseModel):
        ticker: str
        sec_summary: str | None = None
        news_summary: str | None = None
        social_summary: str | None = None
        chart_summary: str | None = None
        analyst_summary: str | None = None

    class GovernorResponse(BaseModel):
        status: str
        ticker: str
        agent: str
        summary: str | None = None
        full_memo: str | None = None
        timestamp: str

    class AnalyzeRequest(BaseModel):
        ticker: str

    class RerunRequest(BaseModel):
        ticker: str
        agent: str

    class AnalyzeResponse(BaseModel):
        status: str
        ticker: str
        workflow_status: str
        timestamp: str
        governor_summary: str | None = None
        risk_summary: str | None = None

    class RerunResponse(BaseModel):
        status: str
        ticker: str
        agent_rerun: str
        workflow_status: str
        governor_summary: str | None = None
        risk_summary: str | None = None

    class MC_RolloutRequest(BaseModel):
        ticker: str
        t: int = 10
        sims: int = 1000

    class MC_RolloutResponse(BaseModel):
        status: str
        ticker: str
        t: int
        sims: int
        days: list[int]
        forecast: list[float]

    class HealthResponse(BaseModel):
        status: str
        service: str

    class ReportResponse(BaseModel):
        ticker: str
        agent: str
        report_type: str
        content: str | None

    # --- INDIVIDUAL AGENT ENDPOINTS ---

    @app.post("/api/agents/sec")
    async def run_sec_agent_endpoint(request: TickerRequest):
        """Run SEC Agent for a ticker and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, run_sec_agent, ticker, True)
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/news")
    async def run_news_agent_endpoint(request: TickerRequest):
        """Run News Sentiment Agent for a ticker and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor,
                lambda: analyze_company_sentiment(
                    ticker=ticker, max_articles=15, lookback_days=7, verbose=True
                ),
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/social")
    async def run_social_agent_endpoint(request: TickerRequest):
        """Run Social Sentiment Agent for a ticker and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, run_social_agent, ticker, True
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/chart")
    async def run_chart_agent_endpoint(request: TickerRequest):
        """Run Chart/Technical Analysis Agent for a ticker and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            state: AnalystSwarmState = {"ticker": ticker}
            result = await loop.run_in_executor(executor, chart_agent_node, state)

            # Return a dictionary with the chart summary and detailed
            return {
                "ticker": ticker,
                "agent": "chart",
                "summary": result.get("chart_agent_summary"),
                "detailed": result.get("chart_agent_detailed"),
                "status": result.get("chart_agent_status", "success"),
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/analyst")
    async def run_analyst_agent_endpoint(request: TickerRequest):
        """Run Professional Analyst Ratings Agent for a ticker and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, run_analyst_agent, ticker, True
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/governor")
    async def run_governor_agent_endpoint(request: GovernorRequest):
        """Run Governor Agent to synthesize reports from other agents and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Build agent_reports dict as expected by run_governor_agent
            agent_reports = {
                "SEC Agent": request.sec_summary or "Not available",
                "News Agent": request.news_summary or "Not available",
                "Social Agent": request.social_summary or "Not available",
                "Chart Agent": request.chart_summary or "Not available",
                "Analyst Agent": request.analyst_summary or "Not available",
            }

            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                executor, run_governor_agent, ticker, agent_reports, True
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/agents/risk")
    async def run_risk_agent_endpoint(request: GovernorRequest):
        """Run Risk Assessment Agent on Governor's memo and return the raw agent response."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # Build agent_reports dict for governor
            agent_reports = {
                "SEC Agent": request.sec_summary or "Not available",
                "News Agent": request.news_summary or "Not available",
                "Social Agent": request.social_summary or "Not available",
                "Chart Agent": request.chart_summary or "Not available",
                "Analyst Agent": request.analyst_summary or "Not available",
            }

            # Run blocking calls in executor
            loop = asyncio.get_event_loop()

            # First run governor to get the memo
            gov_result = await loop.run_in_executor(
                executor, run_governor_agent, ticker, agent_reports, False
            )

            # Get the memo from governor result
            governor_memo = gov_result.get("detailed_report", "")

            # Run risk assessment with the memo
            result = await loop.run_in_executor(
                executor, run_risk_assessment_agent, ticker, governor_memo, True
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- ORCHESTRATED ANALYSIS ENDPOINTS ---

    @app.post("/api/analyze", response_model=AnalyzeResponse)
    async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
        """Analyze a stock ticker."""
        ticker = request.ticker.upper()

        if not ticker:
            raise HTTPException(status_code=400, detail="No ticker provided")

        try:
            # --- UPDATED TO RUN IN BACKGROUND ---
            # This makes the API return instantly
            background_tasks.add_task(run_analyst_swarm, ticker, save_state=True)

            return AnalyzeResponse(
                status="pending",
                ticker=ticker,
                workflow_status="Analysis started in background. Check /api/analysis/{ticker} for status.",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                governor_summary="Analysis pending...",
                risk_summary="Analysis pending...",
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- NEW ENDPOINT: MC ROLLOUT ---
    @app.post("/api/mc_rollout", response_model=MC_RolloutResponse)
    async def mc_rollout(request: MC_RolloutRequest):
        """Run a Monte Carlo simulation for a ticker."""
        try:
            ticker = request.ticker.upper()
            t = request.t
            sims = request.sims

            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(executor, MC_sims, ticker, t, sims)

            return MC_RolloutResponse(
                status="success",
                ticker=ticker,
                t=t,
                sims=sims,
                days=result["days"],
                forecast=result["median"].tolist(),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # --- NEW ENDPOINT: RERUN AGENT ---
    @app.post("/api/rerun", response_model=RerunResponse)
    async def rerun_agent(request: RerunRequest):
        """Re-run a specific agent for a ticker."""
        ticker = request.ticker.upper()
        agent_name = request.agent.lower()

        if not ticker or not agent_name:
            raise HTTPException(
                status_code=400, detail="Ticker and agent name are required"
            )

        try:
            # Run blocking call in executor
            loop = asyncio.get_event_loop()
            new_state = await loop.run_in_executor(
                executor, rerun_agent_and_downstream, ticker, agent_name
            )
            return RerunResponse(
                status="success",
                ticker=ticker,
                agent_rerun=agent_name,
                workflow_status=new_state.get("workflow_status", "unknown"),
                governor_summary=new_state.get("governor_summary"),
                risk_summary=new_state.get("risk_summary"),
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/analyses")
    async def get_analyses():
        """Get list of all analyses."""
        index_file = "workflow_states/analysis_index.json"

        if not os.path.exists(index_file):
            return {"analyses": []}

        try:
            with open(index_file, "r") as f:
                index = json.load(f)
            return index
        except json.JSONDecodeError:
            return {"analyses": []}

    @app.get("/api/analysis/{ticker}")
    async def get_analysis_state(ticker: str):
        """Get detailed analysis state for a specific ticker."""
        state = load_latest_state(ticker.upper())
        if not state:
            raise HTTPException(status_code=404, detail="Analysis not found")

        return state

    @app.get("/api/report/summary/{ticker}/{agent_name}", response_model=ReportResponse)
    async def get_summary_report(ticker: str, agent_name: str):
        """Get a specific agent's summary report."""
        state = load_latest_state(ticker.upper())
        if not state:
            raise HTTPException(status_code=404, detail="Analysis not found")

        key = f"{agent_name.lower()}_summary"
        if key not in state:
            raise HTTPException(
                status_code=404, detail=f"Agent {agent_name} not found in state"
            )

        return ReportResponse(
            ticker=ticker,
            agent=agent_name,
            report_type="summary",
            content=state.get(key),
        )

    @app.get(
        "/api/report/detailed/{ticker}/{agent_name}", response_model=ReportResponse
    )
    async def get_detailed_report(ticker: str, agent_name: str):
        """Get a specific agent's detailed report."""
        state = load_latest_state(ticker.upper())
        if not state:
            raise HTTPException(status_code=404, detail="Analysis not found")

        key = f"{agent_name.lower()}_detailed"
        if key not in state:
            # Fallback for governor/risk
            if agent_name == "governor":
                key = "governor_full_memo"
            elif agent_name == "risk":
                key = "risk_full_report"
            else:
                raise HTTPException(
                    status_code=404, detail=f"Agent {agent_name} not found in state"
                )

        return ReportResponse(
            ticker=ticker,
            agent=agent_name,
            report_type="detailed",
            content=state.get(key),
        )

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        """Health check endpoint."""
        return HealthResponse(status="healthy", service="AgentSeer API")

    print(f"\n{'='*70}")
    print(f"🚀 AgentSeer API Server Starting (FastAPI)")
    print(f"{'='*70}")
    print(f"\n📡 Server running on http://{host}:{port}")
    print(f"📚 Interactive API docs: http://{host}:{port}/docs")
    print(f"📖 Alternative docs: http://{host}:{port}/redoc")
    print(f"\n📝 Individual Agent endpoints:")
    print(f"  POST   /api/agents/sec         (Body: {{'ticker': 'TSLA'}})")
    print(f"  POST   /api/agents/news        (Body: {{'ticker': 'TSLA'}})")
    print(f"  POST   /api/agents/social      (Body: {{'ticker': 'TSLA'}})")
    print(f"  POST   /api/agents/chart       (Body: {{'ticker': 'TSLA'}})")
    print(f"  POST   /api/agents/analyst     (Body: {{'ticker': 'TSLA'}})")
    print(
        f"  POST   /api/agents/governor    (Body: {{'ticker': 'TSLA', 'sec_summary': '...', ...}})"
    )
    print(
        f"  POST   /api/agents/risk        (Body: {{'ticker': 'TSLA', 'sec_summary': '...', ...}})"
    )
    print(f"\n📝 Orchestrated Analysis endpoints:")
    print(f"  POST   /api/analyze           (Body: {{'ticker': 'TSLA'}})")
    print(
        f"  POST   /api/rerun             (Body: {{'ticker': 'TSLA', 'agent': 'social'}})"
    )
    print(f"\n📝 Report & State endpoints:")
    print(f"  GET    /api/analyses")
    print(f"  GET    /api/analysis/{{ticker}}")
    print(f"  GET    /api/report/summary/{{ticker}}/{{agent}}")
    print(f"  GET    /api/report/detailed/{{ticker}}/{{agent}}")
    print(f"  GET    /api/health")
    print(f"\n{'='*70}\n")

    uvicorn.run(app, host=host, port=port, log_level="info")


# --- MAIN CLI ---


def main():
    parser = argparse.ArgumentParser(
        description="AgentSeer - Multi-Agent Financial Analysis System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agentseer.py TSLA                    # Analyze Tesla
  python run_agentseer.py --batch TSLA AAPL MSFT  # Batch analysis
  python run_agentseer.py --list                  # List analyses
  python run_agentseer.py --server                # Start API server
        """,
    )

    parser.add_argument(
        "ticker", nargs="?", help="Stock ticker to analyze (e.g., TSLA)"
    )
    parser.add_argument("--batch", nargs="+", help="Analyze multiple tickers")
    parser.add_argument(
        "--list", action="store_true", help="List all previous analyses"
    )
    parser.add_argument("--server", action="store_true", help="Start API server")
    parser.add_argument(
        "--port", type=int, default=8000, help="API server port (default: 8000)"
    )
    parser.add_argument(
        "--host", default="0.0.0.0", help="API server host (default: 0.0.0.0)"
    )

    args = parser.parse_args()

    if args.list:
        list_analyses()
    elif args.server:
        start_api_server(host=args.host, port=args.port)
    elif args.batch:
        run_batch_analysis(args.batch)
    elif args.ticker:
        run_single_analysis(args.ticker)
    else:
        parser.print_help()
        print("\n💡 Tip: Start with 'python run_agentseer.py TSLA' to analyze Tesla")


if __name__ == "__main__":
    main()
