"""
AgentSeer - Complete Integration Script

This script provides a unified interface to run the entire Analyst Swarm system.
It includes CLI, web API, and batch processing capabilities.

Usage:
    # Single analysis
    python run_agentseer.py TSLA
    
    # Batch analysis
    python run_agentseer.py --batch TSLA AAPL MSFT GOOGL
    
    # Start web API server
    python run_agentseer.py --server
    
    # View existing analyses
    python run_agentseer.py --list
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

# Import the orchestrator
try:
    from orchestrator import run_analyst_swarm
except ImportError:
    print("❌ Error: Could not import orchestrator.py")
    print("Make sure orchestrator.py is in the same directory")
    sys.exit(1)


# --- UTILITIES ---

def setup_directories():
    """Ensure all required directories exist."""
    directories = ['reports', 'workflow_states', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def save_analysis_summary(ticker: str, state: dict):
    """Save a summary of the analysis to a central index file."""
    index_file = "workflow_states/analysis_index.json"
    
    # Load existing index
    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            index = json.load(f)
    else:
        index = {"analyses": []}
    
    # Create summary entry
    summary = {
        "ticker": ticker,
        "timestamp": state.get('timestamp'),
        "workflow_status": state.get('workflow_status'),
        "agents_completed": sum([
            1 for status in [
                state.get('sec_agent_status'),
                state.get('news_agent_status'),
                state.get('social_agent_status'),
                state.get('chart_agent_status'),
                state.get('analyst_agent_status')
            ] if status == 'success'
        ]),
        "governor_status": state.get('governor_status'),
        "risk_status": state.get('risk_status'),
        "errors": state.get('errors', [])
    }
    
    # Add to index
    index["analyses"].insert(0, summary)
    
    # Keep only last 100 analyses
    index["analyses"] = index["analyses"][:100]
    
    # Save index
    with open(index_file, 'w') as f:
        json.dump(index, f, indent=2)


# --- SINGLE ANALYSIS ---

def run_single_analysis(ticker: str):
    """Run analysis for a single ticker."""
    print(f"\n{'='*70}")
    print(f"🎯 Running AgentSeer Analysis: ${ticker}")
    print(f"{'='*70}\n")
    
    setup_directories()
    
    try:
        # Run the workflow
        final_state = run_analyst_swarm(ticker, save_state=True)
        
        # Save to index
        save_analysis_summary(ticker, final_state)
        
        # Print results
        print("\n" + "="*70)
        print("📊 ANALYSIS COMPLETE")
        print("="*70)
        print(f"\n✅ Status: {final_state.get('workflow_status', 'unknown')}")
        
        if final_state.get('errors'):
            print(f"⚠️ Errors: {len(final_state['errors'])} issue(s) encountered")
            for error in final_state['errors']:
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
            results.append((ticker, 'success', state))
            successful += 1
        except Exception as e:
            print(f"❌ Failed to analyze ${ticker}: {e}")
            results.append((ticker, 'failed', str(e)))
            failed += 1
    
    # Print summary
    print("\n" + "="*70)
    print("📊 BATCH ANALYSIS COMPLETE")
    print("="*70)
    print(f"\n✅ Successful: {successful}/{len(tickers)}")
    print(f"❌ Failed: {failed}/{len(tickers)}")
    
    if failed > 0:
        print("\n⚠️ Failed tickers:")
        for ticker, status, error in results:
            if status == 'failed':
                print(f"  - ${ticker}: {error}")
    
    return results


# --- LIST ANALYSES ---

def list_analyses():
    """List all previous analyses."""
    index_file = "workflow_states/analysis_index.json"
    
    if not os.path.exists(index_file):
        print("📭 No analyses found. Run your first analysis to get started!")
        return
    
    with open(index_file, 'r') as f:
        index = json.load(f)
    
    analyses = index.get('analyses', [])
    
    if not analyses:
        print("📭 No analyses found.")
        return
    
    print(f"\n{'='*70}")
    print(f"📊 AgentSeer Analysis History ({len(analyses)} total)")
    print(f"{'='*70}\n")
    
    for i, analysis in enumerate(analyses[:20], 1):  # Show last 20
        status_emoji = {
            'completed_successfully': '✅',
            'completed_with_errors': '⚠️',
            'failed': '❌'
        }.get(analysis.get('workflow_status'), '❓')
        
        print(f"{i}. {status_emoji} ${analysis['ticker']} - {analysis['timestamp']}")
        print(f"   Agents: {analysis.get('agents_completed', 0)}/5 | "
              f"Governor: {analysis.get('governor_status', 'N/A')} | "
              f"Risk: {analysis.get('risk_status', 'N/A')}")
        
        if analysis.get('errors'):
            print(f"   ⚠️ {len(analysis['errors'])} error(s)")
        print()


# --- WEB API SERVER ---

def start_api_server(host='0.0.0.0', port=8000):
    """Start a simple Flask API server for AgentSeer."""
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
    except ImportError:
        print("❌ Flask not installed. Install with: pip install flask flask-cors")
        return
    
    app = Flask(__name__)
    CORS(app)
    
    setup_directories()
    
    @app.route('/api/analyze', methods=['POST'])
    def analyze():
        """Analyze a stock ticker."""
        data = request.json
        ticker = data.get('ticker', '').upper()
        
        if not ticker:
            return jsonify({'error': 'No ticker provided'}), 400
        
        try:
            final_state = run_analyst_swarm(ticker, save_state=True)
            save_analysis_summary(ticker, final_state)
            
            return jsonify({
                'status': 'success',
                'ticker': ticker,
                'workflow_status': final_state.get('workflow_status'),
                'timestamp': final_state.get('timestamp'),
                'governor_summary': final_state.get('governor_summary'),
                'risk_summary': final_state.get('risk_summary')
            })
        except Exception as e:
            return jsonify({'status': 'error', 'message': str(e)}), 500
    
    @app.route('/api/analyses', methods=['GET'])
    def get_analyses():
        """Get list of all analyses."""
        index_file = "workflow_states/analysis_index.json"
        
        if not os.path.exists(index_file):
            return jsonify({'analyses': []})
        
        with open(index_file, 'r') as f:
            index = json.load(f)
        
        return jsonify(index)
    
    @app.route('/api/analysis/<ticker>', methods=['GET'])
    def get_analysis(ticker):
        """Get detailed analysis for a specific ticker."""
        # Find most recent state file for this ticker
        state_dir = Path('workflow_states')
        state_files = list(state_dir.glob(f"{ticker.upper()}_workflow_state_*.json"))
        
        if not state_files:
            return jsonify({'error': 'Analysis not found'}), 404
        
        # Get most recent
        latest_file = max(state_files, key=lambda p: p.stat().st_mtime)
        
        with open(latest_file, 'r') as f:
            state = json.load(f)
        
        return jsonify(state)
    
    @app.route('/api/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({'status': 'healthy', 'service': 'AgentSeer API'})
    
    print(f"\n{'='*70}")
    print(f"🚀 AgentSeer API Server Starting")
    print(f"{'='*70}")
    print(f"\n📡 Server running on http://{host}:{port}")
    print(f"\n📝 Available endpoints:")
    print(f"  POST   /api/analyze       - Analyze a ticker")
    print(f"  GET    /api/analyses      - List all analyses")
    print(f"  GET    /api/analysis/<ticker> - Get specific analysis")
    print(f"  GET    /api/health        - Health check")
    print(f"\n{'='*70}\n")
    
    app.run(host=host, port=port, debug=False)


# --- MAIN CLI ---

def main():
    parser = argparse.ArgumentParser(
        description='AgentSeer - Multi-Agent Financial Analysis System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agentseer.py TSLA                    # Analyze Tesla
  python run_agentseer.py --batch TSLA AAPL MSFT  # Batch analysis
  python run_agentseer.py --list                  # List analyses
  python run_agentseer.py --server                # Start API server
        """
    )
    
    parser.add_argument('ticker', nargs='?', help='Stock ticker to analyze (e.g., TSLA)')
    parser.add_argument('--batch', nargs='+', help='Analyze multiple tickers')
    parser.add_argument('--list', action='store_true', help='List all previous analyses')
    parser.add_argument('--server', action='store_true', help='Start API server')
    parser.add_argument('--port', type=int, default=8000, help='API server port (default: 8000)')
    parser.add_argument('--host', default='0.0.0.0', help='API server host (default: 0.0.0.0)')
    
    args = parser.parse_args()
    
    # Handle different modes
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