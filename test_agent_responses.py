#!/usr/bin/env python3
"""
Test script to verify all agents return proper dictionary responses.
This tests the response structure and format of all agents in the BerkshireBot system.
"""

import json
import sys
from typing import Dict, Any


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*70}")
    print(f"{title}")
    print(f"{'='*70}\n")


def validate_response_structure(agent_name: str, response: Any) -> bool:
    """
    Validate that the agent response is a dictionary with required fields.

    Args:
        agent_name: Name of the agent being tested
        response: The response from the agent

    Returns:
        True if valid, False otherwise
    """
    print(f"🔍 Validating {agent_name} response structure...")

    # Check if response is a dictionary
    if not isinstance(response, dict):
        print(
            f"  ❌ FAIL: Response is not a dictionary (type: {type(response).__name__})"
        )
        return False

    # Check required fields
    required_fields = ["ticker", "agent", "summary_report", "detailed_report"]
    missing_fields = [field for field in required_fields if field not in response]

    if missing_fields:
        print(f"  ❌ FAIL: Missing required fields: {missing_fields}")
        return False

    # Check for error field
    if "error" in response:
        print(f"  ⚠️  WARNING: Error in response: {response['error']}")
        print(f"  ✓ But structure is valid (has required fields)")
        return True

    print(f"  ✅ PASS: All required fields present")
    print(
        f"  📊 Additional fields: {list(set(response.keys()) - set(required_fields))}"
    )
    return True


def print_response_summary(agent_name: str, response: Dict[str, Any]):
    """Print a summary of the response."""
    print(f"\n📋 {agent_name} Response Summary:")
    print(f"  • Ticker: {response.get('ticker', 'N/A')}")
    print(f"  • Agent: {response.get('agent', 'N/A')}")
    print(f"  • Has Error: {'error' in response}")
    print(f"  • Total Fields: {len(response)}")
    print(f"  • Summary Report Length: {len(response.get('summary_report', ''))} chars")
    print(
        f"  • Detailed Report Length: {len(response.get('detailed_report', ''))} chars"
    )


def test_news_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test News Agent response."""
    print_section("Testing News Agent")

    try:
        from news_agent import run_news_agent

        print(f"Running News Agent for {ticker}...")
        response = run_news_agent(
            ticker=ticker,
            max_articles=5,  # Fewer articles for faster testing
            lookback_days=3,
        )

        if validate_response_structure("News Agent", response):
            print_response_summary("News Agent", response)

            # News-specific validation
            news_specific_fields = [
                "weighted_sentiment_score",
                "bullish_pressure",
                "bearish_pressure",
            ]
            found_fields = [f for f in news_specific_fields if f in response]
            print(f"  📈 News-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ News Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "News",
            "summary_report": "",
            "detailed_report": "",
        }


def test_social_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test Social Agent response."""
    print_section("Testing Social Agent")

    try:
        from social_agent import run_social_agent

        print(f"Running Social Agent for {ticker}...")
        response = run_social_agent(ticker, save_to_file=False)

        if validate_response_structure("Social Agent", response):
            print_response_summary("Social Agent", response)

            # Social-specific validation
            social_specific_fields = [
                "overall_sentiment",
                "sentiment_score",
                "freshness_stats",
            ]
            found_fields = [f for f in social_specific_fields if f in response]
            print(f"  💬 Social-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ Social Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "Social",
            "summary_report": "",
            "detailed_report": "",
        }


def test_analyst_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test Analyst Agent response."""
    print_section("Testing Analyst Agent")

    try:
        from analyst_agent import run_analyst_agent

        print(f"Running Analyst Agent for {ticker}...")
        response = run_analyst_agent(ticker, save_to_file=False)

        if validate_response_structure("Analyst Agent", response):
            print_response_summary("Analyst Agent", response)

            # Analyst-specific validation
            analyst_specific_fields = ["consensus", "consensus_score", "average_target"]
            found_fields = [f for f in analyst_specific_fields if f in response]
            print(f"  📊 Analyst-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ Analyst Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "Analyst",
            "summary_report": "",
            "detailed_report": "",
        }


def test_sec_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test SEC Agent response."""
    print_section("Testing SEC Agent")

    try:
        from sec_agent import run_sec_agent

        print(f"Running SEC Agent for {ticker}...")
        print(
            "⚠️  Note: This may take several minutes as it analyzes 5 years of filings..."
        )
        response = run_sec_agent(ticker, save_to_file=False)

        if validate_response_structure("SEC Agent", response):
            print_response_summary("SEC Agent", response)

            # SEC-specific validation
            sec_specific_fields = [
                "filings_analyzed",
                "years_covered",
                "financial_metrics",
            ]
            found_fields = [f for f in sec_specific_fields if f in response]
            print(f"  📄 SEC-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ SEC Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "SEC",
            "summary_report": "",
            "detailed_report": "",
        }


def test_governor_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test Governor Agent response."""
    print_section("Testing Governor Agent")

    try:
        from governor_agent import run_governor_agent

        # Create mock reports for testing
        mock_reports = {
            "News Agent": "Mock news report with positive sentiment",
            "Social Agent": "Mock social report showing bullish sentiment",
            "Analyst Agent": "Mock analyst report with Buy rating",
            "SEC Agent": "Mock SEC report with moderate risk",
        }

        print(f"Running Governor Agent for {ticker} with mock reports...")
        response = run_governor_agent(ticker, mock_reports, save_to_file=False)

        if validate_response_structure("Governor Agent", response):
            print_response_summary("Governor Agent", response)

            # Governor-specific validation
            governor_specific_fields = [
                "executive_summary",
                "confidence_level",
                "validation_results",
            ]
            found_fields = [f for f in governor_specific_fields if f in response]
            print(f"  🎯 Governor-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ Governor Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "Governor",
            "summary_report": "",
            "detailed_report": "",
        }


def test_risk_assessment_agent(ticker: str = "AAPL") -> Dict[str, Any]:
    """Test Risk Assessment Agent response."""
    print_section("Testing Risk Assessment Agent")

    try:
        from risk_assessment_agent import run_risk_assessment_agent

        # Create mock investment memo
        mock_memo = f"""
# Investment Memo: ${ticker}

## Executive Summary
Test investment memo for validating risk assessment agent response structure.

## Key Risks
- Market volatility
- Competition
- Regulatory changes
        """

        print(f"Running Risk Assessment Agent for {ticker}...")
        response = run_risk_assessment_agent(ticker, mock_memo, save_to_file=False)

        if validate_response_structure("Risk Assessment Agent", response):
            print_response_summary("Risk Assessment Agent", response)

            # Risk-specific validation
            risk_specific_fields = [
                "overall_risk_score",
                "overall_risk_level",
                "risk_categories",
            ]
            found_fields = [f for f in risk_specific_fields if f in response]
            print(f"  ⚠️  Risk-specific fields: {found_fields}")

        return response

    except Exception as e:
        print(f"❌ Risk Assessment Agent Error: {e}")
        import traceback

        traceback.print_exc()
        return {
            "error": str(e),
            "ticker": ticker,
            "agent": "Risk_Assessment",
            "summary_report": "",
            "detailed_report": "",
        }


def save_test_results(results: Dict[str, Dict[str, Any]], ticker: str):
    """Save test results to a JSON file."""
    output_file = f"test_results_{ticker}.json"

    # Create a simplified version for JSON serialization
    simplified_results = {}
    for agent_name, response in results.items():
        simplified_results[agent_name] = {
            "ticker": response.get("ticker"),
            "agent": response.get("agent"),
            "has_error": "error" in response,
            "error_message": response.get("error", None),
            "field_count": len(response),
            "fields": list(response.keys()),
            "summary_length": len(response.get("summary_report", "")),
            "detailed_length": len(response.get("detailed_report", "")),
        }

    with open(output_file, "w") as f:
        json.dump(simplified_results, f, indent=2)

    print(f"\n📄 Test results saved to: {output_file}")


def main():
    """Run all agent tests."""
    print(
        """
╔══════════════════════════════════════════════════════════════════╗
║           BerkshireBot Agent Response Structure Test             ║
║                  Testing Dictionary Returns                      ║
╚══════════════════════════════════════════════════════════════════╝
    """
    )

    # Get ticker from command line or use default
    ticker = sys.argv[1].upper() if len(sys.argv) > 1 else "AAPL"
    print(f"Testing with ticker: {ticker}\n")

    results = {}

    # Test each agent
    print("🚀 Starting agent tests...\n")

    # 1. News Agent (fast)
    results["News Agent"] = test_news_agent(ticker)

    # 2. Social Agent (fast)
    results["Social Agent"] = test_social_agent(ticker)

    # 3. Analyst Agent (fast)
    results["Analyst Agent"] = test_analyst_agent(ticker)

    # 4. SEC Agent (slow - may be skipped)
    if "--skip-sec" not in sys.argv:
        results["SEC Agent"] = test_sec_agent(ticker)
    else:
        print_section("Skipping SEC Agent (use without --skip-sec to include)")

    # 5. Governor Agent (fast)
    results["Governor Agent"] = test_governor_agent(ticker)

    # 6. Risk Assessment Agent (fast)
    results["Risk Assessment Agent"] = test_risk_assessment_agent(ticker)

    # Print final summary
    print_section("FINAL SUMMARY")

    total_agents = len(results)
    passed = sum(1 for r in results.values() if "error" not in r)
    failed = total_agents - passed

    print(f"📊 Test Results:")
    print(f"  • Total Agents Tested: {total_agents}")
    print(f"  • Passed: {passed} ✅")
    print(f"  • Failed: {failed} ❌")
    print(f"  • Success Rate: {(passed/total_agents)*100:.1f}%\n")

    for agent_name, response in results.items():
        status = "❌ FAILED" if "error" in response else "✅ PASSED"
        print(f"  {status}: {agent_name}")

    # Save results
    save_test_results(results, ticker)

    print(f"\n{'='*70}")
    print("Test complete! Check the summary above and the JSON file for details.")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
