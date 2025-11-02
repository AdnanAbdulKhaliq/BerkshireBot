#!/usr/bin/env python3
"""
Example usage of the sec_agent module for financial analysis.

This demonstrates how to use the refactored SEC Agent to analyze
SEC filings for any company.
"""

from sec_agent import SecAgent, run_sec_agent


def main():
    """Example using the convenient wrapper function."""
    ticker = "AAPL"
    company_description = "A technology company that designs, manufactures, and markets consumer electronics, computer software, and online services."
    
    # Use the wrapper function (recommended - matches other agents)
    summary, detailed = run_sec_agent(ticker, company_description, save_to_file=True)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE!")
    print("="*60)
    print("\nSummary Report:")
    print(summary)
    print(f"\nDetailed report preview (first 500 chars):\n{detailed[:500]}...")


def example_class_usage():
    """Example using the SecAgent class directly."""
    company_name = "Apple Inc"
    company_description = "A technology company that designs, manufactures, and markets consumer electronics, computer software, and online services."
    
    agent = SecAgent(
        verbose=True,      # Enable detailed logging
        save_final=True,   # Save the final report to a file
        save_trace=True    # Save the execution trace/progress log
    )
    
    report = agent.run(company_name, company_description)
    
    print("\n" + "="*60)
    print("Analysis Complete!")
    print("="*60)
    print(f"\nReport preview (first 500 chars):\n{report[:500]}...")


def example_quiet_mode():
    """Example showing how to run in quiet mode (no logging, no file saves)."""
    agent = SecAgent(verbose=False, save_final=False, save_trace=False)
    report = agent.run("Microsoft Corporation")
    return report


def example_multiple_tickers():
    """Example showing how to analyze multiple companies using the wrapper."""
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    reports = {}
    
    for ticker in tickers:
        print(f"\n{'='*60}")
        print(f"Analyzing ${ticker}...")
        print(f"{'='*60}\n")
        
        summary, detailed = run_sec_agent(ticker, save_to_file=True)
        reports[ticker] = {"summary": summary, "detailed": detailed}
    
    return reports


if __name__ == "__main__":
    main()

