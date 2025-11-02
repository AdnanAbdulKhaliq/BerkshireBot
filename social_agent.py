"""
Social Sentiment Agent - Analyst Swarm Component

This agent analyzes retail investor sentiment from social media and forums
using a dedicated Google Custom Search Engine (CSE) filtered to social sources.

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
    - GOOGLE_API_KEY: Your Google Custom Search API key
    - GOOGLE_CSE_ID_SOCIAL: Your Social Sentiment Engine CSE ID

Setup Instructions:
    1. Create a Google Programmable Search Engine at:
       https://programmablesearchengine.google.com/
    
    2. Add these sites to your "Social Sentiment Engine":
       - reddit.com/r/wallstreetbets/*
       - reddit.com/r/stocks/*
       - reddit.com/r/investing/*
       - twitter.com
       - stocktwits.com/symbol/*
       - seekingalpha.com
    
    3. CRITICAL: Turn "Search the entire web" to OFF
    
    4. Copy your Search Engine ID and set it as GOOGLE_CSE_ID_SOCIAL
"""

import os
import time
import json
from datetime import datetime
from functools import lru_cache
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_community import GoogleSearchAPIWrapper
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


# --- ENVIRONMENT VALIDATION ---

def validate_environment() -> None:
    """Validate all required API keys and configuration are set."""
    required_vars = {
        "GEMINI_API_KEY": "Gemini LLM for analysis",
        "GOOGLE_API_KEY": "Google Search API",
        "GOOGLE_CSE_ID_SOCIAL": "Social Sentiment Engine CSE ID"
    }
    
    missing = []
    for var_name, description in required_vars.items():
        if var_name not in os.environ:
            missing.append(f"  ❌ {var_name}: {description}")
    
    if missing:
        error_msg = "Missing required environment variables:\n" + "\n".join(missing)
        raise EnvironmentError(error_msg)
    
    print("✅ Social_Agent: All environment variables validated")


# Validate at module load time
validate_environment()


# --- SETUP TOOLS & MODELS ---

# Initialize Gemini LLM with low temperature for consistent analysis
llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0.1,
    api_key=os.environ["GEMINI_API_KEY"]
)

# Initialize Google Search tool with Social Sentiment CSE
social_search_tool = GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    google_cse_id=os.environ["GOOGLE_CSE_ID_SOCIAL"]
)


# --- SEARCH LOGIC WITH CACHING ---

# Ticker to company name mapping for better search results
TICKER_TO_NAME = {
    "TSLA": "Tesla",
    "AAPL": "Apple",
    "MSFT": "Microsoft",
    "GOOGL": "Google",
    "AMZN": "Amazon",
    "META": "Meta Facebook",
    "NVDA": "Nvidia",
    "AMD": "AMD",
    "COIN": "Coinbase",
    "PLTR": "Palantir",
    "GME": "GameStop",
    "AMC": "AMC",
    "SPY": "SPY S&P500",
    "QQQ": "QQQ Nasdaq",
}


@lru_cache(maxsize=100)
def cached_search(ticker: str, timestamp_hour: int) -> list:
    """
    Cached search function to avoid hitting API rate limits.
    Cache expires every hour (timestamp_hour changes).
    
    Args:
        ticker: Stock ticker symbol
        timestamp_hour: Current hour timestamp for cache key
    
    Returns:
        List of search result dictionaries with metadata
    """
    # Get company name for better social media results
    company_name = TICKER_TO_NAME.get(ticker, ticker)
    
    # Optimized query: Use company name + ticker for best results
    # Social media users typically use company names, not tickers
    query = f"{company_name} {ticker} stock sentiment discussion"
    
    print(f"🔍 Social_Agent: Searching for '{query}'...")
    
    try:
        # Use .results() instead of .run() to get structured data
        results = social_search_tool.results(query, num_results=10)
        print(f"📥 Retrieved {len(results)} results")
        return results
    except Exception as e:
        print(f"❌ Search API error: {e}")
        raise


def format_sources_for_llm(results: list) -> str:
    """
    Format structured search results into a readable format for the LLM.
    
    Args:
        results: List of search result dictionaries
    
    Returns:
        Formatted string with source citations
    """
    if not results:
        return "No sources found."
    
    formatted = "Below are the social media sources found. Each source has a number, title, URL, and text snippet.\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('link', 'No URL')
        snippet = result.get('snippet', 'No snippet available')
        
        formatted += f"\n{'='*70}\n"
        formatted += f"SOURCE #{i}\n"
        formatted += f"{'='*70}\n"
        formatted += f"TITLE: {title}\n"
        formatted += f"URL: {url}\n"
        formatted += f"SNIPPET:\n{snippet}\n"
    
    formatted += f"\n{'='*70}\n"
    formatted += f"END OF SOURCES (Total: {len(results)} sources)\n"
    formatted += f"{'='*70}\n"
    
    return formatted


def run_search(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute search with caching and error handling.
    
    Args:
        input_dict: Contains 'ticker' key
    
    Returns:
        Dictionary with search results, raw results, and metadata
    """
    ticker = input_dict.get("ticker", "").upper()
    
    if not ticker:
        return {
            "ticker": "",
            "social_data": "Error: No ticker provided",
            "raw_results": [],
            "search_status": "failed",
            "result_count": 0
        }
    
    try:
        # Cache key changes every hour (3600 seconds)
        cache_key = int(time.time() // 3600)
        raw_results = cached_search(ticker, cache_key)
        
        # Validate results quality
        if not raw_results or len(raw_results) == 0:
            return {
                "ticker": ticker,
                "social_data": f"No social data found for {ticker}. Public discussion may be minimal.",
                "raw_results": [],
                "search_status": "partial",
                "result_count": 0
            }
        
        # Format results for LLM
        formatted_data = format_sources_for_llm(raw_results)
        
        print(f"✅ Social_Agent: Retrieved {len(raw_results)} sources")
        
        return {
            "ticker": ticker,
            "social_data": formatted_data,
            "raw_results": raw_results,  # Keep raw results for report generation
            "search_status": "success",
            "result_count": len(raw_results)
        }
        
    except Exception as e:
        print(f"❌ Social_Agent: Search failed - {e}")
        return {
            "ticker": ticker,
            "social_data": f"Error retrieving social data: {str(e)}",
            "raw_results": [],
            "search_status": "failed",
            "result_count": 0
        }


# --- DETAILED ANALYSIS PROMPT ---

detailed_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert "Social-Sentimentalist" financial analyst analyzing retail investor sentiment for ${ticker}.

**CRITICAL INSTRUCTIONS - READ CAREFULLY:**

You will receive sources formatted like this:
======================================================================
SOURCE #1
======================================================================
TITLE: [title here]
URL: [url here]
SNIPPET:
[text here]

For EVERY source, you MUST:
1. Extract the EXACT title and URL provided
2. Quote EXACT phrases from the snippet (use quotation marks)
3. Cite the source number (e.g., "Source #3")
4. Do NOT say "No title" or "No URL" - they are ALWAYS provided
5. Do NOT make up or paraphrase - use the actual text

**YOUR ANALYSIS MUST INCLUDE:**

1. **Executive Summary**: 2-3 sentences covering overall sentiment
2. **Source-by-Source Analysis**: For EACH numbered source:
   - Copy the exact title
   - Copy the exact URL
   - Identify platform (reddit/twitter/stocktwits/seekingalpha from URL)
   - Extract 2-3 direct quotes from the snippet
   - State that source's sentiment
3. **Key Themes**: 3-5 recurring topics with exact quotes and source numbers
4. **Notable Quotes**: 5-10 EXACT quotes with source numbers
5. **Risk Factors**: Concerns mentioned with source citations
6. **Bullish Catalysts**: Positive drivers with source citations

Return your analysis in this EXACT JSON format:
{{
  "executive_summary": "2-3 sentence overview",
  "overall_sentiment": "Bullish" | "Bearish" | "Neutral",
  "sentiment_score": -1.0 to 1.0,
  "sentiment_reasoning": "Explain with source numbers",
  "data_quality": "excellent" | "good" | "limited" | "insufficient",
  "source_analyses": [
    {{
      "source_number": 1,
      "source_title": "EXACT title from source",
      "source_url": "EXACT URL from source",
      "platform": "reddit" | "twitter" | "stocktwits" | "seekingalpha" | "other",
      "direct_quotes": ["exact quote 1", "exact quote 2"],
      "sentiment": "Bullish" | "Bearish" | "Neutral",
      "key_points": ["point 1", "point 2"],
      "analysis": "Your interpretation"
    }}
  ],
  "key_themes": [
    {{
      "theme": "Theme name",
      "description": "Description",
      "sentiment": "Bullish" | "Bearish" | "Neutral",
      "prevalence": "high" | "medium" | "low",
      "supporting_quotes": [
        {{
          "quote": "exact text from snippet",
          "source_number": 1,
          "source_url": "exact URL"
        }}
      ]
    }}
  ],
  "notable_quotes": [
    {{
      "quote": "EXACT text from snippet",
      "source_number": 1,
      "source_url": "exact URL",
      "sentiment": "Bullish" | "Bearish" | "Neutral"
    }}
  ],
  "risk_factors": [
    {{
      "risk": "description",
      "source_numbers": [1, 2],
      "supporting_quote": "exact text"
    }}
  ],
  "bullish_catalysts": [
    {{
      "catalyst": "description",
      "source_numbers": [3],
      "supporting_quote": "exact text"
    }}
  ],
  "consensus_view": "What most sources agree on",
  "discussion_volume": "high" | "medium" | "low"
}}

<search_metadata>
Ticker: {ticker}
Search Status: {search_status}
Number of Sources: {result_count}
</search_metadata>

<social_data>
{social_data}
</social_data>

Return ONLY valid JSON. Do not include any text before or after the JSON object."""
)


# --- CREATE DETAILED ANALYSIS CHAIN ---

# This chain is now just the analysis part, not the search
analysis_chain = detailed_prompt_template | llm | JsonOutputParser()


# --- REPORT GENERATION ---

def generate_detailed_report(ticker: str, analysis: Dict[str, Any], raw_results: list) -> str:
    """
    Generate a comprehensive markdown report from analysis results.
    
    Args:
        ticker: Stock ticker symbol
        analysis: Detailed analysis JSON from LLM
        raw_results: The raw search results list, passed in for the index
    
    Returns:
        Formatted markdown report string
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = f"""
# Social Sentiment Analysis Report: ${ticker}
**Generated:** {timestamp}
**Agent:** Social-Sentimentalist
**Model:** Gemini Pro Latest

---

## Executive Summary

{analysis.get('executive_summary', 'No summary available.')}

---

## Sentiment Overview

**Overall Sentiment:** {analysis.get('overall_sentiment', 'N/A')}  
**Sentiment Score:** {analysis.get('sentiment_score', 0.0):+.2f} / 1.0  
**Data Quality:** {analysis.get('data_quality', 'unknown').capitalize()}  
**Discussion Volume:** {analysis.get('discussion_volume', 'unknown').capitalize()}

### Sentiment Reasoning
{analysis.get('sentiment_reasoning', 'No detailed reasoning provided.')}

---

## Consensus View

{analysis.get('consensus_view', 'No clear consensus identified.')}

---

## Source-by-Source Analysis
"""
    
    sources = analysis.get('source_analyses', [])
    if sources:
        for source in sources:
            report += f"""
### Source {source.get('source_number', 'N/A')}: {source.get('source_title', 'No Title')}

**Sentiment:** {source.get('sentiment', 'N/A')}
**Platform:** {source.get('platform', 'unknown').capitalize()}
**URL:** {source.get('source_url', 'No URL')}

**Key Points:**
"""
            for point in source.get('key_points', []):
                report += f"- {point}\n"
            
            report += f"\n**Direct Quotes:**\n"
            for quote in source.get('direct_quotes', []):
                report += f"- \"{quote}\"\n"
            
            report += f"\n**Analysis:** {source.get('analysis', 'No analysis available.')}\n"
    else:
        report += "\n*No individual sources could be analyzed.*\n"
    
    report += "\n---\n\n## Key Themes\n"
    
    themes = analysis.get('key_themes', [])
    if themes:
        for theme in themes:
            report += f"""
### {theme.get('theme', 'Unknown Theme')}

**Sentiment:** {theme.get('sentiment', 'N/A')}  
**Prevalence:** {theme.get('prevalence', 'unknown').capitalize()}

{theme.get('description', 'No description available.')}

**Supporting Quotes:**
"""
            for quote in theme.get('supporting_quotes', []):
                report += f"- \"{quote.get('quote')}\" (Source #{quote.get('source_number')})\n"
    else:
        report += "\n*No recurring themes identified.*\n"
    
    report += "\n---\n\n## Notable Quotes\n"
    
    quotes = analysis.get('notable_quotes', [])
    if quotes:
        for quote in quotes:
            sentiment_emoji = {
                'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '⚪'
            }.get(quote.get('sentiment', 'Neutral'), '⚪')
            
            report += f"""
{sentiment_emoji} **{quote.get('sentiment', 'N/A')}** > "{quote.get('quote', 'No quote available.')}"
*Source #{quote.get('source_number')} ({quote.get('source_url')})*
"""
    else:
        report += "\n*No notable quotes extracted.*\n"
    
    report += "\n---\n\n## Risk Factors & Concerns\n"
    
    risks = analysis.get('risk_factors', [])
    if risks:
        for risk in risks:
            report += f"- **{risk.get('risk', 'Unknown Risk')}** (Sources: {risk.get('source_numbers')})\n"
            report += f"  > *\"{risk.get('supporting_quote', 'N/A')}\"*\n"
    else:
        report += "\n*No significant risk factors mentioned.*\n"
    
    report += "\n---\n\n## Bullish Catalysts & Drivers\n"
    
    catalysts = analysis.get('bullish_catalysts', [])
    if catalysts:
        for catalyst in catalysts:
            report += f"- **{catalyst.get('catalyst', 'Unknown Catalyst')}** (Sources: {catalyst.get('source_numbers')})\n"
            report += f"  > *\"{catalyst.get('supporting_quote', 'N/A')}\"*\n"
    else:
        report += "\n*No bullish catalysts identified.*\n"
    
    report += f"""

---

## Source Index

Below are all the sources analyzed in this report:

"""
    
    if raw_results:
        for i, result in enumerate(raw_results, 1):
            report += f"{i}. **{result.get('title', 'No title')}**\n"
            report += f"   - URL: {result.get('link', 'No URL')}\n\n"
    else:
        report += "*No raw sources were provided to the report generator.*\n"

    report += f"""
---

## Metadata

**Search Status:** {analysis.get('search_status', 'unknown')}  
**Number of Sources:** {analysis.get('result_count', 0)}  
**Analysis Timestamp:** {timestamp}

---

*This report is generated from public social media, forums, and discussion boards. It represents retail investor sentiment and should not be considered financial advice.*
"""
    
    return report.strip()

def save_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """
    Save the detailed report to a text file.
    
    Args:
        ticker: Stock ticker symbol
        report: Formatted report content
        output_dir: Directory to save reports (created if doesn't exist)
    
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_social_sentiment_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    # Write report to file
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {filepath}")
    return filepath


# --- MAIN AGENT FUNCTION ---

def run_social_agent(ticker: str, save_to_file: bool = True) -> tuple[str, str]:
    """
    Execute the Social-Sentimentalist Agent analysis.
    """
    print(f"\n{'='*60}")
    print(f"🤖 Social-Sentimentalist Agent: Analyzing ${ticker}")
    print(f"{'='*60}\n")
    
    try:
        # --- FIX: Break the chain into search and analysis steps ---
        
        # 1. Run the search part first to get results
        print("Running search...")
        search_output = run_search({"ticker": ticker.upper()})
        
        if search_output["search_status"] == "failed":
            raise Exception(f"Search failed: {search_output['social_data']}")
        
        if search_output["search_status"] == "partial":
            print(f"Warning: {search_output['social_data']}")

        # Store raw_results locally so it's not lost
        raw_results = search_output.get("raw_results", [])

        # 2. Run the analysis part of the chain
        print("Running analysis with LLM...")
        # Pass the search_output dictionary as the input to the prompt
        analysis_json = analysis_chain.invoke(search_output)
        
        # --- End of Fix ---

        # Add search status metadata back into the final JSON for the report
        # The LLM doesn't return these, so we add them back manually
        analysis_json['search_status'] = search_output.get('search_status')
        analysis_json['result_count'] = search_output.get('result_count')

        # Generate detailed report
        # Now, raw_results is correctly passed
        detailed_report = generate_detailed_report(ticker, analysis_json, raw_results)
        
        # Save to file if requested
        if save_to_file:
            save_report(ticker, detailed_report)
        
        # Create summary report for console/orchestrator
        sentiment = analysis_json.get('overall_sentiment', 'N/A')
        score = analysis_json.get('sentiment_score', 0.0)
        quality = analysis_json.get('data_quality', 'unknown')
        exec_summary = analysis_json.get('executive_summary', 'No summary available.')
        source_count = analysis_json.get('result_count', 0)
        
        summary_report = f"""
**Social-Sentimentalist Agent Summary: ${ticker}**

📊 **Quick Overview**
* **Sentiment:** {sentiment} ({score:+.2f})
* **Data Quality:** {quality.capitalize()}
* **Sources Analyzed:** {source_count}

📝 **Executive Summary:**
{exec_summary}

💡 **Key Insight:**
{analysis_json.get('consensus_view', 'See detailed report for full analysis.')}

---
*Full detailed report with source citations saved to file*
*Agent: Social-Sentimentalist | Model: Gemini Pro*
        """
        
        print("✅ Social_Agent: Analysis complete")
        return summary_report.strip(), detailed_report
        
    except Exception as e:
        print(f"❌ Social_Agent: Chain execution failed - {e}")
        import traceback
        traceback.print_exc()
        
        error_report = f"""
**Social-Sentimentalist Agent Report: ${ticker}**

⚠️ **Error:** Analysis could not be completed.
**Details:** {str(e)}
"""
        return error_report.strip(), error_report.strip()

# --- CLI ENTRY POINT ---

if __name__ == "__main__":
    import sys
    
    # Get ticker from command line or use default
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "COIN"  # Default example
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python social_agent.py <TICKER>\n")
    
    # Run the agent
    summary, detailed = run_social_agent(ticker, save_to_file=True)
    
    # Display summary
    print("\n" + "="*60)
    print("SUMMARY OUTPUT (Console)")
    print("="*60 + "\n")
    print(summary)
    print("\n" + "="*60)
    print("\n📄 Full detailed report saved to 'reports/' directory")