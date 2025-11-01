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
    # Optimized query for social sentiment
    # Since CSE is pre-filtered to social sites, we can be more specific
    query = f"{ticker} stock bullish bearish sentiment discussion"
    
    print(f"🔍 Social_Agent: Searching social sentiment for {ticker}...")
    
    try:
        # Use .results() instead of .run() to get structured data
        results = social_search_tool.results(query, num_results=10)
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
    formatted = ""
    for i, result in enumerate(results, 1):
        formatted += f"\n{'='*60}\n"
        formatted += f"SOURCE #{i}\n"
        formatted += f"{'='*60}\n"
        formatted += f"Title: {result.get('title', 'No title')}\n"
        formatted += f"URL: {result.get('link', 'No URL')}\n"
        formatted += f"Snippet:\n{result.get('snippet', 'No snippet available')}\n"
    
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
    """You are an expert "Social-Sentimentalist" financial analyst. Your task is to 
analyze the following collection of public forum snippets, search results, 
and discussions about the stock ${ticker}.

**CRITICAL INSTRUCTIONS:**
1. Each SOURCE is numbered and includes a Title, URL, and Snippet
2. You MUST cite specific SOURCE numbers when making claims
3. You MUST quote actual text from the snippets (use exact phrases)
4. Do NOT paraphrase - use the actual language from the sources
5. Include the SOURCE URL in your analysis for each point

Provide a COMPREHENSIVE, DETAILED analysis that includes:

1. **Executive Summary**: Overall sentiment and key takeaways (2-3 sentences)
2. **Sentiment Breakdown**: Quantified sentiment with score and reasoning
3. **Source-by-Source Analysis**: Analyze EACH numbered source individually with:
   - Source number and URL
   - Direct quotes from the snippet
   - Sentiment of that specific source
4. **Key Themes**: Identify and explain 3-5 recurring topics with direct quotes
5. **Notable Quotes**: Extract 5-10 ACTUAL quotes (exact text) with source numbers
6. **Risk Factors**: What concerns are mentioned (with source citations)
7. **Catalysts**: What positive drivers are mentioned (with source citations)
8. **Consensus View**: What is the crowd's general consensus

Return your analysis in the following JSON format:
{{
  "executive_summary": "2-3 sentence overview",
  "overall_sentiment": "Bullish" | "Bearish" | "Neutral",
  "sentiment_score": 0.0,
  "sentiment_reasoning": "Detailed explanation with source citations",
  "data_quality": "excellent" | "good" | "limited" | "insufficient",
  "source_analyses": [
    {{
      "source_number": 1,
      "source_url": "full URL",
      "source_title": "title of source",
      "platform": "reddit" | "twitter" | "stocktwits" | "seekingalpha" | "other",
      "direct_quotes": ["exact quote 1", "exact quote 2"],
      "sentiment": "Bullish" | "Bearish" | "Neutral",
      "key_points": ["point 1", "point 2"],
      "analysis": "Your interpretation of this source"
    }}
  ],
  "key_themes": [
    {{
      "theme": "Theme name",
      "description": "What this theme is about",
      "sentiment": "Bullish" | "Bearish" | "Neutral",
      "prevalence": "high" | "medium" | "low",
      "supporting_quotes": [
        {{
          "quote": "exact text",
          "source_number": 1,
          "source_url": "URL"
        }}
      ]
    }}
  ],
  "notable_quotes": [
    {{
      "quote": "EXACT text from snippet",
      "source_number": 1,
      "source_url": "URL",
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

Return *only* the JSON object, no additional text."""
)


# --- CREATE DETAILED ANALYSIS CHAIN ---

detailed_chain = (
    RunnablePassthrough.assign(search_results=run_search)
    | RunnablePassthrough.assign(
        ticker=lambda x: x["search_results"]["ticker"],
        social_data=lambda x: x["search_results"]["social_data"],
        raw_results=lambda x: x["search_results"]["raw_results"],
        search_status=lambda x: x["search_results"]["search_status"],
        result_count=lambda x: x["search_results"]["result_count"]
    )
    | detailed_prompt_template
    | llm
    | JsonOutputParser()
)


# --- REPORT GENERATION ---

def generate_detailed_report(ticker: str, analysis: Dict[str, Any]) -> str:
    """
    Generate a comprehensive markdown report from analysis results.
    
    Args:
        ticker: Stock ticker symbol
        analysis: Detailed analysis JSON from LLM
    
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
    
    # Add source analyses
    sources = analysis.get('source_analyses', [])
    if sources:
        for i, source in enumerate(sources, 1):
            report += f"""
### Source {i}: {source.get('source_type', 'Unknown').capitalize()}

**Sentiment:** {source.get('sentiment', 'N/A')}

**Key Points:**
"""
            for point in source.get('key_points', []):
                report += f"- {point}\n"
            
            report += f"\n**Discussion:** {source.get('discussion_snippet', 'No details available.')}\n"
    else:
        report += "\n*No individual sources could be analyzed.*\n"
    
    report += "\n---\n\n## Key Themes\n"
    
    # Add key themes
    themes = analysis.get('key_themes', [])
    if themes:
        for theme in themes:
            report += f"""
### {theme.get('theme', 'Unknown Theme')}

**Sentiment:** {theme.get('sentiment', 'N/A')}  
**Prevalence:** {theme.get('prevalence', 'unknown').capitalize()}

{theme.get('description', 'No description available.')}
"""
    else:
        report += "\n*No recurring themes identified.*\n"
    
    report += "\n---\n\n## Notable Quotes\n"
    
    # Add notable quotes
    quotes = analysis.get('notable_quotes', [])
    if quotes:
        for quote in quotes:
            sentiment_emoji = {
                'Bullish': '🟢',
                'Bearish': '🔴',
                'Neutral': '⚪'
            }.get(quote.get('sentiment', 'Neutral'), '⚪')
            
            report += f"""
{sentiment_emoji} **{quote.get('sentiment', 'N/A')}**  
> "{quote.get('quote', 'No quote available.')}"

*Context:* {quote.get('context', 'No context provided.')}
"""
    else:
        report += "\n*No notable quotes extracted.*\n"
    
    report += "\n---\n\n## Risk Factors & Concerns\n"
    
    # Add risk factors
    risks = analysis.get('risk_factors', [])
    if risks:
        for risk in risks:
            report += f"- {risk}\n"
    else:
        report += "\n*No significant risk factors mentioned.*\n"
    
    report += "\n---\n\n## Bullish Catalysts & Drivers\n"
    
    # Add catalysts
    catalysts = analysis.get('bullish_catalysts', [])
    if catalysts:
        for catalyst in catalysts:
            report += f"- {catalyst}\n"
    else:
        report += "\n*No bullish catalysts identified.*\n"
    
    report += f"""

---

## Metadata

**Search Status:** {analysis.get('search_status', 'unknown')}  
**Data Length:** {analysis.get('result_length', 0)} characters  
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
    
    Args:
        ticker: Stock ticker symbol (e.g., "AAPL", "TSLA")
        save_to_file: Whether to save detailed report to file (default: True)
    
    Returns:
        Tuple of (summary_report, detailed_report)
    
    Example:
        >>> summary, detailed = run_social_agent("COIN")
        >>> print(summary)
    """
    print(f"\n{'='*60}")
    print(f"🤖 Social-Sentimentalist Agent: Analyzing ${ticker}")
    print(f"{'='*60}\n")
    
    try:
        # Execute the detailed analysis chain
        analysis_json = detailed_chain.invoke({"ticker": ticker.upper()})
        
        # Generate detailed report
        detailed_report = generate_detailed_report(ticker, analysis_json)
        
        # Save to file if requested
        if save_to_file:
            save_report(ticker, detailed_report)
        
        # Extract data for summary report
        sentiment = analysis_json.get('overall_sentiment', 'N/A')
        score = analysis_json.get('sentiment_score', 0.0)
        quality = analysis_json.get('data_quality', 'unknown')
        exec_summary = analysis_json.get('executive_summary', 'No summary available.')
        
        # Generate concise summary for console/API
        summary_report = f"""
**Social-Sentimentalist Agent Summary: ${ticker}**

📊 **Quick Overview**
* **Sentiment:** {sentiment} ({score:+.2f})
* **Data Quality:** {quality.capitalize()}

📝 **Executive Summary:**
{exec_summary}

💡 **Key Insight:**
{analysis_json.get('consensus_view', 'See detailed report for full analysis.')}

---
*Full detailed report saved to file*
*Agent: Social-Sentimentalist | Model: Gemini Pro*
        """
        
        print("✅ Social_Agent: Analysis complete")
        return summary_report.strip(), detailed_report
        
    except Exception as e:
        print(f"❌ Social_Agent: Chain execution failed - {e}")
        
        error_report = f"""
**Social-Sentimentalist Agent Report: ${ticker}**

⚠️ **Error:** Analysis could not be completed.

**Details:** {str(e)}

**Possible Causes:**
* API rate limits exceeded
* Network connectivity issues
* Invalid ticker symbol
* Insufficient search results

**Recommendation:** Retry in a few minutes or check environment configuration.
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