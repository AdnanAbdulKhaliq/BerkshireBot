"""
Social Sentiment Agent - Analyst Swarm Component (Enhanced with Result Filtering)

This agent analyzes retail investor sentiment from social media with:
- Freshness filtering (prioritize recent content)
- Result deduplication
- Quality scoring

Environment Variables Required:
    - GEMINI_API_KEY: Your Google Gemini API key
    - GOOGLE_API_KEY: Your Google Custom Search API key
    - GOOGLE_CSE_ID_SOCIAL: Your Social Sentiment Engine CSE ID
"""

import os
import time
import json
import re
from datetime import datetime, timedelta
from functools import lru_cache
from typing import Dict, Any, List, Tuple
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

validate_environment()

# --- SETUP TOOLS & MODELS ---

llm = ChatGoogleGenerativeAI(
    model="gemini-pro-latest",
    temperature=0.1,
    api_key=os.environ["GEMINI_API_KEY"]
)

social_search_tool = GoogleSearchAPIWrapper(
    google_api_key=os.environ["GOOGLE_API_KEY"],
    google_cse_id=os.environ["GOOGLE_CSE_ID_SOCIAL"]
)

# --- TICKER TO COMPANY NAME MAPPING ---

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

# --- RESULT FRESHNESS FILTERING ---

def extract_time_indicators(text: str) -> Tuple[bool, str]:
    """
    Extract time indicators from snippet text to determine freshness.
    
    Args:
        text: Snippet or title text
    
    Returns:
        Tuple of (is_recent, time_indicator_found)
    """
    text_lower = text.lower()
    
    # Very recent indicators (last 24 hours)
    very_recent = [
        r'\d+\s*(hour|hr)s?\s*ago',
        r'\d+\s*(minute|min)s?\s*ago',
        r'just now',
        r'moments ago',
        r'today',
        r'this morning',
        r'this afternoon',
        r'tonight'
    ]
    
    # Recent indicators (last 7 days)
    recent = [
        r'yesterday',
        r'\d+\s*days?\s*ago',
        r'this week',
        r'last week'
    ]
    
    # Check for very recent
    for pattern in very_recent:
        if re.search(pattern, text_lower):
            return True, "very_recent"
    
    # Check for recent
    for pattern in recent:
        if re.search(pattern, text_lower):
            # Extract days if mentioned
            days_match = re.search(r'(\d+)\s*days?\s*ago', text_lower)
            if days_match:
                days = int(days_match.group(1))
                if days <= 7:
                    return True, f"recent_{days}d"
                elif days <= 30:
                    return False, f"moderate_{days}d"
                else:
                    return False, f"old_{days}d"
            return True, "recent"
    
    # No clear time indicator
    return None, "unknown"


def calculate_freshness_score(result: Dict[str, Any]) -> float:
    """
    Calculate a freshness score (0-1) for a search result.
    
    Args:
        result: Search result dictionary
    
    Returns:
        Freshness score between 0 (old) and 1 (very fresh)
    """
    snippet = result.get('snippet', '')
    title = result.get('title', '')
    
    # Check snippet first
    is_recent, indicator = extract_time_indicators(snippet)
    
    # If not found in snippet, check title
    if is_recent is None:
        is_recent, indicator = extract_time_indicators(title)
    
    # Score based on indicator
    if indicator == "very_recent":
        return 1.0
    elif indicator.startswith("recent"):
        if "_" in indicator:
            days = int(indicator.split("_")[1].replace("d", ""))
            return max(0.6, 1.0 - (days / 14.0))  # Decay over 14 days
        return 0.8
    elif indicator.startswith("moderate"):
        return 0.4
    elif indicator.startswith("old"):
        return 0.1
    else:
        # No clear indicator - assume moderate age
        return 0.5


def filter_by_freshness(
    results: List[Dict[str, Any]],
    min_score: float = 0.3,
    days_back: int = 30
) -> List[Dict[str, Any]]:
    """
    Filter search results by freshness score.
    
    Args:
        results: List of search result dictionaries
        min_score: Minimum freshness score to keep (0-1)
        days_back: Maximum days back to consider
    
    Returns:
        Filtered list of results with freshness scores
    """
    scored_results = []
    
    for result in results:
        freshness_score = calculate_freshness_score(result)
        result['freshness_score'] = freshness_score
        result['freshness_tier'] = (
            "very_fresh" if freshness_score >= 0.8 else
            "fresh" if freshness_score >= 0.6 else
            "moderate" if freshness_score >= 0.4 else
            "old"
        )
        
        # Keep if above minimum score
        if freshness_score >= min_score:
            scored_results.append(result)
    
    # Sort by freshness score (highest first)
    scored_results.sort(key=lambda x: x['freshness_score'], reverse=True)
    
    return scored_results


# --- RESULT DEDUPLICATION ---

def calculate_similarity(text1: str, text2: str) -> float:
    """
    Calculate simple text similarity between two strings.
    
    Args:
        text1, text2: Text strings to compare
    
    Returns:
        Similarity score between 0 and 1
    """
    # Simple word-based similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union)


def deduplicate_results(
    results: List[Dict[str, Any]],
    similarity_threshold: float = 0.6
) -> List[Dict[str, Any]]:
    """
    Remove duplicate or very similar results.
    
    Args:
        results: List of search result dictionaries
        similarity_threshold: Similarity threshold for considering duplicates
    
    Returns:
        Deduplicated list
    """
    if not results:
        return []
    
    deduplicated = [results[0]]  # Always keep first result
    
    for result in results[1:]:
        current_snippet = result.get('snippet', '')
        current_title = result.get('title', '')
        current_url = result.get('link', '')
        
        is_duplicate = False
        
        for existing in deduplicated:
            existing_snippet = existing.get('snippet', '')
            existing_title = existing.get('title', '')
            existing_url = existing.get('link', '')
            
            # Check URL similarity (same domain + similar path)
            if current_url and existing_url:
                if current_url == existing_url:
                    is_duplicate = True
                    break
            
            # Check snippet similarity
            snippet_sim = calculate_similarity(current_snippet, existing_snippet)
            if snippet_sim > similarity_threshold:
                is_duplicate = True
                break
            
            # Check title similarity
            title_sim = calculate_similarity(current_title, existing_title)
            if title_sim > similarity_threshold:
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(result)
    
    return deduplicated


# --- RESULT QUALITY SCORING ---

def calculate_quality_score(result: Dict[str, Any]) -> float:
    """
    Calculate overall quality score for a search result.
    
    Args:
        result: Search result dictionary
    
    Returns:
        Quality score between 0 and 1
    """
    score = 0.5  # Base score
    
    snippet = result.get('snippet', '')
    title = result.get('title', '')
    url = result.get('link', '')
    
    # Length indicators (longer snippets tend to have more context)
    if len(snippet) > 100:
        score += 0.1
    if len(snippet) > 200:
        score += 0.1
    
    # Source quality (prioritize known platforms)
    high_quality_domains = ['reddit.com', 'seekingalpha.com', 'twitter.com', 'stocktwits.com']
    if any(domain in url.lower() for domain in high_quality_domains):
        score += 0.2
    
    # Content indicators (specific terms suggest substantive discussion)
    quality_terms = [
        'earnings', 'revenue', 'profit', 'growth', 'valuation',
        'analysis', 'forecast', 'target', 'upgrade', 'downgrade',
        'catalyst', 'risk', 'opportunity', 'investment'
    ]
    
    text_combined = (snippet + ' ' + title).lower()
    quality_term_count = sum(1 for term in quality_terms if term in text_combined)
    score += min(0.2, quality_term_count * 0.05)
    
    # Freshness bonus (from previously calculated)
    freshness = result.get('freshness_score', 0.5)
    score += freshness * 0.2
    
    return min(1.0, score)


def filter_by_quality(
    results: List[Dict[str, Any]],
    min_score: float = 0.4
) -> List[Dict[str, Any]]:
    """
    Filter results by quality score and add quality metadata.
    
    Args:
        results: List of search result dictionaries
        min_score: Minimum quality score to keep
    
    Returns:
        Filtered and scored results
    """
    scored_results = []
    
    for result in results:
        quality_score = calculate_quality_score(result)
        result['quality_score'] = quality_score
        result['quality_tier'] = (
            "high" if quality_score >= 0.7 else
            "medium" if quality_score >= 0.5 else
            "low"
        )
        
        if quality_score >= min_score:
            scored_results.append(result)
    
    # Sort by combined freshness + quality
    scored_results.sort(
        key=lambda x: (x.get('freshness_score', 0) * 0.6 + x.get('quality_score', 0) * 0.4),
        reverse=True
    )
    
    return scored_results


# --- ENHANCED SEARCH WITH FILTERING ---

@lru_cache(maxsize=100)
def cached_search(ticker: str, timestamp_hour: int) -> list:
    """
    Cached search with enhanced filtering.
    Cache expires every hour.
    
    Args:
        ticker: Stock ticker symbol
        timestamp_hour: Current hour timestamp for cache key
    
    Returns:
        List of filtered, scored search results
    """
    company_name = TICKER_TO_NAME.get(ticker, ticker)
    query = f"{company_name} {ticker} stock sentiment discussion"
    
    print(f"🔍 Social_Agent: Searching for '{query}'...")
    
    try:
        # Get raw results
        raw_results = social_search_tool.results(query, num_results=15)  # Get more initially
        print(f"📥 Retrieved {len(raw_results)} raw results")
        
        # Apply freshness filtering
        fresh_results = filter_by_freshness(raw_results, min_score=0.3, days_back=30)
        print(f"✨ {len(fresh_results)} results after freshness filter")
        
        # Apply deduplication
        unique_results = deduplicate_results(fresh_results, similarity_threshold=0.6)
        print(f"🔗 {len(unique_results)} results after deduplication")
        
        # Apply quality filtering
        quality_results = filter_by_quality(unique_results, min_score=0.4)
        print(f"⭐ {len(quality_results)} results after quality filter")
        
        # Keep top 10 highest scoring
        final_results = quality_results[:10]
        print(f"📊 Final: {len(final_results)} high-quality, recent results")
        
        return final_results
        
    except Exception as e:
        print(f"❌ Search API error: {e}")
        raise


def format_sources_for_llm(results: list) -> str:
    """Format search results with quality metadata for LLM."""
    if not results:
        return "No sources found."
    
    formatted = "Below are HIGH-QUALITY, RECENT social media sources. Each has freshness and quality scores.\n\n"
    
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        url = result.get('link', 'No URL')
        snippet = result.get('snippet', 'No snippet available')
        freshness_score = result.get('freshness_score', 0)
        freshness_tier = result.get('freshness_tier', 'unknown')
        quality_score = result.get('quality_score', 0)
        quality_tier = result.get('quality_tier', 'unknown')
        
        formatted += f"\n{'='*70}\n"
        formatted += f"SOURCE #{i}\n"
        formatted += f"{'='*70}\n"
        formatted += f"TITLE: {title}\n"
        formatted += f"URL: {url}\n"
        formatted += f"FRESHNESS: {freshness_tier.upper()} (score: {freshness_score:.2f})\n"
        formatted += f"QUALITY: {quality_tier.upper()} (score: {quality_score:.2f})\n"
        formatted += f"SNIPPET:\n{snippet}\n"
    
    formatted += f"\n{'='*70}\n"
    formatted += f"END OF SOURCES (Total: {len(results)} filtered sources)\n"
    formatted += f"All sources have been filtered for freshness and quality.\n"
    formatted += f"{'='*70}\n"
    
    return formatted


def run_search(input_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Execute enhanced search with filtering and error handling."""
    ticker = input_dict.get("ticker", "").upper()
    
    if not ticker:
        return {
            "ticker": "",
            "social_data": "Error: No ticker provided",
            "raw_results": [],
            "search_status": "failed",
            "result_count": 0,
            "freshness_stats": {},
            "quality_stats": {}
        }
    
    try:
        cache_key = int(time.time() // 3600)
        filtered_results = cached_search(ticker, cache_key)
        
        if not filtered_results:
            return {
                "ticker": ticker,
                "social_data": f"No recent, high-quality social data found for {ticker}. Discussion may be minimal or outdated.",
                "raw_results": [],
                "search_status": "partial",
                "result_count": 0,
                "freshness_stats": {},
                "quality_stats": {}
            }
        
        # Calculate statistics
        freshness_stats = {
            "very_fresh": sum(1 for r in filtered_results if r.get('freshness_tier') == 'very_fresh'),
            "fresh": sum(1 for r in filtered_results if r.get('freshness_tier') == 'fresh'),
            "moderate": sum(1 for r in filtered_results if r.get('freshness_tier') == 'moderate'),
            "avg_freshness_score": sum(r.get('freshness_score', 0) for r in filtered_results) / len(filtered_results)
        }
        
        quality_stats = {
            "high": sum(1 for r in filtered_results if r.get('quality_tier') == 'high'),
            "medium": sum(1 for r in filtered_results if r.get('quality_tier') == 'medium'),
            "low": sum(1 for r in filtered_results if r.get('quality_tier') == 'low'),
            "avg_quality_score": sum(r.get('quality_score', 0) for r in filtered_results) / len(filtered_results)
        }
        
        formatted_data = format_sources_for_llm(filtered_results)
        
        print(f"✅ Social_Agent: Retrieved {len(filtered_results)} filtered sources")
        print(f"   📈 Freshness: {freshness_stats['very_fresh']} very fresh, {freshness_stats['fresh']} fresh")
        print(f"   ⭐ Quality: {quality_stats['high']} high, {quality_stats['medium']} medium")
        
        return {
            "ticker": ticker,
            "social_data": formatted_data,
            "raw_results": filtered_results,
            "search_status": "success",
            "result_count": len(filtered_results),
            "freshness_stats": freshness_stats,
            "quality_stats": quality_stats
        }
        
    except Exception as e:
        print(f"❌ Social_Agent: Search failed - {e}")
        return {
            "ticker": ticker,
            "social_data": f"Error retrieving social data: {str(e)}",
            "raw_results": [],
            "search_status": "failed",
            "result_count": 0,
            "freshness_stats": {},
            "quality_stats": {}
        }


# --- PROMPT TEMPLATE (keeping existing structure) ---

detailed_prompt_template = ChatPromptTemplate.from_template(
    """You are an expert "Social-Sentimentalist" financial analyst analyzing retail investor sentiment for ${ticker}.

**DATA QUALITY NOTE:**
All sources have been filtered for:
- Freshness (recent content prioritized)
- Quality (substantive discussion prioritized)
- Uniqueness (duplicates removed)

Each source includes FRESHNESS and QUALITY scores.

**CRITICAL INSTRUCTIONS:**

For EVERY source, you MUST:
1. Extract the EXACT title and URL provided
2. Quote EXACT phrases from the snippet (use quotation marks)
3. Cite the source number (e.g., "Source #3")
4. Note the freshness and quality tiers
5. Do NOT make up or paraphrase - use actual text

**YOUR ANALYSIS MUST INCLUDE:**

1. **Executive Summary**: Emphasizing data freshness and quality
2. **Source-by-Source Analysis**: For EACH source with freshness/quality notes
3. **Key Themes**: With quotes and source citations
4. **Notable Quotes**: EXACT quotes with source numbers
5. **Risk Factors**: With source citations
6. **Bullish Catalysts**: With source citations

Return analysis in this JSON format:
{{
  "executive_summary": "2-3 sentence overview noting data quality",
  "overall_sentiment": "Bullish" | "Bearish" | "Neutral",
  "sentiment_score": -1.0 to 1.0,
  "sentiment_reasoning": "Explain with source numbers and freshness context",
  "data_quality": "excellent" | "good" | "limited" | "insufficient",
  "data_freshness_assessment": "Assessment of how recent the data is",
  "source_analyses": [
    {{
      "source_number": 1,
      "source_title": "EXACT title",
      "source_url": "EXACT URL",
      "platform": "reddit" | "twitter" | "stocktwits" | "seekingalpha" | "other",
      "freshness_tier": "from source metadata",
      "quality_tier": "from source metadata",
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
          "quote": "exact text",
          "source_number": 1,
          "source_url": "exact URL",
          "freshness_tier": "from metadata"
        }}
      ]
    }}
  ],
  "notable_quotes": [
    {{
      "quote": "EXACT text",
      "source_number": 1,
      "source_url": "exact URL",
      "sentiment": "Bullish" | "Bearish" | "Neutral",
      "freshness_tier": "very_fresh" | "fresh" | "moderate"
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
  "discussion_volume": "high" | "medium" | "low",
  "freshness_summary": "Summary of how fresh the data is",
  "quality_summary": "Summary of data quality"
}}

<search_metadata>
Ticker: {ticker}
Search Status: {search_status}
Number of Sources: {result_count}
Freshness Stats: {freshness_stats}
Quality Stats: {quality_stats}
</search_metadata>

<social_data>
{social_data}
</social_data>

Return ONLY valid JSON."""
)

analysis_chain = detailed_prompt_template | llm | JsonOutputParser()

# --- REPORT GENERATION (keeping existing, adding freshness/quality info) ---

def generate_detailed_report(ticker: str, analysis: Dict[str, Any], raw_results: list, search_metadata: Dict[str, Any]) -> str:
    """Generate comprehensive markdown report with freshness and quality metrics."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    freshness_stats = search_metadata.get('freshness_stats', {})
    quality_stats = search_metadata.get('quality_stats', {})
    
    report = f"""
# Social Sentiment Analysis Report: ${ticker}
**Generated:** {timestamp}
**Agent:** Social-Sentimentalist (Enhanced with Filtering)
**Model:** Gemini Pro Latest

---

## Data Quality & Freshness Report

**Search Results:** {search_metadata.get('result_count', 0)} filtered sources  
**Data Quality:** {analysis.get('data_quality', 'unknown').capitalize()}

### Freshness Distribution:
- 🔥 Very Fresh (< 24h): {freshness_stats.get('very_fresh', 0)}
- ✨ Fresh (< 7 days): {freshness_stats.get('fresh', 0)}
- 📅 Moderate (< 30 days): {freshness_stats.get('moderate', 0)}
- **Average Freshness Score:** {freshness_stats.get('avg_freshness_score', 0):.2f}/1.0

### Quality Distribution:
- ⭐ High Quality: {quality_stats.get('high', 0)}
- ✅ Medium Quality: {quality_stats.get('medium', 0)}
- ⚠️ Low Quality: {quality_stats.get('low', 0)}
- **Average Quality Score:** {quality_stats.get('avg_quality_score', 0):.2f}/1.0

**Freshness Assessment:** {analysis.get('data_freshness_assessment', 'No assessment')}  
**Quality Assessment:** {analysis.get('quality_summary', 'No assessment')}

---

## Executive Summary

{analysis.get('executive_summary', 'No summary available.')}

---

## Sentiment Overview

**Overall Sentiment:** {analysis.get('overall_sentiment', 'N/A')}  
**Sentiment Score:** {analysis.get('sentiment_score', 0.0):+.2f} / 1.0  
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
            freshness_emoji = {
                'very_fresh': '🔥',
                'fresh': '✨',
                'moderate': '📅',
                'old': '⏳'
            }.get(source.get('freshness_tier', ''), '❓')
            
            quality_emoji = {
                'high': '⭐',
                'medium': '✅',
                'low': '⚠️'
            }.get(source.get('quality_tier', ''), '❓')
            
            report += f"""
### Source {source.get('source_number', 'N/A')}: {source.get('source_title', 'No Title')}

**Sentiment:** {source.get('sentiment', 'N/A')}  
**Platform:** {source.get('platform', 'unknown').capitalize()}  
**Freshness:** {freshness_emoji} {source.get('freshness_tier', 'unknown').replace('_', ' ').title()}  
**Quality:** {quality_emoji} {source.get('quality_tier', 'unknown').capitalize()}  
**URL:** {source.get('source_url', 'No URL')}

**Key Points:**
"""
            for point in source.get('key_points', []):
                report += f"- {point}\n"
            
            report += f"\n**Direct Quotes:**\n"
            for quote in source.get('direct_quotes', []):
                report += f"- \"{quote}\"\n"
            
            report += f"\n**Analysis:** {source.get('analysis', 'No analysis.')}\n"
    else:
        report += "\n*No sources could be analyzed.*\n"
    
    report += "\n---\n\n## Key Themes\n"
    
    themes = analysis.get('key_themes', [])
    if themes:
        for theme in themes:
            report += f"""
### {theme.get('theme', 'Unknown Theme')}

**Sentiment:** {theme.get('sentiment', 'N/A')}  
**Prevalence:** {theme.get('prevalence', 'unknown').capitalize()}

{theme.get('description', 'No description.')}

**Supporting Quotes:**
"""
            for quote in theme.get('supporting_quotes', []):
                freshness = quote.get('freshness_tier', 'unknown')
                fresh_emoji = '🔥' if freshness == 'very_fresh' else '✨' if freshness == 'fresh' else '📅'
                report += f"- {fresh_emoji} \"{quote.get('quote')}\" (Source #{quote.get('source_number')})\n"
    else:
        report += "\n*No themes identified.*\n"
    
    report += "\n---\n\n## Notable Quotes\n"
    
    quotes = analysis.get('notable_quotes', [])
    if quotes:
        for quote in quotes:
            sentiment_emoji = {'Bullish': '🟢', 'Bearish': '🔴', 'Neutral': '⚪'}.get(quote.get('sentiment', 'Neutral'), '⚪')
            fresh_emoji = '🔥' if quote.get('freshness_tier') == 'very_fresh' else '✨'
            
            report += f"""
{sentiment_emoji} {fresh_emoji} **{quote.get('sentiment', 'N/A')}** > "{quote.get('quote', 'No quote.')}"
*Source #{quote.get('source_number')} ({quote.get('source_url')})*
"""
    else:
        report += "\n*No notable quotes extracted.*\n"
    
    report += "\n---\n\n## Risk Factors & Concerns\n"
    
    risks = analysis.get('risk_factors', [])
    if risks:
        for risk in risks:
            report += f"- **{risk.get('risk', 'Unknown')}** (Sources: {risk.get('source_numbers')})\n"
            report += f"  > *\"{risk.get('supporting_quote', 'N/A')}\"*\n"
    else:
        report += "\n*No significant risk factors mentioned.*\n"
    
    report += "\n---\n\n## Bullish Catalysts & Drivers\n"
    
    catalysts = analysis.get('bullish_catalysts', [])
    if catalysts:
        for catalyst in catalysts:
            report += f"- **{catalyst.get('catalyst', 'Unknown')}** (Sources: {catalyst.get('source_numbers')})\n"
            report += f"  > *\"{catalyst.get('supporting_quote', 'N/A')}\"*\n"
    else:
        report += "\n*No bullish catalysts identified.*\n"
    
    report += f"""

---

## Methodology & Filtering

This report uses **advanced filtering** to ensure data quality:

### Freshness Filtering
- Time indicators extracted from snippets and titles
- Content prioritized by recency (last 24 hours weighted highest)
- Results scored on freshness scale (0-1)

### Quality Filtering
- Source reputation scoring (known platforms prioritized)
- Content depth analysis (longer, substantive snippets preferred)
- Topic relevance scoring (financial terms weighted)

### Deduplication
- Similar content automatically removed
- URL-based duplicate detection
- Text similarity threshold: 60%

**Final Dataset:** {search_metadata.get('result_count', 0)} high-quality, recent sources

---

## Source Index

Below are all filtered sources analyzed in this report:

"""
    
    if raw_results:
        for i, result in enumerate(raw_results, 1):
            fresh_score = result.get('freshness_score', 0)
            qual_score = result.get('quality_score', 0)
            report += f"{i}. **{result.get('title', 'No title')}**\n"
            report += f"   - URL: {result.get('link', 'No URL')}\n"
            report += f"   - Freshness: {fresh_score:.2f}/1.0 | Quality: {qual_score:.2f}/1.0\n\n"
    else:
        report += "*No sources available.*\n"

    report += f"""
---

## Metadata

**Search Status:** {search_metadata.get('search_status', 'unknown')}  
**Number of Sources:** {search_metadata.get('result_count', 0)}  
**Analysis Timestamp:** {timestamp}

---

*This report is generated from filtered, recent social media and forum discussions.*
*All sources have been validated for freshness and quality.*
*This should not be considered financial advice.*
"""
    
    return report.strip()


def save_report(ticker: str, report: str, output_dir: str = "reports") -> str:
    """Save the detailed report to a text file."""
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_social_sentiment_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"📄 Report saved to: {filepath}")
    return filepath


# --- MAIN AGENT FUNCTION ---

def run_social_agent(ticker: str, save_to_file: bool = True) -> tuple[str, str]:
    """Execute the Social-Sentimentalist Agent analysis with enhanced filtering."""
    print(f"\n{'='*60}")
    print(f"🤖 Social-Sentimentalist Agent (Enhanced): Analyzing ${ticker}")
    print(f"{'='*60}\n")
    
    try:
        # Run search with filtering
        print("Running enhanced search with filtering...")
        search_output = run_search({"ticker": ticker.upper()})
        
        if search_output["search_status"] == "failed":
            raise Exception(f"Search failed: {search_output['social_data']}")
        
        if search_output["search_status"] == "partial":
            print(f"Warning: {search_output['social_data']}")

        raw_results = search_output.get("raw_results", [])
        
        # Run LLM analysis
        print("Running analysis with LLM...")
        analysis_json = analysis_chain.invoke(search_output)
        
        # Add metadata back
        analysis_json['search_status'] = search_output.get('search_status')
        analysis_json['result_count'] = search_output.get('result_count')

        # Generate detailed report with metadata
        search_metadata = {
            'search_status': search_output.get('search_status'),
            'result_count': search_output.get('result_count'),
            'freshness_stats': search_output.get('freshness_stats', {}),
            'quality_stats': search_output.get('quality_stats', {})
        }
        
        detailed_report = generate_detailed_report(ticker, analysis_json, raw_results, search_metadata)
        
        # Save to file if requested
        if save_to_file:
            save_report(ticker, detailed_report)
        
        # Create summary report
        sentiment = analysis_json.get('overall_sentiment', 'N/A')
        score = analysis_json.get('sentiment_score', 0.0)
        quality = analysis_json.get('data_quality', 'unknown')
        exec_summary = analysis_json.get('executive_summary', 'No summary available.')
        source_count = analysis_json.get('result_count', 0)
        
        freshness_stats = search_metadata['freshness_stats']
        quality_stats = search_metadata['quality_stats']
        
        summary_report = f"""
**Social-Sentimentalist Agent Summary: ${ticker}**

📊 **Quick Overview**
* **Sentiment:** {sentiment} ({score:+.2f})
* **Data Quality:** {quality.capitalize()}
* **Sources Analyzed:** {source_count} (filtered)

🔥 **Data Freshness:**
* Very Fresh (< 24h): {freshness_stats.get('very_fresh', 0)}
* Fresh (< 7 days): {freshness_stats.get('fresh', 0)}
* Avg Freshness Score: {freshness_stats.get('avg_freshness_score', 0):.2f}/1.0

⭐ **Data Quality:**
* High Quality: {quality_stats.get('high', 0)}
* Medium Quality: {quality_stats.get('medium', 0)}
* Avg Quality Score: {quality_stats.get('avg_quality_score', 0):.2f}/1.0

📝 **Executive Summary:**
{exec_summary}

💡 **Key Insight:**
{analysis_json.get('consensus_view', 'See detailed report for full analysis.')}

✨ **Enhanced with Advanced Filtering**
This analysis uses freshness filtering, deduplication, and quality scoring
to ensure only recent, high-quality social sentiment data is analyzed.

---
*Full detailed report with filtering metrics saved to file*
*Agent: Social-Sentimentalist (Enhanced) | Model: Gemini Pro*
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
    
    if len(sys.argv) > 1:
        ticker = sys.argv[1].upper()
    else:
        ticker = "COIN"
        print(f"No ticker provided, using default: {ticker}")
        print("Usage: python social_agent.py <TICKER>\n")
    
    summary, detailed = run_social_agent(ticker, save_to_file=True)
    
    print("\n" + "="*60)
    print("SUMMARY OUTPUT (Console)")
    print("="*60 + "\n")
    print(summary)
    print("\n" + "="*60)
    print("\n📄 Full detailed report saved to 'reports/' directory")