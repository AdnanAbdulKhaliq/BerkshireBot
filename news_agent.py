"""
BerkshireBot News Sentiment Analysis Agent
Using LangChain + Gemini + NewsAPI
"""

from newsapi import NewsApiClient
import os
from dotenv import load_dotenv
from typing import Literal, List
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ============================================================================
# CONFIGURATION: Financial Sources and Weights
# ============================================================================

FINANCIAL_SOURCES = "bloomberg,reuters,the-wall-street-journal,business-insider,fortune,associated-press,axios,financial-post"

# Source weights: Tier 1 sources (market-movers) have higher influence
SOURCE_WEIGHTS = {
    # Tier 1 (Market-Movers) - These sources move markets
    "reuters": 1.5,
    "bloomberg": 1.5,
    "the-wall-street-journal": 1.5,
    "associated-press": 1.5,  # Global newswire, high credibility
    # Tier 2 (High-Quality Financial) - Strong financial journalism
    "business-insider": 1.3,
    "fortune": 1.3,
    "axios": 1.3,  # Known for concise, data-driven business news
    "financial-post": 1.2,  # Canadian financial coverage
    # Default for any other source
    "default": 1.0,
}

# Time decay configuration for recency weighting
# Articles older than this lose all weight (linear decay)
RECENCY_DECAY_DAYS = 7.0  # News older than 7 days has zero weight

# Initialize clients
newsapi = NewsApiClient(api_key=NEWS_API_KEY)
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",  # Using the available model
    google_api_key=GEMINI_API_KEY,
    temperature=0.3,  # Lower temperature for more consistent analysis
)


# ============================================================================
# UTILITY: Rate Limit Checker
# ============================================================================


def check_api_rate_limit() -> dict:
    """
    Check if the NewsAPI rate limit has been exceeded.

    Returns:
        Dictionary with 'available' boolean and 'message' string
    """
    try:
        # Make a minimal test call
        result = newsapi.get_everything(
            q="test",
            page_size=1,
        )
        return {"available": True, "message": "API is available"}
    except Exception as e:
        error_str = str(e)
        if "rateLimited" in error_str or "rate" in error_str.lower():
            return {
                "available": False,
                "message": "⚠️  NewsAPI rate limit exceeded. Free accounts: 100 requests/24h (50 every 12h). Wait or upgrade to paid plan.",
            }
        return {"available": False, "message": f"API error: {error_str}"}


def calculate_recency_weight(article_date_str: str) -> float:
    """
    Calculate time decay weight for an article based on its age.

    Args:
        article_date_str: ISO format datetime string from NewsAPI (e.g., "2025-11-01T14:29:28Z")

    Returns:
        Float between 0.0 and 1.0 (1.0 = today, 0.0 = RECENCY_DECAY_DAYS or older)
    """
    from datetime import datetime, timezone

    try:
        # Parse the article date (NewsAPI uses ISO format with Z for UTC)
        article_date = datetime.fromisoformat(article_date_str.replace("Z", "+00:00"))

        # Get current time in UTC
        now = datetime.now(timezone.utc)

        # Calculate age in days
        age_in_days = (now - article_date).total_seconds() / (24 * 60 * 60)

        # Linear decay: weight = 1.0 at day 0, weight = 0.0 at RECENCY_DECAY_DAYS
        recency_weight = max(
            0.0, (RECENCY_DECAY_DAYS - age_in_days) / RECENCY_DECAY_DAYS
        )

        return recency_weight
    except Exception as e:
        # If we can't parse the date, return neutral weight
        return 1.0


# ============================================================================
# STEP 1: Data Models (Pydantic)
# ============================================================================


class ArticleSentiment(BaseModel):
    """Structured output for individual article sentiment analysis"""

    sentiment: Literal["Bullish", "Bearish", "Neutral"] = Field(
        description="The financial sentiment of the article (Bullish, Bearish, or Neutral)."
    )
    reason: str = Field(
        description="A brief, one-sentence explanation for why this sentiment was assigned."
    )
    stock_impact: int = Field(
        ge=0,
        le=10,
        description="The likely impact of this news on the stock price, rated from 0 (no impact) to 10 (maximum impact).",
    )
    article_title: str = Field(description="The title of the analyzed article.")
    url: str = Field(default="", description="The URL of the article (optional field).")
    source_name: str = Field(default="", description="The name of the news source.")
    source_id: str = Field(default="", description="The source ID from NewsAPI.")
    published_at: str = Field(default="", description="Publication date from NewsAPI.")
    recency_weight: float = Field(
        default=1.0, description="Time decay weight (1.0=today, 0.0=old news)."
    )
    weighted_impact: float = Field(
        default=0.0, description="Impact score weighted by source tier and recency."
    )


# ============================================================================
# STEP 2: News Fetching Tool
# ============================================================================


def get_latest_company_news(
    company_name: str,
    lookback_days: int = 7,
    max_articles: int = 50,
    use_financial_sources: bool = True,
) -> dict:
    """
    Fetches the latest news articles for a given company
    from the last 'lookback_days'.

    Args:
        company_name: The company to search for
        lookback_days: Number of days to look back (default: 7)
        max_articles: Maximum number of articles to fetch (default: 50)
        use_financial_sources: If True, only search within high-quality financial sources (default: True)

    Returns:
        Dictionary with 'total_results' and 'articles' list
    """
    try:
        # Calculate the date range for the lookback period
        from datetime import datetime, timedelta

        to_date = datetime.now()
        from_date = to_date - timedelta(days=lookback_days)

        # Format dates as required by NewsAPI (YYYY-MM-DD)
        from_param = from_date.strftime("%Y-%m-%d")
        to_param = to_date.strftime("%Y-%m-%d")

        if use_financial_sources:
            # Fetch from each source separately to ensure diversity
            # Then combine and deduplicate
            all_articles = []
            seen_urls = set()
            sources_list = FINANCIAL_SOURCES.split(",")

            # Calculate articles per source to evenly distribute max_articles
            # Use ceiling division to ensure we get enough articles
            articles_per_source = (max_articles + len(sources_list) - 1) // len(
                sources_list
            )

            for source in sources_list:
                try:
                    result = newsapi.get_everything(
                        q=company_name,
                        sources=source,
                        language="en",
                        sort_by="relevancy",
                        page_size=articles_per_source,
                        from_param=from_param,
                        to=to_param,
                    )

                    for article in result.get("articles", []):
                        url = article.get("url", "")
                        # Deduplicate by URL
                        if url and url not in seen_urls:
                            seen_urls.add(url)
                            all_articles.append(article)

                except Exception as e:
                    # Check for rate limit error
                    error_str = str(e)
                    if "rateLimited" in error_str or "rate" in error_str.lower():
                        return {
                            "error": "NewsAPI rate limit exceeded. Free accounts are limited to 100 requests per 24 hours (50 every 12 hours). Please wait for your quota to reset or upgrade to a paid plan.",
                            "articles": [],
                        }
                    # Skip other errors from individual sources
                    continue

            # Limit to exactly max_articles
            all_articles = all_articles[:max_articles]

            return {
                "total_results": len(all_articles),
                "articles": all_articles,
            }
        else:
            # Original behavior: no source filtering
            query_params = {
                "q": company_name,
                "language": "en",
                "sort_by": "relevancy",
                "page_size": max_articles,
                "from_param": from_param,
                "to": to_param,
            }

            all_articles = newsapi.get_everything(**query_params)

            return {
                "total_results": all_articles["totalResults"],
                "articles": all_articles["articles"],
            }
    except Exception as e:
        return {"error": str(e), "articles": []}


# ============================================================================
# STEP 3: Sentiment Analysis Chain (Map Step)
# ============================================================================

# Define the system prompt for individual article analysis
system_prompt = """
You are an expert financial analyst. Your task is to analyze a news article 
about a specific company and determine its sentiment from an investor's perspective.

Company in focus: {company_name}

Analyze the article's title and description, then provide your analysis in the 
requested JSON format.

- **Bullish**: The news is good for the company's stock price (e.g., strong earnings, 
  new partnerships, product launches, positive guidance, market share gains).
- **Bearish**: The news is bad for the company's stock price (e.g., earnings miss, 
  layoffs, lawsuits, regulatory issues, executive departures, declining sales).
- **Neutral**: The news is informational but has no clear impact on the stock price 
  (e.g., CEO speaking at a conference, general market trends, routine announcements).

For the stock_impact rating, use a scale from 0 to 10:
- **0-2**: Minimal impact (minor news, unlikely to affect stock price)
- **3-4**: Low impact (some relevance but limited market reaction expected)
- **5-6**: Moderate impact (notable news that could influence trading)
- **7-8**: High impact (significant news likely to move the stock)
- **9-10**: Major impact (game-changing news with substantial market implications)

Consider both the explicit content and the implied market impact.
"""

article_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        (
            "human",
            "Please analyze this article:\n\nTitle: {article_title}\nDescription: {article_description}\nURL: {url}",
        ),
    ]
)

# Create the analysis chain
analysis_chain = article_prompt | llm.with_structured_output(ArticleSentiment)


# ============================================================================
# STEP 4: Summary Chain (Reduce Step)
# ============================================================================

summary_prompt_template = """
You are a senior investment analyst. I have analyzed {num_articles} news articles 
about {company_name} and classified their sentiment.

Here is the sentiment breakdown:
- Bullish: {bullish_count} articles
- Bearish: {bearish_count} articles
- Neutral: {neutral_count} articles

Bullish Articles:
{bullish_articles}

Bearish Articles:
{bearish_articles}

Based on this data, provide a comprehensive investment dashboard summary.

Please provide your analysis in the following format:

**Overall Sentiment**: [Bullish/Bearish/Neutral/Mixed]

**Sentiment Score**: [Calculate as: (Bullish - Bearish) / Total, range -1 to +1]

**Key Bullish Drivers** (1-3 main themes from Bullish articles):
- [Theme 1]
- [Theme 2]
- [Theme 3]

**Key Bearish Drivers** (1-3 main themes from Bearish articles):
- [Theme 1]
- [Theme 2]
- [Theme 3]

**High Impact News** (Top 2-3 most important articles):
- [Article 1]
- [Article 2]
- [Article 3]

**Investment Recommendation**: [Brief 2-3 sentence summary of what this news means for investors]

Keep your analysis concise, actionable, and focused on investment implications.
"""

summary_prompt = ChatPromptTemplate.from_template(summary_prompt_template)
summary_chain = summary_prompt | llm


# ============================================================================
# STEP 5: Main Agent Function
# ============================================================================


def analyze_company_sentiment(
    ticker: str,
    max_articles: int = 30,
    lookback_days: int = 7,
    verbose: bool = True,
) -> dict:
    """
    Main function to fetch news and analyze sentiment for a company.

    Args:
        company_name: The company to analyze
        max_articles: Maximum number of articles to analyze (default: 30)
        lookback_days: Number of days to look back for news (default: 7)
        verbose: Whether to print progress updates

    Returns:
        Dictionary containing sentiment analysis results and summary
    """

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

    company_name = TICKER_TO_NAME.get(ticker.upper(), ticker)

    if verbose:
        print(f"\n{'='*70}")
        print(f"BerkshireBot News Sentiment Analysis")
        print(f"{'='*70}")
        print(f"Company: {company_name}")
        print(f"Max Articles: {max_articles}")
        print(f"Lookback Period: {lookback_days} days")
        print(f"{'='*70}\n")

    # STEP 1: Fetch news articles
    if verbose:
        print("📰 Fetching latest news articles...")

    news_data = get_latest_company_news(
        company_name, lookback_days=lookback_days, max_articles=max_articles
    )

    if "error" in news_data:
        return {"error": news_data["error"]}

    articles = news_data["articles"]

    if not articles:
        return {"error": "No articles found for this company"}

    if verbose:
        print(f"✓ Found {len(articles)} articles\n")

    # STEP 2: Analyze each article (Map step)
    if verbose:
        print("🤖 Analyzing sentiment for each article...")

    analyzed_articles = []

    for i, article in enumerate(articles, 1):
        try:
            if verbose:
                print(
                    f"  [{i}/{len(articles)}] Analyzing: {article.get('title', 'No title')[:60]}..."
                )

            # Extract source information
            source = article.get("source", {})
            source_id = source.get("id", "unknown")
            source_name = source.get("name", "Unknown Source")
            published_at = article.get("publishedAt", "")

            # Prepare the article data
            article_data = {
                "company_name": company_name,
                "article_title": article.get("title", "No title"),
                "article_description": article.get("description", "No description"),
                "url": article.get("url", ""),
            }

            # Run the analysis chain
            sentiment_result = analysis_chain.invoke(article_data)

            # Add source and date information
            sentiment_result.source_name = source_name
            sentiment_result.source_id = source_id
            sentiment_result.published_at = published_at

            # Calculate recency weight (time decay)
            recency_weight = calculate_recency_weight(published_at)
            sentiment_result.recency_weight = recency_weight

            # Calculate weighted impact using source tier AND recency
            source_weight = SOURCE_WEIGHTS.get(source_id, SOURCE_WEIGHTS["default"])
            sentiment_result.weighted_impact = (
                sentiment_result.stock_impact * source_weight * recency_weight
            )

            if verbose:
                age_indicator = (
                    "🔥"
                    if recency_weight > 0.8
                    else "📅" if recency_weight > 0.4 else "🕐"
                )
                print(
                    f"      {age_indicator} Source: {source_name} (SW: {source_weight}x, RW: {recency_weight:.2f}) | Impact: {sentiment_result.stock_impact} → Weighted: {sentiment_result.weighted_impact:.1f}"
                )

            analyzed_articles.append(sentiment_result)

        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error analyzing article {i}: {str(e)}")
            continue

    if verbose:
        print(f"\n✓ Analyzed {len(analyzed_articles)} articles successfully\n")

    # STEP 3: Categorize results
    bullish = [a for a in analyzed_articles if a.sentiment == "Bullish"]
    bearish = [a for a in analyzed_articles if a.sentiment == "Bearish"]
    neutral = [a for a in analyzed_articles if a.sentiment == "Neutral"]

    # Calculate source distribution
    source_distribution = {}
    for article in analyzed_articles:
        source_name = article.source_name
        if source_name not in source_distribution:
            source_distribution[source_name] = 0
        source_distribution[source_name] += 1

    # ========================================================================
    # ENHANCED SENTIMENT SCORING: Bull vs. Bear Pressure Analysis
    # ========================================================================

    # Calculate total possible impact (denominator for normalization)
    max_possible_impact = sum(a.weighted_impact for a in analyzed_articles)

    # Calculate separate bullish and bearish pressure components
    total_bullish_impact = sum(
        a.weighted_impact for a in analyzed_articles if a.sentiment == "Bullish"
    )

    total_bearish_impact = sum(
        a.weighted_impact for a in analyzed_articles if a.sentiment == "Bearish"
    )

    # Normalize each component to get pressure scores (0.0 to 1.0 scale)
    bullish_pressure_score = (
        total_bullish_impact / max_possible_impact if max_possible_impact > 0 else 0
    )

    bearish_pressure_score = (
        total_bearish_impact / max_possible_impact if max_possible_impact > 0 else 0
    )

    # Net sentiment score is the difference between bull and bear pressure
    normalized_sentiment_score = bullish_pressure_score - bearish_pressure_score

    # Calculate high-impact article count (impact >= 7)
    high_impact_articles = [a for a in analyzed_articles if a.stock_impact >= 7]

    if verbose:
        print(f"� SENTIMENT ANALYSIS RESULTS")
        print(f"{'='*70}")
        print(
            f"📈 Overall Sentiment Score: {normalized_sentiment_score:.2f} (range: -1 to +1)"
        )
        print(f"\n💪 Bull vs. Bear Pressure:")
        print(f"   🐂 Bullish Pressure: {bullish_pressure_score*100:.1f}%")
        print(f"   🐻 Bearish Pressure: {bearish_pressure_score*100:.1f}%")
        print(f"\n📰 News Volume:")
        print(f"   Total Articles: {len(analyzed_articles)}")
        print(f"   High-Impact (7+): {len(high_impact_articles)} articles")
        print(f"   (Positive = Bullish, Negative = Bearish, Zero = Neutral)\n")

    # STEP 4: Generate summary (Reduce step)
    if verbose:
        print("📊 Generating investment dashboard summary...\n")

    # Format bullish articles for summary (sorted by weighted impact)
    bullish_sorted = sorted(bullish, key=lambda x: x.weighted_impact, reverse=True)
    bullish_text = (
        "\n".join(
            [
                f"- {a.article_title} (Impact: {a.stock_impact}, Weighted: {a.weighted_impact:.1f}, Source: {a.source_name}) - {a.reason}"
                for a in bullish_sorted[:10]  # Limit to top 10
            ]
        )
        if bullish
        else "None"
    )

    # Format bearish articles for summary (sorted by weighted impact)
    bearish_sorted = sorted(bearish, key=lambda x: x.weighted_impact, reverse=True)
    bearish_text = (
        "\n".join(
            [
                f"- {a.article_title} (Impact: {a.stock_impact}, Weighted: {a.weighted_impact:.1f}, Source: {a.source_name}) - {a.reason}"
                for a in bearish_sorted[:10]  # Limit to top 10
            ]
        )
        if bearish
        else "None"
    )

    # Create summary
    summary_data = {
        "company_name": company_name,
        "num_articles": len(analyzed_articles),
        "bullish_count": len(bullish),
        "bearish_count": len(bearish),
        "neutral_count": len(neutral),
        "bullish_articles": bullish_text,
        "bearish_articles": bearish_text,
    }

    summary_response = summary_chain.invoke(summary_data)

    # STEP 5: Return comprehensive results
    return {
        "company": company_name,
        "total_articles_analyzed": len(analyzed_articles),
        "sentiment_breakdown": {
            "bullish": len(bullish),
            "bearish": len(bearish),
            "neutral": len(neutral),
        },
        "source_distribution": source_distribution,
        "weighted_sentiment_score": normalized_sentiment_score,
        "bullish_pressure": bullish_pressure_score,
        "bearish_pressure": bearish_pressure_score,
        "high_impact_count": len(high_impact_articles),
        "bullish_articles": [
            {
                "title": a.article_title,
                "sentiment": a.sentiment,
                "impact": a.stock_impact,
                "weighted_impact": a.weighted_impact,
                "recency_weight": a.recency_weight,
                "published_at": a.published_at,
                "reason": a.reason,
                "url": a.url,
                "source": a.source_name,
                "source_id": a.source_id,
            }
            for a in bullish_sorted  # Use sorted version
        ],
        "bearish_articles": [
            {
                "title": a.article_title,
                "sentiment": a.sentiment,
                "impact": a.stock_impact,
                "weighted_impact": a.weighted_impact,
                "recency_weight": a.recency_weight,
                "published_at": a.published_at,
                "reason": a.reason,
                "url": a.url,
                "source": a.source_name,
                "source_id": a.source_id,
            }
            for a in bearish_sorted  # Use sorted version
        ],
        "neutral_articles": [
            {
                "title": a.article_title,
                "sentiment": a.sentiment,
                "impact": a.stock_impact,
                "weighted_impact": a.weighted_impact,
                "recency_weight": a.recency_weight,
                "published_at": a.published_at,
                "reason": a.reason,
                "url": a.url,
                "source": a.source_name,
                "source_id": a.source_id,
            }
            for a in neutral
        ],
        "high_impact_articles": [
            {
                "title": a.article_title,
                "sentiment": a.sentiment,
                "impact": a.stock_impact,
                "weighted_impact": a.weighted_impact,
                "recency_weight": a.recency_weight,
                "published_at": a.published_at,
                "reason": a.reason,
                "url": a.url,
                "source": a.source_name,
                "source_id": a.source_id,
            }
            for a in sorted(
                high_impact_articles, key=lambda x: x.weighted_impact, reverse=True
            )
        ],
        "dashboard_summary": summary_response.content,
    }


# ============================================================================
# STEP 5.5: Wrapper Function for Orchestrator
# ============================================================================


def run_news_agent(ticker: str, max_articles: int = 30, lookback_days: int = 7) -> dict:
    """
    Wrapper function to match the interface of other agents.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        max_articles: Maximum number of articles to analyze (default: 30)
        lookback_days: Number of days to look back for news (default: 7)

    Returns:
        Dictionary containing:
        - ticker: Stock ticker analyzed
        - agent: Agent name
        - summary_report: Brief summary of sentiment analysis
        - detailed_report: Full analysis with all articles
        - All fields from analyze_company_sentiment()

    Example:
        >>> result = run_news_agent("AAPL", max_articles=20)
        >>> print(result["summary_report"])
        >>> print(result["weighted_sentiment_score"])
    """
    try:
        print(f"\n{'='*60}")
        print(f"🤖 News Agent (Enhanced): Analyzing ${ticker}")
        print(f"{'='*60}\n")

        # Run the sentiment analysis
        results = analyze_company_sentiment(
            company_name=ticker, max_articles=max_articles, lookback_days=lookback_days
        )

        # Create summary report
        sentiment_score = results.get("weighted_sentiment_score", 0)
        bullish_count = results.get("sentiment_breakdown", {}).get("bullish", 0)
        bearish_count = results.get("sentiment_breakdown", {}).get("bearish", 0)
        neutral_count = results.get("sentiment_breakdown", {}).get("neutral", 0)
        total_articles = results.get("total_articles_analyzed", 0)

        summary_report = f"""**News Agent Summary: ${ticker}**

📊 **Sentiment Analysis:**
* Weighted Sentiment Score: {sentiment_score:.2f}
* Articles Analyzed: {total_articles}
* Bullish: {bullish_count} | Bearish: {bearish_count} | Neutral: {neutral_count}

💡 **Key Finding:**
{results.get("dashboard_summary", "No summary available")[:300]}...

---
*Agent: News Agent (Enhanced) | Model: Gemini Pro*
"""

        detailed_report = results.get(
            "dashboard_summary", "No detailed analysis available"
        )

        # Return comprehensive dictionary with all original data
        return {
            "ticker": ticker,
            "agent": "News",
            "summary_report": summary_report.strip(),
            "detailed_report": detailed_report,
            **results,  # Include all original fields from analyze_company_sentiment
        }

    except Exception as e:
        print(f"❌ News_Agent: Analysis failed - {e}")
        import traceback

        traceback.print_exc()

        error_msg = str(e)
        return {
            "ticker": ticker,
            "agent": "News",
            "error": error_msg,
            "summary_report": f"**Error analyzing ${ticker}**: {error_msg}",
            "detailed_report": f"**Error analyzing ${ticker}**: {error_msg}\n\n{traceback.format_exc()}",
        }


# ============================================================================
# STEP 6: Excel Export Function
# ============================================================================


def export_to_excel(results: dict, filename: str = None) -> str:
    """
    Export sentiment analysis results to an Excel file.

    Args:
        results: The results dictionary from analyze_company_sentiment
        filename: Optional custom filename (default: {company}_sentiment_analysis.xlsx)

    Returns:
        The filename of the created Excel file
    """
    try:
        import pandas as pd
    except ImportError:
        raise ImportError(
            "pandas and openpyxl are required for Excel export. "
            "Install with: pip install pandas openpyxl"
        )

    # Prepare data for Excel
    data = []
    serial_no = 1

    # Add all articles (bullish, bearish, neutral) to the list
    all_articles = (
        results.get("bullish_articles", [])
        + results.get("bearish_articles", [])
        + results.get("neutral_articles", [])
    )

    for article in all_articles:
        data.append(
            {
                "Serial No.": serial_no,
                "Article Heading": article.get("title", "N/A"),
                "Source": article.get("source", "N/A"),
                "Published": article.get("published_at", "N/A"),
                "Sentiment": article.get("sentiment", "N/A"),
                "Impact Score (0-10)": article.get("impact", 0),
                "Recency Weight": round(article.get("recency_weight", 1.0), 2),
                "Weighted Impact": round(article.get("weighted_impact", 0), 2),
                "Reason": article.get("reason", "N/A"),
                "URL": article.get("url", "N/A"),
            }
        )
        serial_no += 1

    # Create DataFrame
    df = pd.DataFrame(data)

    # Generate filename if not provided
    if filename is None:
        company_name = results.get("company", "company").replace(" ", "_")
        filename = f"{company_name}_sentiment_analysis.xlsx"

    # Export to Excel
    df.to_excel(filename, index=False, sheet_name="Sentiment Analysis")

    return filename


# ============================================================================
# STEP 7: CLI Interface
# ============================================================================


def main():
    """Command-line interface for the sentiment analysis agent"""

    print("\n" + "=" * 70)
    print("🚀 BerkshireBot - AI-Powered News Sentiment Analysis")
    print("=" * 70)

    # Get user input
    company_name = input("\nEnter company name to analyze: ").strip()

    if not company_name:
        print("❌ Error: Company name cannot be empty")
        return

    try:
        max_articles = int(
            input("Max articles to analyze (default 30): ").strip() or "30"
        )
    except ValueError:
        max_articles = 30

    try:
        lookback_days = int(input("Days to look back (default 7): ").strip() or "7")
    except ValueError:
        lookback_days = 7

    # Run analysis
    try:
        results = analyze_company_sentiment(
            company_name,
            max_articles=max_articles,
            lookback_days=lookback_days,
            verbose=True,
        )

        if "error" in results:
            print(f"\n❌ Error: {results['error']}")
            return

        # Display results
        print("\n" + "=" * 70)
        print("📊 INVESTMENT DASHBOARD")
        print("=" * 70)
        print(f"\nCompany: {results['company']}")
        print(f"Articles Analyzed: {results['total_articles_analyzed']}")

        print(f"\nSentiment Breakdown:")
        print(f"  🟢 Bullish: {results['sentiment_breakdown']['bullish']}")
        print(f"  🔴 Bearish: {results['sentiment_breakdown']['bearish']}")
        print(f"  ⚪ Neutral: {results['sentiment_breakdown']['neutral']}")

        # Display enhanced metrics: Bull vs. Bear Pressure
        print(f"\n💪 Bull vs. Bear Pressure Analysis:")
        bullish_pct = results["bullish_pressure"] * 100
        bearish_pct = results["bearish_pressure"] * 100
        print(f"  🐂 Bullish Pressure: {bullish_pct:.1f}%")
        print(f"  🐻 Bearish Pressure: {bearish_pct:.1f}%")

        # Display news volume metrics
        print(f"\n📰 News Volume:")
        print(f"  Total Articles: {results['total_articles_analyzed']}")
        print(f"  High-Impact (7+): {results['high_impact_count']} articles")

        # Display source distribution
        print(f"\n� Source Distribution:")
        for source, count in sorted(
            results["source_distribution"].items(), key=lambda x: x[1], reverse=True
        ):
            print(f"  • {source}: {count} article(s)")

        # Display weighted sentiment score
        score = results["weighted_sentiment_score"]
        score_emoji = "🟢" if score > 0.2 else "🔴" if score < -0.2 else "⚪"
        score_label = (
            "BULLISH" if score > 0.2 else "BEARISH" if score < -0.2 else "NEUTRAL"
        )
        print(f"\n{score_emoji} Overall Sentiment Score: {score:.2f} ({score_label})")
        print(f"   Range: -1.0 (max bearish) to +1.0 (max bullish)")

        # Display top high-impact articles
        if results["high_impact_articles"]:
            print(f"\n🔥 Top High-Impact News:")
            for i, article in enumerate(results["high_impact_articles"][:3], 1):
                sentiment_emoji = (
                    "🐂"
                    if article["sentiment"] == "Bullish"
                    else "🐻" if article["sentiment"] == "Bearish" else "⚪"
                )
                print(f"  {i}. {sentiment_emoji} {article['title'][:60]}...")
                print(
                    f"     Impact: {article['impact']}/10 | Weighted: {article['weighted_impact']:.1f}"
                )

        print("\n" + "-" * 70)
        print("\n" + results["dashboard_summary"])
        print("\n" + "=" * 70)

        # Ask user for export options
        print("\n📁 Export Options:")
        print("  1. Save as JSON (detailed results)")
        print("  2. Save as Excel (article list with sentiment)")
        print("  3. Save both JSON and Excel")
        print("  4. Don't save")

        choice = input("\nSelect option (1-4): ").strip()

        if choice == "1":
            # Save JSON only
            import json

            filename = f"{company_name.replace(' ', '_')}_sentiment_analysis.json"
            with open(filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ Results saved to {filename}")

        elif choice == "2":
            # Save Excel only
            try:
                filename = export_to_excel(results)
                print(f"✓ Results saved to {filename}")
            except ImportError as e:
                print(f"❌ Error: {e}")
                print("Install required packages: pip install pandas openpyxl")

        elif choice == "3":
            # Save both
            import json

            json_filename = f"{company_name.replace(' ', '_')}_sentiment_analysis.json"
            with open(json_filename, "w") as f:
                json.dump(results, f, indent=2, default=str)
            print(f"✓ JSON saved to {json_filename}")

            try:
                excel_filename = export_to_excel(results)
                print(f"✓ Excel saved to {excel_filename}")
            except ImportError as e:
                print(f"❌ Error exporting Excel: {e}")
                print("Install required packages: pip install pandas openpyxl")

        elif choice == "4":
            print("✓ Results not saved")
        else:
            print("⚠️  Invalid choice. Results not saved")

    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis cancelled by user")
    except Exception as e:
        print(f"\n❌ Error during analysis: {str(e)}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
