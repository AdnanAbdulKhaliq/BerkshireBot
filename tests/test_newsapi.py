#!/usr/bin/env python3
"""Quick test script to verify NewsAPI is working"""

from newsapi import NewsApiClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
NEWS_API_KEY = os.getenv("NEWS_API_KEY")


def test_newsapi():
    """Test NewsAPI connection and fetch top headlines"""

    # Check if API key is loaded
    if not NEWS_API_KEY:
        print("❌ ERROR: NEWS_API_KEY not found in .env file")
        return False

    print(f"✓ API Key loaded: {NEWS_API_KEY[:8]}...")

    # Initialize NewsAPI client
    try:
        newsapi = NewsApiClient(api_key=NEWS_API_KEY)
        print("✓ NewsAPI client initialized")
    except Exception as e:
        print(f"❌ ERROR initializing NewsAPI client: {e}")
        return False

    # Fetch top headlines
    try:
        print("\nFetching top business headlines...")
        response = newsapi.get_top_headlines(
            category="business", language="en", page_size=5
        )

        if response["status"] == "ok":
            print(f"✓ Successfully fetched {response['totalResults']} articles")
            print(f"\nShowing first {len(response['articles'])} headlines:\n")

            for i, article in enumerate(response["articles"], 1):
                print(f"{i}. {article['title']}")
                print(f"   Source: {article['source']['name']}")
                print(f"   Published: {article['publishedAt']}")
                print()

            return True
        else:
            print(f"❌ ERROR: API returned status '{response['status']}'")
            return False

    except Exception as e:
        print(f"❌ ERROR fetching headlines: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("NewsAPI Test")
    print("=" * 60)

    success = test_newsapi()

    print("=" * 60)
    if success:
        print("✅ All tests passed! NewsAPI is working correctly.")
    else:
        print("❌ Tests failed. Please check the errors above.")
    print("=" * 60)
