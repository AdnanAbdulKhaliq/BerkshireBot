#!/usr/bin/env python3
"""Quick test script to verify Gemini API with LangChain is working"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def test_gemini_langchain():
    """Test Gemini API connection via LangChain"""

    # Check if API key is loaded
    if not GEMINI_API_KEY:
        print("❌ ERROR: GEMINI_API_KEY not found in .env file")
        return False

    print(f"✓ API Key loaded: {GEMINI_API_KEY[:10]}...")

    # Try importing LangChain Gemini
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI

        print("✓ LangChain Google GenAI module imported")
    except ImportError as e:
        print(f"❌ ERROR: Could not import LangChain Google GenAI: {e}")
        print("\nTry installing: pip install langchain-google-genai")
        return False

    # Initialize Gemini model
    try:
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY, temperature=0.7
        )
        print("✓ Gemini model initialized (gemini-2.0-flash)")
    except Exception as e:
        print(f"❌ ERROR initializing Gemini model: {e}")
        return False

    # Test a simple prompt
    try:
        print("\nSending test prompt to Gemini...")
        test_prompt = "What is 2+2? Answer in one short sentence."

        response = llm.invoke(test_prompt)

        print("✓ Successfully received response from Gemini")
        print(f"\nPrompt: {test_prompt}")
        print(f"Response: {response.content}")

        return True

    except Exception as e:
        print(f"❌ ERROR calling Gemini API: {e}")
        return False


def test_gemini_with_messages():
    """Test Gemini with message history"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.messages import HumanMessage, SystemMessage

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", google_api_key=GEMINI_API_KEY
        )

        print("\nTesting with message history...")
        messages = [
            SystemMessage(
                content="You are a helpful assistant that provides brief answers."
            ),
            HumanMessage(content="Name one famous AI company."),
        ]

        response = llm.invoke(messages)
        print("✓ Message history test passed")
        print(f"Response: {response.content}")

        return True

    except Exception as e:
        print(f"❌ ERROR in message history test: {e}")
        return False


if __name__ == "__main__":
    print("=" * 60)
    print("Gemini API + LangChain Test")
    print("=" * 60)

    test1 = test_gemini_langchain()

    if test1:
        test2 = test_gemini_with_messages()
    else:
        test2 = False

    print("\n" + "=" * 60)
    if test1 and test2:
        print("✅ All tests passed! Gemini API with LangChain is working.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)
