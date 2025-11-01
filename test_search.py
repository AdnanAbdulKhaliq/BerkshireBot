import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load the .env file to get the key
load_dotenv()

print("Checking for GEMINI_API_KEY...")
if "GEMINI_API_KEY" not in os.environ:
    print("❌ FAILED: GEMINI_API_KEY not found in .env file.")
    exit()

print("✅ GEMINI_API_KEY found. Configuring API...")

try:
    # Configure the client with the API key
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])

    print("\nFetching available models...\n")
    
    # This is how you call "ListModels"
    for model in genai.list_models():
        # We only care about models that can 'generateContent'
        if 'generateContent' in model.supported_generation_methods:
            print(f"✅ Model Name: {model.name}")

    print("\n--- End of List ---")
    print("If you see 'models/gemini-1.5-pro-latest' in this list,")
    print("the fix to 'social_agent.py' (adding api_version='v1') will work.")


except Exception as e:
    print(f"\n❌ FAILED to list models. Error: {e}")