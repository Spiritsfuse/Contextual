import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def list_embedding_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY not found")
        return
    
    client = genai.Client(api_key=api_key)
    print("Available embedding models:")
    try:
        # The new SDK might not have list_models on the client directly in the same way
        # But we can try to iterate or check
        for model in client.models.list():
            if "embed" in model.name.lower() or "embedding" in model.name.lower():
                print(f"- {model.name} (Supported: {model.supported_actions})")
    except Exception as e:
        print(f"Error listing models: {e}")

if __name__ == "__main__":
    list_embedding_models()
