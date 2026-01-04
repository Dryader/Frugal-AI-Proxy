import requests
import time

BASE_URL = "http://localhost:8000"

queries = [
    "What is the capital of France?",  # Should route to 'sonar'
    "How does a quantum computer work?", # Should route to 'sonar-reasoning-pro'
    "Compare the iPhone 15 and Samsung S24.", # Should route to 'sonar-pro'
    "What is the capital of France?",  # Should be a CACHE HIT
]

def test_proxy():
    print(f"🚀 Starting Frugal AI Proxy Test...")
    for query in queries:
        print(f"\nPrompt: {query}")
        try:
            start = time.time()
            response = requests.post(f"{BASE_URL}/chat", json={"message": query})
            duration = time.time() - start
            
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success ({duration:.2f}s)")
                print(f"   Source: {data.get('source')}")
                print(f"   Model: {data.get('model', 'N/A')}")
                print(f"   Savings: ${data.get('savings_usd', 0):.4f}")
            else:
                print(f"❌ Error: {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Connection failed: {e}. Is the server running on port 8000?")
        
        time.sleep(1) # Small delay between requests

if __name__ == "__main__":
    test_proxy()
