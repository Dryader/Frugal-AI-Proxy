import os
import time
import json
import uuid
import hashlib
from datetime import datetime
from typing import Optional, List, Dict

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlmodel import SQLModel, create_engine, Session, select
import duckdb
from openai import OpenAI
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from models import LLMLog

load_dotenv()

# --- Configuration & Setup ---
DB_FILE = "frugal_proxy.db"
VECTOR_DB_FILE = "vector_cache.duckdb"
PRICING_FILE = "pricing.json"
PERPLEXITY_API_KEY = os.getenv("PERPLEXITY_API_KEY", "your_key_here")

# Load pricing
with open(PRICING_FILE, "r") as f:
    PRICING = json.load(f)

# Initialize SQLite
sqlite_url = f"sqlite:///{DB_FILE}"
engine = create_engine(sqlite_url)
SQLModel.metadata.create_all(engine)

# Initialize DuckDB for Vector Cache
vdb = duckdb.connect(VECTOR_DB_FILE)
vdb.execute("""
    CREATE TABLE IF NOT EXISTS semantic_cache (
        id VARCHAR PRIMARY KEY,
        prompt TEXT,
        response TEXT,
        model VARCHAR,
        embedding FLOAT[]
    )
""")

# Initialize Local Embedding Model
model_name = "all-MiniLM-L6-v2"
encoder = SentenceTransformer(model_name)

# Initialize Perplexity (OpenAI compatible)
client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

app = FastAPI(title="Frugal AI Proxy")

# --- Helper Functions ---

def calculate_cost(model: str, input_tokens: int, output_tokens: int, search_performed: bool = True) -> float:
    if model not in PRICING["models"]:
        return 0.0
    config = PRICING["models"][model]
    cost = (input_tokens / 1_000_000 * config["input_1m"]) + (output_tokens / 1_000_000 * config["output_1m"])
    if search_performed:
        cost += (config["search_fee_1k"] / 1000)
    return cost

def classify_intent(prompt: str) -> str:
    prompt_lower = prompt.lower()
    if any(word in prompt_lower for word in ["why", "how", "solve", "debug", "reason"]):
        return "sonar-reasoning-pro"
    if any(word in prompt_lower for word in ["compare", "research", "summarize", "trends"]):
        return "sonar-pro"
    return "sonar"

# --- API Endpoints ---

class ChatRequest(BaseModel):
    message: str
    stream: bool = False

@app.post("/chat")
async def chat(request: ChatRequest, background_tasks: BackgroundTasks):
    start_time = time.time()
    prompt = request.message
    
    # 1. Semantic Cache Check (using DuckDB)
    query_embedding = encoder.encode(prompt).tolist()
    
    # Calculate cosine similarity in DuckDB
    # Cosine Similarity = (A . B) / (||A|| * ||B||)
    # Since all-MiniLM-L6-v2 embeddings are normalized, we can just use dot product
    search_query = """
        SELECT response, model, 
               list_dot_product(embedding, ?::FLOAT[]) as similarity
        FROM semantic_cache
        ORDER BY similarity DESC
        LIMIT 1
    """
    results = vdb.execute(search_query, [query_embedding]).fetchone()
    
    # Distance = 1 - Similarity. We want distance < 0.2, so similarity > 0.8
    if results and results[2] > 0.8:
        # Cache Hit
        latency = int((time.time() - start_time) * 1000)
        response_text, model_used, similarity = results
        
        # Calculate what we saved (compared to premium baseline)
        premium_model = PRICING["premium_baseline"]
        est_input = len(prompt) // 4
        est_output = len(response_text) // 4
        savings = calculate_cost(premium_model, est_input, est_output)
        
        log_entry = LLMLog(
            prompt=prompt,
            response=response_text,
            model_used=f"Cache ({model_used})",
            cache_hit=True,
            cost_usd=0.0,
            savings_usd=savings,
            latency_ms=latency,
            routing_justification=f"Semantic match found (similarity: {similarity:.2f})"
        )
        
        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
            
        return {
            "response": response_text,
            "source": "cache",
            "latency_ms": latency,
            "savings_usd": savings
        }

    # 2. Cache Miss - Routing Logic
    target_model = classify_intent(prompt)
    justification = f"Intent classified as {target_model} based on query keywords."
    
    # 3. Call Perplexity
    try:
        response = client.chat.completions.create(
            model=target_model,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.choices[0].message.content or ""
        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        
        actual_cost = calculate_cost(target_model, input_tokens, output_tokens)
        premium_cost = calculate_cost(PRICING["premium_baseline"], input_tokens, output_tokens)
        savings = max(0, premium_cost - actual_cost)
        
        latency = int((time.time() - start_time) * 1000)
        
        # 4. Store in Cache (DuckDB)
        vdb.execute(
            "INSERT INTO semantic_cache VALUES (?, ?, ?, ?, ?)",
            [str(uuid.uuid4()), prompt, response_text, target_model, query_embedding]
        )
        
        # 5. Log to SQLite
        log_entry = LLMLog(
            prompt=prompt,
            response=response_text,
            model_used=target_model,
            cache_hit=False,
            tokens_input=input_tokens,
            tokens_output=output_tokens,
            cost_usd=actual_cost,
            savings_usd=savings,
            latency_ms=latency,
            routing_justification=justification
        )
        
        # 6. Optional Shadow Call
        if target_model != PRICING["premium_baseline"]:
            try:
                shadow_res = client.chat.completions.create(
                    model=PRICING["premium_baseline"],
                    messages=[{"role": "user", "content": prompt}]
                )
                log_entry.shadow_response = shadow_res.choices[0].message.content
                log_entry.shadow_model = PRICING["premium_baseline"]
            except Exception as e:
                print(f"Shadow call failed: {e}")

        with Session(engine) as session:
            session.add(log_entry)
            session.commit()
            
        return {
            "response": response_text,
            "source": "api",
            "model": target_model,
            "latency_ms": latency,
            "cost_usd": actual_cost,
            "savings_usd": savings
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
