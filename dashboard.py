import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sqlmodel import create_engine, Session, select
from datetime import datetime
import json
from models import LLMLog

# --- Setup ---
DB_FILE = "frugal_proxy.db"
PRICING_FILE = "pricing.json"
engine = create_engine(f"sqlite:///{DB_FILE}")

st.set_page_config(page_title="Frugal AI Proxy Dashboard", layout="wide")

# Load pricing for reference
with open(PRICING_FILE, "r") as f:
    PRICING = json.load(f)

def get_data():
    with Session(engine) as session:
        statement = select(LLMLog)
        results = session.exec(statement).all()
        return pd.DataFrame([r.dict() for r in results])

# --- Sidebar ---
st.sidebar.title("Frugal AI Proxy")
view_mode = st.sidebar.radio("Select View", ["Executive ROI", "Technical Deep-Dive"])

if st.sidebar.button("Refresh Data"):
    st.rerun()

# --- Main Dashboard ---
df = get_data()

if df.empty:
    st.warning("No data found in the database. Start chatting with the proxy to see metrics!")
else:
    # Pre-processing
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    if view_mode == "Executive ROI":
        st.title("📈 Executive ROI Dashboard")
        st.markdown("### *Focus: Cost Savings & Value Leakage Prevention*")
        
        # Top Metrics
        col1, col2, col3, col4 = st.columns(4)
        total_spend = df['cost_usd'].sum()
        total_savings = df['savings_usd'].sum()
        hit_rate = (df['cache_hit'].mean() * 100)
        
        col1.metric("Total Actual Spend", f"${total_spend:.4f}")
        col2.metric("Total Money Saved", f"${total_savings:.4f}", delta=f"{total_savings/(total_spend+total_savings)*100:.1f}% ROI")
        col3.metric("Cache Hit Rate", f"{hit_rate:.1f}%")
        col4.metric("Queries Handled", len(df))

        # Savings Chart
        st.subheader("Cumulative Savings Over Time")
        df_sorted = df.sort_values('timestamp')
        df_sorted['cum_savings'] = df_sorted['savings_usd'].cumsum()
        fig_savings = px.area(df_sorted, x='timestamp', y='cum_savings', title="Total Dollars Saved (Cumulative)")
        st.plotly_chart(fig_savings, use_container_width=True)

        # Value Leakage Section
        st.divider()
        st.subheader("🛡️ Value Leakage Prevention")
        st.write("How much would have been wasted if we used the most expensive model for every query?")
        
        # Model Distribution
        model_counts = df['model_used'].value_counts().reset_index()
        model_counts.columns = ['Model', 'Count']
        fig_models = px.pie(model_counts, values='Count', names='Model', title="Model Usage Distribution")
        st.plotly_chart(fig_models, use_container_width=True)

        # Shadow Call Comparison
        st.divider()
        st.subheader("⚖️ Quality Check: Shadow Call Comparison")
        st.write("Comparing our 'Frugal' choice vs. the 'Premium' baseline.")
        
        shadow_df = df[df['shadow_response'].notnull()].tail(3)
        if not shadow_df.empty:
            for _, row in shadow_df.iterrows():
                with st.expander(f"Query: {row['prompt'][:50]}..."):
                    c1, c2 = st.columns(2)
                    c1.info(f"**Frugal Choice ({row['model_used']})**\n\n{row['response']}")
                    c2.success(f"**Premium Baseline ({row['shadow_model']})**\n\n{row['shadow_response']}")
                    st.caption(f"Routing Justification: {row['routing_justification']}")
        else:
            st.info("No shadow calls recorded yet.")

        # Modern Model Showcase
        st.divider()
        st.subheader("🚀 AI Market Intelligence")
        st.write("Use Perplexity to research the latest pricing and benchmarks for OpenAI, Anthropic, and Google.")
        
        if st.button("Run Market Research"):
            with st.spinner("Perplexity is researching the latest model data..."):
                try:
                    import requests
                    research_prompt = "Provide a concise comparison table of the latest flagship models from OpenAI (GPT-4), Anthropic (Claude 3), and Google (Gemini 1.5). Include input/output pricing per 1M tokens and a key operational strength for each."
                    res = requests.post("http://localhost:8000/chat", json={"message": research_prompt})
                    if res.status_code == 200:
                        st.success("Research complete! Refreshing dashboard...")
                        st.rerun()
                    else:
                        st.error(f"Research failed: {res.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

        # Display the latest research result if it exists
        research_logs = df[df['prompt'].str.contains("OpenAI", na=False) & df['prompt'].str.contains("Anthropic", na=False)].tail(1)
        if not research_logs.empty:
            st.markdown("#### Latest Market Research Result:")
            st.info(research_logs.iloc[0]['response'])
            st.caption(f"Research conducted via {research_logs.iloc[0]['model_used']} on {research_logs.iloc[0]['timestamp']}")
        else:
            st.info("Click the button above to run your first market research report.")

        st.divider()
        st.subheader("📊 Modern Model Landscape (Marketed)")
        st.write("Current state of the art models and their target use cases.")
        
        showcase_data = [
            {"Model": "Sonar", "Tier": "Efficiency", "Cost": "$", "Best For": "Real-time facts"},
            {"Model": "Sonar Pro", "Tier": "Research", "Cost": "$$$", "Best For": "Deep synthesis"},
            {"Model": "Sonar Reasoning Pro", "Tier": "Logic", "Cost": "$$", "Best For": "Math/Coding"},
            {"Model": "Deep Research", "Tier": "Autonomous", "Cost": "$$$$$", "Best For": "10+ page reports"}
        ]
        st.table(pd.DataFrame(showcase_data))

    else:
        st.title("🛠️ Technical Deep-Dive")
        st.markdown("### *Focus: Latency, Cache Performance, and Routing Logs*")

        # Latency Comparison
        st.subheader("Latency Distribution: Cache vs. API")
        fig_latency = px.histogram(df, x='latency_ms', color='cache_hit', 
                                   barmode='overlay', title="Latency (ms) - Cache Hits vs. API Calls",
                                   labels={'cache_hit': 'Is Cache Hit?'})
        st.plotly_chart(fig_latency, use_container_width=True)

        # Token Usage
        st.subheader("Token Consumption")
        token_df = df[df['cache_hit'] == False]
        if not token_df.empty:
            fig_tokens = px.scatter(token_df, x='tokens_input', y='tokens_output', 
                                    size='cost_usd', color='model_used',
                                    title="Input vs. Output Tokens by Model")
            st.plotly_chart(fig_tokens, use_container_width=True)
        else:
            st.info("No API token data available yet (all hits were cache hits).")

        # Raw Logs
        st.subheader("Detailed Transaction Logs")
        st.dataframe(df[['timestamp', 'prompt', 'model_used', 'cache_hit', 'latency_ms', 'cost_usd', 'routing_justification']].sort_values('timestamp', ascending=False))
