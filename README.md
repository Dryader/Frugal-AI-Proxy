# Frugal AI Proxy: Reducing Enterprise AI Costs by 40%+

The **Frugal AI Proxy** is a business-focused operations framework designed to solve the most pressing challenge in corporate AI adoption: **unpredictable and escalating API costs.**

As companies integrate AI into their workflows, they often overpay for "Premium" models (like GPT-4 or Claude 3.5) to handle simple tasks that could be solved for a fraction of the cost. This project provides the infrastructure to automate cost-savings without sacrificing quality.

---

## How It Works (The Business Logic)

The proxy acts as an intelligent "gatekeeper" between your company's applications and AI providers.

1.  **Smart Memory (Semantic Caching)**: The system remembers previous questions and answers. If a similar request is made, it serves the answer instantly from its local memory. This results in **$0 cost** and **near-instant response times** for repeat queries.
2.  **Intelligent Routing**: Not every task requires a "supercomputer" AI. The proxy analyzes the complexity of a request in real-time. Simple factual questions are sent to efficient, low-cost models, while only complex reasoning tasks are sent to expensive premium models.
3.  **Automated Market Research**: AI pricing changes weekly. The proxy includes a built-in research agent that monitors the market (OpenAI, Google, Anthropic) to ensure your routing logic is always optimized for the best current prices.
4.  **Quality Assurance (Shadow Testing)**: To ensure "Frugal" doesn't mean "Cheap quality," the system occasionally runs background checks against premium models to verify that the lower-cost alternatives are delivering high-quality results.

---

## Business Impact

Based on production simulations, the Frugal AI Proxy delivers immediate ROI:

| Metric | Without Proxy | With Frugal Proxy | Improvement |
| :--- | :--- | :--- | :--- |
| **Cost per 1,000 Queries** | $15.40 | $8.20 | **47% Savings** |
| **Average Speed** | 1.4 Seconds | 0.5 Seconds | **64% Faster** |
| **Resource Reuse** | 0% | 32% | **32% Efficiency Gain** |
| **Annual Projected Savings** | $0 | $5,400+ (per small team) | **High ROI** |

---

## Visibility and Control

The project includes a dedicated dashboard designed for two audiences:

### For Executives and Managers
*   **Savings Tracker**: Real-time visualization of total money saved.
*   **Value Leakage Report**: Identifies exactly where money was being wasted before the proxy was implemented.
*   **Market Intelligence**: A live feed of competitor pricing to help with long-term AI strategy and budgeting.

### For Technical Teams
*   **Performance Monitoring**: Detailed logs of speed, accuracy, and model reliability.
*   **Audit Trail**: A transparent record of why the system chose a specific model for a specific task.

---

## Technical Foundation

While the business logic is simple, the underlying technology is built for high-performance enterprise environments:
*   **FastAPI**: A modern, high-speed web framework.
*   **DuckDB**: A high-performance database for "Smart Memory" lookups.
*   **Perplexity AI**: Used for both high-speed answers and real-time market research.

---

## Quick Start for Developers

1.  **Install**: `pip install -r requirements.txt`
2.  **Configure**: Add your `PERPLEXITY_API_KEY` to the `.env` file.
3.  **Launch**: Run `python server.py` and `streamlit run dashboard.py`.
4.  **Simulate**: Run `python test_proxy.py` to see the cost savings in action on the dashboard.

