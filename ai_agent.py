"""
AI Advisor Report — Groq LLM · SEBI Compliant
Generates a professional investment report incorporating agent reasoning traces.
"""

import os
import uuid
import random
import time
from datetime import datetime
from groq import Groq
from dotenv import load_dotenv

load_dotenv(override=True)


def get_ai_recommendations(
    risk_profile: dict,
    weights: dict,
    performance: tuple,
    reasoning_trace: list[dict] | None = None,
    macro_result: dict | None = None,
    force_refresh: bool = True,  # Force new generation each time
) -> str:
    """
    Calls Groq LLM to generate a professional investment advisory report
    tailored to the Indian market context and INR denomination.

    Now also includes agent reasoning for richer, data-driven reports.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return (
            "⚠️ **GROQ_API_KEY not found.**\n\n"
            "Please create a `.env` file in the project root with:\n"
            "```\nGROQ_API_KEY=your_key_here\n```\n"
            "Get a free key at https://console.groq.com"
        )

    client = Groq(api_key=api_key)

    weights_str = "\n".join([
        f"  - {k}: {v*100:.2f}%"
        for k, v in weights.items() if v > 0.005
    ])

    if not weights_str:
        weights_str = "  (No significant allocations)"

    # ── Build agent reasoning summary ────────────────────────────────────────
    agent_section = ""
    if reasoning_trace:
        lines = []
        for entry in reasoning_trace:
            ticker = entry.get("ticker", "—")
            agent = entry.get("agent", "Unknown")
            signal = entry.get("signal", "N/A")
            score = entry.get("score", "N/A")
            reason = entry.get("reasoning", "")
            # Truncate long reasoning for prompt size
            if len(reason) > 200:
                reason = reason[:200] + "..."
            lines.append(f"  [{agent}] {ticker}: {signal} (score={score}) — {reason}")
        agent_section = "\n    AGENT ANALYSIS SUMMARY:\n" + "\n".join(lines[:15])  # Reduced from 30 to 15

    # ── Macro context ────────────────────────────────────────────────────────
    macro_section = ""
    if macro_result:
        env = macro_result.get("environment", "N/A")
        m_score = macro_result.get("score", "N/A")
        macro_data = macro_result.get("macro_data", {})
        macro_section = f"""
    MACRO ENVIRONMENT: {env} (score: {m_score}/100)
    - RBI Repo Rate: {macro_data.get('rbi_repo_rate', 'N/A')}%
    - CPI Inflation: {macro_data.get('cpi_inflation', 'N/A')}%
    - GDP Growth: {macro_data.get('gdp_growth', 'N/A')}%
    - Crude Oil: ${macro_data.get('crude_oil_usd', 'N/A')}/bbl
    """

    # Generate multiple random elements
    random_num = random.randint(100000, 999999)
    timestamp_ms = datetime.now().timestamp() * 1000
    
    prompt = f"""
    [REQUEST #{random_num} at {timestamp_ms}]
    UUID: {uuid.uuid4()}
    
    CRITICAL: Generate UNIQUE content. Use different wording, perspectives, and structure than any previous response.
    
    You are a SEBI-registered investment advisor specialising in Indian equity markets.
    Provide a fresh professional report.

    CLIENT PROFILE:
    - Age: {risk_profile['age']}
    - Risk Tolerance: {risk_profile['risk_level']}
    - Investment Amount: ₹{risk_profile['amount']:,}
    - Investment Horizon: {risk_profile['horizon']} years
    - Financial Goal: {risk_profile['goal']}

    OPTIMISED PORTFOLIO (NSE stocks, conviction-weighted):
    {weights_str}

    PORTFOLIO METRICS:
    - Expected Annual Return: {performance[0]*100:.2f}%
    - Annual Volatility (Risk): {performance[1]*100:.2f}%
    - Sharpe Ratio: {performance[2]:.2f}
    {macro_section}
    {agent_section}

    Please provide the following sections:

    1. **Portfolio Summary** (2–3 lines, mention Indian market context and how agent signals shaped allocation)
    2. **Rationale for Key Holdings** (why each major stock suits this client, reference agent findings where possible)
    3. **Risk Assessment** (short-term market risks, long-term sector risks for India, macro environment impact)
    4. **Projected Growth in ₹** (estimate portfolio value at 1yr, 3yr, 5yr based on expected return)
    5. **Indian Tax Implications** (mention STCG 15%, LTCG 10% above ₹1L briefly)
    6. **Rebalancing Strategy** (when and how to rebalance, incorporating agent signal reviews)
    7. **Regulatory Disclaimer** (SEBI disclaimer, past performance caveat)

    Keep it professional, specific to Indian markets (NSE/BSE context), and data-driven.
    Use ₹ symbol for all monetary values.
    """

    try:
        time.sleep(0.1)  # Prevent API caching
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": (
                        f"Request ID: {uuid.uuid4()}. "
                        "You are a SEBI-registered expert investment advisor for Indian equity markets. "
                        "Be professional, specific, and always denominate values in Indian Rupees (₹). "
                        "IMPORTANT: Vary your language, structure, and perspectives. Do not repeat previous analyses."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=1500,
            temperature=1.0,
            top_p=0.9,
            frequency_penalty=0.5,
            presence_penalty=0.5,
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"⚠️ **AI report generation failed:** {e}"