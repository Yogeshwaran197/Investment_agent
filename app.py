import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv(override=True)

from market_data import get_tickers_for_sectors, get_stock_data, TICKERS
from optimizer import optimize_portfolio, fetch_benchmark_data, compute_benchmark_comparison
from orchestrator import orchestrate
from ai_agent import get_ai_recommendations

st.set_page_config(
    page_title="AI Investment Advisor (India) — V3",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Clean Dark Theme CSS ---
st.markdown('''
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Dark background */
    .stApp {
        background-color: #0a0a0a;
    }
    
    .main .block-container {
        background-color: #111111;
        padding: 2rem;
        max-width: 100%;
    }
    
    /* Headers */
    h1 {
        color: #ffffff !important;
        font-weight: 700 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2 {
        color: #e5e5e5 !important;
        font-weight: 600 !important;
        font-size: 1.8rem !important;
        margin-top: 2rem !important;
        margin-bottom: 1rem !important;
    }
    
    h3 {
        color: #d4d4d4 !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
    }
    
    /* Text */
    p, span, label {
        color: #a3a3a3 !important;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #1a1a1a;
        border-right: 1px solid #2a2a2a;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] label {
        color: #d4d4d4 !important;
    }
    
    /* Button */
    .stButton>button {
        background-color: #2a2a2a;
        color: #ffffff;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.75rem 1.5rem;
        transition: all 0.2s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        background-color: #3a3a3a;
        border-color: #4a4a4a;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-weight: 500 !important;
        color: #737373 !important;
        font-size: 0.85rem !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    [data-testid="stMetricDelta"] {
        color: #a3a3a3 !important;
    }
    
    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        font-weight: 600;
        color: #ffffff;
        padding: 1rem;
    }
    
    .streamlit-expanderHeader:hover {
        background-color: #222222;
        border-color: #3a3a3a;
    }
    
    /* Alerts */
    .stAlert {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
        color: #d4d4d4;
    }
    
    .stSuccess {
        background-color: #1a2a1a;
        border-color: #2a3a2a;
    }
    
    .stWarning {
        background-color: #2a2a1a;
        border-color: #3a3a2a;
    }
    
    .stError {
        background-color: #2a1a1a;
        border-color: #3a2a2a;
    }
    
    /* Dataframes */
    [data-testid="stDataFrame"] {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 8px;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 1px;
        background-color: #2a2a2a;
    }
    
    /* Input fields */
    .stTextInput>div>div>input,
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>select {
        background-color: #1a1a1a;
        border: 1px solid #2a2a2a;
        border-radius: 6px;
        color: #ffffff;
    }
    
    .stTextInput>div>div>input:focus,
    .stNumberInput>div>div>input:focus,
    .stSelectbox>div>div>select:focus {
        border-color: #3a3a3a;
        background-color: #222222;
    }
    
    /* Radio & Checkbox */
    .stRadio>div,
    .stCheckbox>div {
        color: #d4d4d4;
    }
    
    /* Slider */
    .stSlider>div>div>div>div {
        background-color: #3a3a3a;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1a1a1a;
        border-bottom: 1px solid #2a2a2a;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: #737373;
        background-color: transparent;
    }
    
    .stTabs [aria-selected="true"] {
        color: #ffffff;
        border-bottom-color: #ffffff;
    }
    
    /* Markdown */
    .stMarkdown {
        color: #a3a3a3;
    }
    
</style>
''', unsafe_allow_html=True)


st.title("AI Investment Advisory Agent")
st.markdown(
    "Multi-agent Indian equity portfolio powered by Groq AI • "
    "Technical · Sentiment · Fundamental · Macro analysis • NSE Stocks • ₹ INR"
)

# ── Sidebar: Risk Profile ──────────────────────────────────────────────────────
st.sidebar.header("Your Investment Profile")

age = st.sidebar.slider("Age", 18, 70, 28)

amount = st.sidebar.number_input(
    "Investment Amount (₹)",
    min_value=10_000,
    max_value=10_000_000,
    value=100_000,
    step=5_000,
    help="Minimum ₹10,000 recommended for meaningful diversification"
)

horizon = st.sidebar.selectbox("Investment Horizon (Years)", [1, 3, 5, 10, 20])

risk_level = st.sidebar.radio(
    "Risk Tolerance",
    ["Conservative", "Moderate", "Aggressive"],
    help=(
        "Conservative = min volatility | "
        "Moderate = ~15% target return | "
        "Aggressive = max Sharpe ratio"
    )
)

goal = st.sidebar.selectbox(
    "Financial Goal",
    ["Wealth Growth", "Retirement", "Passive Income / Dividends", "Short-term Gains", "Child Education"]
)

all_sectors = list(TICKERS.keys())
sectors = st.sidebar.multiselect(
    "Sectors of Interest",
    all_sectors,
    default=["Tech", "Finance", "Index ETF"]
)

st.sidebar.markdown("---")
st.sidebar.subheader("SIP Mode")
sip_enabled = st.sidebar.toggle("Enable Monthly SIP", value=False)
sip_amount = None
if sip_enabled:
    sip_amount = st.sidebar.number_input(
        "Monthly SIP Amount (₹)",
        min_value=1_000,
        max_value=1_000_000,
        value=10_000,
        step=1_000,
        help="Amount to invest each month via SIP"
    )

run = st.sidebar.button("Generate Recommendations", use_container_width=True)


# ── Helper: format INR ─────────────────────────────────────────────────────────
def fmt_inr(value: float) -> str:
    """Format a number as Indian Rupees with lakh/crore notation."""
    if value >= 1e7:
        return f"₹{value/1e7:.2f} Cr"
    elif value >= 1e5:
        return f"₹{value/1e5:.2f} L"
    else:
        return f"₹{value:,.0f}"


# ── Main Logic ─────────────────────────────────────────────────────────────────
if run:
    if not sectors:
        st.warning("Please select at least one sector.")
        st.stop()

    # ── Step 1: Fetch Market Data ──────────────────────────────────────────────
    with st.spinner("Fetching NSE market data..."):
        tickers = get_tickers_for_sectors(sectors)
        prices = get_stock_data(tuple(tickers))

        if prices.empty or prices.shape[1] < 2:
            st.error(
                "Could not fetch sufficient stock data. "
                "Try selecting more sectors or check your internet connection."
            )
            st.stop()

    # ── Step 2: Run All Agents via Orchestrator ────────────────────────────────
    with st.spinner("Running 5 specialist agents (Technical · Sentiment · Fundamental · Macro · Risk Profiler)..."):
        orchestrator_result = orchestrate(
            prices=prices,
            tickers=list(prices.columns),
            age=age,
            horizon=horizon,
            risk_level=risk_level,
            goal=goal,
        )
        conviction_scores = orchestrator_result["conviction_scores"]
        agent_results = orchestrator_result["agent_results"]
        reasoning_trace = orchestrator_result["reasoning_trace"]

    # ── Step 3: Conviction-Weighted Optimisation ───────────────────────────────
    with st.spinner("Running Conviction-Weighted Mean-Variance Optimisation..."):
        risk_profile = {
            "age":        age,
            "amount":     amount,
            "horizon":    horizon,
            "risk_level": risk_level,
            "goal":       goal,
        }
        try:
            weights, performance, allocation, leftover, sip_schedule = optimize_portfolio(
                prices, risk_level, amount,
                conviction_scores=conviction_scores,
                sip_monthly=sip_amount,
            )
        except Exception as e:
            st.error(f"Optimisation failed: {e}. Try a different risk level or add more sectors.")
            st.stop()

    # ── Step 4: Benchmark Data ─────────────────────────────────────────────────
    with st.spinner("Fetching benchmark data (Nifty 50 / Sensex)..."):
        benchmark_prices = fetch_benchmark_data()
        bench_comparison = None
        if not benchmark_prices.empty:
            bench_comparison = compute_benchmark_comparison(weights, prices, benchmark_prices)

    # ── Step 5: AI Report with Agent Reasoning ─────────────────────────────────
    with st.spinner("Generating AI advisory report via Groq..."):
        ai_report = get_ai_recommendations(
            risk_profile, weights, performance,
            reasoning_trace=reasoning_trace,
            macro_result=agent_results.get("Macro"),
        )

    # ══════════════════════════════════════════════════════════════════════════
    #   RESULTS LAYOUT
    # ══════════════════════════════════════════════════════════════════════════
    st.success("Multi-Agent Analysis Complete!")

    # ── Key Metrics ────────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Expected Annual Return",  f"{performance[0]*100:.2f}%")
    col2.metric("Annual Volatility",        f"{performance[1]*100:.2f}%")
    col3.metric("Sharpe Ratio",             f"{performance[2]:.2f}")
    projected_5yr = amount * ((1 + performance[0]) ** 5)
    col4.metric("Projected Value (5yr)",    fmt_inr(projected_5yr))

    st.divider()

    # ── Agent Conviction Heatmap ───────────────────────────────────────────────
    st.subheader("Agent Conviction Scores per Stock")

    # Build heatmap data
    heatmap_data = []
    tech_results = agent_results.get("Technical", {})
    sent_results = agent_results.get("Sentiment", {})
    fund_results = agent_results.get("Fundamental", {})
    macro_score = agent_results.get("Macro", {}).get("score", 50)
    risk_score = agent_results.get("Risk Profiler", {}).get("risk_score", 50)

    for ticker in list(weights.keys()):
        if weights.get(ticker, 0) < 0.001:
            continue
        row = {"Stock": ticker}
        row["Technical"] = tech_results.get(ticker, {}).get("score", "—")
        row["Sentiment"] = sent_results.get(ticker, {}).get("score", "—")
        row["Fundamental"] = fund_results.get(ticker, {}).get("score", "—")
        row["Macro"] = macro_score
        row["Risk Profile"] = risk_score
        row["Conviction"] = conviction_scores.get(ticker, 50)
        heatmap_data.append(row)

    if heatmap_data:
        heatmap_df = pd.DataFrame(heatmap_data).set_index("Stock")

        # Convert to numeric for the heatmap, replacing "—" with NaN
        heatmap_numeric = heatmap_df.apply(pd.to_numeric, errors="coerce")

        fig_heatmap = go.Figure(data=go.Heatmap(
            z=heatmap_numeric.values,
            x=heatmap_numeric.columns.tolist(),
            y=heatmap_numeric.index.tolist(),
            colorscale=[
                [0, "#ef4444"],      # red (bearish)
                [0.35, "#f97316"],   # orange
                [0.5, "#eab308"],    # yellow (neutral)
                [0.65, "#84cc16"],   # lime
                [1, "#22c55e"],      # green (bullish)
            ],
            zmin=0, zmax=100,
            text=heatmap_numeric.values.astype(int),
            texttemplate="%{text}",
            textfont={"size": 13},
            hovertemplate="Stock: %{y}<br>Agent: %{x}<br>Score: %{z}<extra></extra>",
        ))
        fig_heatmap.update_layout(
            height=max(300, len(heatmap_data) * 40 + 100),
            margin=dict(l=10, r=10, t=10, b=10),
            xaxis_title="Agent",
            yaxis_title="",
            yaxis_autorange="reversed",
        )
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.info("No stocks with significant weights to display.")

    st.divider()

    # ── Projected Growth Chart ─────────────────────────────────────────────────
    st.subheader("Projected Portfolio Growth (₹)")
    years     = list(range(0, horizon + 1))
    projected = [amount * ((1 + performance[0]) ** y) for y in years]
    pessimist = [amount * ((1 + max(performance[0] - performance[1], -0.99)) ** y) for y in years]
    optimist  = [amount * ((1 + performance[0] + performance[1]) ** y) for y in years]

    fig_growth = go.Figure()
    fig_growth.add_trace(go.Scatter(
        x=years, y=optimist, name="Optimistic",
        line=dict(dash="dot", color="green")
    ))
    fig_growth.add_trace(go.Scatter(
        x=years, y=projected, name="Expected",
        line=dict(width=3, color="royalblue")
    ))
    fig_growth.add_trace(go.Scatter(
        x=years, y=pessimist, name="Pessimistic",
        line=dict(dash="dot", color="red")
    ))
    fig_growth.update_layout(
        xaxis_title="Years",
        yaxis_title="Portfolio Value (₹)",
        yaxis_tickprefix="₹",
        yaxis_tickformat=",",
        hovermode="x unified"
    )
    st.plotly_chart(fig_growth, use_container_width=True)

    # ── Allocation + History ───────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Portfolio Allocation")
        filtered = {k: v for k, v in weights.items() if v > 0.001}
        fig_pie = px.pie(
            values=list(filtered.values()),
            names=list(filtered.keys()),
            title="Recommended Allocation",
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("1-Year NSE Price History (₹)")
        cols_to_show = [c for c in list(filtered.keys()) if c in prices.columns]
        if cols_to_show:
            norm = prices[cols_to_show] / prices[cols_to_show].iloc[0] * 100
            fig_hist = px.line(
                norm,
                title="Normalised Price History (Base = 100)",
                labels={"value": "Indexed Price", "variable": "Stock"}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        else:
            st.info("No price history available for selected stocks.")

    st.divider()

    # ── Benchmark Comparison ───────────────────────────────────────────────────
    if bench_comparison and not bench_comparison.get("normalized", pd.DataFrame()).empty:
        st.subheader("Benchmark Comparison (vs Nifty 50 / Sensex)")

        # Metrics row
        bench_cols = st.columns(1 + len(bench_comparison.get("benchmarks", {})))
        bench_cols[0].metric(
            "Your Portfolio (1yr)",
            f"{bench_comparison['portfolio_return']*100:.2f}%"
        )
        for i, (bench_name, bench_data) in enumerate(bench_comparison.get("benchmarks", {}).items()):
            alpha = bench_data.get("alpha", 0)
            bench_cols[i + 1].metric(
                f"vs {bench_name}",
                f"{bench_data['return']*100:.2f}%",
                delta=f"Alpha: {alpha*100:+.2f}%",
                delta_color="normal" if alpha >= 0 else "inverse"
            )

        # Normalised comparison chart
        norm_df = bench_comparison.get("normalized", pd.DataFrame())
        if not norm_df.empty:
            fig_bench = px.line(
                norm_df,
                title="Portfolio vs Benchmarks (Normalised to 100)",
                labels={"value": "Indexed Value", "variable": ""}
            )
            fig_bench.update_layout(hovermode="x unified")
            st.plotly_chart(fig_bench, use_container_width=True)

        st.divider()

    # ── AI Report ──────────────────────────────────────────────────────────────
    st.subheader("AI Advisor Report (Groq — SEBI Compliant)")
    st.markdown(ai_report)

    st.divider()

    # ── Reasoning Trace ────────────────────────────────────────────────────────
    st.subheader("Reasoning Trace — Why Each Agent Decided")

    # Macro & Risk Profiler (portfolio-level)
    with st.expander("Macro Environment Assessment", expanded=False):
        macro_res = agent_results.get("Macro", {})
        env = macro_res.get("environment", "N/A")
        m_score = macro_res.get("score", "N/A")
        env_colors = {"FAVORABLE": "Favorable", "NEUTRAL": "Neutral", "UNFAVORABLE": "Unfavorable"}
        st.markdown(f"### Environment: **{env}** (Score: {m_score}/100)")

        components = macro_res.get("components", {})
        if components:
            macro_cols = st.columns(len(components))
            for i, (name, comp) in enumerate(components.items()):
                macro_cols[i].metric(name.replace("_", " ").title(), f"{comp['score']}/100")
                macro_cols[i].caption(comp.get("detail", ""))

        macro_data = macro_res.get("macro_data", {})
        if macro_data:
            st.markdown("**Key Indicators:**")
            ind_cols = st.columns(4)
            ind_cols[0].metric("RBI Repo Rate", f"{macro_data.get('rbi_repo_rate', 'N/A')}%")
            ind_cols[1].metric("CPI Inflation", f"{macro_data.get('cpi_inflation', 'N/A')}%")
            ind_cols[2].metric("GDP Growth", f"{macro_data.get('gdp_growth', 'N/A')}%")
            ind_cols[3].metric("Crude Oil", f"${macro_data.get('crude_oil_usd', 'N/A')}/bbl")

    with st.expander("Risk Profile Assessment", expanded=False):
        risk_res = agent_results.get("Risk Profiler", {})
        r_score = risk_res.get("risk_score", "N/A")
        r_cat = risk_res.get("category", "N/A")
        max_eq = risk_res.get("max_equity_pct", 0)
        st.markdown(f"### Risk Score: **{r_score}/100** → Category: **{r_cat}**")
        st.markdown(f"Maximum recommended equity allocation: **{max_eq*100:.0f}%**")

        components = risk_res.get("components", {})
        if components:
            risk_cols = st.columns(len(components))
            for i, (name, comp) in enumerate(components.items()):
                risk_cols[i].metric(
                    name.replace("_", " ").title(),
                    f"{comp.get('value', 'N/A')}",
                    delta=f"Score: {comp.get('score', 'N/A')}/100"
                )

    # Per-stock agent reasoning
    with st.expander("Per-Stock Agent Signals", expanded=True):
        # Build a summary table
        trace_rows = []
        for entry in reasoning_trace:
            if entry.get("ticker") == "—":
                continue  # Skip portfolio-level entries (shown above)
            trace_rows.append({
                "Agent": entry.get("agent", ""),
                "Stock": entry.get("ticker", ""),
                "Signal": entry.get("signal", ""),
                "Score": entry.get("score", ""),
                "Reasoning": entry.get("reasoning", "")[:120],
            })

        if trace_rows:
            trace_df = pd.DataFrame(trace_rows)
            # Color-code signals
            st.dataframe(
                trace_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score": st.column_config.ProgressColumn(
                        "Score",
                        min_value=0,
                        max_value=100,
                        format="%d",
                    ),
                },
            )
        else:
            st.info("No per-stock reasoning data available.")

    st.divider()

    # ── Share Allocation Table ─────────────────────────────────────────────────
    st.subheader("Share Allocation Breakdown")
    if allocation:
        alloc_rows = []
        for stock, shares in allocation.items():
            price_per_share = float(prices[stock].iloc[-1]) if stock in prices.columns else 0
            value = shares * price_per_share
            conv = conviction_scores.get(stock, "—")
            alloc_rows.append({
                "Stock (NSE)":       stock,
                "Shares":            shares,
                "Price / Share (₹)": f"₹{price_per_share:,.2f}",
                "Invested Value":    fmt_inr(value),
                "Portfolio Weight":  f"{weights.get(stock, 0)*100:.1f}%",
                "Conviction":        f"{conv}/100",
            })
        alloc_df = pd.DataFrame(alloc_rows)
        st.dataframe(alloc_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No full shares could be allocated with the given amount. Try increasing the investment.")

    st.info(f"Uninvested Cash: {fmt_inr(leftover)}")

    # ── SIP Schedule ───────────────────────────────────────────────────────────
    if sip_schedule:
        st.divider()
        st.subheader("SIP Monthly Allocation Schedule")
        st.markdown(f"*Monthly SIP: {fmt_inr(sip_amount)} × 12 months = {fmt_inr(sip_amount * 12)} total*")

        sip_rows = []
        for entry in sip_schedule:
            stocks_str = ", ".join([
                f"{t}: {s} shares" for t, s in entry["stocks"].items()
            ]) if entry["stocks"] else "—"
            sip_rows.append({
                "Month":     entry["month"],
                "Stocks":    stocks_str,
                "Invested":  fmt_inr(entry["invested"]),
                "Cash Left": fmt_inr(entry["cash"]),
            })

        sip_df = pd.DataFrame(sip_rows)
        st.dataframe(sip_df, use_container_width=True, hide_index=True)

        # Total invested vs cash over 12 months
        total_invested = sum(e["invested"] for e in sip_schedule)
        total_cash = sum(e["cash"] for e in sip_schedule)
        sip_cols = st.columns(3)
        sip_cols[0].metric("Total SIP (12m)", fmt_inr(sip_amount * 12))
        sip_cols[1].metric("Total Invested", fmt_inr(total_invested))
        sip_cols[2].metric("Total Uninvested", fmt_inr(total_cash))

    # ── Tax Note ──────────────────────────────────────────────────────────────
    with st.expander("Indian Tax Implications (Quick Reference)"):
        st.markdown("""
        | Holding Period | Gain Type | Tax Rate |
        |---|---|---|
        | < 12 months | Short-Term Capital Gain (STCG) | **15%** |
        | ≥ 12 months | Long-Term Capital Gain (LTCG) | **10%** (above ₹1 Lakh exempt) |

        > **Note:** LTCG above ₹1,00,000 per financial year is taxable at 10% without indexation benefit for listed equity.
        > Consult a CA or tax advisor for your specific situation.
        """)
