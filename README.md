# 🚀 AI Investment Advisory Agent

A sophisticated multi-agent investment advisory system for Indian equity markets, powered by Groq AI and built with Streamlit.

## 📋 Overview

This application uses 5 specialized AI agents to analyze stocks and generate personalized investment recommendations for the Indian market (NSE/BSE). It provides conviction-weighted portfolio optimization, risk profiling, and SEBI-compliant investment reports.

## ✨ Features

### 🤖 Multi-Agent Analysis System
- **Technical Agent**: Analyzes price trends, RSI, MACD, and moving averages
- **Sentiment Agent**: Evaluates market sentiment and news (simulated)
- **Fundamental Agent**: Assesses financial metrics like P/E ratio, debt-to-equity
- **Macro Agent**: Monitors RBI repo rate, inflation, GDP growth, crude oil prices
- **Risk Profiler**: Evaluates user risk tolerance based on age, horizon, and goals

### 📊 Portfolio Optimization
- Conviction-weighted mean-variance optimization
- Sharpe ratio maximization
- Risk-adjusted returns
- Benchmark comparison (Nifty 50 / Sensex)

### 💼 Investment Features
- Personalized risk profiling
- SIP (Systematic Investment Plan) mode
- Share allocation breakdown
- Projected growth scenarios (optimistic, expected, pessimistic)
- Indian tax implications (STCG/LTCG)
- Rebalancing strategies

### 🎨 User Interface
- Clean dark theme
- Interactive charts and heatmaps
- Real-time NSE stock data
- Agent conviction scores visualization
- Detailed reasoning traces

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **AI/LLM**: Groq (llama-3.1-8b-instant)
- **Data**: yfinance (Yahoo Finance API)
- **Optimization**: scipy, numpy, pandas
- **Visualization**: plotly

## 📦 Installation

### Prerequisites
- Python 3.8+
- Groq API Key (free at https://console.groq.com)

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/Yogeshwaran197/Investment_agent.git
cd Investment_agent
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Create `.env` file**
```bash
# Create .env file in project root
echo GROQ_API_KEY=your_groq_api_key_here > .env
```

4. **Run the application**
```bash
streamlit run app.py
```

## 🚀 Usage

1. **Configure Your Profile** (Sidebar)
   - Set your age
   - Enter investment amount (₹)
   - Choose investment horizon (1-20 years)
   - Select risk tolerance (Conservative/Moderate/Aggressive)
   - Pick financial goal
   - Select sectors of interest

2. **Optional: Enable SIP Mode**
   - Toggle SIP mode
   - Set monthly SIP amount

3. **Generate Recommendations**
   - Click "Generate Recommendations"
   - Wait for multi-agent analysis
   - Review portfolio allocation and AI report

## 📁 Project Structure

```
Investment_Agent/
├── agents/
│   ├── __init__.py
│   ├── technical_agent.py      # Technical analysis
│   ├── sentiment_agent.py      # Sentiment analysis
│   ├── fundamental_agent.py    # Fundamental analysis
│   ├── macro_agent.py          # Macro environment analysis
│   └── risk_profiler.py        # Risk profiling
├── app.py                      # Main Streamlit application
├── ai_agent.py                 # Groq LLM integration
├── orchestrator.py             # Agent coordination
├── optimizer.py                # Portfolio optimization
├── market_data.py              # Stock data fetching
├── requirements.txt            # Dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # This file
```

## 🔑 Environment Variables

Create a `.env` file with:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your free API key at: https://console.groq.com

## 📊 Supported Sectors

- Technology (TCS, Infosys, Wipro, HCL Tech)
- Finance (HDFC Bank, ICICI Bank, SBI, Kotak Bank)
- Energy (Reliance, ONGC, BPCL, IOC)
- FMCG (HUL, ITC, Nestle, Britannia)
- Pharma (Sun Pharma, Dr. Reddy's, Cipla, Lupin)
- Auto (Maruti, Tata Motors, M&M, Bajaj Auto)
- Index ETF (Nifty BeES, Junior BeES)

## ⚠️ Disclaimer

**This application is for educational and informational purposes only.**

- Not a substitute for professional financial advice
- Past performance does not guarantee future results
- Investments in equity markets are subject to market risks
- Consult a SEBI-registered investment advisor before making investment decisions
- The developers are not responsible for any financial losses

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is open source and available under the MIT License.

## 👨‍💻 Author

**Yogeshwaran**
- GitHub: [@Yogeshwaran197](https://github.com/Yogeshwaran197)

## 🙏 Acknowledgments

- Groq for providing fast LLM inference
- Yahoo Finance for market data
- Streamlit for the amazing framework
- Indian stock market community

## 📞 Support

For issues, questions, or suggestions:
- Open an issue on GitHub
- Contact via GitHub profile

---

**Made with ❤️ for Indian investors**
