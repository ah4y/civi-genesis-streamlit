# 🏙️ CIVI-GENESIS: AI-Powered Policy Simulator

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://civi-genesis.streamlit.app)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Powered by Gemini](https://img.shields.io/badge/Powered%20by-Google%20Gemini-orange.svg)](https://ai.google.dev/)

Simulate how **50,000 synthetic citizens** react to your policies using hybrid AI (LLM + Neural Network). Perfect for policy analysis, business strategy, and understanding social dynamics.

**[🚀 Try Live Demo](https://civi-genesis.streamlit.app)**

---

## 🎯 What Can You Do?

- **Test Policies**: See how citizens react to fuel subsidies, education programs, or business changes
- **Scale Insights**: Start with detailed LLM analysis, then scale to 50K citizens using neural networks
- **Track Impact**: Monitor happiness, policy support, and income across demographics over time
- **Expert Analysis**: Get AI-generated insights from economist, activist, and business perspectives

## ✨ Key Features

**🤖 Hybrid AI System**
- Google Gemini LLM for nuanced citizen reactions
- Neural network learns from LLM to scale simulations
- Intelligent fallback system ensures reliability

**👥 Synthetic Population**
- Up to 50,000 diverse citizens with unique profiles
- Demographics, economics, personality, and political views
- Realistic social dynamics and group interactions

**📊 Rich Analytics**
- Real-time dashboards with interactive charts
- Income inequality tracking and group analysis
- Individual citizen timelines and diary entries
- Export data for further analysis

**⚙️ Three Simulation Modes**
- **LLM_ONLY**: Maximum accuracy with detailed AI analysis
- **HYBRID**: Balanced approach combining LLM + Neural Network  
- **NN_ONLY**: Lightning fast simulations for large populations

---

## 🚀 Quick Start

### Option 1: Try Online (Easiest)
Just click **[Launch App](https://civi-genesis.streamlit.app)** - no installation needed!

### Option 2: Run Locally

1. **Clone the repository**
   ```bash
   git clone https://github.com/YOUR_USERNAME/civi-genesis-streamlit.git
   cd civi-genesis-streamlit
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free API key**
   - Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Create your free Gemini API key (200 requests/day)

4. **Set up environment**
   ```bash
   # Create .env file
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

5. **Launch the app**
   ```bash
   streamlit run app.py
   ```

Open your browser to `http://localhost:8501` and start simulating!

---

## 🎮 How to Use

### 1. Configure Your Simulation
- **Population Size**: 100 - 50,000 citizens
- **Time Steps**: 1-10 steps to see evolution over time
- **Policy**: Choose preset or create custom policy
- **Mode**: Start with LLM_ONLY for best accuracy

### 2. Run Your First Simulation
- Click "🚀 Run Simulation"
- Watch as citizens react to your policy
- Explore results in Overview, Groups, Citizens, and Expert tabs

### 3. Scale with Neural Networks
- Run a few LLM simulations to generate training data
- Train the neural network to learn LLM behavior
- Switch to HYBRID or NN_ONLY for massive simulations

### 4. Analyze Results
- **Overview**: Population-wide trends and statistics
- **Groups**: Compare reactions by income level and location
- **Citizens**: Browse individual citizen profiles and timelines
- **Experts**: AI-generated analysis from multiple perspectives
- **Scenarios**: Compare different policy options side-by-side

---

## 🧠 How It Works

### The Science Behind CIVI-GENESIS

1. **Synthetic Population Generation**
   - Creates diverse citizens with realistic demographic distributions
   - Each citizen has unique personality, economic status, and political views
   - Population reflects real-world social complexity

2. **LLM-Powered Micro-Simulations**
   - Google Gemini analyzes how individual citizens react to policies
   - Considers personal circumstances, personality, and social context
   - Generates detailed reasoning and diary entries

3. **Neural Network Learning**
   - Learns to approximate LLM behavior from collected examples
   - Enables scaling to tens of thousands of citizens
   - Maintains quality while dramatically improving speed

4. **Intelligent Fallback System**
   - LLM → Neural Network → Rule-based progression
   - Ensures simulations continue even with API limits
   - Graceful degradation maintains meaningful results

### Example Policy Presets

- **Fuel Subsidy Removal**: Economic policy with immediate price impacts
- **Student Scholarship Program**: Education targeting low-income families  
- **Universal Basic Income**: Social policy with income redistribution
- **AI Tutor Initiative**: EdTech innovation for educational equity
- **Startup Pricing Strategy**: Business decisions affecting stakeholders

---

## 🔧 Technical Architecture

```
📁 Project Structure
├── app.py              # Main Streamlit application
├── simulation.py       # Core simulation engine
├── llm_client.py       # Gemini LLM integration
├── nn_model.py         # Neural network model
├── population.py       # Citizen generation
├── stats.py           # Analytics computation
├── ui_sections.py     # Interface components
└── data_models.py     # Core data structures
```

**Tech Stack**
- **Frontend**: Streamlit with Plotly visualizations
- **AI/ML**: Google Gemini API + Scikit-learn neural networks
- **Data**: Pandas + NumPy for processing
- **Deployment**: Streamlit Cloud ready

---

## 🚀 Deploy Your Own

### Streamlit Cloud (Recommended)

1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub account
4. Select this repository
5. Add your `GEMINI_API_KEY` in secrets
6. Click Deploy!

### Local Development

```bash
# Install in development mode
pip install -r requirements.txt

# Set environment variables
export GEMINI_API_KEY="your_key_here"  # Linux/Mac
# or
$env:GEMINI_API_KEY="your_key_here"    # Windows PowerShell

# Run with auto-reload
streamlit run app.py --server.runOnSave true
```

---

## ⚠️ Important Notes

**This is a Synthetic Simulation Tool**
- Creates fictional citizens and scenarios for exploratory purposes
- **Does NOT predict real-world behavior** - use for thought experiments
- Supplement, don't replace, real data and expert analysis
- Consider as a tool for identifying potential scenarios and blind spots

**Limitations**
- Simplified model of complex social dynamics  
- LLM outputs may reflect training biases
- Neural network introduces additional approximation layers
- Results are for educational/research purposes only

---

## 🤝 Contributing

We welcome contributions! Areas for improvement:

- **New Policy Domains**: Healthcare, environment, technology
- **Enhanced Citizen Models**: More personality types and demographics
- **Better Visualizations**: Additional chart types and insights
- **Performance Optimization**: Faster simulations and better scaling
- **Documentation**: Tutorials, examples, and case studies

## 📄 License

This project is open source and available under the MIT License.

---

## 🙋‍♂️ Support

- **Issues**: Report bugs or request features on GitHub
- **Discussions**: Share use cases and get community support
- **API Help**: Check the [Google AI documentation](https://ai.google.dev/docs)

---

**Built with ❤️ using Streamlit, Google Gemini, and Scikit-learn**

*Explore the future of policy simulation with CIVI-GENESIS - where AI meets social science!*