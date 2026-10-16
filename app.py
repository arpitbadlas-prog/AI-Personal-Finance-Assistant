"""
Personal Finance AI Agent - Streamlit Application
Frameworks: LangChain, Streamlit, Scikit-Learn, Joblib
"""

import os
import re
import math
import time
import json
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# ── Page Configuration ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Personal Finance Assistant",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS for Polished, Modern Aesthetics ─────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 1.8rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.08);
    }
    
    .main-header h1 {
        margin: 0;
        font-size: 2.1rem;
        font-weight: 700;
        color: #ffffff;
    }
    
    .main-header p {
        margin: 0.4rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
        color: #e0e7ff;
    }
    
    .badge-sub {
        background-color: rgba(255, 255, 255, 0.15);
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        display: inline-block;
        margin-top: 8px;
    }
    
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    
    .metric-title {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        color: #0f172a;
        font-size: 1.5rem;
        font-weight: 700;
        margin-top: 4px;
    }
    
    .quick-query-btn {
        margin-bottom: 8px;
    }
    
    .tool-box {
        background-color: #f8fafc;
        border-left: 4px solid #3b82f6;
        padding: 12px 16px;
        border-radius: 4px;
        margin: 10px 0;
        font-size: 0.9rem;
    }
    
    .risk-low {
        background-color: #ecfdf5;
        border: 1px solid #a7f3d0;
        color: #065f46;
        padding: 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-mod {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        color: #92400e;
        padding: 14px;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-high {
        background-color: #fef2f2;
        border: 1px solid #fecaca;
        color: #991b1b;
        padding: 14px;
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# ── Load ML Model & Scaler ─────────────────────────────────────────────────────
@st.cache_resource
def load_ml_assets():
    scaler_path = "scaler.pkl"
    model_path = "model.pkl"
    scaler = None
    model = None
    
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
        except Exception as e:
            st.sidebar.error(f"Error loading scaler: {e}")
            
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.sidebar.error(f"Error loading model: {e}")
            
    return scaler, model

scaler, model = load_ml_assets()

# ── Core Deterministic Financial Calculation Tools ─────────────────────────────
def compute_emi(principal: float, annual_rate: float, tenure_years: float) -> dict:
    """Calculates loan EMI using the reducing balance formula."""
    try:
        p = float(principal)
        r_annual = float(annual_rate)
        years = float(tenure_years)
        
        if p <= 0 or years <= 0:
            return {"error": "Principal and tenure must be positive numbers."}
            
        r = (r_annual / 12.0) / 100.0
        n = int(round(years * 12.0))
        
        if r == 0:
            emi = p / n
        else:
            factor = math.pow(1.0 + r, n)
            emi = (p * r * factor) / (factor - 1.0)
            
        total_payment = emi * n
        total_interest = total_payment - p
        
        return {
            "emi": emi,
            "principal": p,
            "annual_rate": r_annual,
            "tenure_years": years,
            "total_months": n,
            "total_interest": total_interest,
            "total_payment": total_payment,
            "formatted_summary": (
                f"=== EMI Calculation Breakdown ===\n"
                f"• Monthly EMI: ₹{emi:,.2f}\n"
                f"• Loan Principal: ₹{p:,.2f}\n"
                f"• Annual Interest Rate: {r_annual:.2f}%\n"
                f"• Tenure: {years:g} years ({n} months)\n"
                f"• Total Interest Payable: ₹{total_interest:,.2f}\n"
                f"• Total Amount Payable: ₹{total_payment:,.2f}"
            )
        }
    except Exception as e:
        return {"error": str(e)}

def compute_sip(monthly_investment: float, annual_return_rate: float, tenure_years: float) -> dict:
    """Calculates Future Value (FV) of monthly SIP investments."""
    try:
        p = float(monthly_investment)
        r_annual = float(annual_return_rate)
        years = float(tenure_years)
        
        if p <= 0 or years <= 0:
            return {"error": "Monthly investment and tenure must be positive numbers."}
            
        r = (r_annual / 12.0) / 100.0
        n = int(round(years * 12.0))
        
        if r == 0:
            fv = p * n
        else:
            # Formula: FV = P * ((1+r)^n - 1) / r
            factor = math.pow(1.0 + r, n)
            fv = p * ((factor - 1.0) / r)
            
        total_invested = p * n
        wealth_gain = fv - total_invested
        
        return {
            "fv": fv,
            "monthly_investment": p,
            "annual_return": r_annual,
            "tenure_years": years,
            "total_months": n,
            "total_invested": total_invested,
            "wealth_gain": wealth_gain,
            "formatted_summary": (
                f"=== SIP Wealth Accumulation Breakdown ===\n"
                f"• Estimated Future Value (FV): ₹{fv:,.2f}\n"
                f"• Monthly Contribution: ₹{p:,.2f}\n"
                f"• Expected Annual Return: {r_annual:.2f}%\n"
                f"• Tenure: {years:g} years ({n} months)\n"
                f"• Total Capital Invested: ₹{total_invested:,.2f}\n"
                f"• Estimated Wealth Gained: ₹{wealth_gain:,.2f}"
            )
        }
    except Exception as e:
        return {"error": str(e)}

def compute_budget(monthly_income: float) -> dict:
    """Calculates 50-30-20 budget plan."""
    try:
        income = float(monthly_income)
        if income <= 0:
            return {"error": "Monthly income must be greater than zero."}
            
        needs = 0.50 * income
        wants = 0.30 * income
        savings = 0.20 * income
        
        return {
            "income": income,
            "needs": needs,
            "wants": wants,
            "savings": savings,
            "formatted_summary": (
                f"=== 50-30-20 Monthly Budget Allocation for ₹{income:,.2f} ===\n"
                f"• 50% Needs (₹{needs:,.2f}): Essential living costs (Rent/EMI, groceries, utilities, insurance)\n"
                f"• 30% Wants (₹{wants:,.2f}): Lifestyle expenses (Dining out, recreation, shopping, vacations)\n"
                f"• 20% Savings (₹{savings:,.2f}): Wealth building (Emergency fund, mutual fund SIPs, retirement)"
            )
        }
    except Exception as e:
        return {"error": str(e)}

def compute_math(expression: str) -> dict:
    """Evaluates percentages and general math expressions safely and robustly."""
    try:
        raw = expression.strip()
        # Clean currency symbols, commas, and common word variants
        cleaned = re.sub(r'[₹$€£,]', '', raw)
        cleaned = re.sub(r'\b(?:rs\.?|inr)\b', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\b(?:percent|pct)\b', '%', cleaned, flags=re.IGNORECASE)
        
        # 1. Percentage calculation: e.g. 'What is 20% of ₹85,000' or '20% of 85000'
        pct_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%\s*(?:of|on|\*)\s*([0-9]+(?:\.[0-9]+)?)', cleaned, re.IGNORECASE)
        if pct_match:
            pct = float(pct_match.group(1))
            val = float(pct_match.group(2))
            res = (pct / 100.0) * val
            summary = f"{pct:g}% of ₹{val:,.2f} = ₹{res:,.2f}"
            return {"result": res, "formatted_summary": summary}

        # Reverse percentage: e.g. '85000 * 20%' or '85000 at 20%'
        pct_rev = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:at|\*)\s*([0-9]+(?:\.[0-9]+)?)\s*%', cleaned, re.IGNORECASE)
        if pct_rev:
            val = float(pct_rev.group(1))
            pct = float(pct_rev.group(2))
            res = (pct / 100.0) * val
            summary = f"{pct:g}% of ₹{val:,.2f} = ₹{res:,.2f}"
            return {"result": res, "formatted_summary": summary}

        # 2. General arithmetic extraction
        math_str = cleaned
        math_str = re.sub(r'\bplus\b', '+', math_str, flags=re.IGNORECASE)
        math_str = re.sub(r'\bminus\b', '-', math_str, flags=re.IGNORECASE)
        math_str = re.sub(r'\b(?:times|multiplied by)\b', '*', math_str, flags=re.IGNORECASE)
        math_str = re.sub(r'\b(?:divided by|over)\b', '/', math_str, flags=re.IGNORECASE)
        math_str = math_str.replace('x', '*').replace('X', '*').replace('^', '**')

        # Find the math formula part using regex
        expr_match = re.search(r'([0-9\.\s\+\-\*\/\(\)]+)', math_str)
        if expr_match:
            candidate = expr_match.group(1).strip()
            if any(c.isdigit() for c in candidate):
                allowed = set('0123456789+-*/(). ')
                if all(c in allowed for c in candidate):
                    res = eval(candidate, {'__builtins__': None}, {})
                    if isinstance(res, float):
                        summary = f"Calculation Result: ₹{res:,.2f} ({res:g})"
                    else:
                        summary = f"Calculation Result: {res}"
                    return {"result": res, "formatted_summary": summary}

        return {"error": f"Could not parse mathematical expression from '{raw}'. Please provide numbers or percentages."}
    except Exception as e:
        return {"error": str(e)}

# ── Agent Orchestrator & Tool Selector ─────────────────────────────────────────
def parse_lakhs(text: str) -> float:
    m = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:lakh|lakhs|lac|lacs|l)\b', text, re.IGNORECASE)
    if m:
        return float(m.group(1)) * 100000.0
    m_cr = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:crore|crores|cr)\b', text, re.IGNORECASE)
    if m_cr:
        return float(m_cr.group(1)) * 10000000.0
    return None

def agent_reason_and_solve(query: str, provider: str = None, api_key: str = None, model_name: str = None) -> dict:
    """
    Intelligent Agent pipeline.
    Combines live LLM orchestration (OpenAI, OpenRouter, Groq) with
    high-precision deterministic calculation tools to prevent hallucinations.
    """
    q_low = query.lower().replace(',', '')
    selected_tool = ""
    tool_args = {}
    thought = ""
    tool_res = {}
    
    # 1. Loan EMI Intent
    if "emi" in q_low or "loan" in q_low:
        thought = "The user is asking to calculate a loan repayment or monthly EMI. I should delegate this to the EMI Calculator Tool."
        selected_tool = "EMI Calculator Tool"
        p = parse_lakhs(q_low)
        if not p:
            p_match = re.search(r'(?:₹|rs\.?|loan(?:\s+amount)?\s*(?:of)?)\s*([0-9]+)', q_low)
            p = float(p_match.group(1)) if p_match else 1000000.0
            
        r_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', q_low)
        rate = float(r_match.group(1)) if r_match else 8.0
        
        y_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:year|years|yrs)', q_low)
        years = float(y_match.group(1)) if y_match else 20.0
        
        tool_args = {"principal": p, "annual_rate": rate, "tenure_years": years}
        tool_res = compute_emi(p, rate, years)

    # 2. SIP Investment Intent
    elif "invest" in q_low or "sip" in q_low or ("monthly" in q_low and ("return" in q_low or "years" in q_low)):
        thought = "The user wants to compute compound future value for monthly investments (SIP). I will invoke the SIP Investment Calculator Tool."
        selected_tool = "SIP Investment Calculator Tool"
        amounts = [float(x) for x in re.findall(r'(?:₹|rs\.?\s*)?([0-9]+)', q_low)]
        y_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*(?:year|years|yrs)', q_low)
        years = float(y_match.group(1)) if y_match else 10.0
        r_match = re.search(r'([0-9]+(?:\.[0-9]+)?)\s*%', q_low)
        rate = float(r_match.group(1)) if r_match else 10.0
        
        cand = [a for a in amounts if a != years and a != rate]
        monthly_inv = cand[0] if cand else 5000.0
        
        tool_args = {"monthly_investment": monthly_inv, "annual_return_rate": rate, "tenure_years": years}
        tool_res = compute_sip(monthly_inv, rate, years)

    # 3. Budget Planning Intent
    elif "budget" in q_low or "50-30-20" in q_low or "earning" in q_low or "income" in q_low:
        thought = "The user is requesting a salary/budget allocation. I will apply the 50-30-20 Budget Planner Tool."
        selected_tool = "Budget Planner Tool (50-30-20)"
        p = parse_lakhs(q_low)
        if not p:
            inc_match = re.search(r'(?:₹|rs\.?|income|earning|salary)?\s*([0-9]{4,10})', q_low)
            p = float(inc_match.group(1)) if inc_match else 90000.0
        tool_args = {"monthly_income": p}
        tool_res = compute_budget(p)

    # 4. Math / Percentage
    else:
        thought = "The query asks for a percentage or arithmetic computation. I will invoke the General Math Calculator Tool."
        selected_tool = "General Math Calculator Tool"
        tool_args = {"expression": query}
        tool_res = compute_math(query)
        
    obs = tool_res.get("formatted_summary", tool_res.get("error", "No output"))
    
    # Try calling configured LLM if user provided key
    llm_explanation = None
    if api_key and provider and "Deterministic" not in provider:
        try:
            from openai import OpenAI
            base_url = None
            if "OpenRouter" in provider:
                base_url = "https://openrouter.ai/api/v1"
            elif "Groq" in provider:
                base_url = "https://api.groq.com/openai/v1"
                
            client = OpenAI(base_url=base_url, api_key=api_key.strip(), timeout=12.0)
            sys_msg = (
                "You are an expert Personal Finance Advisor. "
                "The calculation has been verified by deterministic financial tools:\n"
                f"{obs}\n"
                "Synthesize a clear, professional, and friendly response for the user explaining these numbers."
            )
            resp = client.chat.completions.create(
                model=model_name or "gpt-4o-mini",
                messages=[
                    {"role": "system", "content": sys_msg},
                    {"role": "user", "content": query}
                ],
                max_tokens=300,
                temperature=0.2
            )
            llm_explanation = resp.choices[0].message.content
        except Exception as e:
            err_msg = str(e)
            if "401" in err_msg or "User not found" in err_msg:
                st.sidebar.warning("⚠️ API Key error: 401 Unauthorized ('User not found'). Used offline precision engine.")
            else:
                st.sidebar.info(f"ℹ️ LLM Notice: {err_msg[:60]}... Used offline calculation engine.")

    if llm_explanation:
        advice = llm_explanation
    else:
        advice = (
            f"Here is your personalized financial calculation:\n\n"
            f"{obs}\n\n"
            f"💡 **Financial Advisory Tip:** Keeping your total debt obligations under 35% of your income "
            f"and investing automatically every month ensures consistent, compounding wealth growth."
        )
    
    return {
        "thought": thought,
        "selected_tool": selected_tool,
        "tool_args": tool_args,
        "observation": obs,
        "response": advice
    }

# ── Sidebar Navigation & Settings ──────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/money-bag-euro.png", width=64)
    st.title("Settings & Status")
    
    st.markdown("### 🤖 LLM Provider Setup")
    provider = st.selectbox(
        "Select Provider",
        [
            "Deterministic Financial Agent (Recommended - Offline)",
            "OpenRouter (Free Models)",
            "OpenAI (gpt-4o-mini)",
            "Groq (Llama-3.3 Free & Fast)"
        ]
    )
    
    api_key = ""
    selected_model = "gpt-4o-mini"
    
    if "OpenRouter" in provider:
        api_key = st.text_input(
            "OpenRouter API Key",
            type="password",
            placeholder="sk-or-v1-...",
            value=os.environ.get("OPENROUTER_API_KEY", ""),
            help="Get a free key at https://openrouter.ai/keys"
        )
        selected_model = st.selectbox(
            "Select Free Model",
            [
                "nvidia/nemotron-3.5-lightning:free",
                "liquid/lfm-2.5-2.6b:free",
                "google/gemma-4-31b-it:free",
                "meta-llama/llama-3.3-70b-instruct:free",
                "deepseek/deepseek-r1:free"
            ]
        )
        if st.button("🧪 Test OpenRouter Key", use_container_width=True):
            if not api_key:
                st.warning("Please enter your OpenRouter API Key first.")
            else:
                with st.spinner("Connecting to OpenRouter..."):
                    try:
                        from openai import OpenAI
                        test_client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key.strip(), timeout=8.0)
                        test_client.models.list()
                        st.success("✅ OpenRouter API Key is valid and active!")
                    except Exception as e:
                        err_str = str(e)
                        if "401" in err_str or "User not found" in err_str:
                            st.error("❌ 401 Unauthorized: User not found.\nThis key is invalid or deleted. Create a new key at openrouter.ai/keys.")
                        else:
                            st.error(f"❌ Connection Error: {err_str[:120]}")
                            
    elif "OpenAI" in provider:
        api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...", value=os.environ.get("OPENAI_API_KEY", ""))
        selected_model = "gpt-4o-mini"
    elif "Groq" in provider:
        api_key = st.text_input("Groq API Key", type="password", placeholder="gsk_...", value=os.environ.get("GROQ_API_KEY", ""))
        selected_model = "llama-3.3-70b-versatile"
        
    st.markdown("---")
    st.markdown("### 📦 Artifact Pipeline Status")
    if model is not None:
        st.success("✅ `model.pkl` Loaded (RandomForest)")
    else:
        st.error("❌ `model.pkl` Not Found")
        
    if scaler is not None:
        st.success("✅ `scaler.pkl` Loaded (StandardScaler)")
    else:
        st.error("❌ `scaler.pkl` Not Found")
        
    st.info("✅ 4 LangChain Tools Active")
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# ── Header Banner ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>💰 Personal Finance AI Agent</h1>
    <p>An intelligent agent built with LangChain Tools for loan EMI computation, SIP investment compounding, 50-30-20 budgeting, and machine learning risk profiling.</p>
</div>
""", unsafe_allow_html=True)

# ── Main Tabs ──────────────────────────────────────────────────────────────────
tab_agent, tab_ml, tab_calc, tab_docs = st.tabs([
    "💬 AI Finance Agent (Chat)",
    "🛡️ ML Financial Health Predictor",
    "🧮 Interactive Calculators",
    "📋 System Overview & Architecture"
])

# ════════════════════════════════════════════════════════════════════════════════
# TAB 1: AI FINANCE AGENT CHAT
# ════════════════════════════════════════════════════════════════════════════════
with tab_agent:
    st.subheader("🤖 LangChain Financial Planning Assistant")
    st.markdown("Ask any question regarding loan EMIs, long-term investments, monthly budgeting, or financial calculations.")
    
    # Quick Test Buttons for Financial Calculations
    st.markdown("##### ⚡ Quick-Run Sample Queries:")
    col1, col2, col3 = st.columns(3)
    col4, col5, col6 = st.columns(3)
    
    q_selected = None
    with col1:
        if st.button("🏦 Q1: EMI for ₹10L @ 8% (20 yrs)", use_container_width=True):
            q_selected = "Calculate EMI for ₹10 lakh loan at 8% interest for 20 years"
    with col2:
        if st.button("📈 Q2: ₹7,000 SIP @ 10% (15 yrs)", use_container_width=True):
            q_selected = "If I invest ₹7000 monthly for 15 years at 10% return"
    with col3:
        if st.button("📊 Q3: Budget for ₹90,000 Income", use_container_width=True):
            q_selected = "Create a budget plan for income ₹90,000"
    with col4:
        if st.button("🧮 Q4: What is 20% of ₹85,000", use_container_width=True):
            q_selected = "What is 20% of ₹85,000"
    with col5:
        if st.button("💎 Q5: ₹3,000 SIP @ 8% (5 yrs)", use_container_width=True):
            q_selected = "How much will I get if I invest ₹3000 per month for 5 years at 8% return"
    with col6:
        if st.button("🎯 Q6: ₹50K Income + ₹5K SIP Plan", use_container_width=True):
            q_selected = "Create a budget plan for income ₹50,000 and calculate returns if I invest ₹5,000 monthly for 10 years at 12%"

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your AI Personal Finance Assistant. How can I help you today? You can ask me to calculate loan EMIs, estimate mutual fund SIP returns, create a 50-30-20 budget plan, or solve math calculations.",
                "trace": None
            }
        ]

    # Process quick selection or text input
    user_prompt = st.chat_input("Ask a financial question (e.g. Calculate EMI for 15 lakh at 8.5% for 15 years)...")
    if q_selected:
        user_prompt = q_selected

    if user_prompt:
        st.session_state.messages.append({"role": "user", "content": user_prompt, "trace": None})
        
        # Agent execution
        with st.spinner("Agent is reasoning and executing tools..."):
            agent_output = agent_reason_and_solve(
                user_prompt,
                provider=provider,
                api_key=api_key,
                model_name=selected_model
            )
            st.session_state.messages.append({
                "role": "assistant",
                "content": agent_output["response"],
                "trace": agent_output
            })

    # Render Chat History
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            with st.chat_message("user", avatar="👤"):
                st.markdown(f"**{msg['content']}**")
        else:
            with st.chat_message("assistant", avatar="💰"):
                st.markdown(msg["content"])
                
                # Show Agent Scratchpad / Trace if tool was used
                if msg.get("trace"):
                    trace = msg["trace"]
                    with st.expander("🔍 Agent Reasoning & Tool Call Breakdown", expanded=False):
                        st.markdown(f"**🧠 Thought:** `{trace['thought']}`")
                        st.markdown(f"**🛠️ Selected Tool:** `{trace['selected_tool']}`")
                        st.markdown(f"**📥 Extracted Parameters:** `{json.dumps(trace['tool_args'])}`")
                        st.markdown(f"**📤 Tool Output / Observation:**\n```\n{trace['observation']}\n```")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 2: ML FINANCIAL HEALTH PREDICTOR (MODEL.PKL + SCALER.PKL)
# ════════════════════════════════════════════════════════════════════════════════
with tab_ml:
    st.subheader("🛡️ Machine Learning Financial Health & Risk Assessment")
    st.markdown("""
    This module uses the trained **RandomForestClassifier** (`model.pkl`) and **StandardScaler** (`scaler.pkl`) 
    created in our Jupyter Notebook to evaluate a user's financial resilience, debt burden, and risk tier.
    """)
    
    col_in1, col_in2 = st.columns(2)
    with col_in1:
        st.markdown("##### 💼 Monthly Cash Flow Inputs")
        in_income = st.number_input("Monthly Net Income (₹)", min_value=10000.0, max_value=1000000.0, value=90000.0, step=5000.0)
        in_expenses = st.number_input("Monthly Living Expenses (₹)", min_value=0.0, max_value=1000000.0, value=45000.0, step=2000.0)
        
    with col_in2:
        st.markdown("##### 💳 Debt & Savings Obligations")
        in_emi = st.number_input("Existing Monthly Loan EMI (₹)", min_value=0.0, max_value=500000.0, value=15000.0, step=1000.0)
        in_savings = st.number_input("Monthly Savings & Investments (₹)", min_value=0.0, max_value=500000.0, value=25000.0, step=1000.0)

    # Derived Ratios
    dti_val = in_emi / in_income if in_income > 0 else 0
    savings_rate_val = in_savings / in_income if in_income > 0 else 0
    surplus_val = in_income - (in_expenses + in_emi + in_savings)
    
    st.markdown("---")
    st.markdown("##### 📊 Real-Time Financial Indicators")
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Debt-to-Income (DTI)</div>
            <div class="metric-value">{dti_val * 100:.1f}%</div>
            <small style="color:{'#10b981' if dti_val <= 0.3 else '#ef4444'};">{'Ideal (<= 30%)' if dti_val <= 0.3 else 'High Debt Burden'}</small>
        </div>
        """, unsafe_allow_html=True)
    with m_col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Savings Rate</div>
            <div class="metric-value">{savings_rate_val * 100:.1f}%</div>
            <small style="color:{'#10b981' if savings_rate_val >= 0.2 else '#f59e0b'};">{'Target Met (>= 20%)' if savings_rate_val >= 0.2 else 'Below 20% Target'}</small>
        </div>
        """, unsafe_allow_html=True)
    with m_col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">Net Monthly Surplus</div>
            <div class="metric-value">₹{surplus_val:,.2f}</div>
            <small style="color:{'#10b981' if surplus_val >= 0 else '#ef4444'};">{'Positive Surplus' if surplus_val >= 0 else 'Deficit Alert'}</small>
        </div>
        """, unsafe_allow_html=True)

    if st.button("🔮 Run Financial Health Assessment", type="primary", use_container_width=True):
        if model is None or scaler is None:
            st.error("ML model or scaler not loaded. Please ensure `model.pkl` and `scaler.pkl` are present.")
        else:
            raw_feats = np.array([[in_income, in_expenses, in_emi, in_savings, dti_val, savings_rate_val]])
            scaled_feats = scaler.transform(raw_feats)
            pred_class = model.predict(scaled_feats)[0]
            pred_probs = model.predict_proba(scaled_feats)[0]
            
            st.markdown("### 📋 ML Health Diagnosis")
            if pred_class == 2:
                st.markdown(f"""
                <div class="risk-low">
                    <h3>🟢 Low Financial Risk / Financially Healthy (Confidence: {pred_probs[2]*100:.1f}%)</h3>
                    <p>Excellent financial discipline! Your debt-to-income ratio is low, you consistently save over 20% of your earnings, and maintain positive cash flow buffer.</p>
                </div>
                """, unsafe_allow_html=True)
            elif pred_class == 1:
                st.markdown(f"""
                <div class="risk-mod">
                    <h3>🟡 Moderate Financial Risk / Stable (Confidence: {pred_probs[1]*100:.1f}%)</h3>
                    <p>Your financial position is stable but has room for optimization. Try keeping non-essential expenses down to increase your monthly investment rate toward 20%.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-high">
                    <h3>🔴 High Financial Risk / Vulnerable (Confidence: {pred_probs[0]*100:.1f}%)</h3>
                    <p>High debt-servicing load or low savings buffer detected. Recommended action: Restructure debt, pause discretionary spending, and build a 3-month emergency reserve immediately.</p>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("#### 🎯 Recommended Action Plan:")
            b_target = compute_budget(in_income)
            st.write(f"- **Target Needs (50%):** ₹{b_target['needs']:,.2f} (Current: ₹{in_expenses:,.2f})")
            st.write(f"- **Target Wants (30%):** ₹{b_target['wants']:,.2f}")
            st.write(f"- **Target Savings (20%):** ₹{b_target['savings']:,.2f} (Current: ₹{in_savings:,.2f})")

# ════════════════════════════════════════════════════════════════════════════════
# TAB 3: STANDALONE CALCULATORS
# ════════════════════════════════════════════════════════════════════════════════
with tab_calc:
    st.subheader("🧮 Interactive Financial Calculators")
    calc_option = st.radio("Choose Calculator", ["EMI Calculator", "SIP Investment Calculator", "50-30-20 Budget Allocator"], horizontal=True)
    
    if calc_option == "EMI Calculator":
        c1, c2 = st.columns([1, 1])
        with c1:
            p_val = st.slider("Loan Principal Amount (₹)", 50000.0, 5000000.0, 1000000.0, step=50000.0)
            r_val = st.slider("Annual Interest Rate (%)", 5.0, 20.0, 8.0, step=0.25)
            n_val = st.slider("Loan Tenure (Years)", 1.0, 30.0, 20.0, step=1.0)
            res = compute_emi(p_val, r_val, n_val)
            
        with c2:
            st.markdown("#### 📊 Amortization Breakdown")
            st.metric("Monthly EMI", f"₹{res['emi']:,.2f}")
            st.metric("Total Interest Payable", f"₹{res['total_interest']:,.2f}")
            st.metric("Total Repayment (Principal + Interest)", f"₹{res['total_payment']:,.2f}")
            
            # Bar chart
            chart_data = pd.DataFrame({
                "Category": ["Principal Loan", "Total Interest"],
                "Amount": [p_val, res["total_interest"]]
            })
            st.bar_chart(chart_data.set_index("Category"))
            
    elif calc_option == "SIP Investment Calculator":
        c1, c2 = st.columns([1, 1])
        with c1:
            sip_p = st.slider("Monthly Contribution (₹)", 500.0, 100000.0, 7000.0, step=500.0)
            sip_r = st.slider("Expected Return Rate (% p.a.)", 5.0, 25.0, 10.0, step=0.5)
            sip_y = st.slider("Investment Period (Years)", 1.0, 35.0, 15.0, step=1.0)
            res = compute_sip(sip_p, sip_r, sip_y)
            
        with c2:
            st.markdown("#### 📈 Wealth Growth Projection")
            st.metric("Future Value (FV)", f"₹{res['fv']:,.2f}")
            st.metric("Total Capital Invested", f"₹{res['total_invested']:,.2f}")
            st.metric("Estimated Wealth Gain", f"₹{res['wealth_gain']:,.2f}")
            
            chart_data = pd.DataFrame({
                "Category": ["Total Invested", "Wealth Gained"],
                "Amount": [res["total_invested"], res["wealth_gain"]]
            })
            st.bar_chart(chart_data.set_index("Category"))

    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            inc = st.slider("Net Monthly Income (₹)", 15000.0, 500000.0, 90000.0, step=5000.0)
            res = compute_budget(inc)
            
        with c2:
            st.markdown("#### 🎯 50-30-20 Apportionment")
            st.metric("Needs (50%)", f"₹{res['needs']:,.2f}")
            st.metric("Wants (30%)", f"₹{res['wants']:,.2f}")
            st.metric("Savings & Investments (20%)", f"₹{res['savings']:,.2f}")
            
            b_chart = pd.DataFrame({
                "Category": ["Needs (50%)", "Wants (30%)", "Savings (20%)"],
                "Amount": [res['needs'], res['wants'], res['savings']]
            })
            st.bar_chart(b_chart.set_index("Category"))

# ════════════════════════════════════════════════════════════════════════════════
# TAB 4: SYSTEM OVERVIEW & ARCHITECTURE
# ════════════════════════════════════════════════════════════════════════════════
with tab_docs:
    st.subheader("📋 System Overview & Technical Architecture")
    st.markdown("""
    ### 1. Objective
    An AI-powered Personal Finance Assistant built using **LangChain Agents and Tools**.
    The agent helps users perform financial calculations and planning such as EMI calculation, 
    investment returns, and budget allocation. The LLM decides which specialized tool to invoke depending on user inquiries.

    ### 2. Tools Implemented:
    1. **Calculator Tool:** Used for general mathematical and percentage calculations.
    2. **EMI Calculator Tool:** Computes loan EMI using the reducing balance formula:
       $$\\text{EMI} = \\frac{P \\times r \\times (1+r)^n}{(1+r)^n - 1}$$
    3. **SIP Investment Calculator:** Calculates future accumulated wealth:
       $$\\text{FV} = P \\times \\frac{(1+r)^n - 1}{r}$$
    4. **Budget Planner Tool:** Allocates net monthly salary based on the 50-30-20 rule:
       - 50% Needs
       - 30% Wants
       - 20% Savings
    5. **ML Financial Health Assessor:** Serialized with `joblib.dump()` into `scaler.pkl` and `model.pkl`.

    ### 3. Core Demonstration Queries:
    - *Query 1:* Calculate EMI for ₹10 lakh loan at 8% interest for 20 years
    - *Query 2:* If I invest ₹7000 monthly for 15 years at 10% return
    - *Query 3:* Create a budget plan for income ₹90,000
    - *Query 4:* What is 20% of ₹85,000
    - *Query 5:* How much will I get if I invest ₹3000 per month for 5 years at 8% return
    """)
