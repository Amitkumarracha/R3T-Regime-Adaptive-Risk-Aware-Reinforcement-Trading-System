"""Streamlit Dashboard for R3T."""

import streamlit as st
import pandas as pd
import json
import os
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="R3T Trading Dashboard", layout="wide")

def load_data():
    metrics = {}
    if os.path.exists("results/metrics/r3t.json"):
        with open("results/metrics/r3t.json") as f:
            metrics = json.load(f)
            
    explanations = []
    if os.path.exists("results/trades/r3t_explanations.json"):
        with open("results/trades/r3t_explanations.json") as f:
            explanations = json.load(f)
            
    equity = None
    if os.path.exists("results/r3t_equity_curve.csv"):
        equity = pd.read_csv("results/r3t_equity_curve.csv")
        
    return metrics, explanations, equity

def main():
    st.title("🤖 R3T Autonomous Trading Dashboard")
    
    metrics, explanations, equity = load_data()
    
    if not metrics and not equity:
        st.warning("No data found. Run `python -m experiments.train_r3t` first.")
        return
        
    st.sidebar.header("System Status")
    st.sidebar.success("Mode: PAPER TRADING")
    if explanations:
        latest = explanations[-1]
        st.sidebar.info(f"Current Regime: {latest.get('regime', 'UNKNOWN')}")
        
    # Layout
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Return", f"{metrics.get('total_return_pct', 0):.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{metrics.get('sharpe_ratio', 0):.2f}")
    with col3:
        st.metric("Max Drawdown", f"{metrics.get('max_drawdown_pct', 0):.2f}%")
    with col4:
        st.metric("Win Rate", f"{metrics.get('win_rate_pct', 0):.2f}%")
        
    # Equity Curve
    if equity is not None:
        st.subheader("Equity Curve")
        fig = px.line(equity, y="portfolio_value", title="Portfolio Value over Time")
        st.plotly_chart(fig, use_container_width=True)
        
    # AI Decision Panel
    st.subheader("Latest AI Decisions")
    if explanations:
        for exp in reversed(explanations[-5:]):
            with st.expander(f"{exp['action']} - {exp['confidence_pct']:.1f}% Confidence (Risk Score: {exp['risk_score']:.1f})"):
                for reason in exp['primary_reasons']:
                    st.write(f"• {reason}")
                st.caption(exp['disclaimer'])
    else:
        st.info("No trade explanations available.")

if __name__ == "__main__":
    main()

