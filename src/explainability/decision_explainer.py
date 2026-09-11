"""
Generates rule-based explanations for RL actions.
"""
import json

class DecisionExplainer:
    def __init__(self):
        pass

    def explain(
        self, 
        ticker: str, 
        action_name: str, 
        target_allocation: float, 
        confidence: float, 
        regime_name: str, 
        risk_score: float, 
        q_values: dict, 
        state_dict: dict, 
        risk_checks: dict
    ) -> dict:
        
        reasons = []
        
        # Trend
        if state_dict.get("EMA_slope", 0) > 0.001:
            reasons.append("Short-term trend is strongly positive (EMA sloping up).")
        elif state_dict.get("EMA_slope", 0) < -0.001:
            reasons.append("Short-term trend is strongly negative (EMA sloping down).")
            
        # Momentum
        rsi = state_dict.get("RSI", 50)
        if rsi > 70:
            reasons.append("RSI indicates overbought conditions (>70).")
        elif rsi < 30:
            reasons.append("RSI indicates oversold conditions (<30).")
            
        macd_hist = state_dict.get("MACD_hist", 0)
        if macd_hist > 0:
            reasons.append("MACD histogram is positive (bullish momentum).")
        elif macd_hist < 0:
            reasons.append("MACD histogram is negative (bearish momentum).")
            
        # Regime context
        reasons.append(f"Market regime classified as {regime_name}.")
        
        # Confidence
        if confidence > 0.8:
            reasons.append(f"High confidence in model prediction ({confidence*100:.1f}%).")
        elif confidence < 0.6:
            reasons.append(f"Low confidence in model prediction ({confidence*100:.1f}%), limiting exposure.")
            
        # Circuit breakers
        if "BLOCKED" in risk_checks.get("circuit_breaker", ""):
            reasons.append(f"Action overridden by Risk Engine: {risk_checks['circuit_breaker']}")
            
        explanation = {
            "symbol": ticker,
            "action": action_name,
            "target_allocation_pct": target_allocation * 100,
            "confidence_pct": confidence * 100,
            "regime": regime_name,
            "risk_score": risk_score,
            "primary_reasons": reasons,
            "disclaimer": "This is an AI-generated explanation. Not financial advice. Trading involves risk."
        }
        
        return explanation

    def format_human_readable(self, explanation: dict) -> str:
        lines = [
            f"AI Decision: {explanation['action']} ({explanation['target_allocation_pct']}% allocation)",
            f"Confidence: {explanation['confidence_pct']:.1f}% | Regime: {explanation['regime']} | Risk Score: {explanation['risk_score']:.1f}",
            "",
            "Why did the AI make this decision?"
        ]
        for idx, reason in enumerate(explanation["primary_reasons"], 1):
            lines.append(f"  {idx}. {reason}")
            
        lines.append("")
        lines.append(f"⚠️ {explanation['disclaimer']}")
        return "\n".join(lines)

