"""
Risk Profiler Agent — Quiz → Quantified Risk Score
Converts user profile inputs into a 0-100 risk score.
"""


class RiskProfiler:
    """Converts age, horizon, goal, and risk tolerance into a quantified
    risk score (0-100) with reasoning. Higher score = more aggressive."""

    NAME = "Risk Profiler"

    # Goal aggressiveness mapping
    GOAL_SCORES = {
        "Wealth Growth":              70,
        "Retirement":                 55,
        "Passive Income / Dividends": 40,
        "Short-term Gains":           80,
        "Child Education":            50,
    }

    RISK_LEVEL_SCORES = {
        "Conservative": 25,
        "Moderate":     55,
        "Aggressive":   85,
    }

    def analyze(self, age: int, horizon: int, risk_level: str, goal: str) -> dict:
        """
        Compute a risk score from user profile.

        Parameters:
            age        – investor's age (18-70)
            horizon    – investment horizon in years
            risk_level – "Conservative" / "Moderate" / "Aggressive"
            goal       – financial goal string

        Returns:
            {
                "risk_score": 0-100,
                "category": "Conservative" / "Moderate" / "Aggressive",
                "max_equity_pct": float,
                "reasoning": str,
                "components": { ... }
            }
        """
        reasoning_parts = []
        scores = []

        # ── Age component (younger = can take more risk) ─────────────────────
        # 18 → 90, 70 → 20, linear interpolation
        age_score = max(20, min(90, 90 - (age - 18) * (70 / 52)))
        age_score = int(age_score)
        scores.append(age_score)
        reasoning_parts.append(
            f"Age {age} → score {age_score}/100 "
            f"({'long runway for recovery' if age < 35 else 'shorter horizon, prefer stability' if age > 50 else 'balanced phase'})"
        )

        # ── Horizon component (longer = more risk capacity) ──────────────────
        if horizon >= 10:
            horizon_score = 85
            reasoning_parts.append(f"Horizon {horizon}yr → score 85/100 (long-term, can ride volatility)")
        elif horizon >= 5:
            horizon_score = 65
            reasoning_parts.append(f"Horizon {horizon}yr → score 65/100 (medium-term)")
        elif horizon >= 3:
            horizon_score = 45
            reasoning_parts.append(f"Horizon {horizon}yr → score 45/100 (short-medium term)")
        else:
            horizon_score = 25
            reasoning_parts.append(f"Horizon {horizon}yr → score 25/100 (short-term, prefer safety)")
        scores.append(horizon_score)

        # ── Stated risk tolerance ────────────────────────────────────────────
        risk_score = self.RISK_LEVEL_SCORES.get(risk_level, 55)
        scores.append(risk_score)
        reasoning_parts.append(f"Stated risk tolerance: {risk_level} → score {risk_score}/100")

        # ── Goal component ───────────────────────────────────────────────────
        goal_score = self.GOAL_SCORES.get(goal, 55)
        scores.append(goal_score)
        reasoning_parts.append(f"Goal '{goal}' → score {goal_score}/100")

        # ── Weighted aggregate ───────────────────────────────────────────────
        # Risk tolerance and age get higher weight
        final_score = int(
            0.30 * risk_score +
            0.25 * age_score +
            0.25 * horizon_score +
            0.20 * goal_score
        )
        final_score = max(0, min(100, final_score))

        # Category
        if final_score >= 65:
            category = "Aggressive"
            max_equity = 0.90
        elif final_score >= 40:
            category = "Moderate"
            max_equity = 0.70
        else:
            category = "Conservative"
            max_equity = 0.50

        return {
            "risk_score": final_score,
            "category": category,
            "max_equity_pct": max_equity,
            "reasoning": " | ".join(reasoning_parts),
            "components": {
                "age": {"value": age, "score": age_score},
                "horizon": {"value": horizon, "score": horizon_score},
                "risk_level": {"value": risk_level, "score": risk_score},
                "goal": {"value": goal, "score": goal_score},
            },
        }
