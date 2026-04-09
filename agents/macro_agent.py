"""
Macro Agent — RBI Repo Rate · CPI Inflation · GDP Growth
Provides a macro-environment assessment that adjusts overall risk appetite.
Uses hardcoded recent data (updated periodically) for reliability.
"""


class MacroAgent:
    """Evaluates the Indian macroeconomic environment using RBI repo rate,
    CPI inflation, and GDP growth. Returns FAVORABLE / NEUTRAL / UNFAVORABLE
    with a 0-100 score and suggested adjustments."""

    NAME = "Macro"

    # ── Latest Indian macro data (update these periodically) ─────────────────
    # Source: RBI, MOSPI, World Bank — as of Q1 FY2026-27
    MACRO_DATA = {
        "rbi_repo_rate": 6.25,          # RBI repo rate (%)  — Apr 2026
        "rbi_prev_rate": 6.50,          # Previous rate for direction
        "cpi_inflation": 4.5,           # CPI YoY (%) — latest
        "rbi_target_inflation": 4.0,    # RBI target
        "rbi_tolerance_upper": 6.0,     # RBI upper band
        "gdp_growth": 6.5,             # Real GDP growth (%) — latest quarter
        "gdp_prev": 6.7,              # Previous quarter for trend
        "iip_growth": 4.2,            # Industrial production growth (%)
        "fii_flow_trend": "MIXED",     # FII flow direction: INFLOW / OUTFLOW / MIXED
        "usd_inr": 86.5,              # USD/INR exchange rate
        "crude_oil_usd": 75.0,        # Brent crude ($/barrel)
    }

    def _assess_rates(self) -> tuple[int, str]:
        """Score based on RBI repo rate direction and level."""
        rate = self.MACRO_DATA["rbi_repo_rate"]
        prev = self.MACRO_DATA["rbi_prev_rate"]

        if rate < prev:
            # Rate cut — positive for equity
            score = 75
            reason = f"RBI repo rate cut to {rate}% from {prev}% (dovish, positive for equities)"
        elif rate == prev:
            score = 55
            reason = f"RBI repo rate unchanged at {rate}% (neutral stance)"
        else:
            score = 35
            reason = f"RBI repo rate hiked to {rate}% from {prev}% (hawkish, negative for equities)"

        # Additional penalty/bonus based on absolute level
        if rate > 7.0:
            score -= 10
            reason += " — high absolute rate"
        elif rate < 5.5:
            score += 10
            reason += " — accommodative level"

        return max(0, min(100, score)), reason

    def _assess_inflation(self) -> tuple[int, str]:
        """Score based on CPI inflation vs RBI target band."""
        cpi = self.MACRO_DATA["cpi_inflation"]
        target = self.MACRO_DATA["rbi_target_inflation"]
        upper = self.MACRO_DATA["rbi_tolerance_upper"]

        if cpi <= target:
            score = 80
            reason = f"CPI inflation {cpi}% at/below RBI target {target}% (ideal for equity)"
        elif cpi <= (target + upper) / 2:
            score = 60
            reason = f"CPI inflation {cpi}% within comfort zone (manageable)"
        elif cpi <= upper:
            score = 40
            reason = f"CPI inflation {cpi}% near upper tolerance {upper}% (watch for rate hikes)"
        else:
            score = 20
            reason = f"CPI inflation {cpi}% above RBI tolerance {upper}% (risk of tightening)"

        return score, reason

    def _assess_gdp(self) -> tuple[int, str]:
        """Score based on GDP growth rate and trend."""
        gdp = self.MACRO_DATA["gdp_growth"]
        prev = self.MACRO_DATA["gdp_prev"]

        if gdp >= 7.0:
            score = 85
            reason = f"GDP growth {gdp}% — strong (India outperforming peers)"
        elif gdp >= 6.0:
            score = 70
            reason = f"GDP growth {gdp}% — healthy"
        elif gdp >= 4.5:
            score = 50
            reason = f"GDP growth {gdp}% — moderate, below potential"
        else:
            score = 30
            reason = f"GDP growth {gdp}% — sluggish"

        # Trend adjustment
        if gdp > prev:
            score = min(100, score + 5)
            reason += " (accelerating ↑)"
        elif gdp < prev:
            score = max(0, score - 5)
            reason += " (decelerating ↓)"

        return score, reason

    def _assess_external(self) -> tuple[int, str]:
        """Score based on external factors: crude oil, FII flows, rupee."""
        crude = self.MACRO_DATA["crude_oil_usd"]
        fii = self.MACRO_DATA["fii_flow_trend"]
        usd_inr = self.MACRO_DATA["usd_inr"]

        scores = []
        reasons = []

        # Crude oil — India is net importer, high crude is negative
        if crude < 65:
            scores.append(80)
            reasons.append(f"Crude oil ${crude}/bbl — favorable for India")
        elif crude < 80:
            scores.append(60)
            reasons.append(f"Crude oil ${crude}/bbl — manageable")
        elif crude < 100:
            scores.append(40)
            reasons.append(f"Crude oil ${crude}/bbl — pressure on trade deficit")
        else:
            scores.append(20)
            reasons.append(f"Crude oil ${crude}/bbl — significant headwind")

        # FII flows
        if fii == "INFLOW":
            scores.append(75)
            reasons.append("FII flows: net inflow (positive for markets)")
        elif fii == "OUTFLOW":
            scores.append(30)
            reasons.append("FII flows: net outflow (selling pressure)")
        else:
            scores.append(50)
            reasons.append("FII flows: mixed")

        avg_score = int(sum(scores) / len(scores))
        return avg_score, " | ".join(reasons)

    # ── Public API ───────────────────────────────────────────────────────────
    def analyze(self) -> dict:
        """
        Assess the Indian macroeconomic environment.

        Returns:
            {
                "environment": "FAVORABLE" / "NEUTRAL" / "UNFAVORABLE",
                "score": 0-100,
                "reasoning": str,
                "components": { ... },
                "adjustments": { "equity_tilt": float, "debt_tilt": float }
            }
        """
        rate_score, rate_reason = self._assess_rates()
        infl_score, infl_reason = self._assess_inflation()
        gdp_score, gdp_reason = self._assess_gdp()
        ext_score, ext_reason = self._assess_external()

        # Weighted average: GDP and inflation matter most for equities
        final_score = int(
            0.20 * rate_score +
            0.25 * infl_score +
            0.30 * gdp_score +
            0.25 * ext_score
        )
        final_score = max(0, min(100, final_score))

        if final_score >= 65:
            environment = "FAVORABLE"
            equity_tilt = 1.10   # 10% boost to equity conviction
            debt_tilt = 0.90
        elif final_score >= 40:
            environment = "NEUTRAL"
            equity_tilt = 1.00
            debt_tilt = 1.00
        else:
            environment = "UNFAVORABLE"
            equity_tilt = 0.85   # 15% reduction in equity conviction
            debt_tilt = 1.15

        all_reasons = [rate_reason, infl_reason, gdp_reason, ext_reason]

        return {
            "environment": environment,
            "score": final_score,
            "reasoning": " | ".join(all_reasons),
            "components": {
                "rbi_rate": {"score": rate_score, "detail": rate_reason},
                "inflation": {"score": infl_score, "detail": infl_reason},
                "gdp": {"score": gdp_score, "detail": gdp_reason},
                "external": {"score": ext_score, "detail": ext_reason},
            },
            "adjustments": {
                "equity_tilt": equity_tilt,
                "debt_tilt": debt_tilt,
            },
            "macro_data": self.MACRO_DATA,
        }
