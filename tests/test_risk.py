"""tests/test_risk.py — Risk engine and circuit breaker tests."""
import pytest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk.risk_engine import PortfolioRiskEngine
from src.risk.circuit_breaker import DailyCircuitBreaker


class TestDailyCircuitBreaker:
    def setup_method(self):
        self.cb = DailyCircuitBreaker({
            "max_daily_loss_pct": 1.5,
            "max_drawdown_pct": 5.0,
        })

    def test_allows_normal_trading(self):
        """Should allow trading when within limits."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        allowed, reason = self.cb.check(99000)  # only -1% daily loss
        assert allowed, f"Should allow trading: {reason}"

    def test_blocks_on_daily_loss_breach(self):
        """Should block trading when daily loss exceeds limit."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        # -2% daily loss, exceeds 1.5% limit
        allowed, reason = self.cb.check(98000)
        assert not allowed, "Should block trading on daily loss breach"
        assert "daily" in reason.lower() or "loss" in reason.lower()

    def test_blocks_on_max_drawdown_breach(self):
        """Should block trading when max drawdown exceeds limit."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(110000)  # peak was 110k
        # Now at 103k: drawdown from peak = 6.4%, exceeds 5% limit
        allowed, reason = self.cb.check(103000)
        assert not allowed, "Should block trading on max drawdown breach"

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        # Trigger block
        self.cb.check(97000)
        # Reset
        self.cb.reset()
        # Now within limits after reset
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        allowed, _ = self.cb.check(99500)
        assert allowed, "After reset, should allow trading again"

    def test_get_status_returns_dict(self):
        """get_status must return a dict with required keys."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        self.cb.check(99000)
        status = self.cb.get_status()
        assert isinstance(status, dict)
        assert "blocked" in status

    def test_is_active_when_blocked(self):
        """is_active property should be True when circuit breaker has fired."""
        self.cb.reset()
        self.cb.set_day_start(100000)
        self.cb.update_peak(100000)
        allowed, _ = self.cb.check(97000)
        if not allowed:
            assert self.cb.is_active, "is_active should be True when blocked"


class TestPortfolioRiskEngine:
    def setup_method(self):
        self.engine = PortfolioRiskEngine({
            "max_position_pct": 25.0,
            "max_sector_exposure_pct": 50.0,
        })

    def test_position_within_limit_allowed(self):
        """Position within max_position_pct should be allowed."""
        self.engine.update_portfolio_state(100000, 80000, {}, 100000, 100000)
        allowed, reason = self.engine.check_position_allowed("TCS", 20000, 100000)
        assert allowed, f"Should allow position within 25%: {reason}"

    def test_position_exceeding_limit_rejected(self):
        """Position exceeding max_position_pct should be rejected."""
        self.engine.update_portfolio_state(100000, 80000, {}, 100000, 100000)
        # 30% of 100000 = 30000 > 25% limit
        allowed, reason = self.engine.check_position_allowed("TCS", 30000, 100000)
        assert not allowed, "Should reject position exceeding 25%"

    def test_risk_score_in_bounds(self):
        """Risk score must be in [0, 100]."""
        self.engine.update_portfolio_state(95000, 50000, {}, 100000, 100000)
        score = self.engine.get_risk_score()
        assert 0 <= score <= 100, f"Risk score {score} out of [0,100]"

    def test_get_risk_summary_keys(self):
        """Risk summary must contain required keys."""
        self.engine.update_portfolio_state(100000, 80000, {}, 100000, 100000)
        summary = self.engine.get_risk_summary()
        assert isinstance(summary, dict)
        # Must have at least risk_score
        assert "risk_score" in summary
