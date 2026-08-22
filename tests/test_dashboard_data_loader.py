"""Tests for dashboard.data_loader.fetch_phase9a_summary (mocked psycopg)."""
from __future__ import annotations

import psycopg
import pytest

from dashboard import data_loader


class FakeCursor:
    def __init__(self, ones, alls):
        self._ones = list(ones)
        self._alls = list(alls)
        self._oi = 0
        self._ai = 0

    def execute(self, sql, params=None):
        pass

    def fetchone(self):
        v = self._ones[self._oi]
        self._oi += 1
        return v

    def fetchall(self):
        v = self._alls[self._ai]
        self._ai += 1
        return v

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class FakeConn:
    def __init__(self, cursor):
        self._cursor = cursor

    def cursor(self):
        return self._cursor

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def _patch(monkeypatch, ones, alls):
    cursor = FakeCursor(ones, alls)
    monkeypatch.setattr(data_loader, "get_db_url", lambda: "postgresql://dummy")
    monkeypatch.setattr(psycopg, "connect", lambda *a, **kw: FakeConn(cursor))


def test_table_not_yet_created(monkeypatch):
    _patch(monkeypatch, ones=[{"exists": None}], alls=[])
    summary = data_loader.fetch_phase9a_summary()
    assert summary["status_message"] == "Table not yet created"
    assert summary["total_breakouts"] == 0
    assert summary["hit_rate_t1"] is None


def test_collecting_data_status(monkeypatch):
    _patch(
        monkeypatch,
        ones=[
            {"exists": "phase9a_forward_returns"},  # table exists
            {"n": 12, "up": 7, "down": 5},          # breakouts
            {"filled": 20, "pending": 10},          # filled/pending
        ],
        alls=[[{"horizon": 1, "n": 5, "hit_rate": 0.6}]],  # < 10 -> None
    )
    summary = data_loader.fetch_phase9a_summary()
    assert summary["total_breakouts"] == 12
    assert summary["up_breakouts"] == 7
    assert summary["down_breakouts"] == 5
    assert summary["forward_returns_filled"] == 20
    assert summary["hit_rate_t1"] is None  # only 5 filled < 10
    assert "Collecting data" in summary["status_message"]


def test_enough_data_with_hit_rates(monkeypatch):
    _patch(
        monkeypatch,
        ones=[
            {"exists": "phase9a_forward_returns"},
            {"n": 40, "up": 22, "down": 18},
            {"filled": 110, "pending": 10},
        ],
        alls=[[{"horizon": 1, "n": 40, "hit_rate": 0.625},
               {"horizon": 2, "n": 40, "hit_rate": 0.55}]],
    )
    summary = data_loader.fetch_phase9a_summary()
    assert summary["total_breakouts"] == 40
    assert summary["hit_rate_t1"] == pytest.approx(0.625)
    assert summary["hit_rate_t2"] == pytest.approx(0.55)
    assert summary["status_message"] == "Enough data — ready for analysis"
