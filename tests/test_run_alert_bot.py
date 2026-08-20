"""Step 6 tests - run_alert_bot entry point.

Covers: argparse defaults and every flag, environment validation (token/chat
presence, directory creation), status printing on empty and populated DBs,
the no-telegram placeholder config, and main() routing for every mode
(status/dry-run/once/calibrate/forever) including missing-env exit code 1,
the --db override reaching SchedulerConfig, and the guarantee that dry-run
never touches the Telegram API.

Deterministic, no network, no real Telegram: scheduler/telegram functions are
patched at the run_alert_bot module level, DBs live in tmp_path, env vars are
controlled via monkeypatch, and load_dotenv is a no-op in every main() test
(a real .env on disk can never influence results).
"""
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

# --- same path bootstrap as scripts/run_alert_bot.py itself -----------------
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import scripts.run_alert_bot as rab  # noqa: E402
from kronos_trading.alerts.prediction_logger import (  # noqa: E402
    initialize_db,
    log_prediction,
    update_actual,
)
from kronos_trading.alerts.scheduler import CycleResult  # noqa: E402
from kronos_trading.alerts.telegram_sender import TelegramConfig  # noqa: E402
from kronos_trading.alerts.har_forecaster import HarForecast  # noqa: E402

UTC = timezone.utc
MOD = "scripts.run_alert_bot"


def make_cycle_result():
    return CycleResult(timestamp="2024-01-15T15:00:00Z", success=True,
                       assets_processed=["BTC/USDT", "ETH/USDT"], errors=[],
                       forecasts={}, breakouts={}, send_results={},
                       duration_seconds=0.05)


def clean_cycle():
    return make_cycle_result()


# ---------------------------------------------------------------------------
# parse_args
# ---------------------------------------------------------------------------

class TestParseArgs:
    def test_parse_args_defaults(self):
        args = rab.parse_args([])
        assert args.dry_run is False
        assert args.once is False
        assert args.calibrate is False
        assert args.status is False
        assert args.db is None
        assert args.log_level == "INFO"
        assert args.no_telegram is False

    def test_parse_args_dry_run(self):
        assert rab.parse_args(["--dry-run"]).dry_run is True

    def test_parse_args_once(self):
        assert rab.parse_args(["--once"]).once is True

    def test_parse_args_calibrate(self):
        assert rab.parse_args(["--calibrate"]).calibrate is True

    def test_parse_args_status(self):
        assert rab.parse_args(["--status"]).status is True

    def test_parse_args_db_override(self):
        assert rab.parse_args(["--db", "/tmp/test.db"]).db == "/tmp/test.db"

    def test_parse_args_log_level(self):
        assert rab.parse_args(["--log-level", "DEBUG"]).log_level == "DEBUG"

    def test_parse_args_no_telegram(self):
        assert rab.parse_args(["--no-telegram"]).no_telegram is True


# ---------------------------------------------------------------------------
# validate_environment
# ---------------------------------------------------------------------------

class TestValidateEnvironment:
    def test_validate_environment_no_telegram_mode(self, tmp_path):
        (tmp_path / "data").mkdir()
        (tmp_path / "data" / "db").mkdir()
        (tmp_path / "logs").mkdir()
        errors = rab.validate_environment(no_telegram=True, project_root=tmp_path)
        assert errors == []

    def test_validate_environment_missing_token(self, tmp_path, monkeypatch):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "chat_1")
        errors = rab.validate_environment(no_telegram=False, project_root=tmp_path)
        assert any("TELEGRAM_BOT_TOKEN" in e for e in errors)
        assert not any("TELEGRAM_CHAT_ID" in e for e in errors)

    def test_validate_environment_missing_chat_id(self, tmp_path, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "token_1")
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        errors = rab.validate_environment(no_telegram=False, project_root=tmp_path)
        assert any("TELEGRAM_CHAT_ID" in e for e in errors)
        assert not any("TELEGRAM_BOT_TOKEN" in e for e in errors)

    def test_validate_environment_creates_dirs(self, tmp_path):
        (tmp_path / "data").mkdir()  # data/ exists, data/db/ and logs/ missing
        errors = rab.validate_environment(no_telegram=True, project_root=tmp_path)
        assert errors == []
        assert (tmp_path / "data" / "db").is_dir()
        assert (tmp_path / "logs").is_dir()


# ---------------------------------------------------------------------------
# print_status
# ---------------------------------------------------------------------------

class TestPrintStatus:
    def test_print_status_empty_db(self, tmp_path, capsys):
        db = str(tmp_path / "status.db")
        initialize_db(db)
        rab.print_status(db)  # must not raise
        out = capsys.readouterr().out
        assert "HAR Alert Bot — Status Report" in out
        assert "Last prediction: none" in out
        assert "Total predictions logged: 0" in out
        assert "HAR MAE:         N/A" in out

    def test_print_status_with_data(self, tmp_path, capsys):
        db = str(tmp_path / "status.db")
        initialize_db(db)
        for i in range(5):
            ts = (datetime(2024, 1, 15, tzinfo=UTC) + timedelta(hours=i)) \
                .strftime("%Y-%m-%dT%H:%M:%SZ")
            log_prediction(db, ts, "BTC/USDT", "1h",
                           HarForecast(100.0 + i, (1.0, 0.5, 0.3, 0.1), 178))
        update_actual(db, "2024-01-15T00:00:00Z", "BTC/USDT", "1h", 101.0)
        update_actual(db, "2024-01-15T01:00:00Z", "BTC/USDT", "1h", 102.0)
        update_actual(db, "2024-01-15T02:00:00Z", "BTC/USDT", "1h", 103.0)
        rab.print_status(db)
        out = capsys.readouterr().out
        assert "Total predictions logged: 5" in out
        assert "Completed (actual filled): 3" in out
        assert "Pending (awaiting close): 2" in out
        assert "Last prediction: 2024-01-15T04:00:00Z" in out
        assert "BTC/USDT 1h:" in out and "ETH/USDT 1h:" in out
        # Only 3 completed rows -> calibration N/A (needs 24).
        assert "HAR MAE:         N/A" in out


# ---------------------------------------------------------------------------
# make_no_telegram_config
# ---------------------------------------------------------------------------

class TestMakeNoTelegramConfig:
    def test_make_no_telegram_config(self):
        cfg = rab.make_no_telegram_config()
        assert isinstance(cfg, TelegramConfig)
        assert cfg.bot_token == "no-telegram-mode"
        assert cfg.chat_id == "0"


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

class TestMain:
    def test_main_status_mode(self, capsys):
        with patch(f"{MOD}.print_status") as ps:
            code = rab.main(["--status"])
        assert code == 0
        ps.assert_called_once()
        assert "TELEGRAM_BOT_TOKEN" not in capsys.readouterr().out  # no env needed

    def test_main_dry_run_mode(self):
        with patch(f"{MOD}.run_single_cycle", return_value=clean_cycle()), \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--dry-run"])
        assert code == 0

    def test_main_once_mode(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        with patch(f"{MOD}.run_single_cycle", return_value=clean_cycle()) as rc, \
             patch(f"{MOD}.initialize_db") as init, \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--once"])
        assert code == 0
        rc.assert_called_once()
        init.assert_called_once()

    def test_main_missing_env_returns_1(self, monkeypatch, capsys):
        monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
        monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
        with patch(f"{MOD}.load_dotenv"):
            code = rab.main([])
        assert code == 1
        out = capsys.readouterr().out
        assert "TELEGRAM_BOT_TOKEN" in out and "TELEGRAM_CHAT_ID" in out

    def test_main_calibrate_mode(self, monkeypatch):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        with patch(f"{MOD}.run_calibration_cycle") as cal, \
             patch(f"{MOD}.initialize_db"), \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--calibrate"])
        assert code == 0
        cal.assert_called_once()

    def test_main_db_override_passed_to_config(self, monkeypatch, tmp_path):
        monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
        monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
        db = str(tmp_path / "override.db")
        with patch(f"{MOD}.run_single_cycle", return_value=clean_cycle()) as rc, \
             patch(f"{MOD}.initialize_db"), \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--once", "--db", db])
        assert code == 0
        assert rc.call_args.args[1].db_path == db  # config passed to cycle

    def test_dry_run_does_not_call_telegram(self):
        with patch(f"{MOD}.run_single_cycle", return_value=clean_cycle()), \
             patch("kronos_trading.alerts.telegram_sender.requests.post") as post, \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--dry-run"])
        assert code == 0
        post.assert_not_called()  # no Telegram API call in dry-run mode

    def test_main_no_telegram_forever_mode(self):
        # --no-telegram routes to run_forever with suppressed sends.
        with patch(f"{MOD}.run_forever") as rf, \
             patch(f"{MOD}.load_dotenv"):
            code = rab.main(["--no-telegram"])
        assert code == 0
        rf.assert_called_once()
        cfg = rf.call_args.args[0]
        assert cfg.bot_token == "no-telegram-mode"  # placeholder, never sent
