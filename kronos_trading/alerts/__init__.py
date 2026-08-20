"""Phase 1 - Telegram volatility alert bot (observation only, no trading).

Subpackage layout (built in dependency order):

* ``har_forecaster``    - HAR volatility computation (validated model, past-only)
* ``prediction_logger`` - SQLite persistence of predictions/outcomes
* ``breakout_detector`` - actual-vs-predicted breakout checks + calibration stats
* ``telegram_sender``   - Telegram Bot API message formatting/sending
* ``scheduler``         - hourly main loop (fetch -> compute -> log -> send)

This is a monitoring system only. It never places orders and never predicts
price direction.
"""
