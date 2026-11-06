# Exness MT5 agent

Research and demo execution for EURUSD and XAUUSD. The M15 setup has to agree with the closed H1 trend. A tree model then estimates whether that setup reaches a 1:2 target before the stop inside 12 candles. Orders are sized so a stop costs about 1% of equity. A day that is down 3% sends no new trades.

This is not a profitable-bot claim. No script can know its live win rate before it is trained on your terminal's own history and judged on a later period. With a 1:2 payoff the breakeven hit rate is about 33% before spread and commission. The 0.70 cutoff is a confidence filter. It is not a forecast that 70% of trades will win. If the holdout never reaches 0.70, leave the threshold alone.

Use an Exness demo. `LIVE_TRADING` defaults to false, and a real account is refused unless you set `ALLOW_REAL_ACCOUNT=true`.

## Layout

| Piece | Role |
| --- | --- |
| `exness_mt5_agent/mt5_client.py` | Attach to the terminal, resolve Exness symbol names, download bars |
| `exness_mt5_agent/features.py` | EMA 50/200, RSI, ATR, tick-volume ratio, failed-break flags, H1 merge |
| `exness_mt5_agent/labels.py` | Binary target: 1:2 hit inside 12 bars |
| `exness_mt5_agent/path.py` | One stop/target scan shared by the label and the backtest |
| `exness_mt5_agent/modeling.py` | XGBoost or RandomForest, 5 purged walk-forward folds |
| `exness_mt5_agent/signals.py` | BUY / SELL / HOLD |
| `exness_mt5_agent/risk.py` | 1% lots, rounded stops, persisted 3% daily halt |
| `exness_mt5_agent/news.py` | 30-minute blackout around high-impact events |
| `exness_mt5_agent/execution.py` | Market order with SL, TP, and lot size |
| `exness_mt5_agent/backtest.py` | Bar replay with the same gates |
| `exness_mt5_agent/pipeline.py` | `check`, `train`, `backtest`, `live` |

## How a trade is allowed

All of these have to be true on a closed M15 bar:

1. The 50 EMA is on the same side of the 200 EMA as the last closed H1 bar, and that relationship is not brand new (3 M15 bars, 2 H1 bars). The crossover is the regime, not the entry.
2. Price pulled back through the 50 EMA and closed back with the trend. The bar that breaks the prior 20-bar high or low is skipped. A break that has already closed back through its level is skipped.
3. RSI(14) sits in a rolling band: longs between the 20th and 55th percentile of the last 100 RSI values and below 65; shorts between the 45th and 80th and above 35.
4. Tick volume is at least 1.2 times its 20-bar average. Forex tick volume is broker activity, not exchange volume.
5. The model trained on that direction assigns probability of at least 0.70. If both directions qualify, the bar is a HOLD.
6. Spread is inside the limit, the stop clears the broker's stop level, and the lot size risks at most 1% including the commission you configured. A new trade is skipped when that 1% would push the day through a 3% loss from the morning equity. Once the day is already down 3%, the halt stays on until the next broker date.
7. No high-impact event for the symbol's currencies is inside a 30-minute window.

Stop distance is 1.5 ATR. Target distance is 3.0 ATR. If the stop and the target both print on one candle, the result is a loss, because OHLC cannot tell which one happened first. A gap through the stop is filled at the open, not at the stop price.

The H1 value used on an M15 bar is the hour that has already closed. The 10:15 candle cannot see the 10:00 hour while that hour is still forming.

## Install

The MetaTrader 5 package publishes a 64-bit Windows wheel for this machine's Python 3.14. The package talks to the Exness terminal on this PC. It does not log in from a server. If a future Python release has no wheel, use 64-bit Python 3.12.

```powershell
cd $env:USERPROFILE\Projects\exness-mt5-agent
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env
python -m unittest discover -s tests
```

`vectorbt` is optional and is not installed by the requirements file. The built-in backtest is the one that matches the stops.

## Demo account

1. Open an Exness demo and pick Standard or Raw Spread. Create the MT5 account, not only the client-area login.
2. Install the MT5 terminal from the Exness personal area and log in. Leave that terminal open. File → Login to Trade Account, then check the server name. Trial servers often look like `Exness-MT5Trial`.
3. In Market Watch, confirm the symbol names. Standard accounts often use `EURUSDm` and `XAUUSDm`. Raw Spread often uses `EURUSD` and `XAUUSD`. Put those exact names in `SYMBOLS` if the resolver picks the wrong one.
4. Tools → Options → Charts → set **Max bars in chart** high enough for the history you want (50,000 or more). Open EURUSD and XAUUSD on M15 and H1 once so the terminal downloads them.
5. Click **Algo Trading** in the toolbar so it is green before you ever set `LIVE_TRADING=true`.
6. In `.env`, leave `MT5_LOGIN` blank to attach to the account already logged in. Set `MT5_PATH` only if initialize cannot find the terminal.
7. Compare the clock in the MT5 toolbar with UTC. Set `BROKER_UTC_OFFSET` to server hours minus UTC hours. Trial servers are often +2 or +3.
8. Run `python -m exness_mt5_agent check`. You want `real=False`, a resolved symbol, and a spread you recognize. Fix the offset before trusting the news filter.

Raw Spread accounts charge a commission. Copy that round-turn cost per 1.00 lot into `COMMISSION_PER_LOT`. Standard accounts usually keep it at 0 because the cost is in the spread. Wrong commission makes the reported risk look cleaner than the account.

## News calendar

Copy the example and replace it with real high-impact events:

```powershell
copy data\news_events.example.csv data\news_events.csv
```

Columns are `time_utc`, `currency`, `impact`, `title`. Times are UTC. Impact `high` is the only row that blocks. EURUSD listens to EUR and USD. XAUUSD listens to USD and XAU. A missing file, an unreadable file, or a file whose newest event is more than seven days old blocks new live trades. Refresh it before the week starts. The example file is not loaded automatically.

## Train

```powershell
python -m exness_mt5_agent train
```

Training is separate for long and short, and separate for each symbol. Rows are kept only when the rule setup is present, because that is the distribution the live loop asks about. Five expanding folds walk forward in time. Twelve source bars are dropped before each test block so a training label cannot be built from the same future the test features can see. The saved model is fit on the earlier portion only. The last 15% of setups is a holdout, and the log reports precision at 0.70 there.

Read the holdout line before anything else. Precision with a handful of signals is noise. Zero signals means the model is not willing to trade that history at this threshold. A shallow XGBoost model (depth 3) is the default. Set `MODEL_TYPE=random_forest` for the forest alternative: depth 5, at least 20 samples per leaf.

The terminal must actually contain the bars. `HISTORY_BARS=20000` is only a request. A selective setup can need more than that before it has 200 labeled examples. The command fails clearly when it does not.

## Backtest

```powershell
python -m exness_mt5_agent backtest --symbol EURUSD
python -m exness_mt5_agent backtest --symbol XAUUSD --rules-only
```

`--rules-only` trades every setup and ignores the model, which shows whether the filter itself has an edge before you trust a fit. The replay enters on the next bar, allows one position, applies the symbol's contract size, spread, commission, and the 3% day stop, and writes `data/<symbol>_backtest_trades.csv`.

Judge the log by net PnL, max drawdown, profit factor, and how many trades there were. A high win rate with a tiny sample, or a high win rate that disappears once spread is included, is not a reason to go live.

### vectorbt

Use vectorbt when you want a second view of the signal timestamps. It will not reproduce the intrabar stop rule above. Install it in the same environment (`pip install vectorbt`) and point it at the closes plus the trade log:

```python
import pandas as pd
import vectorbt as vbt

trades = pd.read_csv("data/EURUSD_backtest_trades.csv", parse_dates=["decision_time", "exit_time"])
# Build a close series from the MT5 export you trust, then:
entries = pd.Series(False, index=close.index)
exits = pd.Series(False, index=close.index)
for row in trades.itertuples():
    if row.side != "BUY":
        continue
    entries.loc[row.decision_time] = True
    exits.loc[row.exit_time] = True
portfolio = vbt.Portfolio.from_signals(close, entries, exits, freq="15min", init_cash=10_000)
print(portfolio.stats())
```

Treat differences versus `backtest` as expected. The agent's own summary is the one aligned with SL, TP, spread, and the daily halt. The MT5 Strategy Tester is a third check only if you re-code these rules in MQL5. This repository does not emit an Expert Advisor, and a Tester run of a different EA is not a test of this Python agent.

## Live demo

Keep `LIVE_TRADING=false` for the first session.

```powershell
python -m exness_mt5_agent live
```

The process polls every 20 seconds and acts once per closed M15 bar. Dry-run lines show the lot, stop, and target it would have sent. When the log is behaving and the backtest on the holdout period is something you can tolerate losing, set `LIVE_TRADING=true` on the demo only. Leave the terminal logged in. Stopping the script does not close an open position; the stop and target on the order remain at the broker.

The daily halt is stored in `data/runtime_state.json`. Deleting that file to "reset" a red day removes the circuit breaker. Do that only when you understand which day you are erasing.

## Operating limits

- One agent position per symbol, identified by magic number `260106`.
- Lot size is floored to the volume step. If the minimum lot would risk more than 1%, the trade is skipped.
- `MAX_LOTS` defaults to 20. A calculated size above that is skipped rather than silently cut down.
- Restart the trainer on a schedule you choose, such as once a month, and keep a fresh holdout. Do not fit on the same window you just used to congratulate the model.
- Slippage beyond the modeled spread is not fully known. Gold around news can move through the stop even when the calendar filter is right.
- Nothing here is financial advice. Demo losses are the cheap version of the lesson.
