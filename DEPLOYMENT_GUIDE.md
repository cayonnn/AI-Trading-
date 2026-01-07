# 🚀 Production Deployment Guide
## Gold Trading System - MT5 Integration

**Version**: 1.0.0
**Status**: ✅ PRODUCTION READY
**Last Updated**: 2025-12-30
**Current Gold Price**: $4,370 (Live MT5)

---

## 📋 Quick Navigation

- [System Overview](#system-overview)
- [Pre-Deployment Checklist](#pre-deployment-checklist)
- [Installation Guide](#installation-guide)
- [Configuration](#configuration)
- [Testing Protocol](#testing-protocol)
- [Deployment Modes](#deployment-modes)
- [Monitoring & Alerts](#monitoring--alerts)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## 🎯 System Overview

### Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   PRODUCTION TRADING SYSTEM                      │
│                         (24/7 Operation)                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐         ┌────────────────────┐               │
│  │ MT5 Terminal │────────→│ MT5 Data Provider  │               │
│  │              │         │                    │               │
│  │ - Live Data  │         │ - 6 Timeframes     │               │
│  │ - Execution  │         │ - Auto-Reconnect   │               │
│  └──────────────┘         │ - Error Handling   │               │
│                           └─────────┬──────────┘               │
│                                     │                           │
│                                     ▼                           │
│                           ┌────────────────────┐               │
│                           │ Master Integration │               │
│                           │     System         │               │
│                           │                    │               │
│                           │ ✓ Regime Detection │               │
│                           │ ✓ MTF Analysis     │               │
│                           │ ✓ Sentiment        │               │
│                           │ ✓ Alt Data         │               │
│                           │ ✓ Ensemble (5)     │               │
│                           └─────────┬──────────┘               │
│                                     │                           │
│                                     ▼                           │
│                           ┌────────────────────┐               │
│                           │  Risk Monitor      │               │
│                           │                    │               │
│                           │ ✓ Daily Limits     │               │
│                           │ ✓ Drawdown Check   │               │
│                           │ ✓ Position Limits  │               │
│                           │ ✓ Circuit Breakers │               │
│                           └─────────┬──────────┘               │
│                                     │                           │
│         ┌───────────────────────────┼───────────────────┐      │
│         │                           │                   │      │
│         ▼                           ▼                   ▼      │
│  ┌─────────────┐         ┌──────────────┐    ┌──────────────┐│
│  │ Database    │         │ MT5 Trade    │    │ Performance  ││
│  │ Manager     │         │ Executor     │    │ Monitor      ││
│  │             │         │              │    │              ││
│  │ - Signals   │         │ - Orders     │    │ - Dashboard  ││
│  │ - Trades    │         │ - Positions  │    │ - Alerts     ││
│  │ - Analytics │         │ - Risk Calc  │    │ - Reports    ││
│  └─────────────┘         └──────────────┘    └──────────────┘│
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

### System Capabilities

✅ **Real-Time Trading**
- Live MT5 market data (6 timeframes)
- Professional order execution
- Position management
- Auto-reconnection

✅ **Advanced Analytics**
- Multi-timeframe analysis (W1, D1, H4, H1, M30, M15)
- Regime detection (Trending/Ranging)
- Sentiment analysis (5 sources)
- Alternative data integration (6 sources)
- Ensemble strategies (5 models)

✅ **Risk Management**
- Daily loss limits (5%)
- Maximum drawdown protection (15%)
- Position size calculation
- Circuit breakers
- Pre-trade validation

✅ **Monitoring & Control**
- Real-time logging
- Performance tracking
- Database persistence
- Alert system ready
- Web dashboard ready

---

## ✅ Pre-Deployment Checklist

### System Requirements

#### Hardware
- [ ] CPU: 4+ cores (Intel i5/AMD Ryzen 5 or better)
- [ ] RAM: 8GB minimum (16GB recommended)
- [ ] Storage: 100GB free space (SSD recommended)
- [ ] Network: Stable internet (10 Mbps+, low latency)

#### Software
- [ ] Windows 10/11 (64-bit)
- [ ] Python 3.9+ installed (tested on 3.12)
- [ ] MetaTrader 5 Terminal installed (Build 5474+)
- [ ] Git (optional, for version control)

#### MT5 Account
- [ ] MT5 account active
- [ ] Login credentials available
- [ ] Server name known (e.g., "XMGlobal-MT5")
- [ ] Algorithmic trading enabled
- [ ] Minimum balance: $1,000+ (recommended: $10,000+)
- [ ] Symbol "GOLD" available
- [ ] Leverage: 1:100 or higher

### Pre-Deployment Tests

- [ ] ✅ All component imports working
- [ ] ✅ MT5 connection successful
- [ ] ✅ Multi-timeframe data fetching (6 TFs)
- [ ] ✅ Signal generation working
- [ ] ✅ Database operations functional
- [ ] ✅ Paper trading simulation tested
- [ ] ⏳ 30-day paper trading completed
- [ ] ⏳ Performance metrics acceptable

### Documentation Review

- [ ] Read this deployment guide completely
- [ ] Understand risk management rules
- [ ] Know emergency shutdown procedures
- [ ] Understand circuit breaker triggers
- [ ] Review troubleshooting section

---

## 📦 Installation Guide

### Step 1: System Preparation

```bash
# Navigate to project directory
cd "F:\Mobile App\AI Trade"

# Verify Python version
python --version
# Expected: Python 3.9.x or higher

# Verify MT5 Python module
python -c "import MetaTrader5 as mt5; print('MT5 Version:', mt5.__version__)"
# Expected: MT5 Version: 5.0.5430
```

### Step 2: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt

# Verify critical packages
python -c "import MetaTrader5, pandas, numpy, yaml, loguru; print('All imports OK')"
```

**Key Dependencies**:
```
MetaTrader5==5.0.5430
pandas>=2.0.0
numpy>=1.24.0
PyYAML>=6.0
loguru>=0.7.0
scikit-learn>=1.3.0
python-dateutil>=2.8.0
```

### Step 3: MT5 Terminal Setup

1. **Open MetaTrader 5**
2. **Go to**: Tools → Options → Expert Advisors
3. **Enable**:
   - ✅ Allow algorithmic trading
   - ✅ Allow DLL imports
   - ✅ Allow WebRequest for listed URLs
4. **Click**: OK

5. **Add Symbol**:
   - Open Market Watch (Ctrl+M)
   - Right-click → Symbols
   - Search for "GOLD"
   - Click "Show" if hidden
   - Verify it appears in Market Watch

### Step 4: Environment Configuration

**Option A: Using Environment Variables (Recommended)**

Windows PowerShell:
```powershell
$env:MT5_LOGIN="your_account_number"
$env:MT5_PASSWORD="your_password"
$env:MT5_SERVER="YourBroker-MT5"
```

Windows CMD:
```cmd
set MT5_LOGIN=your_account_number
set MT5_PASSWORD=your_password
set MT5_SERVER=YourBroker-MT5
```

Linux/Mac:
```bash
export MT5_LOGIN="your_account_number"
export MT5_PASSWORD="your_password"
export MT5_SERVER="YourBroker-MT5"
```

**Option B: Using .env File**

Create `.env` file in project root:
```env
MT5_LOGIN=your_account_number
MT5_PASSWORD=your_password
MT5_SERVER=YourBroker-MT5

# Optional API Keys
NEWSAPI_KEY=your_news_api_key
TWITTER_API_KEY=your_twitter_key
FRED_API_KEY=your_fred_key
```

### Step 5: Configuration File Review

Edit `config.yaml`:

```yaml
# CRITICAL SETTINGS - Review Before Deployment

system:
  environment: "production"  # production, staging, development

trading:
  symbol: "GOLD"             # Your broker's gold symbol
  timeframe: "H1"            # Primary timeframe
  min_confidence: 0.85       # 85% minimum signal confidence
  max_open_positions: 3      # Maximum concurrent trades

risk:
  max_daily_loss_pct: 0.05   # 5% - HARD LIMIT
  max_drawdown_pct: 0.15     # 15% - EMERGENCY STOP
  max_consecutive_losses: 3   # Pause after 3 losses

paper_trading:
  enabled: true              # SET TO false FOR LIVE TRADING
  initial_capital: 10000.0   # Paper trading starting balance
```

**⚠️ CRITICAL**: For live trading, set `paper_trading.enabled: false`

---

## 🧪 Testing Protocol

### Phase 1: Component Verification (10 minutes)

```bash
# Test 1: MT5 Connection
python test_mt5_connection.py
```

**Expected Output**:
```
[PASS] Connection successful
[PASS] Symbol: GOLD
[PASS] Current Price: Bid: 4366.89, Ask: 4367.36
[PASS] Fetched 100 H1 bars
[PASS] Fetched 3/3 timeframes
[PASS] Reconnection successful
ALL TESTS PASSED
```

```bash
# Test 2: Full Integration
python test_mt5_integration.py
```

**Expected Output**:
```
[PASS] Connected to MT5
[PASS] Fetched 6/6 timeframes (6000 bars total)
[PASS] Signal generated successfully
  Action: LONG
  Confidence: 98%
[PASS] Signal saved to database
[PASS] Paper trade simulated
ALL INTEGRATION TESTS PASSED
```

**If any test fails**: Stop and troubleshoot before proceeding.

### Phase 2: Quick Paper Trading Test (30 minutes)

```bash
python run_paper_trading_test.py
```

**Monitor For**:
- ✅ No errors in console
- ✅ Signals generated (1-2 expected in 30 min)
- ✅ Paper trades executed
- ✅ Balance tracked correctly

**Stop with**: Ctrl+C

**Review**:
```bash
# Check logs
type logs\production_mt5_*.log

# Check database
python -c "from database_manager import DatabaseManager; db = DatabaseManager(); print(db.get_recent_signals(5))"
```

### Phase 3: Extended Paper Trading (7-30 days) ⭐ CRITICAL

```bash
python production_system_mt5.py --paper-trading --interval 300
```

**Daily Monitoring Checklist**:
- [ ] System still running (check console)
- [ ] No errors in logs
- [ ] Signals being generated (1-5 per day expected)
- [ ] Win rate trending > 55%
- [ ] No circuit breaker activations

**Weekly Performance Review**:
```python
from database_manager import DatabaseManager
db = DatabaseManager()
stats = db.get_performance_stats()
print(stats)
```

**Minimum Requirements to Proceed to Live**:
- ✅ 30+ days paper trading
- ✅ Win rate > 60%
- ✅ Profit factor > 2.0
- ✅ Max drawdown < 10%
- ✅ Sharpe ratio > 1.5
- ✅ Zero system crashes
- ✅ Stable performance (no degradation)

---

## 🚀 Deployment Modes

### Mode 1: Paper Trading (START HERE)

**Purpose**: Risk-free testing with real data and real signals

**Duration**: Minimum 30 days (recommended 60-90 days)

**Command**:
```bash
python production_system_mt5.py --paper-trading --interval 300
```

**What Happens**:
- ✅ Connects to MT5 for live data
- ✅ Generates real signals every 5 minutes
- ✅ Simulates trade execution
- ✅ Tracks performance
- ❌ No real money at risk

**Success Criteria**:
| Metric | Target | Your Result |
|--------|--------|-------------|
| Win Rate | > 60% | ___% |
| Profit Factor | > 2.0 | ___ |
| Max Drawdown | < 10% | ___% |
| Sharpe Ratio | > 1.5 | ___ |
| Uptime | > 99% | ___% |

**When to Proceed**: All criteria met for 30+ consecutive days

### Mode 2: Small Capital Live Test

**Purpose**: Verify real execution with minimal risk

**Capital**: 5-10% of total account

**Pre-requisites**:
- ✅ Phase 1 (Paper Trading) completed successfully
- ✅ All performance targets met
- ✅ System stable for 30+ days

**Configuration Changes**:

1. Edit `config.yaml`:
```yaml
paper_trading:
  enabled: false  # DISABLE PAPER TRADING

trading:
  max_position_size: 0.1      # 10% max position
  default_position_size: 0.05 # 5% default
  max_open_positions: 1       # Only 1 position
```

2. **Command**:
```bash
python production_system_mt5.py --live --interval 300
```

**Monitor Closely**:
- ✅ First 10 trades execute correctly
- ✅ Stop loss/take profit levels accurate
- ✅ Slippage < 2 points average
- ✅ Performance matches paper trading (±10%)
- ✅ No execution errors

**Duration**: 10-20 trades or 2-4 weeks

**Success Criteria**:
- ✅ Win rate within 5% of paper trading
- ✅ Average slippage < 3 points
- ✅ No failed orders
- ✅ Profit factor > 1.8

### Mode 3: Full Production

**⚠️ Only proceed after Small Capital Test succeeds!**

**Configuration**:

```yaml
paper_trading:
  enabled: false

trading:
  max_position_size: 0.5      # 50% max
  default_position_size: 0.3  # 30% default
  max_open_positions: 3       # Up to 3 positions
```

**Command**:
```bash
python production_system_mt5.py --live --interval 300
```

**Continuous Monitoring Required**:
- **Daily**: Check logs, review trades, monitor P&L
- **Weekly**: Analyze performance vs targets
- **Monthly**: Full review and optimization

---

## 📊 Monitoring & Alerts

### Real-Time Console Monitoring

**Healthy System Indicators**:
```
23:55:17 | INFO | [OK] MT5 connection established
23:55:17 | INFO | [OK] Fetched 6/6 timeframes
23:55:17 | INFO | [OK] Signal generated
23:55:17 | INFO |   Action: LONG
23:55:17 | INFO |   Confidence: 98%
23:55:17 | INFO | [OK] All risk checks passed
23:55:17 | INFO | [OK] Order executed successfully
```

**Warning Signs**:
```
WARNING | Daily loss limit approaching (4.2% of 5%)
WARNING | Low confidence signal (82% < 85% min)
WARNING | High volatility detected
```

**Critical Alerts**:
```
ERROR | CIRCUIT BREAKER ACTIVATED
ERROR | Max drawdown exceeded: 15.2%
ERROR | Order execution failed
ERROR | MT5 connection lost
```

### Log File Monitoring

**Location**: `logs/production_mt5_YYYY-MM-DD.log`

**View Real-Time**:
```bash
# Windows PowerShell
Get-Content logs\production_mt5_*.log -Wait -Tail 50

# Command Prompt
powershell Get-Content logs\production_mt5_*.log -Wait -Tail 50
```

**Check for Errors**:
```bash
findstr /i "error" logs\production_mt5_*.log
findstr /i "circuit" logs\production_mt5_*.log
findstr /i "failed" logs\production_mt5_*.log
```

**Check Trade Execution**:
```bash
findstr /i "Order executed" logs\production_mt5_*.log
```

### Database Monitoring

**Quick Stats Query**:
```python
from database_manager import DatabaseManager

db = DatabaseManager()

# Today's signals
signals = db.get_recent_signals(limit=10)
print(f"Signals today: {len(signals)}")

# Recent trades
trades = db.get_recent_trades(limit=10)
print(f"Trades: {len(trades)}")

# Win rate
executed = [s for s in signals if s['executed']]
print(f"Execution rate: {len(executed)/len(signals)*100:.1f}%")
```

### Performance Dashboard

**Key Metrics to Track**:

| Metric | Formula | Target | Alert If |
|--------|---------|--------|----------|
| Daily P&L | Today's profit/loss | Positive | < -5% |
| Win Rate | Wins / Total Trades | > 60% | < 55% |
| Profit Factor | Gross Profit / Gross Loss | > 2.0 | < 1.5 |
| Max Drawdown | Peak to Trough | < 15% | > 10% |
| Sharpe Ratio | (Return - Risk Free) / StdDev | > 2.0 | < 1.0 |
| Average R:R | Avg Win / Avg Loss | > 2.5 | < 2.0 |

---

## 🔧 Troubleshooting

### Issue 1: MT5 Connection Failed

**Error**:
```
ERROR: MT5 init failed: (-1, 'Terminal: Call failed')
```

**Solutions**:

1. **Check MT5 is Running**:
   - Open Task Manager (Ctrl+Shift+Esc)
   - Look for "terminal64.exe" process
   - If not found, open MT5 Terminal

2. **Verify Algorithmic Trading Enabled**:
   - MT5: Tools → Options → Expert Advisors
   - ✅ "Allow algorithmic trading" must be checked

3. **Check Credentials**:
   ```bash
   # Verify environment variables are set
   echo %MT5_LOGIN%
   echo %MT5_SERVER%
   ```

4. **Try Manual Connection First**:
   - Login to MT5 manually with your credentials
   - If manual login fails, contact broker

5. **Restart MT5**:
   - Close MT5 completely
   - Wait 10 seconds
   - Open MT5
   - Try again

### Issue 2: Symbol Not Found

**Error**:
```
ERROR: Symbol GOLD not found
```

**Solutions**:

1. **Find Correct Symbol Name**:
   ```python
   python -c "import MetaTrader5 as mt5; mt5.initialize(); symbols = mt5.symbols_get(); gold = [s.name for s in symbols if 'GOLD' in s.name.upper() or 'XAU' in s.name.upper()]; print('Gold symbols:', gold); mt5.shutdown()"
   ```

2. **Common Gold Symbol Names**:
   - GOLD
   - XAUUSD
   - GOLD.
   - XAUUSDm
   - (Check your broker's naming)

3. **Update config.yaml**:
   ```yaml
   trading:
     symbol: "YOUR_BROKER_GOLD_SYMBOL"
   ```

4. **Make Symbol Visible**:
   - MT5: View → Market Watch (Ctrl+M)
   - Right-click → Symbols
   - Find your gold symbol
   - Click "Show"

### Issue 3: Order Execution Failed

**Common Error Codes**:

| Code | Meaning | Solution |
|------|---------|----------|
| 10004 | Requote | Retry automatically (system does this) |
| 10006 | Invalid request | Check lot size, SL, TP levels |
| 10013 | Invalid volume | Use 0.01-1.0 lots, check broker minimums |
| 10014 | Invalid price | Check market hours, spread |
| 10015 | Invalid stops | SL/TP too close to price |
| 10016 | Market closed | Trade outside market hours |
| 10019 | No money | Insufficient balance |
| 10027 | Autotrading disabled | Enable in MT5 settings |

**Solutions**:

1. **Check Account Balance**:
   ```python
   from mt5_trade_executor import MT5TradeExecutor
   executor = MT5TradeExecutor()
   if executor.connect():
       account = executor.get_account_info()
       print(f"Balance: ${account['balance']}")
       print(f"Free Margin: ${account['margin_free']}")
   ```

2. **Verify Lot Size**:
   ```python
   symbol_info = executor.get_symbol_info("GOLD")
   print(f"Min lot: {symbol_info['volume_min']}")
   print(f"Max lot: {symbol_info['volume_max']}")
   print(f"Step: {symbol_info['volume_step']}")
   ```

3. **Check Market Hours**:
   - Gold trades 24/5 (Sunday 5pm - Friday 5pm EST)
   - Avoid trading during rollover (5-6pm EST)

### Issue 4: High Memory Usage

**Symptoms**: Python process > 2GB RAM

**Solutions**:

1. **Reduce Data Fetching**:
   ```python
   # In production_system_mt5.py, reduce bars
   mtf_data = provider.get_multi_timeframe(symbol, timeframes, bars=500)  # Instead of 1000
   ```

2. **Clear Old Data**:
   ```sql
   DELETE FROM trading_signals WHERE timestamp < date('now', '-30 days');
   ```

3. **Restart Daily**:
   - Set up scheduled task to restart system at 00:00 UTC

### Issue 5: Circuit Breaker Activated

**Triggers**:
- Max drawdown exceeded (15%)
- Daily loss limit (5%)
- 3 consecutive losses

**What to Do**:

1. **DO NOT Override Manually** - Circuit breakers are safety features

2. **Analyze Root Cause**:
   ```python
   # Check recent trades
   from database_manager import DatabaseManager
   db = DatabaseManager()
   trades = db.get_recent_trades(limit=20)
   losses = [t for t in trades if t['profit'] < 0]
   print(f"Recent losses: {len(losses)}")
   for trade in losses:
       print(f"{trade['action']} {trade['entry_price']} → Loss: ${trade['profit']}")
   ```

3. **Review Market Conditions**:
   - Was there major news event?
   - Market volatility spike?
   - System parameters still appropriate?

4. **Wait for Reset**:
   - System automatically resets after configured duration
   - Use this time to analyze and improve

5. **Consider Adjustments**:
   ```yaml
   # Reduce risk if needed
   trading:
     default_position_size: 0.2  # Reduce from 0.5
     max_open_positions: 1       # Reduce from 3

   # Increase quality threshold
     min_confidence: 0.90        # Increase from 0.85
   ```

---

## ⚡ Performance Optimization

### System Performance

**Target Benchmarks**:
- Signal generation: < 3 seconds
- Data fetch (6 TFs): < 5 seconds
- Order execution: < 1 second
- Total cycle: < 10 seconds

**If Slower**:

1. **Reduce Timeframes**:
   ```yaml
   multi_timeframe:
     timeframes:
       D1: { weight: 0.40 }
       H4: { weight: 0.30 }
       H1: { weight: 0.30 }
       # Remove M30, M15 for speed
   ```

2. **Reduce Bars**:
   ```python
   # Minimum 200-500 bars still effective
   mtf_data = provider.get_multi_timeframe(symbol, timeframes, bars=500)
   ```

3. **Disable ML** (if enabled):
   ```yaml
   ml:
     enabled: false
   ```

4. **Use SSD**:
   - Move database to SSD
   - Improves query performance

### Trading Performance

**Target Metrics**:
- Win rate: > 60%
- Profit factor: > 2.0
- Sharpe ratio: > 2.0
- Max drawdown: < 15%

**If Underperforming**:

1. **Increase Quality Threshold**:
   ```yaml
   trading:
     min_confidence: 0.90  # From 0.85
   ```

2. **Reduce Position Size**:
   ```yaml
   trading:
     default_position_size: 0.3  # From 0.5
   ```

3. **Tighter Stop Loss**:
   ```yaml
   trading:
     stop_loss:
       default: 0.010  # From 0.012 (1.0% instead of 1.2%)
   ```

4. **Filter by Regime**:
   - System performs better in trending markets
   - Consider pausing in ranging markets

5. **Review Signal Quality**:
   ```python
   # Analyze signal distribution
   signals = db.get_recent_signals(limit=100)
   high_conf = [s for s in signals if s['adjusted_confidence'] > 0.90]
   print(f"High confidence signals: {len(high_conf)/len(signals)*100:.1f}%")
   ```

---

## 📝 Production Operations Checklist

### Daily Tasks (5 minutes)

**Morning Check** (before market open):
- [ ] Verify system is running (check console)
- [ ] Review overnight logs for errors
- [ ] Check open positions in MT5
- [ ] Verify account balance/equity
- [ ] Check for circuit breaker status

**Evening Review** (after market close):
- [ ] Review day's signals and trades
- [ ] Check win rate and P&L
- [ ] Analyze any losses
- [ ] Review performance metrics
- [ ] Plan for next day

### Weekly Tasks (30 minutes)

**Sunday Preparation**:
- [ ] Review week's performance vs targets
- [ ] Calculate win rate, profit factor, drawdown
- [ ] Analyze signal quality and distribution
- [ ] Check for any degradation in performance
- [ ] Review and update parameters if needed
- [ ] Backup database
- [ ] Check disk space and logs rotation

### Monthly Tasks (2 hours)

**End of Month Review**:
- [ ] Full performance analysis
- [ ] Compare vs benchmarks
- [ ] Calculate Sharpe ratio
- [ ] Review max drawdown
- [ ] Analyze trade distribution
- [ ] Regime detection accuracy review
- [ ] Parameter optimization assessment
- [ ] System reliability check
- [ ] Update documentation
- [ ] Plan improvements for next month

---

## 🔒 Safety Protocols

### Emergency Shutdown Procedures

**Method 1: Graceful Shutdown**
```bash
# Press Ctrl+C in running terminal
# System will:
# - Stop accepting new signals
# - Close database connections
# - Disconnect from MT5
```

**Method 2: Force Kill**
```bash
# Windows
taskkill /F /IM python.exe

# Then manually close positions in MT5 if needed
```

**Method 3: Close All Positions in MT5**
- Right-click on position → Close
- Or use "Close All" if available

### Risk Limits Summary

| Limit Type | Threshold | Action | Override |
|------------|-----------|--------|----------|
| Max Drawdown | 15% | STOP ALL | ❌ Never |
| Daily Loss | 5% | Stop Today | ❌ No |
| Consecutive Losses | 3 | Pause 24h | ⚠️ Review First |
| Position Limits | 3 positions | Block New | ✅ Config |
| Min Confidence | 85% | Skip Signal | ✅ Config |

### Backup Strategy

**Automated Backups** (built-in):
- Database backed up before major operations
- Location: `backup_trading_db_TIMESTAMP.db`
- Retention: Last 5 backups kept

**Manual Backup**:
```bash
# Daily backup
copy trading_data.db backups\trading_data_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%.db

# Weekly full backup
xcopy /E /I "F:\Mobile App\AI Trade" "F:\Backups\AI_Trade_%DATE:~-4,4%%DATE:~-10,2%%DATE:~-7,2%"
```

---

## 🎯 Success Criteria

### Week 1-4: Paper Trading Phase

**Targets**:
- ✅ System uptime > 99%
- ✅ 1-5 signals per day
- ✅ Win rate > 55%
- ✅ No critical errors
- ✅ No circuit breaker triggers

### Month 2-3: Small Capital Phase

**Targets**:
- ✅ Performance matches paper (±10%)
- ✅ Win rate > 60%
- ✅ Profit factor > 2.0
- ✅ Avg slippage < 3 points
- ✅ No failed executions

### Month 4+: Full Production

**Targets**:
- ✅ Consistent profitability
- ✅ Win rate > 65%
- ✅ Profit factor > 2.5
- ✅ Sharpe ratio > 2.0
- ✅ Max drawdown < 10%
- ✅ System reliability 99.9%

---

## 📞 Quick Command Reference

```bash
# TESTING
python test_mt5_connection.py                    # Connection test
python test_mt5_integration.py                   # Full integration test
python run_paper_trading_test.py                 # 30-min paper test

# PAPER TRADING
python production_system_mt5.py --paper-trading  # Start paper trading

# LIVE TRADING
python production_system_mt5.py --live           # Start live trading

# WITH OPTIONS
python production_system_mt5.py --live --interval 600                    # Custom interval
python production_system_mt5.py --live --login 12345 --password "pass"  # With credentials

# MONITORING
type logs\production_mt5_*.log                   # View logs
python -c "from database_manager import DatabaseManager; db = DatabaseManager(); print(db.get_recent_signals(10))"  # Recent signals
```

---

## ✅ Final Pre-Launch Checklist

### Before Paper Trading
- [ ] All tests passing
- [ ] MT5 connection verified
- [ ] Configuration reviewed
- [ ] `paper_trading.enabled: true`
- [ ] Logs directory exists
- [ ] Database initialized

### Before Live Trading
- [ ] 30+ days paper trading completed
- [ ] All performance targets met
- [ ] `paper_trading.enabled: false`
- [ ] Risk limits configured
- [ ] Account funded adequately
- [ ] Emergency procedures understood
- [ ] Monitoring system in place
- [ ] Backup strategy active

---

**Deployment Status**: ✅ READY FOR PRODUCTION

**Recommended Next Step**: Start 30-day paper trading

**Support**: Review troubleshooting section and check logs

---

*Remember: Paper trade first, start small, scale gradually. Never risk more than you can afford to lose.* 🚀

**Good luck with your trading system!**
