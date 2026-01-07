"""
AI Trading System - Production Server
======================================
Production-ready server with:
- Waitress WSGI server (Windows compatible)
- Security headers
- Rate limiting
- File logging
- Error handling
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, g
from flask_cors import CORS
import threading
import time
from datetime import datetime
from functools import wraps
from collections import defaultdict
import logging
from logging.handlers import RotatingFileHandler

# ============================================================
# Production Configuration
# ============================================================

class Config:
    """Production configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'ai-trading-secret-key-change-in-production')
    RATE_LIMIT = 100  # requests per minute
    LOG_FILE = 'logs/trading_server.log'
    LOG_MAX_SIZE = 10 * 1024 * 1024  # 10MB
    LOG_BACKUP_COUNT = 5
    DEBUG = False


# ============================================================
# Initialize Flask App
# ============================================================

app = Flask(__name__)
app.config.from_object(Config)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Create logs directory
os.makedirs('logs', exist_ok=True)

# Setup logging
file_handler = RotatingFileHandler(
    Config.LOG_FILE,
    maxBytes=Config.LOG_MAX_SIZE,
    backupCount=Config.LOG_BACKUP_COUNT
)
file_handler.setFormatter(logging.Formatter(
    '%(asctime)s - %(levelname)s - %(message)s'
))
file_handler.setLevel(logging.INFO)
app.logger.addHandler(file_handler)
app.logger.setLevel(logging.INFO)

# Console logging (for production monitoring)
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(
    '%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
))
app.logger.addHandler(console_handler)


# ============================================================
# Rate Limiting
# ============================================================

rate_limit_data = defaultdict(list)

def rate_limit(limit_per_minute=100):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def decorated_function(*args, **kwargs):
            client_ip = request.remote_addr
            now = time.time()
            
            # Clean old entries
            rate_limit_data[client_ip] = [
                t for t in rate_limit_data[client_ip] 
                if now - t < 60
            ]
            
            if len(rate_limit_data[client_ip]) >= limit_per_minute:
                return jsonify({
                    "error": "Rate limit exceeded",
                    "retry_after": 60
                }), 429
            
            rate_limit_data[client_ip].append(now)
            return f(*args, **kwargs)
        return decorated_function
    return decorator


# ============================================================
# Security Headers
# ============================================================

@app.after_request
def add_security_headers(response):
    """Add security headers to all responses"""
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Cache-Control'] = 'no-store'
    return response


# ============================================================
# Request Logging
# ============================================================

@app.before_request
def log_request():
    """Log incoming requests"""
    g.start_time = time.time()


@app.after_request
def log_response(response):
    """Log response with timing"""
    if hasattr(g, 'start_time'):
        elapsed = (time.time() - g.start_time) * 1000
        app.logger.info(f"{request.method} {request.path} - {response.status_code} ({elapsed:.0f}ms)")
    return response


# ============================================================
# Error Handlers
# ============================================================

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}")
    return jsonify({"error": "Internal server error"}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled exception: {e}")
    return jsonify({"error": str(e)}), 500


# ============================================================
# Health Check
# ============================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    })


# ============================================================
# AI Components
# ============================================================

# Local imports (after Flask setup)
from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment
from ai_agent.ppo_agent import PPOAgent
from ai_agent.trade_memory import TradeMemory
import pandas as pd
import numpy as np

# Global state
agent = None
memory = None
trading_active = False
trading_thread = None


def initialize():
    """Initialize AI components"""
    global agent, memory
    
    state_dim = 8 + 3
    
    # Try Walk-Forward model first
    agent = PPOAgentWalkForward(state_dim=state_dim)
    loaded = agent.load("best_wf")
    
    if not loaded:
        # Try regular model
        agent = PPOAgent(state_dim=state_dim)
        loaded = agent.load("best")
    
    memory = TradeMemory()
    
    app.logger.info(f"AI initialized: Model loaded = {loaded}")
    return loaded


# ============================================================
# API Routes
# ============================================================

@app.route('/api/status', methods=['GET'])
@rate_limit(limit_per_minute=60)
def get_status():
    """Get system status"""
    global agent, memory
    
    if agent is None:
        initialize()
    
    try:
        stats = memory.get_performance_stats() if memory else {}
        
        try:
            df = pd.read_csv("data/training/GOLD_H1.csv")
            df.columns = [c.lower() for c in df.columns]
            
            env = TradingEnvironment(df.tail(500).reset_index(drop=True))
            state = env.reset()
            
            while True:
                action, _, _ = agent.select_action(state)
                next_state, reward, done, info = env.step(action)
                if done:
                    break
                state = next_state
            
            performance = env.get_performance()
        except Exception as e:
            performance = {"n_trades": 0, "win_rate": 0, "total_pnl": 0}
        
        return jsonify({
            "status": "online",
            "model_loaded": agent is not None,
            "episodes": agent.training_episodes if agent else 0,
            "performance": performance,
            "memory_stats": stats,
            "trading_active": trading_active,
            "timestamp": datetime.now().isoformat(),
        })
    except Exception as e:
        app.logger.error(f"Status error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/backtest', methods=['POST'])
@rate_limit(limit_per_minute=10)
def run_backtest():
    """Run backtest"""
    global agent
    
    if agent is None:
        initialize()
    
    try:
        df = pd.read_csv("data/training/GOLD_H1.csv")
        df.columns = [c.lower() for c in df.columns]
        
        env = TradingEnvironment(df)
        state = env.reset()
        
        while True:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
        
        performance = env.get_performance()
        
        app.logger.info(f"Backtest complete: {performance['n_trades']} trades, ${performance['total_pnl']:.2f}")
        
        return jsonify({
            "success": True,
            "trades": performance["n_trades"],
            "win_rate": performance["win_rate"],
            "pnl": performance["total_pnl"],
            "return_pct": performance["return_pct"],
        })
    except Exception as e:
        app.logger.error(f"Backtest error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/paper/start', methods=['POST'])
@rate_limit(limit_per_minute=10)
def start_paper_trading():
    """Start paper trading"""
    global trading_active, trading_thread, agent
    
    if trading_active:
        return jsonify({"error": "Trading already active"})
    
    if agent is None:
        initialize()
    
    trading_active = True
    
    def paper_trade_loop():
        global trading_active
        
        while trading_active:
            try:
                df = pd.read_csv("data/training/GOLD_H1.csv")
                df.columns = [c.lower() for c in df.columns]
                
                env = TradingEnvironment(df.tail(100).reset_index(drop=True))
                state = env.reset()
                
                action, _, _ = agent.select_action(state)
                
                app.logger.info(f"Paper Trade: Action={action}, Price={df['close'].iloc[-1]:.2f}")
                
                time.sleep(60)
                
            except Exception as e:
                app.logger.error(f"Paper trading error: {e}")
                time.sleep(60)
    
    trading_thread = threading.Thread(target=paper_trade_loop, daemon=True)
    trading_thread.start()
    
    app.logger.info("Paper trading started")
    return jsonify({"success": True, "message": "Paper trading started"})


@app.route('/api/paper/stop', methods=['POST'])
@rate_limit(limit_per_minute=10)
def stop_paper_trading():
    """Stop paper trading"""
    global trading_active
    
    trading_active = False
    app.logger.info("Paper trading stopped")
    
    return jsonify({"success": True, "message": "Paper trading stopped"})


@app.route('/api/train', methods=['POST'])
@rate_limit(limit_per_minute=5)
def train_ai():
    """Train AI"""
    global agent
    
    data = request.get_json() or {}
    episodes = data.get('episodes', 50)
    
    app.logger.info(f"Training started with {episodes} episodes")
    
    try:
        from ai_agent.ppo_walk_forward import train_walk_forward
        
        agent, history = train_walk_forward(
            n_folds=3,
            episodes_per_fold=episodes // 3
        )
        
        app.logger.info(f"Training complete: {episodes} episodes")
        
        return jsonify({
            "success": True,
            "episodes": episodes,
            "win_rate": history[-1]["win_rate"] if history else 0,
        })
    except Exception as e:
        app.logger.error(f"Training error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/api/trades', methods=['GET'])
@rate_limit(limit_per_minute=30)
def get_trades():
    """Get trade history"""
    global memory
    
    if memory is None:
        memory = TradeMemory()
    
    try:
        trades = memory.recall_recent(limit=50)
        return jsonify({
            "trades": [
                {
                    "id": t.trade_id,
                    "time": t.timestamp.isoformat() if t.timestamp else None,
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "pnl": t.pnl,
                }
                for t in trades
            ]
        })
    except Exception as e:
        app.logger.error(f"Trades error: {e}")
        return jsonify({"error": str(e)}), 500


# ============================================================
# MT5 Live Trading
# ============================================================

MT5_AVAILABLE = False
mt5 = None

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    app.logger.info("MetaTrader5 module loaded successfully")
except ImportError:
    MT5_AVAILABLE = False
    app.logger.warning("MetaTrader5 not installed - Live trading disabled")
    
    # Create mock mt5 for route registration
    class MockMT5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        TRADE_ACTION_DEAL = 0
        ORDER_TIME_GTC = 0
        ORDER_FILLING_IOC = 0
        TRADE_RETCODE_DONE = 0
        TIMEFRAME_H1 = 16385
        
        def initialize(self): return False
        def shutdown(self): pass
        def account_info(self): return None
        def symbol_info(self, s): return None
        def symbol_info_tick(self, s): return None
        def symbol_select(self, s, v): pass
        def order_send(self, r): return None
        def positions_get(self, **k): return None
        def copy_rates_from_pos(self, *a): return None
    
    mt5 = MockMT5()


@app.route('/api/mt5/status', methods=['GET'])
@rate_limit(limit_per_minute=30)
def mt5_status():
    """Get MT5 connection status"""
    if not MT5_AVAILABLE:
        return jsonify({"connected": False, "error": "MT5 not installed"})
    
    if not mt5.initialize():
        return jsonify({"connected": False, "error": "MT5 not running"})
    
    account_info = mt5.account_info()
    if account_info is None:
        mt5.shutdown()
        return jsonify({"connected": False, "error": "Not logged in"})
    
    info = {
        "connected": True,
        "login": account_info.login,
        "server": account_info.server,
        "balance": account_info.balance,
        "equity": account_info.equity,
        "margin": account_info.margin,
        "free_margin": account_info.margin_free,
        "profit": account_info.profit,
    }
    
    mt5.shutdown()
    return jsonify(info)


@app.route('/api/mt5/connect', methods=['POST'])
@rate_limit(limit_per_minute=10)
def mt5_connect():
    """Connect to MT5"""
    if not MT5_AVAILABLE:
        return jsonify({"error": "MT5 not installed"}), 400
    
    if not mt5.initialize():
        return jsonify({"error": "Failed to initialize MT5"}), 500
    
    account_info = mt5.account_info()
    if account_info:
        app.logger.info(f"MT5 connected: {account_info.login}")
        return jsonify({
            "success": True,
            "login": account_info.login,
            "balance": account_info.balance,
        })
    
    return jsonify({"error": "Not logged in to MT5"}), 400


@app.route('/api/mt5/trade', methods=['POST'])
@rate_limit(limit_per_minute=10)
def mt5_trade():
    """Execute real trade on MT5"""
    if not MT5_AVAILABLE:
        return jsonify({"error": "MT5 not installed"}), 400
    
    data = request.get_json() or {}
    symbol = data.get('symbol', 'XAUUSD')
    lot_size = float(data.get('lot_size', 0.01))
    action = data.get('action', 'BUY')
    
    # Safety checks
    if lot_size > 1.0:
        return jsonify({"error": "Lot size too large (max 1.0)"}), 400
    
    if lot_size < 0.01:
        return jsonify({"error": "Lot size too small (min 0.01)"}), 400
    
    if not mt5.initialize():
        return jsonify({"error": "MT5 not running"}), 500
    
    try:
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            return jsonify({"error": f"Symbol {symbol} not found"}), 400
        
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)
        
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return jsonify({"error": "Failed to get price"}), 500
        
        point = symbol_info.point
        if action == 'BUY':
            price = tick.ask
            sl = price - 200 * point
            tp = price + 600 * point
            order_type = mt5.ORDER_TYPE_BUY
        else:
            price = tick.bid
            sl = price + 200 * point
            tp = price - 600 * point
            order_type = mt5.ORDER_TYPE_SELL
        
        request_order = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot_size,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "magic": 123456,
            "comment": "AI Trading System",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request_order)
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            return jsonify({
                "error": f"Order failed: {result.comment}",
                "retcode": result.retcode,
            }), 500
        
        app.logger.info(f"Trade executed: {action} {lot_size} {symbol} @ {price}")
        
        return jsonify({
            "success": True,
            "order_id": result.order,
            "symbol": symbol,
            "action": action,
            "lot_size": lot_size,
            "price": price,
            "sl": sl,
            "tp": tp,
        })
        
    except Exception as e:
        app.logger.error(f"Trade error: {e}")
        return jsonify({"error": str(e)}), 500
    finally:
        mt5.shutdown()


@app.route('/api/mt5/auto-trade', methods=['POST'])
@rate_limit(limit_per_minute=5)
def mt5_auto_trade():
    """Start AI auto-trading on MT5"""
    global agent, trading_active
    
    if not MT5_AVAILABLE:
        return jsonify({"error": "MT5 not installed"}), 400
    
    data = request.get_json() or {}
    symbol = data.get('symbol', 'XAUUSD')
    lot_size = float(data.get('lot_size', 0.01))
    
    if trading_active:
        return jsonify({"error": "Trading already active"})
    
    if agent is None:
        initialize()
    
    trading_active = True
    
    def auto_trade_loop():
        global trading_active
        
        while trading_active:
            try:
                if not mt5.initialize():
                    time.sleep(60)
                    continue
                
                rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 500)
                if rates is None:
                    mt5.shutdown()
                    time.sleep(60)
                    continue
                
                df = pd.DataFrame(rates)
                df.columns = [c.lower() for c in df.columns]
                
                env = TradingEnvironment(df)
                state = env.reset()
                action, _, _ = agent.select_action(state)
                
                current_positions = mt5.positions_get(symbol=symbol)
                has_position = current_positions and len(current_positions) > 0
                
                if action == 1 and not has_position:
                    tick = mt5.symbol_info_tick(symbol)
                    symbol_info = mt5.symbol_info(symbol)
                    point = symbol_info.point
                    
                    request_order = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": lot_size,
                        "type": mt5.ORDER_TYPE_BUY,
                        "price": tick.ask,
                        "sl": tick.ask - 200 * point,
                        "tp": tick.ask + 600 * point,
                        "magic": 123456,
                        "comment": "AI Auto Trade",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request_order)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        app.logger.info(f"AI Auto Trade: BUY {lot_size} {symbol}")
                
                elif action == 2 and has_position:
                    pos = current_positions[0]
                    tick = mt5.symbol_info_tick(symbol)
                    
                    request_close = {
                        "action": mt5.TRADE_ACTION_DEAL,
                        "symbol": symbol,
                        "volume": pos.volume,
                        "type": mt5.ORDER_TYPE_SELL,
                        "position": pos.ticket,
                        "price": tick.bid,
                        "magic": 123456,
                        "comment": "AI Auto Close",
                        "type_time": mt5.ORDER_TIME_GTC,
                        "type_filling": mt5.ORDER_FILLING_IOC,
                    }
                    
                    result = mt5.order_send(request_close)
                    if result.retcode == mt5.TRADE_RETCODE_DONE:
                        app.logger.info(f"AI Auto Trade: CLOSE {symbol} profit={pos.profit}")
                
                mt5.shutdown()
                time.sleep(300)
                
            except Exception as e:
                app.logger.error(f"Auto trade error: {e}")
                time.sleep(60)
    
    trading_thread = threading.Thread(target=auto_trade_loop, daemon=True)
    trading_thread.start()
    
    app.logger.info(f"Auto trading started: {symbol} with {lot_size} lots")
    
    return jsonify({
        "success": True, 
        "message": f"Auto trading started on {symbol} with {lot_size} lots"
    })


# ============================================================
# Production Server Entry Point
# ============================================================

def run_production(host='0.0.0.0', port=5000, threads=4):
    """Run production server with Waitress"""
    from waitress import serve
    
    print("=" * 60)
    print("   AI TRADING SYSTEM - PRODUCTION SERVER")
    print("=" * 60)
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Threads: {threads}")
    print("=" * 60)
    
    # Initialize AI
    initialize()
    
    app.logger.info(f"Production server starting on {host}:{port}")
    
    # Run with Waitress WSGI
    serve(app, host=host, port=port, threads=threads)


def run_development():
    """Run development server"""
    print("Running in DEVELOPMENT mode...")
    initialize()
    app.run(host='0.0.0.0', port=5000, debug=True)


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='AI Trading Server')
    parser.add_argument('--dev', action='store_true', help='Run in development mode')
    parser.add_argument('--port', type=int, default=5000, help='Port number')
    parser.add_argument('--threads', type=int, default=4, help='Number of threads')
    
    args = parser.parse_args()
    
    if args.dev:
        run_development()
    else:
        run_production(port=args.port, threads=args.threads)
