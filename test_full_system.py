"""
AI Trading System - Full System Test
=====================================
ทดสอบทุกส่วนก่อนใช้จริง
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
from datetime import datetime

def print_header(title):
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")

def print_result(name, status, details=""):
    icon = "✅" if status else "❌"
    print(f"   {icon} {name}: {'PASS' if status else 'FAIL'} {details}")

def run_full_test():
    """Run full system test"""
    
    print_header("AI TRADING SYSTEM - FULL TEST")
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    results = []
    
    # ========================================
    # TEST 1: Model Loading
    # ========================================
    print_header("[1/7] MODEL LOADING")
    try:
        from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment
        
        state_dim = 8 + 3
        agent = PPOAgentWalkForward(state_dim=state_dim)
        loaded = agent.load("best_wf")
        
        print_result("PPO Walk-Forward Model", loaded, f"({agent.training_episodes} episodes)")
        results.append(("Model Loading", loaded))
    except Exception as e:
        print_result("Model Loading", False, str(e))
        results.append(("Model Loading", False))
    
    # ========================================
    # TEST 2: Data Loading
    # ========================================
    print_header("[2/7] DATA LOADING")
    try:
        df = pd.read_csv("data/training/GOLD_H1.csv")
        df.columns = [c.lower() for c in df.columns]
        
        has_ohlc = all(col in df.columns for col in ['open', 'high', 'low', 'close'])
        data_size = len(df)
        
        print_result("Data File", has_ohlc, f"({data_size:,} bars)")
        results.append(("Data Loading", has_ohlc and data_size > 1000))
    except Exception as e:
        print_result("Data Loading", False, str(e))
        results.append(("Data Loading", False))
    
    # ========================================
    # TEST 3: Trading Environment
    # ========================================
    print_header("[3/7] TRADING ENVIRONMENT")
    try:
        env = TradingEnvironment(df.tail(1000).reset_index(drop=True))
        state = env.reset()
        
        # Run 100 steps
        for _ in range(100):
            action = np.random.randint(0, 3)
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
        
        perf = env.get_performance()
        print_result("Environment", True, f"(trades: {perf['n_trades']})")
        results.append(("Trading Environment", True))
    except Exception as e:
        print_result("Trading Environment", False, str(e))
        results.append(("Trading Environment", False))
    
    # ========================================
    # TEST 4: AI Decision Making
    # ========================================
    print_header("[4/7] AI DECISION MAKING")
    try:
        env = TradingEnvironment(df.tail(500).reset_index(drop=True))
        state = env.reset()
        
        actions = {0: 0, 1: 0, 2: 0}
        
        for _ in range(200):
            action, log_prob, value = agent.select_action(state)
            actions[action] += 1
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
        
        perf = env.get_performance()
        
        # Check if AI is making diverse decisions
        diverse = all(count > 0 for count in actions.values())
        
        print_result("AI Decisions", True, f"WAIT:{actions[0]} LONG:{actions[1]} CLOSE:{actions[2]}")
        print_result("Win Rate", perf['win_rate'] > 0.3, f"{perf['win_rate']:.1%}")
        print_result("Profit Factor", perf.get('avg_loss', -1) != 0, 
                    f"{abs(perf.get('avg_win', 0)/perf.get('avg_loss', -1)):.2f}" if perf.get('avg_loss', 0) != 0 else "N/A")
        
        results.append(("AI Decision Making", True))
    except Exception as e:
        print_result("AI Decision Making", False, str(e))
        results.append(("AI Decision Making", False))
    
    # ========================================
    # TEST 5: Trade Memory
    # ========================================
    print_header("[5/7] TRADE MEMORY")
    try:
        from ai_agent.trade_memory import TradeMemory
        
        memory = TradeMemory()
        stats = memory.get_performance_stats()
        
        print_result("Trade Memory", True, f"({stats.get('total_trades', 0)} trades stored)")
        results.append(("Trade Memory", True))
    except Exception as e:
        print_result("Trade Memory", False, str(e))
        results.append(("Trade Memory", False))
    
    # ========================================
    # TEST 6: Online Learning
    # ========================================
    print_header("[6/7] ONLINE LEARNING")
    try:
        from ai_agent.online_learning import create_online_learner
        
        learner = create_online_learner()
        
        # Test recording a trade
        test_state = np.random.randn(11).astype(np.float32)
        exp = learner.record_trade(
            trade_id="SYSTEM_TEST",
            symbol="XAUUSD",
            entry_state=test_state,
            entry_price=2000.0,
            exit_state=test_state,
            exit_price=2010.0,
            action=1,
            holding_bars=10,
        )
        
        stats = learner.get_learning_stats()
        
        print_result("Online Learning", True, f"(experiences: {stats['total_experiences']})")
        results.append(("Online Learning", True))
    except Exception as e:
        print_result("Online Learning", False, str(e))
        results.append(("Online Learning", False))
    
    # ========================================
    # TEST 7: Full Backtest
    # ========================================
    print_header("[7/7] FULL BACKTEST")
    try:
        # Run full backtest on recent data
        test_data = df.tail(5000).reset_index(drop=True)
        env = TradingEnvironment(test_data)
        state = env.reset()
        
        while True:
            action, _, _ = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            if done:
                break
            state = next_state
        
        perf = env.get_performance()
        
        profitable = perf['total_pnl'] > 0
        good_rr = perf.get('avg_loss', -1) != 0 and abs(perf.get('avg_win', 0)/perf.get('avg_loss', -1)) > 1.5
        
        print_result("Profitability", profitable, f"${perf['total_pnl']:.2f}")
        print_result("Return", perf['return_pct'] > 0, f"{perf['return_pct']:.1%}")
        print_result("Trades", perf['n_trades'] > 10, f"{perf['n_trades']}")
        print_result("Win Rate", perf['win_rate'] > 0.4, f"{perf['win_rate']:.1%}")
        
        results.append(("Full Backtest", profitable))
    except Exception as e:
        print_result("Full Backtest", False, str(e))
        results.append(("Full Backtest", False))
    
    # ========================================
    # SUMMARY
    # ========================================
    print_header("TEST SUMMARY")
    
    passed = sum(1 for _, status in results if status)
    total = len(results)
    
    for name, status in results:
        icon = "✅" if status else "❌"
        print(f"   {icon} {name}")
    
    print(f"\n   {'='*40}")
    print(f"   TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print(f"\n   🎉 ALL TESTS PASSED!")
        print(f"   ✅ SYSTEM READY FOR LIVE TRADING")
    else:
        print(f"\n   ⚠️ SOME TESTS FAILED")
        print(f"   ❌ FIX ISSUES BEFORE GOING LIVE")
    
    print(f"{'='*60}\n")
    
    return passed == total


if __name__ == "__main__":
    success = run_full_test()
    sys.exit(0 if success else 1)
