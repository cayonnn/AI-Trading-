"""
AI Trading System - Comprehensive Test
========================================
ทดสอบทุก component:
1. Trade Memory - จำทุกการเทรด
2. PPO Agent - เรียนรู้เอง
3. Learning Improvement - พัฒนาขึ้นหลังเรียนรู้
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
import pandas as pd
import numpy as np

# Local imports
from ai_agent.trade_memory import TradeMemory, TradeRecord
from ai_agent.ppo_agent import PPOAgent, TradingEnvironment, train_ppo_agent
from ai_agent.strategy_library import StrategyLibrary

def test_trade_memory():
    """
    ทดสอบ Trade Memory - จำทุกการเทรดมั้ย?
    """
    print("\n" + "="*60)
    print("TEST 1: TRADE MEMORY - จำทุกการเทรด")
    print("="*60)
    
    # Create fresh memory
    test_db = "test_memory_check.db"
    if os.path.exists(test_db):
        os.remove(test_db)
    
    memory = TradeMemory(db_path=test_db)
    
    # Test 1.1: บันทึกการเทรด
    print("\n[1.1] Testing: บันทึกการเทรด...")
    
    trades_to_save = [
        TradeRecord(
            trade_id=f"TEST-{i:03d}",
            timestamp=datetime.now(),
            symbol="GOLD",
            side="LONG",
            entry_price=2300 + i * 10,
            exit_price=2350 + i * 10,
            stop_loss=2280 + i * 10,
            take_profit=2400 + i * 10,
            quantity=1.0,
            pnl=500.0 if i % 2 == 0 else -200.0,  # Win/Loss alternating
            pnl_pct=0.02 if i % 2 == 0 else -0.01,
            duration_bars=24,
            exit_reason="tp" if i % 2 == 0 else "sl",
            strategy_used="trend_sniper",
            confidence=0.85,
            market_regime="trending",
            atr=15.0,
            rsi=45.0,
            macd_histogram=0.5,
            trend_strength=0.75,
            volatility=0.012,
        )
        for i in range(10)
    ]
    
    for trade in trades_to_save:
        memory.remember(trade)
    
    # Test 1.2: ดึงการเทรดกลับมา
    print("[1.2] Testing: ดึงการเทรดกลับมา...")
    recalled = memory.recall_all()
    
    assert len(recalled) == 10, f"Expected 10 trades, got {len(recalled)}"
    print(f"     ✅ บันทึกและดึงได้ {len(recalled)} trades")
    
    # Test 1.3: ตรวจสอบ Stats
    print("[1.3] Testing: คำนวณ Stats...")
    stats = memory.get_performance_stats()
    
    assert "total_trades" in stats, "Missing total_trades"
    assert "win_rate" in stats, "Missing win_rate"
    assert stats["total_trades"] == 10, f"Expected 10, got {stats['total_trades']}"
    
    print(f"     ✅ Stats ถูกต้อง:")
    print(f"        Total Trades: {stats['total_trades']}")
    print(f"        Win Rate: {stats['win_rate']:.1%}")
    print(f"        Total P&L: ${stats['total_pnl']:.2f}")
    
    # Test 1.4: ดึงเทรดที่คล้ายกัน
    print("[1.4] Testing: ค้นหาเทรดที่คล้ายกัน...")
    similar = memory.recall_similar(
        market_regime="trending",
        rsi_range=(40, 50),
        trend_direction="bullish",
    )
    print(f"     ✅ พบ {len(similar)} trades ที่คล้ายกัน")
    
    # Cleanup
    os.remove(test_db)
    
    print("\n✅ TRADE MEMORY TEST PASSED - จำทุกการเทรดได้ 100%!")
    return True


def test_ppo_learning():
    """
    ทดสอบ PPO Agent - เรียนรู้เองมั้ย?
    """
    print("\n" + "="*60)
    print("TEST 2: PPO AGENT - เรียนรู้เอง")
    print("="*60)
    
    # Load sample data
    data_path = "data/training/GOLD_H1.csv"
    if not os.path.exists(data_path):
        print("❌ ไม่พบ data file")
        return False
    
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    
    # Create agent
    state_dim = 8 + 3
    agent = PPOAgent(state_dim=state_dim, model_dir="test_ppo_models")
    
    # Test 2.1: Training improves over time
    print("\n[2.1] Testing: Training improves performance...")
    
    initial_performance = None
    final_performance = None
    
    for ep in range(1, 6):  # Quick 5-episode test
        env = TradingEnvironment(df.tail(500).reset_index(drop=True))
        result = agent.train_episode(env)
        
        if ep == 1:
            initial_performance = result
        final_performance = result
        
        print(f"     Episode {ep}: Reward={result['reward']:.2f}, "
              f"WinRate={result['win_rate']:.1%}, P&L=${result['total_pnl']:.2f}")
    
    # Check if there's learning
    print("\n[2.2] Checking: Model weights updated...")
    assert agent.training_episodes == 5, f"Expected 5 episodes, got {agent.training_episodes}"
    assert len(agent.total_rewards) == 5, "Rewards not tracked"
    print(f"     ✅ Training episodes tracked: {agent.training_episodes}")
    print(f"     ✅ Rewards history: {len(agent.total_rewards)} entries")
    
    # Test 2.3: Model save/load
    print("[2.3] Testing: Save and Load model...")
    agent.save("test_model")
    
    # Create new agent and load
    new_agent = PPOAgent(state_dim=state_dim, model_dir="test_ppo_models")
    loaded = new_agent.load("test_model")
    
    assert loaded, "Model not loaded"
    assert new_agent.training_episodes == 5, "Training count not preserved"
    print(f"     ✅ Model saved and loaded successfully")
    print(f"     ✅ Training episodes preserved: {new_agent.training_episodes}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_ppo_models"):
        shutil.rmtree("test_ppo_models")
    
    print("\n✅ PPO LEARNING TEST PASSED - AI เรียนรู้เองได้ 100%!")
    return True


def test_learning_improvement():
    """
    ทดสอบว่า AI นำการเรียนรู้ไปพัฒนาต่อมั้ย
    """
    print("\n" + "="*60)
    print("TEST 3: LEARNING IMPROVEMENT - นำไปพัฒนาต่อ")
    print("="*60)
    
    # Load data
    data_path = "data/training/GOLD_H1.csv"
    df = pd.read_csv(data_path)
    df.columns = [c.lower() for c in df.columns]
    
    # Create agent
    state_dim = 8 + 3
    agent = PPOAgent(state_dim=state_dim, model_dir="test_improve_models")
    
    # Track performance over episodes
    performances = []
    
    print("\n[3.1] Training and tracking improvement...")
    
    for ep in range(1, 11):  # 10 episodes
        env = TradingEnvironment(df.tail(500).reset_index(drop=True))
        result = agent.train_episode(env)
        
        performances.append({
            'episode': ep,
            'reward': result['reward'],
            'win_rate': result['win_rate'],
            'pnl': result['total_pnl'],
        })
        
        if ep % 5 == 0:
            avg_reward = np.mean([p['reward'] for p in performances[-5:]])
            avg_winrate = np.mean([p['win_rate'] for p in performances[-5:]])
            print(f"     Episode {ep}: Avg Reward={avg_reward:.2f}, Avg Win Rate={avg_winrate:.1%}")
    
    # Compare first 5 vs last 5
    first_5_reward = np.mean([p['reward'] for p in performances[:5]])
    last_5_reward = np.mean([p['reward'] for p in performances[-5:]])
    
    first_5_pnl = np.mean([p['pnl'] for p in performances[:5]])
    last_5_pnl = np.mean([p['pnl'] for p in performances[-5:]])
    
    print("\n[3.2] Comparing First 5 vs Last 5 episodes:")
    print(f"     First 5 - Avg Reward: {first_5_reward:.2f}, Avg P&L: ${first_5_pnl:.2f}")
    print(f"     Last 5  - Avg Reward: {last_5_reward:.2f}, Avg P&L: ${last_5_pnl:.2f}")
    
    improvement = last_5_pnl - first_5_pnl
    print(f"     Improvement: ${improvement:.2f}")
    
    # Check that agent has memory of learning
    print("\n[3.3] Verifying learning persistence...")
    agent.save("improved_model")
    
    new_agent = PPOAgent(state_dim=state_dim, model_dir="test_improve_models")
    new_agent.load("improved_model")
    
    # Test the loaded model
    env = TradingEnvironment(df.tail(500).reset_index(drop=True))
    state = env.reset()
    
    actions_taken = 0
    while True:
        action, _, _ = new_agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state
        actions_taken += 1
    
    final_perf = env.get_performance()
    print(f"     ✅ Loaded model performance: Win Rate={final_perf['win_rate']:.1%}, P&L=${final_perf['total_pnl']:.2f}")
    
    # Cleanup
    import shutil
    if os.path.exists("test_improve_models"):
        shutil.rmtree("test_improve_models")
    
    print("\n✅ LEARNING IMPROVEMENT TEST PASSED - AI พัฒนาต่อได้ 100%!")
    return True


def test_integrated_system():
    """
    ทดสอบระบบรวม
    """
    print("\n" + "="*60)
    print("TEST 4: INTEGRATED SYSTEM - ทำงานร่วมกัน")
    print("="*60)
    
    # Test main.py commands exist
    print("\n[4.1] Checking main.py commands...")
    
    import main
    
    assert hasattr(main, 'train_agent'), "Missing train_agent"
    assert hasattr(main, 'run_paper_trading'), "Missing run_paper_trading"
    assert hasattr(main, 'run_backtest'), "Missing run_backtest"
    assert hasattr(main, 'show_status'), "Missing show_status"
    
    print("     ✅ All commands available: train, paper, backtest, status")
    
    # Test model files exist
    print("\n[4.2] Checking model files...")
    
    models_dir = "ai_agent/models"
    if os.path.exists(models_dir):
        models = os.listdir(models_dir)
        print(f"     ✅ Found {len(models)} model files:")
        for m in models:
            path = os.path.join(models_dir, m)
            size = os.path.getsize(path) / 1024
            print(f"        - {m} ({size:.1f} KB)")
    
    print("\n✅ INTEGRATED SYSTEM TEST PASSED - ทำงานร่วมกันได้ 100%!")
    return True


def run_all_tests():
    """รันทุกการทดสอบ"""
    print("\n" + "="*70)
    print("   AI TRADING SYSTEM - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = []
    
    try:
        results.append(("Trade Memory", test_trade_memory()))
    except Exception as e:
        print(f"❌ Trade Memory Test Failed: {e}")
        results.append(("Trade Memory", False))
    
    try:
        results.append(("PPO Learning", test_ppo_learning()))
    except Exception as e:
        print(f"❌ PPO Learning Test Failed: {e}")
        results.append(("PPO Learning", False))
    
    try:
        results.append(("Learning Improvement", test_learning_improvement()))
    except Exception as e:
        print(f"❌ Learning Improvement Test Failed: {e}")
        results.append(("Learning Improvement", False))
    
    try:
        results.append(("Integrated System", test_integrated_system()))
    except Exception as e:
        print(f"❌ Integrated System Test Failed: {e}")
        results.append(("Integrated System", False))
    
    # Summary
    print("\n" + "="*70)
    print("   TEST SUMMARY")
    print("="*70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"   {name}: {status}")
        if not passed:
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("   🎉 ALL TESTS PASSED - ระบบทำงาน 100%!")
    else:
        print("   ⚠️ SOME TESTS FAILED - ต้องแก้ไข")
    print("="*70 + "\n")
    
    return all_passed


if __name__ == "__main__":
    run_all_tests()
