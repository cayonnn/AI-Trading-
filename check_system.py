"""
System Check - Verify 100% functionality
==========================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def check_system():
    results = []
    
    print("\n" + "="*60)
    print("   AI TRADING SYSTEM - 100% CHECK")
    print("="*60)
    
    # 1. Check imports
    print("\n[1] Checking imports...")
    try:
        from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment, TradingState
        from ai_agent.trade_memory import TradeMemory
        from ai_agent.strategy_library import StrategyLibrary
        from ai_agent.autonomous_ai import AutonomousAI
        import pandas as pd
        import numpy as np
        import torch
        print("    OK: All imports successful")
        results.append(("Imports", True))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("Imports", False))
    
    # 2. Check data
    print("\n[2] Checking data...")
    try:
        import pandas as pd
        df = pd.read_csv("data/training/GOLD_H1.csv")
        print(f"    OK: Data loaded ({len(df)} bars)")
        results.append(("Data", True))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("Data", False))
    
    # 3. Check models
    print("\n[3] Checking models...")
    try:
        models_dir = "ai_agent/models"
        if os.path.exists(models_dir):
            models = [f for f in os.listdir(models_dir) if f.endswith('.pt')]
            print(f"    OK: Found {len(models)} models")
            for m in models[:5]:  # Show first 5
                size = os.path.getsize(os.path.join(models_dir, m)) / 1024
                print(f"        - {m} ({size:.1f} KB)")
            if len(models) > 5:
                print(f"        ... and {len(models) - 5} more")
            results.append(("Models", True))
        else:
            print("    FAIL: Models directory not found")
            results.append(("Models", False))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("Models", False))
    
    # 4. Check PPO model loading
    print("\n[4] Checking PPO model loading...")
    try:
        from ai_agent.ppo_walk_forward import PPOAgentWalkForward
        agent = PPOAgentWalkForward(state_dim=11)
        agent.load("final")
        print("    OK: PPO model loaded (final.pt)")
        results.append(("PPO Model", True))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("PPO Model", False))
    
    # 5. Check trade memory
    print("\n[5] Checking trade memory...")
    try:
        from ai_agent.trade_memory import TradeMemory
        memory = TradeMemory()
        stats = memory.get_performance_stats()
        print(f"    OK: Memory working ({stats.get('total_trades', 0)} trades)")
        results.append(("Memory", True))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("Memory", False))
    
    # 6. Check LSTM/XGBoost models
    print("\n[6] Checking LSTM/XGBoost models...")
    try:
        checkpoints = "models/checkpoints"
        lstm_exists = os.path.exists(f"{checkpoints}/lstm_best.pt")
        xgb_exists = os.path.exists(f"{checkpoints}/xgboost_best.json")
        if lstm_exists and xgb_exists:
            print("    OK: LSTM and XGBoost models found")
            results.append(("ML Models", True))
        else:
            print(f"    WARN: LSTM={lstm_exists}, XGB={xgb_exists}")
            results.append(("ML Models", lstm_exists or xgb_exists))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("ML Models", False))
    
    # 7. Check AutonomousAI
    print("\n[7] Checking AutonomousAI...")
    try:
        from ai_agent.autonomous_ai import AutonomousAI
        ai = AutonomousAI()
        print("    OK: AutonomousAI initialized")
        results.append(("AutonomousAI", True))
    except Exception as e:
        print(f"    FAIL: {e}")
        results.append(("AutonomousAI", False))
    
    # Summary
    print("\n" + "="*60)
    print("   SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"   {name}: {status}")
    
    print(f"\n   Total: {passed}/{total} ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n   ✅ SYSTEM STATUS: 100% READY")
    elif passed >= total * 0.8:
        print(f"\n   ⚠️ SYSTEM STATUS: {passed/total*100:.0f}% - Minor issues")
    else:
        print(f"\n   ❌ SYSTEM STATUS: {passed/total*100:.0f}% - Needs fixes")
    
    print("="*60 + "\n")
    
    return passed == total


if __name__ == "__main__":
    check_system()

