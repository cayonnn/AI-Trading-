"""Test Online Learning"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from ai_agent.online_learning import create_online_learner

print("="*60)
print("   ONLINE LEARNING TEST")
print("="*60)

learner = create_online_learner()

# Simulate 10 trades
print("\nSimulating 10 trades...")
for i in range(10):
    state = np.random.randn(11).astype(np.float32)
    entry = 2000 + np.random.randn() * 10
    exit_price = entry + np.random.randn() * 20
    
    exp = learner.record_trade(
        trade_id=f"TEST_{i+1}",
        symbol="XAUUSD",
        entry_state=state,
        entry_price=entry,
        exit_state=state,
        exit_price=exit_price,
        action=1,
        holding_bars=np.random.randint(1, 50),
    )
    print(f"  Trade {i+1}: Entry=${entry:.2f}, Exit=${exit_price:.2f}, P&L=${exp.pnl:.2f}")

print("\n" + "="*60)
print("   LEARNING STATS")
print("="*60)

stats = learner.get_learning_stats()
print(f"Total Experiences: {stats['total_experiences']}")
print(f"Model Updates: {stats['updates_count']}")
exp_stats = stats['experience_stats']
print(f"Win Rate: {exp_stats['win_rate']:.1%}")
print(f"Avg P&L: ${exp_stats['avg_pnl']:.2f}")
print(f"Total P&L: ${exp_stats['total_pnl']:.2f}")

print("\n" + "="*60)
print("   ONLINE LEARNING TEST COMPLETE!")
print("="*60)
