"""
AI Trading System - Capability Analysis
=========================================
ประเมินและวิเคราะห์ความสามารถของ AI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment
from ai_agent.trade_memory import TradeMemory

def analyze_ai_capabilities():
    """ประเมินความสามารถของ AI"""
    
    print("\n" + "="*70)
    print("    AI TRADING SYSTEM - CAPABILITY ANALYSIS")
    print("    ประเมินและวิเคราะห์ความสามารถของ AI")
    print("="*70)
    
    # 1. Load trained model
    print("\n[1] MODEL ANALYSIS")
    print("-"*50)
    
    state_dim = 8 + 3
    agent = PPOAgentWalkForward(state_dim=state_dim)
    loaded = agent.load("best_wf")
    
    if loaded:
        print(f"    Model Status: LOADED")
        print(f"    Training Episodes: {agent.training_episodes}")
        print(f"    Network Parameters: {sum(p.numel() for p in agent.policy.parameters()):,}")
        print(f"    Architecture: Actor-Critic (PPO)")
    else:
        print("    Model Status: NOT FOUND - Need training")
        return
    
    # 2. Load data and run evaluation
    print("\n[2] PERFORMANCE EVALUATION")
    print("-"*50)
    
    df = pd.read_csv("data/training/GOLD_H1.csv")
    df.columns = [c.lower() for c in df.columns]
    
    env = TradingEnvironment(df)
    state = env.reset()
    
    # Track actions
    actions_count = {0: 0, 1: 0, 2: 0}
    action_names = {0: "WAIT", 1: "LONG", 2: "CLOSE"}
    
    while True:
        action, _, _ = agent.select_action(state)
        actions_count[action] += 1
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state
    
    perf = env.get_performance()
    
    print(f"    Total Bars Analyzed: {len(df)}")
    print(f"    Total Trades: {perf['n_trades']}")
    print(f"    Win Rate: {perf['win_rate']:.1%}")
    print(f"    Total P&L: ${perf['total_pnl']:.2f}")
    print(f"    Return: {perf['return_pct']:.2%}")
    print(f"    Final Capital: ${perf['final_capital']:.2f}")
    
    if perf['n_trades'] > 0:
        print(f"    Avg Win: ${perf['avg_win']:.2f}")
        print(f"    Avg Loss: ${perf['avg_loss']:.2f}")
        
        if perf['avg_loss'] != 0:
            profit_factor = abs(perf['avg_win'] / perf['avg_loss'])
            print(f"    Profit Factor: {profit_factor:.2f}")
    
    # Action distribution
    print("\n[3] DECISION ANALYSIS")
    print("-"*50)
    
    total_actions = sum(actions_count.values())
    for action, count in actions_count.items():
        pct = count / total_actions * 100
        print(f"    {action_names[action]}: {count} ({pct:.1f}%)")
    
    # Calculate trade frequency
    if perf['n_trades'] > 0:
        bars_per_trade = total_actions / perf['n_trades']
        print(f"\n    Bars per Trade: {bars_per_trade:.0f}")
        print(f"    Trade Frequency: Every {bars_per_trade:.0f} hours (H1 data)")
    
    # 4. Learning curve analysis
    print("\n[4] LEARNING CAPABILITY")
    print("-"*50)
    
    if agent.total_rewards:
        rewards = agent.total_rewards
        print(f"    Total Training Rewards: {len(rewards)} episodes")
        
        if len(rewards) >= 10:
            first_10 = np.mean(rewards[:10])
            last_10 = np.mean(rewards[-10:])
            improvement = last_10 - first_10
            
            print(f"    First 10 Episodes Avg: {first_10:.2f}")
            print(f"    Last 10 Episodes Avg: {last_10:.2f}")
            print(f"    Improvement: {improvement:+.2f} ({improvement/abs(first_10)*100:+.1f}%)")
    
    # 5. Memory analysis
    print("\n[5] TRADE MEMORY")
    print("-"*50)
    
    memory = TradeMemory()
    stats = memory.get_performance_stats()
    
    print(f"    Total Trades in Memory: {stats.get('total_trades', 0)}")
    print(f"    Historical Win Rate: {stats.get('win_rate', 0):.1%}")
    print(f"    Total Historical P&L: ${stats.get('total_pnl', 0):.2f}")
    
    # 6. Strengths and Weaknesses
    print("\n[6] CAPABILITY ASSESSMENT")
    print("-"*50)
    
    strengths = []
    weaknesses = []
    
    # Win Rate assessment
    if perf['win_rate'] >= 0.55:
        strengths.append(f"Good Win Rate ({perf['win_rate']:.1%})")
    elif perf['win_rate'] >= 0.45:
        weaknesses.append(f"Average Win Rate ({perf['win_rate']:.1%}) - needs improvement")
    else:
        weaknesses.append(f"Low Win Rate ({perf['win_rate']:.1%}) - needs more training")
    
    # Profit Factor assessment
    if perf['n_trades'] > 0 and perf['avg_loss'] != 0:
        pf = abs(perf['avg_win'] / perf['avg_loss'])
        if pf >= 1.5:
            strengths.append(f"Good Risk/Reward ({pf:.2f})")
        elif pf >= 1.0:
            weaknesses.append(f"Marginal Risk/Reward ({pf:.2f})")
        else:
            weaknesses.append(f"Poor Risk/Reward ({pf:.2f})")
    
    # Trade frequency
    if perf['n_trades'] > 50:
        strengths.append(f"Active trading ({perf['n_trades']} trades)")
    elif perf['n_trades'] < 10:
        weaknesses.append("Too conservative (few trades)")
    
    # Return assessment
    if perf['return_pct'] > 0.05:
        strengths.append(f"Positive Return ({perf['return_pct']:.1%})")
    elif perf['return_pct'] > 0:
        weaknesses.append(f"Low Return ({perf['return_pct']:.1%})")
    else:
        weaknesses.append(f"Negative Return ({perf['return_pct']:.1%})")
    
    print("\n    STRENGTHS (+):")
    if strengths:
        for s in strengths:
            print(f"      + {s}")
    else:
        print("      No significant strengths detected")
    
    print("\n    WEAKNESSES (-):")
    if weaknesses:
        for w in weaknesses:
            print(f"      - {w}")
    else:
        print("      No significant weaknesses detected")
    
    # 7. Overall rating
    print("\n[7] OVERALL RATING")
    print("-"*50)
    
    score = 0
    max_score = 100
    
    # Win rate (30 points)
    wr_score = min(30, perf['win_rate'] * 50)
    score += wr_score
    
    # Profit Factor (25 points)
    if perf['n_trades'] > 0 and perf['avg_loss'] != 0:
        pf = abs(perf['avg_win'] / perf['avg_loss'])
        pf_score = min(25, pf * 12.5)
    else:
        pf_score = 0
    score += pf_score
    
    # Return (25 points)
    ret_score = min(25, perf['return_pct'] * 500)
    ret_score = max(0, ret_score)
    score += ret_score
    
    # Trade activity (20 points)
    if perf['n_trades'] >= 100:
        trade_score = 20
    else:
        trade_score = perf['n_trades'] / 5
    score += trade_score
    
    print(f"    Win Rate Score:      {wr_score:.0f}/30")
    print(f"    Risk/Reward Score:   {pf_score:.0f}/25")
    print(f"    Return Score:        {ret_score:.0f}/25")
    print(f"    Activity Score:      {trade_score:.0f}/20")
    print(f"    ----------------------------")
    print(f"    TOTAL SCORE:         {score:.0f}/100")
    
    # Rating
    if score >= 80:
        rating = "EXCELLENT"
        emoji = "A+"
    elif score >= 65:
        rating = "GOOD"
        emoji = "B+"
    elif score >= 50:
        rating = "AVERAGE"
        emoji = "C"
    elif score >= 35:
        rating = "NEEDS IMPROVEMENT"
        emoji = "D"
    else:
        rating = "POOR"
        emoji = "F"
    
    print(f"\n    RATING: {rating} ({emoji})")
    
    # 8. Recommendations
    print("\n[8] RECOMMENDATIONS")
    print("-"*50)
    
    recommendations = []
    
    if perf['win_rate'] < 0.55:
        recommendations.append("Train more episodes (200-500) to improve Win Rate")
    
    if perf['n_trades'] > 0 and perf['avg_loss'] != 0:
        pf = abs(perf['avg_win'] / perf['avg_loss'])
        if pf < 1.5:
            recommendations.append("Adjust SL/TP ratios for better Risk/Reward")
    
    if perf['n_trades'] < 50:
        recommendations.append("Lower confidence threshold to increase trading activity")
    
    if agent.training_episodes < 100:
        recommendations.append("Model needs more training (at least 100 episodes)")
    
    if not recommendations:
        recommendations.append("Continue monitoring and fine-tuning")
    
    for i, rec in enumerate(recommendations, 1):
        print(f"    {i}. {rec}")
    
    # Summary
    print("\n" + "="*70)
    print(f"    SUMMARY: AI has scored {score:.0f}/100 ({rating})")
    print("="*70 + "\n")
    
    return {
        'score': score,
        'rating': rating,
        'performance': perf,
        'strengths': strengths,
        'weaknesses': weaknesses,
        'recommendations': recommendations,
    }


if __name__ == "__main__":
    analyze_ai_capabilities()
