"""
AI System Comprehensive Analysis
=================================
วิเคราะห์ความพร้อมของ AI ทั้งระบบ
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
from datetime import datetime

def print_section(title):
    print(f"\n{'='*60}")
    print(f"   {title}")
    print(f"{'='*60}")

def score_to_grade(score):
    if score >= 90: return "A+ (Excellent)"
    elif score >= 80: return "A (Very Good)"
    elif score >= 70: return "B+ (Good)"
    elif score >= 60: return "B (Satisfactory)"
    elif score >= 50: return "C (Needs Work)"
    else: return "D (Poor)"

def analyze_ai_system():
    """วิเคราะห์ AI ทั้งระบบ"""
    
    print("="*60)
    print("   AI TRADING SYSTEM - COMPREHENSIVE ANALYSIS")
    print("   วิเคราะห์ความพร้อมของ AI ทั้งระบบ")
    print("="*60)
    print(f"   Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    scores = {}
    
    # ========================================
    # 1. TRADING CAPABILITY (ความพร้อมในการเทรด)
    # ========================================
    print_section("[1] TRADING CAPABILITY (ความพร้อมในการเทรด)")
    
    trading_score = 0
    
    # 1.1 Model loaded
    try:
        from ai_agent.ppo_walk_forward import PPOAgentWalkForward
        agent = PPOAgentWalkForward(state_dim=11)
        loaded = agent.load("best_wf")
        
        if loaded:
            print(f"   ✅ Model Loaded: {agent.training_episodes} episodes")
            trading_score += 20
            
            if agent.training_episodes >= 500:
                print(f"   ✅ Sufficient Training: {agent.training_episodes} >= 500")
                trading_score += 10
            else:
                print(f"   ⚠️ Low Training: {agent.training_episodes} < 500")
                trading_score += 5
        else:
            print("   ❌ Model Not Loaded")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # 1.2 Backtest performance
    try:
        from ai_agent.ppo_walk_forward import TradingEnvironment
        df = pd.read_csv("data/training/GOLD_H1.csv")
        df.columns = [c.lower() for c in df.columns]
        
        env = TradingEnvironment(df.tail(5000))
        state = env.reset()
        
        for _ in range(4999):
            action, _, _ = agent.select_action(state)
            state, _, done, _ = env.step(action)
            if done: break
        
        perf = env.get_performance()
        
        # Win Rate
        if perf['win_rate'] >= 0.55:
            print(f"   ✅ Win Rate: {perf['win_rate']:.1%} (Good)")
            trading_score += 15
        elif perf['win_rate'] >= 0.45:
            print(f"   ⚠️ Win Rate: {perf['win_rate']:.1%} (Average)")
            trading_score += 10
        else:
            print(f"   ❌ Win Rate: {perf['win_rate']:.1%} (Low)")
            trading_score += 5
        
        # Profitability
        if perf['total_pnl'] > 0:
            print(f"   ✅ Profitable: ${perf['total_pnl']:.2f}")
            trading_score += 15
        else:
            print(f"   ❌ Not Profitable: ${perf['total_pnl']:.2f}")
        
        # Risk/Reward
        if perf.get('avg_loss', 0) != 0:
            rr = abs(perf.get('avg_win', 0) / perf.get('avg_loss', 1))
            if rr >= 1.5:
                print(f"   ✅ Risk/Reward: 1:{rr:.1f} (Good)")
                trading_score += 10
            elif rr >= 1.0:
                print(f"   ⚠️ Risk/Reward: 1:{rr:.1f} (Average)")
                trading_score += 5
            else:
                print(f"   ❌ Risk/Reward: 1:{rr:.1f} (Poor)")
    except Exception as e:
        print(f"   ❌ Backtest Error: {e}")
    
    # 1.3 Advanced Features
    try:
        from ai_agent.ai_full_control import AIFullController
        controller = AIFullController(symbol="GOLD")
        
        if controller.use_advanced:
            print("   ✅ Advanced Features: Enabled (MTF, News, Session)")
            trading_score += 10
        else:
            print("   ⚠️ Advanced Features: Disabled")
            trading_score += 5
    except Exception as e:
        print(f"   ❌ Advanced Features Error: {e}")
    
    # 1.4 Risk Management
    try:
        from ai_agent.ai_full_control import RiskManager
        rm = RiskManager()
        print(f"   ✅ Risk Management: Max {rm.max_risk_per_trade:.0%}/trade, {rm.max_drawdown:.0%} drawdown")
        trading_score += 10
    except Exception as e:
        print(f"   ❌ Risk Management Error: {e}")
    
    scores['trading'] = min(100, trading_score)
    print(f"\n   📊 Trading Capability Score: {scores['trading']}/100")
    
    # ========================================
    # 2. LEARNING CAPABILITY (การเรียนรู้หลังเทรด)
    # ========================================
    print_section("[2] LEARNING CAPABILITY (การเรียนรู้หลังเทรด)")
    
    learning_score = 0
    
    # 2.1 Online Learning
    try:
        from ai_agent.online_learning import create_online_learner
        learner = create_online_learner()
        
        print("   ✅ Online Learning: Enabled")
        learning_score += 20
        
        stats = learner.get_learning_stats()
        print(f"   📊 Experiences: {stats['total_experiences']}")
        print(f"   📊 Updates: {stats['updates_count']}")
        
        if stats['total_experiences'] > 0:
            learning_score += 10
    except Exception as e:
        print(f"   ❌ Online Learning Error: {e}")
    
    # 2.2 Trade Memory
    try:
        from ai_agent.trade_memory import TradeMemory
        memory = TradeMemory()
        mem_stats = memory.get_performance_stats()
        
        print(f"   ✅ Trade Memory: {mem_stats.get('total_trades', 0)} trades stored")
        learning_score += 20
        
        if mem_stats.get('total_trades', 0) > 10:
            learning_score += 10
    except Exception as e:
        print(f"   ❌ Trade Memory Error: {e}")
    
    # 2.3 Learning Curve (from training)
    try:
        if agent.total_rewards and len(agent.total_rewards) >= 10:
            first_10 = np.mean(agent.total_rewards[:10])
            last_10 = np.mean(agent.total_rewards[-10:])
            improvement = last_10 - first_10
            
            if improvement > 0:
                print(f"   ✅ Learning Improvement: {improvement:+.2f}")
                learning_score += 20
            else:
                print(f"   ⚠️ Learning Improvement: {improvement:+.2f}")
                learning_score += 10
        else:
            print("   ⚠️ No learning history available")
            learning_score += 5
    except Exception as e:
        print(f"   ❌ Learning Curve Error: {e}")
    
    # 2.4 Auto Re-train capability
    try:
        if hasattr(learner, 'should_retrain'):
            print("   ✅ Auto Re-train: Capability available")
            learning_score += 10
    except:
        pass
    
    # 2.5 Experience Buffer
    try:
        from ai_agent.online_learning import ExperienceBuffer
        buffer = ExperienceBuffer()
        print(f"   ✅ Experience Buffer: {len(buffer)} experiences")
        learning_score += 10
    except Exception as e:
        print(f"   ⚠️ Experience Buffer: {e}")
    
    scores['learning'] = min(100, learning_score)
    print(f"\n   📊 Learning Capability Score: {scores['learning']}/100")
    
    # ========================================
    # 3. OPERATIONAL CORRECTNESS (ความถูกต้องในการทำงาน)
    # ========================================
    print_section("[3] OPERATIONAL CORRECTNESS (ความถูกต้องในการทำงาน)")
    
    correctness_score = 0
    
    # 3.1 Decision Making
    try:
        env = TradingEnvironment(df.tail(200))
        state = env.reset()
        
        actions = {0: 0, 1: 0, 2: 0}
        for _ in range(100):
            action, _, _ = agent.select_action(state)
            actions[action] += 1
            state, _, done, _ = env.step(action)
            if done: break
        
        # Check if AI makes diverse decisions
        min_action = min(actions.values())
        if min_action > 0:
            print(f"   ✅ Diverse Decisions: WAIT={actions[0]}, BUY={actions[1]}, CLOSE={actions[2]}")
            correctness_score += 20
        else:
            print(f"   ⚠️ Limited Decisions: WAIT={actions[0]}, BUY={actions[1]}, CLOSE={actions[2]}")
            correctness_score += 10
    except Exception as e:
        print(f"   ❌ Decision Error: {e}")
    
    # 3.2 Multi-Timeframe Alignment Check
    try:
        result = controller.analyze(
            df=df.tail(500),
            balance=1000,
            point=0.01,
            ai_action=1,
            ai_confidence=0.7,
        )
        
        if controller.market_context:
            ctx = controller.market_context
            print(f"   ✅ MTF Analysis: M15={ctx.trend_m15}, H1={ctx.trend_h1}, H4={ctx.trend_h4}, D1={ctx.trend_d1}")
            correctness_score += 20
            
            if ctx.trade_allowed == False and ctx.trend_alignment < 0.5:
                print(f"   ✅ Correctly Blocked: {result['reason']}")
                correctness_score += 10
    except Exception as e:
        print(f"   ⚠️ MTF Error: {e}")
    
    # 3.3 Risk Management Check
    try:
        rm = RiskManager()
        rm.peak_balance = 1000
        can_trade, reason = rm.can_trade(950)  # 5% drawdown
        
        if can_trade:
            print("   ✅ Risk Check (5% DD): Trading allowed")
            correctness_score += 10
        
        can_trade, reason = rm.can_trade(850)  # 15% drawdown
        if not can_trade:
            print("   ✅ Risk Check (15% DD): Correctly blocked")
            correctness_score += 10
    except Exception as e:
        print(f"   ⚠️ Risk Check Error: {e}")
    
    # 3.4 Strategy Selection
    try:
        from ai_agent.ai_full_control import MarketRegime, TradingStrategy, StrategySelector
        selector = StrategySelector()
        
        # Check strategy for uptrend
        strategy, _ = selector.select(MarketRegime.STRONG_UPTREND, {'trend_strength': 2.5, 'rsi': 55}, 10, 0.01)
        if strategy == TradingStrategy.TREND_FOLLOW:
            print("   ✅ Strategy Selection: Correct for UPTREND")
            correctness_score += 10
        
        # Check strategy for ranging
        strategy, _ = selector.select(MarketRegime.RANGING, {'trend_strength': 0, 'rsi': 50}, 10, 0.01)
        if strategy == TradingStrategy.SCALP:
            print("   ✅ Strategy Selection: Correct for RANGING")
            correctness_score += 10
    except Exception as e:
        print(f"   ⚠️ Strategy Selection Error: {e}")
    
    # 3.5 Session/News Filter
    try:
        from ai_agent.advanced_features import SessionAnalyzer, NewsEventFilter
        
        session = SessionAnalyzer()
        news_filter = NewsEventFilter()
        
        print("   ✅ Session Analyzer: Available")
        print("   ✅ News Filter: Available")
        correctness_score += 10
    except Exception as e:
        print(f"   ⚠️ Filters Error: {e}")
    
    scores['correctness'] = min(100, correctness_score)
    print(f"\n   📊 Operational Correctness Score: {scores['correctness']}/100")
    
    # ========================================
    # 4. OVERALL ASSESSMENT
    # ========================================
    print_section("[4] OVERALL ASSESSMENT")
    
    # Calculate overall score
    overall = (scores['trading'] + scores['learning'] + scores['correctness']) / 3
    
    print(f"\n   📊 SCORES:")
    print(f"   ├─ Trading Capability:     {scores['trading']}/100 ({score_to_grade(scores['trading'])})")
    print(f"   ├─ Learning Capability:    {scores['learning']}/100 ({score_to_grade(scores['learning'])})")
    print(f"   ├─ Operational Correctness: {scores['correctness']}/100 ({score_to_grade(scores['correctness'])})")
    print(f"   └─────────────────────────────────────")
    print(f"       OVERALL SCORE:         {overall:.0f}/100 ({score_to_grade(overall)})")
    
    # Readiness assessment
    print(f"\n   🎯 READINESS ASSESSMENT:")
    
    if overall >= 80:
        print("   ✅ PRODUCTION READY")
        print("   └─ AI พร้อมสำหรับการเทรดจริง")
    elif overall >= 70:
        print("   ⚠️ ALMOST READY")
        print("   └─ AI เกือบพร้อม ควร monitor 1-2 สัปดาห์")
    elif overall >= 60:
        print("   ⚠️ NEEDS IMPROVEMENT")
        print("   └─ AI ต้องปรับปรุงก่อนใช้งานจริง")
    else:
        print("   ❌ NOT READY")
        print("   └─ AI ยังไม่พร้อม ต้องพัฒนาเพิ่ม")
    
    # Recommendations
    print(f"\n   📋 RECOMMENDATIONS:")
    
    if scores['trading'] < 80:
        print("   • Train more episodes for better Win Rate")
    if scores['learning'] < 80:
        print("   • Run more live/paper trades to build experience")
    if scores['correctness'] < 80:
        print("   • Verify MTF alignment logic")
    
    if overall >= 80:
        print("   • Start with Demo account for 1-2 weeks")
        print("   • Then move to Small Live ($100-500)")
        print("   • Gradually increase as confidence builds")
    
    print("\n" + "="*60)
    print(f"   ANALYSIS COMPLETE - Overall: {overall:.0f}/100")
    print("="*60)
    
    return scores, overall


if __name__ == "__main__":
    analyze_ai_system()
