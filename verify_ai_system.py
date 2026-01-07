"""
AI System Verification
======================
ตรวจสอบว่าระบบ AI ใช้งาน modules ครบถ้วนและใช้ข้อมูลจริง
"""

import os
import sys
from datetime import datetime

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def verify_ai_system():
    """ตรวจสอบระบบ AI ทั้งหมด"""
    
    print("="*70)
    print("   AI TRADING SYSTEM VERIFICATION")
    print("="*70)
    print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    results = {
        "modules_loaded": [],
        "modules_failed": [],
        "trained_models": [],
        "real_data_used": False,
    }
    
    # ==========================================
    # 1. Check All AI Modules
    # ==========================================
    print("\n📦 CHECKING AI MODULES...")
    
    modules_to_check = [
        ("AutonomousAI", "ai_agent.autonomous_ai", "create_autonomous_ai"),
        ("PPOWalkForward", "ai_agent.ppo_walk_forward", "create_ppo_walk_forward"),
        ("MetaLearner", "ai_agent.meta_learner", "create_meta_learner"),
        ("EnhancedOnlineLearner", "ai_agent.enhanced_online_learning", "create_enhanced_online_learner"),
        ("ErrorAnalyzer", "ai_agent.error_analyzer", "create_error_analyzer"),
        ("SelfCorrector", "ai_agent.self_corrector", "create_self_corrector"),
        ("StrategyEvolution", "ai_agent.strategy_evolution", "create_strategy_evolution"),
        ("PositionOptimizer", "ai_agent.position_optimizer", "create_position_optimizer"),
        ("AutoTrainer", "ai_agent.auto_trainer", "create_auto_trainer"),
        ("ShadowTrader", "ai_agent.shadow_trader", "create_shadow_trader"),
        ("KnowledgeBase", "ai_agent.knowledge_base", "create_knowledge_base"),
        ("EvolutionEngine", "ai_agent.evolution_engine", "create_evolution_engine"),
        ("MarketIntelligence", "ai_agent.market_intelligence", "create_market_intelligence"),
        ("DecisionBrain", "ai_agent.decision_brain", "create_decision_brain"),
        ("AdaptiveConfidence", "ai_agent.adaptive_confidence", "create_adaptive_confidence"),
        ("PatternAI", "ai_agent.pattern_ai", "create_pattern_ai"),
        ("MultiTimeframe", "ai_agent.multi_timeframe", "create_multi_timeframe"),
        ("RiskBrain", "ai_agent.risk_brain", "create_risk_brain"),
        ("EnsembleBrain", "ai_agent.ensemble_brain", "create_ensemble_brain"),
        ("MarketContext", "ai_agent.market_context", "create_market_context"),
        ("SmartTiming", "ai_agent.smart_timing", "create_smart_timing"),
        ("TrainedPredictor", "ai_agent.trained_predictor", "create_trained_predictor"),
        ("UnifiedEnsemble", "ai_agent.unified_ensemble", "create_unified_ensemble"),
    ]
    
    for name, module_path, factory in modules_to_check:
        try:
            module = __import__(module_path, fromlist=[factory])
            results["modules_loaded"].append(name)
            print(f"  ✅ {name}")
        except Exception as e:
            results["modules_failed"].append((name, str(e)))
            print(f"  ❌ {name}: {e}")
    
    # ==========================================
    # 2. Check Trained Models
    # ==========================================
    print("\n🧠 CHECKING TRAINED MODELS...")
    
    model_paths = [
        ("LSTM", "models/checkpoints/lstm_best.pt"),
        ("XGBoost", "models/checkpoints/xgboost_best.json"),
        ("PPO WalkForward", "ai_agent/models/best_wf.pt"),
    ]
    
    for name, path in model_paths:
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            mtime = datetime.fromtimestamp(os.path.getmtime(path))
            results["trained_models"].append(name)
            print(f"  ✅ {name}: {size:.1f} KB (modified: {mtime.strftime('%Y-%m-%d %H:%M')})")
        else:
            print(f"  ❌ {name}: Not found at {path}")
    
    # ==========================================
    # 3. Check Training Data  
    # ==========================================
    print("\n📊 CHECKING TRAINING DATA...")
    
    data_path = "data/training/GOLD_H1.csv"
    if os.path.exists(data_path):
        import pandas as pd
        df = pd.read_csv(data_path)
        results["real_data_used"] = True
        print(f"  ✅ Real data: {len(df):,} rows")
        print(f"  ✅ Date range: {df.iloc[0]['datetime'][:10]} to {df.iloc[-1]['datetime'][:10]}")
        print(f"  ✅ Columns: {len(df.columns)}")
    else:
        print(f"  ❌ Training data not found at {data_path}")
    
    # ==========================================
    # 4. Check Autonomous AI Integration
    # ==========================================
    print("\n🔗 CHECKING AUTONOMOUS AI INTEGRATION...")
    
    try:
        from ai_agent.autonomous_ai import create_autonomous_ai
        ai = create_autonomous_ai()
        
        # Check which modules are initialized
        ai_modules = []
        for attr in dir(ai):
            if not attr.startswith('_'):
                obj = getattr(ai, attr)
                if hasattr(obj, '__class__') and not callable(obj):
                    ai_modules.append(attr)
        
        print(f"  ✅ AutonomousAI initialized with {len(ai_modules)} components")
        
        # Check specific modules
        checks = [
            ("market_intel", "MarketIntelligence"),
            ("decision_brain", "DecisionBrain"),
            ("adaptive_confidence", "AdaptiveConfidence"),
            ("pattern_ai", "PatternAI"),
            ("multi_tf", "MultiTimeframe"),
            ("risk_brain", "RiskBrain"),
            ("ensemble_brain", "EnsembleBrain"),
            ("market_context", "MarketContext"),
            ("smart_timing", "SmartTiming"),
            ("trained_predictor", "TrainedPredictor"),
            ("unified_ensemble", "UnifiedEnsemble"),
        ]
        
        print("\n  Module Integration:")
        for attr, name in checks:
            if hasattr(ai, attr):
                obj = getattr(ai, attr)
                print(f"    ✅ {name}: {type(obj).__name__}")
            else:
                print(f"    ❌ {name}: Not integrated")
        
        # Check trained predictor status
        if hasattr(ai, 'trained_predictor'):
            status = ai.trained_predictor.get_status()
            print(f"\n  Trained Predictor Status:")
            print(f"    LSTM loaded: {status['lstm_loaded']}")
            print(f"    XGBoost loaded: {status['xgb_loaded']}")
            
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ==========================================
    # 5. Check make_decision Flow
    # ==========================================
    print("\n🎯 CHECKING DECISION FLOW...")
    
    try:
        # Read make_decision to verify modules are used
        with open("ai_agent/autonomous_ai.py", "r", encoding="utf-8") as f:
            content = f.read()
        
        decision_checks = [
            ("market_intel.analyze", "MarketIntelligence"),
            ("pattern_ai.analyze", "PatternAI"),
            ("multi_tf.analyze", "MultiTimeframe"),
            ("risk_brain.evaluate", "RiskBrain"),
            ("adaptive_confidence.calculate", "AdaptiveConfidence"),
            ("decision_brain.analyze", "DecisionBrain"),
            ("trained_predictor.predict", "TrainedPredictor"),
            ("unified_ensemble.predict", "UnifiedEnsemble"),
        ]
        
        for method, name in decision_checks:
            if method in content:
                print(f"  ✅ {name} used in decision flow")
            else:
                print(f"  ⚠️ {name} NOT found in decision flow")
                
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # ==========================================
    # Summary
    # ==========================================
    print("\n" + "="*70)
    print("   VERIFICATION SUMMARY")
    print("="*70)
    
    print(f"\n  📦 Modules Loaded: {len(results['modules_loaded'])}/{len(modules_to_check)}")
    print(f"  🧠 Trained Models: {len(results['trained_models'])}/3")
    print(f"  📊 Real Data: {'Yes' if results['real_data_used'] else 'No'}")
    
    if results["modules_failed"]:
        print(f"\n  ⚠️ Failed Modules:")
        for name, error in results["modules_failed"]:
            print(f"     - {name}")
    
    # Final verdict
    all_ok = (
        len(results["modules_loaded"]) >= 20 and
        len(results["trained_models"]) >= 2 and
        results["real_data_used"]
    )
    
    print("\n" + "="*70)
    if all_ok:
        print("  ✅ SYSTEM VERIFICATION PASSED")
    else:
        print("  ⚠️ SYSTEM NEEDS ATTENTION")
    print("="*70)
    
    return results


if __name__ == "__main__":
    verify_ai_system()
