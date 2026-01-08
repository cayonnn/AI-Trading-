"""
Autonomous AI Trading Controller
=================================
ระบบ AI เทรดอัตโนมัติ 100% ที่รวมทุก modules เข้าด้วยกัน

Features:
1. Full Autonomous Trading - ตัดสินใจเองทั้งหมด
2. Self-Learning - เรียนรู้และปรับตัวตลอดเวลา
3. Self-Correcting - แก้ไขข้อผิดพลาดอัตโนมัติ
4. Self-Evolving - พัฒนาตัวเองอัตโนมัติ
5. Production Ready - พร้อมใช้งานจริง
"""

import numpy as np
import pandas as pd
import torch
import os
import json
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger

# Import all modules
from ai_agent.enhanced_online_learning import EnhancedOnlineLearner, create_enhanced_online_learner
from ai_agent.error_analyzer import ErrorAnalyzer, create_error_analyzer
from ai_agent.self_corrector import SelfCorrector, create_self_corrector
from ai_agent.meta_learner import MetaLearner, create_meta_learner
from ai_agent.curiosity_module import CuriosityModule, create_curiosity_module
from ai_agent.strategy_evolution import StrategyEvolution, create_strategy_evolution
from ai_agent.position_optimizer import PositionOptimizer, create_position_optimizer
from ai_agent.auto_trainer import AutoTrainer, create_auto_trainer
from ai_agent.shadow_trader import ShadowTrader, create_shadow_trader
from ai_agent.knowledge_base import KnowledgeBase, create_knowledge_base
from ai_agent.evolution_engine import EvolutionEngine, create_evolution_engine

# Intelligence Enhancement Modules
from ai_agent.market_intelligence import MarketIntelligence, create_market_intelligence
from ai_agent.decision_brain import DecisionBrain, create_decision_brain
from ai_agent.adaptive_confidence import AdaptiveConfidence, create_adaptive_confidence

# Advanced AI Modules
from ai_agent.pattern_ai import PatternAI, create_pattern_ai
from ai_agent.multi_timeframe import MultiTimeframe, create_multi_timeframe
from ai_agent.risk_brain import RiskBrain, create_risk_brain
from ai_agent.risk_manager import AIRiskManager, create_ai_risk_manager

# Ultimate Brain Modules
from ai_agent.ensemble_brain import EnsembleBrain, create_ensemble_brain
from ai_agent.market_context import MarketContext, create_market_context

# v3.1: Database for persistent trade history
from database_manager import DatabaseManager
from ai_agent.smart_timing import SmartTiming, create_smart_timing
from ai_agent.master_brain import MasterBrain, create_master_brain  # Supreme AI
from ai_agent.backtest_loader import preload_master_brain_experience  # Preload experience

# Trained Models
from ai_agent.trained_predictor import TrainedModelPredictor, create_trained_predictor

# Unified Ensemble (LSTM + XGBoost + PPO)
from ai_agent.unified_ensemble import UnifiedEnsemble, create_unified_ensemble

# Trained PPO Agent (v2.0)
from ai_agent.ppo_walk_forward import PPOAgentWalkForward as PPOAgent, TradingState

# v3.3: New modules
from ai_agent.news_filter import NewsFilter, get_news_filter
from ai_agent.telegram_alert import TelegramAlert, get_telegram
from ai_agent.model_monitor import ModelMonitor, get_model_monitor

# v3.4: Intelligence Enhancements
from ai_agent.dynamic_thresholds import DynamicRegimeThresholds, create_dynamic_thresholds
from ai_agent.sentiment_integration import SentimentIntegration, get_sentiment


@dataclass
class TradeDecision:
    """การตัดสินใจเทรด"""
    action: str  # 'LONG', 'WAIT', 'CLOSE'
    confidence: float
    position_size: float
    stop_loss: float
    take_profit: float
    reason: str
    
    # Risk info
    risk_pct: float = 0.0
    reward_ratio: float = 0.0
    
    # Metadata
    timestamp: datetime = None
    strategy_version: str = ""
    regime: str = ""
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class MarketState:
    """สถานะตลาดปัจจุบัน"""
    price: float
    trend: float
    volatility: float
    regime: str
    rsi: float = 50.0
    atr: float = 0.0
    momentum: float = 0.0
    volume_ratio: float = 1.0


class AutonomousAI:
    """
    Unified Autonomous AI Trading Controller
    =========================================
    
    ความสามารถ:
    1. วิเคราะห์ตลาดและตัดสินใจเอง 100%
    2. เรียนรู้จากทุก trade
    3. แก้ไขความผิดพลาดอัตโนมัติ
    4. คำนวณ position size อัตโนมัติ
    5. A/B test กลยุทธ์ใหม่
    6. พัฒนาตัวเองต่อเนื่อง
    """
    
    def __init__(
        self,
        capital: float = 10000.0,
        symbol: str = "XAUUSD",
        model_dir: str = "ai_agent/models",
        data_dir: str = "ai_agent/data",
    ):
        self.capital = capital
        self.symbol = symbol
        self.model_dir = model_dir
        self.data_dir = data_dir
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        logger.info("="*60)
        logger.info("   AUTONOMOUS AI TRADING SYSTEM")
        logger.info("="*60)
        
        # Initialize all modules
        logger.info("Loading modules...")
        
        # Core learning
        self.learner = create_enhanced_online_learner()
        self.meta_learner = create_meta_learner()
        self.curiosity = create_curiosity_module()
        
        # Error handling
        self.error_analyzer = create_error_analyzer()
        self.self_corrector = create_self_corrector(self.error_analyzer)
        
        # Strategy
        self.strategy_evolution = create_strategy_evolution()
        self.position_optimizer = create_position_optimizer()
        
        # Training
        self.auto_trainer = create_auto_trainer()
        
        # Testing & Evolution
        self.shadow_trader = create_shadow_trader()
        self.knowledge_base = create_knowledge_base()
        self.evolution_engine = create_evolution_engine()
        
        # Intelligence Enhancement
        self.market_intel = create_market_intelligence()
        self.decision_brain = create_decision_brain()
        self.adaptive_confidence = create_adaptive_confidence()
        
        # Advanced AI
        self.pattern_ai = create_pattern_ai()
        self.multi_tf = create_multi_timeframe()
        self.risk_brain = create_risk_brain()
        self.ai_risk_manager = create_ai_risk_manager()  # AI controls lot/SL/TP
        
        # Ultimate Brain
        self.ensemble_brain = create_ensemble_brain()
        self.market_context = create_market_context()
        self.smart_timing = create_smart_timing()
        self.master_brain = create_master_brain()  # Supreme AI - human-like thinking
        
        # Preload MasterBrain with backtest experience
        try:
            preload_master_brain_experience(self.master_brain, max_trades=100)
            logger.info("✅ MasterBrain preloaded with 100 backtest trades")
        except Exception as e:
            logger.warning(f"Could not preload experience: {e}")
        
        # Trained Models (LSTM + XGBoost)
        self.trained_predictor = create_trained_predictor()
        
        # Trained PPO Agent
        self.ppo_agent = PPOAgent(state_dim=11)  # 8 features + 3 state vars
        ppo_loaded = self.ppo_agent.load("final")  # Load final.pt (latest trained)
        if ppo_loaded:
            logger.info("Loaded trained PPO model (final.pt)")
        else:
            logger.warning("No trained PPO model found, using untrained")
        
        # Unified Ensemble (LSTM + XGBoost + PPO)
        self.unified_ensemble = create_unified_ensemble()
        
        # v3.1: Database for persistent trade history
        self.db = DatabaseManager("trading_data.db")
        logger.info("✅ Database connected: trading_data.db")
        
        # v3.3: New modules
        self.news_filter = get_news_filter()
        self.telegram = get_telegram()  # Set token/chat_id via .env or config
        self.model_monitor = get_model_monitor()
        logger.info("✅ v3.3 modules: NewsFilter, Telegram, ModelMonitor")
        
        # v3.4: Intelligence Enhancements
        self.dynamic_thresholds = create_dynamic_thresholds()
        self.sentiment = get_sentiment()
        logger.info("✅ v3.4 modules: DynamicRegimeThresholds, SentimentIntegration")
        
        # State
        self.current_position = 0  # 0 = flat, 1 = long
        self.entry_price = 0.0
        self.entry_state = None
        self.consecutive_losses = 0
        self.daily_trades = 0
        self.daily_pnl = 0.0
        
        # Trade history
        self.trade_history: List[Dict] = []
        
        # v3.1: Load trade history from DB and teach MasterBrain
        self._load_history_from_db()
        
        logger.info("All modules loaded successfully")
        logger.info(f"Capital: ${capital:,.2f}")
        logger.info(f"Symbol: {symbol}")
        logger.info("="*60)
    
    def _load_history_from_db(self):
        """โหลดประวัติเทรดจาก DB และสอน MasterBrain"""
        try:
            # Get recent closed trades
            cursor = self.db.conn.cursor()
            cursor.execute("""
                SELECT entry_price, exit_price, pnl, pnl_pct, status, entry_time
                FROM trades 
                WHERE status = 'CLOSED'
                ORDER BY entry_time DESC
                LIMIT 200
            """)
            trades = cursor.fetchall()
            
            if trades:
                logger.info(f"📚 Loading {len(trades)} trades from DB to MasterBrain...")
                for trade in trades:
                    entry_price, exit_price, pnl, pnl_pct, status, entry_time = trade
                    
                    # Teach MasterBrain
                    market_state = {
                        'price': exit_price or entry_price,
                        'regime': 'unknown',
                        'volatility': 0.5,
                    }
                    self.master_brain.record_trade_result(
                        action='LONG',
                        result='win' if (pnl or 0) > 0 else 'loss',
                        pnl=pnl or 0,
                        market_state=market_state,
                    )
                
                logger.info(f"✅ MasterBrain learned from {len(trades)} historical trades")
            else:
                logger.info("📭 No historical trades in DB")
                
        except Exception as e:
            logger.warning(f"Could not load history from DB: {e}")
    
    def analyze_market(self, data: pd.DataFrame) -> MarketState:
        """วิเคราะห์สถานะตลาดปัจจุบัน"""
        
        if len(data) < 50:
            return MarketState(
                price=data['close'].iloc[-1],
                trend=0,
                volatility=0.02,
                regime="unknown",
            )
        
        close = data['close']
        
        # Current price
        price = close.iloc[-1]
        
        # Trend
        trend = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
        
        # Volatility
        returns = close.pct_change().dropna()
        volatility = returns.iloc[-20:].std()
        
        # RSI
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ATR
        if 'high' in data.columns and 'low' in data.columns:
            high_low = data['high'] - data['low']
            atr = high_low.rolling(14).mean().iloc[-1]
        else:
            atr = volatility * price
        
        # Momentum
        momentum = (close.iloc[-1] - close.iloc[-10]) / close.iloc[-10]
        
        # Regime detection
        regime = self.meta_learner.detect_regime(trend, volatility, momentum)
        
        # Update meta-learner
        self.meta_learner.update_regime(trend, volatility, momentum)
        
        return MarketState(
            price=price,
            trend=trend,
            volatility=volatility,
            regime=regime,
            rsi=rsi,
            atr=atr,
            momentum=momentum,
        )
    
    def make_decision(
        self,
        data: pd.DataFrame,
        market: MarketState = None,
    ) -> TradeDecision:
        """
        ตัดสินใจเทรดอัตโนมัติ 100%
        
        Flow:
        1. วิเคราะห์ตลาด
        2. ตรวจสอบกับ self-corrector
        3. คำนวณ confidence
        4. คำนวณ position size
        5. สร้าง trade decision
        """
        
        # 1. Analyze market
        if market is None:
            market = self.analyze_market(data)
        
        # ============================================
        # v3.3: Pre-trade checks
        # ============================================
        
        # Check news filter (pause during high-impact events)
        can_trade_news, news_reason = self.news_filter.should_trade()
        if not can_trade_news:
            logger.info(f"📰 {news_reason}")
            return TradeDecision(
                action='WAIT', confidence=0.0, should_execute=False,
                entry_price=market.price, stop_loss=0, take_profit=0,
                position_size=0.0, reasoning=news_reason,
            )
        
        # Check daily loss limit
        can_trade_daily, daily_reason = self.master_brain.check_daily_limit(self.capital)
        if not can_trade_daily:
            logger.info(f"🛑 {daily_reason}")
            return TradeDecision(
                action='WAIT', confidence=0.0, should_execute=False,
                entry_price=market.price, stop_loss=0, take_profit=0,
                position_size=0.0, reasoning=daily_reason,
            )
        
        # Check if good trading hour
        is_good_hour, hour_reason = self.master_brain.is_good_trading_hour()
        if not is_good_hour:
            logger.debug(f"⏰ {hour_reason}")
            # Don't block, just reduce confidence later
        
        # ============================================
        # v3.4: Dynamic Thresholds & Sentiment Check
        # ============================================
        
        # Get regime-specific thresholds
        regime_thresholds = self.dynamic_thresholds.get_thresholds(
            regime=market.regime,
            volatility=market.volatility,
        )
        
        if not regime_thresholds["can_trade"]:
            logger.debug(f"📊 {regime_thresholds['reason']}")
            return TradeDecision(
                action='WAIT', confidence=0.0,
                position_size=0, stop_loss=0, take_profit=0,
                reason=regime_thresholds['reason'],
                regime=market.regime,
            )
        
        # Analyze market sentiment
        sentiment_data = {
            "trend": market.trend,
            "momentum": market.momentum,
            "volatility": market.volatility,
            "regime": market.regime,
        }
        sentiment_summary = self.sentiment.analyze(sentiment_data)
        
        if sentiment_summary.should_avoid:
            logger.info(f"⚠️ Sentiment: {sentiment_summary.avoid_reason}")
            return TradeDecision(
                action='WAIT', confidence=0.0,
                position_size=0, stop_loss=0, take_profit=0,
                reason=f"Sentiment: {sentiment_summary.avoid_reason}",
                regime=market.regime,
            )
        
        # 2. Get ADVANCED Market Intelligence
        intel_analysis = self.market_intel.analyze(data)
        
        # 3. Pattern AI Analysis
        pattern_analysis = self.pattern_ai.analyze(data)
        
        # 4. Multi-Timeframe Analysis (using same data as proxy)
        mtf_signal = self.multi_tf.analyze(base_data=data)
        
        # 5. Get TRAINED PPO prediction
        state = self._extract_state(data)
        ppo_state = TradingState(
            price_features=state[:8],
            position=self.current_position,
            unrealized_pnl=0.0,
            portfolio_value=self.capital,
        )
        try:
            ppo_action, ppo_log_prob, ppo_action_prob = self.ppo_agent.select_action(ppo_state)
            # Use actual action probability as confidence (now directly from PPO)
            ppo_confidence = ppo_action_prob  # Direct probability, no scaling needed
        except Exception as e:
            logger.debug(f"PPO fallback to meta_learner: {e}")
            ppo_action, ppo_confidence = self.meta_learner.select_action(state)
        
        # 6. Get TRAINED model predictions (LSTM + XGBoost)
        trained_pred = self.trained_predictor.predict(data)
        lstm_pred = (trained_pred.lstm_prediction, trained_pred.lstm_confidence)
        xgb_pred = (trained_pred.xgb_prediction, trained_pred.xgb_confidence)
        ppo_pred = (ppo_action, ppo_confidence)
        
        # 7. UNIFIED ENSEMBLE - Combine LSTM + XGBoost + PPO (as advisors)
        ensemble_result = self.unified_ensemble.predict(lstm_pred, xgb_pred, ppo_pred)
        
        # 8. MASTER BRAIN - Supreme AI with human-like thinking
        # Prepare model votes for Master
        model_votes = {
            'lstm': ('LONG' if trained_pred.lstm_prediction == 1 else 'WAIT', trained_pred.lstm_confidence),
            'xgb': ('LONG' if trained_pred.xgb_prediction == 1 else 'WAIT', trained_pred.xgb_confidence),
            'ppo': ('LONG' if ppo_action == 1 else 'WAIT' if ppo_action == 0 else 'CLOSE', ppo_confidence),
        }
        
        # Prepare market data and indicators for Master
        master_market = {
            'price': market.price,
            'trend': market.trend,
            'volatility': market.volatility,
            'regime': market.regime,
            'atr': market.atr,
        }
        
        master_indicators = {
            'rsi': market.rsi,
            'macd': intel_analysis.get('macd', 0),
            'bb_position': intel_analysis.get('bb_position', 0.5),
        }
        
        # Master thinks and decides
        master_thought = self.master_brain.think(master_market, model_votes, master_indicators)
        
        # v3.4: Master's decision has priority when confident
        # Use Master's decision if:
        # 1. Master explicitly overrides, OR
        # 2. Master decides LONG/SHORT with confidence > 55%
        use_master_decision = (
            master_thought.override_models or
            (master_thought.suggested_action in ["LONG", "SHORT"] and master_thought.confidence >= 0.55)
        )
        
        if use_master_decision:
            if master_thought.suggested_action == "LONG":
                action = 1
            elif master_thought.suggested_action == "SHORT":
                action = 2
            else:
                action = 0
            ai_confidence = master_thought.confidence
            logger.info(f"🧠 MASTER DECISION: {master_thought.suggested_action} ({master_thought.confidence:.0%})")
        else:
            # Fallback to ensemble
            if ensemble_result.action == "LONG":
                action = 1
            elif ensemble_result.action == "SHORT":
                action = 2
            else:
                action = 0
            ai_confidence = ensemble_result.confidence
        
        logger.debug(f"Ensemble: {ensemble_result.reasoning}")
        logger.debug(f"Master: {master_thought.reasoning}")
        
        # 8. Risk Brain evaluation
        risk_decision = self.risk_brain.evaluate_risk(
            equity=self.capital + self.daily_pnl,
            volatility=market.volatility,
            confidence=ai_confidence,
            regime=market.regime,
        )
        
        # 9. Apply Adaptive Confidence
        adjusted_confidence, conf_factors = self.adaptive_confidence.calculate_confidence(
            raw_confidence=ai_confidence,
            regime=market.regime,
            volatility=market.volatility,
        )
        
        # 10. Combine all signals for final confidence
        pattern_boost = pattern_analysis['confidence'] * 0.15
        mtf_boost = mtf_signal.confluence_score * 0.10
        ensemble_boost = 0.05 if ensemble_result.consensus else 0
        combined_confidence = min(0.95, adjusted_confidence + pattern_boost + mtf_boost + ensemble_boost)
        
        # 9. Get Decision Brain analysis
        trade_setup = self.decision_brain.analyze_setup(
            current_price=market.price,
            market_intel=intel_analysis,
            atr=market.atr,
        )
        
        # 6. Check with self-corrector
        can_trade, reason = self.self_corrector.should_trade(
            confidence=adjusted_confidence,
            volatility=market.volatility,
            trend=market.trend,
            regime=market.regime,
            rsi=market.rsi,
        )
        
        # 7. Handle current position
        if self.current_position == 1:
            return self._check_exit(data, market, state)
        
        # 8. Entry decision using Decision Brain
        should_enter, enter_reason = self.decision_brain.should_enter(trade_setup)
        
        if not can_trade or not should_enter:
            # Use Master's human-like reasoning for blocked decisions
            blocked_reason = (
                f"{master_thought.reasoning} | "
                f"BLOCKED: {reason if not can_trade else enter_reason}"
            )
            
            logger.debug(
                f"Entry blocked: CanTrade={can_trade} ({reason}) | "
                f"ShouldEnter={should_enter} ({enter_reason}) | "
                f"Confidence={adjusted_confidence:.1%}"
            )
            return TradeDecision(
                action="WAIT",
                confidence=adjusted_confidence,
                position_size=0,
                stop_loss=0,
                take_profit=0,
                reason=blocked_reason,
                regime=market.regime,
                strategy_version=self.evolution_engine.active_version or "default",
            )
        
        # 9. Use intelligence-enhanced confidence
        final_confidence = max(adjusted_confidence, trade_setup.confidence)
        
        # 10. Check regime knowledge
        regime_params = self.knowledge_base.get_regime_parameters(market.regime)
        
        # 11. Check matching patterns  
        patterns = self.knowledge_base.find_matching_patterns({
            'trend': market.trend,
            'volatility': market.volatility,
            'rsi': market.rsi,
            'regime': market.regime,
        })
        
        pattern_boost = len([p for p in patterns if p.success_rate > 0.6]) * 0.02
        final_confidence = min(0.95, final_confidence + pattern_boost)
        
        # 7. AI CONTROLS LOT SIZE, SL, TP DYNAMICALLY
        # Calculate trend strength from market analysis
        trend_strength = abs(intel_analysis.get('trend_score', 0.5))
        
        risk_params = self.ai_risk_manager.calculate_parameters(
            entry_price=market.price,
            atr=market.atr,
            confidence=final_confidence,
            regime=market.regime,
            volatility=market.volatility,
            equity=self.capital + self.daily_pnl,
            is_long=(action == 1),
            trend_strength=trend_strength,
        )
        
        # Use AI-calculated values
        stop_loss = risk_params.stop_loss
        take_profit = risk_params.take_profit
        reward_ratio = risk_params.reward_ratio
        position_value = risk_params.lot_size * market.price * 100  # Gold: 100 oz per lot
        
        logger.debug(f"AI Risk: Lot={risk_params.lot_size}, SL={risk_params.sl_pips}pips, TP={risk_params.tp_pips}pips, R:R={reward_ratio}")
        
        # 9. Final decision - Log thresholds for visibility
        optimal_threshold = regime_params['optimal_confidence']
        logger.debug(
            f"Thresholds: Final={final_confidence:.1%} vs Optimal={optimal_threshold:.1%} | "
            f"Regime={market.regime} | Action={'LONG' if action == 1 else 'WAIT'} | "
            f"CanTrade={can_trade}"
        )
        
        # Build human-like reasoning
        thinking = self._build_reasoning(
            market=market,
            intel=intel_analysis,
            confidence=final_confidence,
            regime=market.regime,
            risk_params=risk_params,
            patterns=patterns,
            action=action,
        )
        
        # LONG decision
        if final_confidence >= optimal_threshold and action == 1:
            return TradeDecision(
                action="LONG",
                confidence=final_confidence,
                position_size=position_value,
                stop_loss=stop_loss,
                take_profit=take_profit,
                reason=thinking,
                risk_pct=risk_params.risk_pct,
                reward_ratio=reward_ratio,
                regime=market.regime,
                strategy_version=self.evolution_engine.active_version or "default",
            )
        
        # SHORT decision (action == 2)
        if final_confidence >= optimal_threshold and action == 2:
            # For SHORT: swap SL and TP direction
            short_sl = market.price + abs(market.price - stop_loss)
            short_tp = market.price - abs(take_profit - market.price)
            
            return TradeDecision(
                action="SHORT",
                confidence=final_confidence,
                position_size=position_value,
                stop_loss=short_sl,
                take_profit=short_tp,
                reason=thinking,
                risk_pct=risk_params.risk_pct,
                reward_ratio=reward_ratio,
                regime=market.regime,
                strategy_version=self.evolution_engine.active_version or "default",
            )
        
        return TradeDecision(
            action="WAIT",
            confidence=final_confidence,
            position_size=0,
            stop_loss=0,
            take_profit=0,
            reason=thinking,
            regime=market.regime,
            strategy_version=self.evolution_engine.active_version or "default",
        )
    
    def _build_reasoning(
        self,
        market: MarketState,
        intel: Dict,
        confidence: float,
        regime: str,
        risk_params,
        patterns: List,
        action: int,
    ) -> str:
        """
        Build human-like reasoning for trading decision.
        
        AI explains WHY it's making this decision like a human trader would.
        """
        reasoning_parts = []
        
        # 1. Market Assessment
        trend_desc = "bullish" if market.trend > 0.5 else "bearish" if market.trend < -0.5 else "neutral"
        vol_desc = "high" if market.volatility > 0.6 else "low" if market.volatility < 0.3 else "moderate"
        reasoning_parts.append(f"Market is {trend_desc} with {vol_desc} volatility ({regime} regime)")
        
        # 2. Technical Analysis
        if market.rsi > 70:
            reasoning_parts.append("RSI overbought - caution for longs")
        elif market.rsi < 30:
            reasoning_parts.append("RSI oversold - potential reversal")
        else:
            reasoning_parts.append(f"RSI neutral at {market.rsi:.0f}")
        
        # 3. Pattern Recognition
        matching_patterns = len([p for p in patterns if p.success_rate > 0.6])
        if matching_patterns > 0:
            reasoning_parts.append(f"{matching_patterns} high-probability patterns detected")
        
        # 4. Model Consensus
        intel_score = intel.get('trend_score', 0)
        if intel_score > 0.3:
            reasoning_parts.append("Models agree on bullish direction")
        elif intel_score < -0.3:
            reasoning_parts.append("Models agree on bearish direction")
        else:
            reasoning_parts.append("Mixed signals from models")
        
        # 5. Risk Assessment
        if risk_params:
            reasoning_parts.append(
                f"Risk: {risk_params.risk_pct:.1%} of equity, R:R={risk_params.reward_ratio:.1f}"
            )
        
        # 6. Final Decision
        if action == 1 and confidence >= 0.6:
            reasoning_parts.append(f"DECISION: LONG with {confidence:.0%} confidence")
        elif confidence < 0.6:
            reasoning_parts.append(f"DECISION: WAIT - confidence {confidence:.0%} below threshold")
        else:
            reasoning_parts.append(f"DECISION: WAIT - conditions not favorable")
        
        return " | ".join(reasoning_parts)
    
    def _check_exit(
        self,
        data: pd.DataFrame,
        market: MarketState,
        state: np.ndarray,
    ) -> TradeDecision:
        """ตรวจสอบว่าควร exit หรือไม่"""
        
        config = self.evolution_engine.get_active_config()
        
        # Calculate current P&L
        pnl_pct = (market.price - self.entry_price) / self.entry_price
        
        # Get action from meta-learner
        action, confidence = self.meta_learner.select_action(state)
        
        # Exit conditions
        should_exit = False
        exit_reason = ""
        
        # Action says close
        if action == 2:  # CLOSE
            should_exit = True
            exit_reason = "AI signals close"
        
        # Take profit
        elif pnl_pct >= config.get('target_atr', 3.0) * 0.01:
            should_exit = True
            exit_reason = f"Take profit reached ({pnl_pct:.2%})"
        
        # Stop loss
        elif pnl_pct <= -config.get('stop_atr', 1.5) * 0.01:
            should_exit = True
            exit_reason = f"Stop loss hit ({pnl_pct:.2%})"
        
        # Regime changed to unfavorable
        elif market.regime in ['volatile', 'ranging'] and pnl_pct > 0:
            should_exit = True
            exit_reason = f"Unfavorable regime change ({market.regime})"
        
        if should_exit:
            return TradeDecision(
                action="CLOSE",
                confidence=confidence,
                position_size=0,
                stop_loss=0,
                take_profit=0,
                reason=exit_reason,
                regime=market.regime,
                strategy_version=self.evolution_engine.active_version or "default",
            )
        
        return TradeDecision(
            action="HOLD",
            confidence=confidence,
            position_size=0,
            stop_loss=0,
            take_profit=0,
            reason=f"Holding position ({pnl_pct:+.2%})",
            regime=market.regime,
            strategy_version=self.evolution_engine.active_version or "default",
        )
    
    def execute_decision(
        self,
        decision: TradeDecision,
        data: pd.DataFrame,
    ) -> Dict:
        """Execute trade decision และบันทึกผล"""
        
        result = {
            "action": decision.action,
            "timestamp": datetime.now(),
            "executed": False,
        }
        
        if decision.action == "LONG" and self.current_position == 0:
            # Open position
            self.current_position = 1
            self.entry_price = data['close'].iloc[-1]
            self.entry_state = self._extract_state(data)
            self.daily_trades += 1
            
            # Record shadow trade if testing
            if self.shadow_trader.shadow_strategy:
                self.shadow_trader.record_trade_decision(
                    self.shadow_trader.get_active_strategy(),
                    "LONG",
                    self.entry_price,
                )
            
            result["executed"] = True
            result["entry_price"] = self.entry_price
            result["position_size"] = decision.position_size
            result["stop_loss"] = decision.stop_loss
            result["take_profit"] = decision.take_profit
            
            logger.info(
                f"📈 LONG @ {self.entry_price:.2f} | "
                f"Size: ${decision.position_size:.0f} | "
                f"SL: {decision.stop_loss:.2f} | "
                f"TP: {decision.take_profit:.2f}"
            )
        
        elif decision.action == "CLOSE" and self.current_position == 1:
            # Close position
            exit_price = data['close'].iloc[-1]
            exit_state = self._extract_state(data)
            
            pnl_pct = (exit_price - self.entry_price) / self.entry_price
            pnl = pnl_pct * 1000  # Simplified
            
            self.current_position = 0
            self.daily_pnl += pnl
            
            # Record trade
            trade_result = {
                "entry_price": self.entry_price,
                "exit_price": exit_price,
                "pnl": pnl,
                "pnl_pct": pnl_pct,
                "regime": decision.regime,
                "confidence": decision.confidence,
            }
            
            # Update all modules
            self._record_trade_result(trade_result, exit_state)
            
            # Update consecutive losses
            if pnl < 0:
                self.consecutive_losses += 1
            else:
                self.consecutive_losses = 0
            
            result["executed"] = True
            result["exit_price"] = exit_price
            result["pnl"] = pnl
            result["pnl_pct"] = pnl_pct
            
            emoji = "✅" if pnl > 0 else "❌"
            logger.info(
                f"{emoji} CLOSE @ {exit_price:.2f} | "
                f"P&L: ${pnl:.2f} ({pnl_pct:+.2%}) | "
                f"Reason: {decision.reason}"
            )
        
        return result
    
    def _record_trade_result(self, result: Dict, exit_state: np.ndarray):
        """บันทึกผล trade ไปยังทุก modules (Online Learning)"""
        
        # Learner
        self.learner.record_trade(
            trade_id=f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            symbol=self.symbol,
            entry_state=self.entry_state,
            entry_price=result['entry_price'],
            exit_state=exit_state,
            exit_price=result['exit_price'],
            action=1,  # LONG
            holding_bars=1,
            volatility=0.02,
            trend=0.01,
            regime=result['regime'],
            confidence=result['confidence'],
        )
        
        # Auto trainer
        self.auto_trainer.record_trade(result['pnl'], result['pnl_pct'])
        
        # Position optimizer
        self.position_optimizer.record_trade(
            result['pnl'], result['pnl_pct'],
            current_equity=self.capital + self.daily_pnl,
        )
        
        # Evolution engine
        self.evolution_engine.update_performance(
            self.evolution_engine.active_version,
            result,
        )
        
        # Knowledge base
        self.knowledge_base.update_regime_knowledge(result['regime'], result)
        
        # ============================================
        # v3.1: Online Learning for all models
        # ============================================
        
        # MasterBrain learns from trade
        market_state = {
            'price': result['exit_price'],
            'regime': result['regime'],
            'volatility': 0.5,
            'sl_atr': result.get('sl_pips', 15) / 15,  # Convert back to ATR
            'tp_atr': result.get('tp_pips', 30) / 15,
        }
        self.master_brain.record_trade_result(
            action='LONG',
            result='win' if result['pnl'] > 0 else 'loss',
            pnl=result['pnl'],
            market_state=market_state,
        )
        
        # Trained predictor learns (LSTM & XGBoost)
        if hasattr(self, 'trained_predictor') and self.entry_state is not None:
            target = 1 if result['pnl'] > 0 else 0
            self.trained_predictor.record_trade_result(
                features=self.entry_state,
                target=target,
                model_prediction=1,  # Was LONG
                actual_direction='LONG',
            )
        
        logger.debug(f"📚 All models learned from trade: {result['pnl']:.2f}")
        
        # v3.1: Save trade to database for persistence
        try:
            trade_id = f"TRADE_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.db.save_trade({
                'trade_id': trade_id,
                'signal_id': result.get('signal_id', trade_id),
                'symbol': self.symbol,
                'entry_time': result.get('entry_time', datetime.now()),
                'entry_price': result['entry_price'],
                'position_size': result.get('position_size', 0.01),
                'exit_time': datetime.now(),
                'exit_price': result['exit_price'],
                'exit_reason': result.get('exit_reason', 'TP' if result['pnl'] > 0 else 'SL'),
                'pnl': result['pnl'],
                'pnl_pct': result.get('pnl_pct', 0),
                'stop_loss': result.get('stop_loss', 0),
                'take_profit': result.get('take_profit', 0),
                'status': 'CLOSED',
            })
            logger.info(f"💾 Trade saved to DB: {trade_id}, PnL: ${result['pnl']:.2f}")
        except Exception as e:
            logger.warning(f"Could not save trade to DB: {e}")
        
        # Add to history
        self.trade_history.append(result)
    
    def _extract_state(self, data: pd.DataFrame) -> np.ndarray:
        """สร้าง state vector จาก market data"""
        
        if len(data) < 20:
            return np.zeros(11, dtype=np.float32)
        
        close = data['close']
        returns = close.pct_change().dropna()
        
        state = np.array([
            returns.iloc[-1] if len(returns) > 0 else 0,  # Last return
            returns.iloc[-5:].mean() if len(returns) >= 5 else 0,  # 5-bar avg
            returns.iloc[-20:].std() if len(returns) >= 20 else 0.02,  # Volatility
            (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20] if len(close) >= 20 else 0,  # Trend
            0.5,  # Placeholder
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
        ], dtype=np.float32)
        
        return state
    
    def run_autonomous_session(
        self,
        data: pd.DataFrame,
        simulation: bool = True,
    ) -> Dict:
        """
        รัน session การเทรดอัตโนมัติ
        
        Args:
            data: Historical data สำหรับ simulation
            simulation: True = backtest, False = live (not implemented)
        """
        
        if not simulation:
            raise NotImplementedError("Live trading not yet integrated")
        
        logger.info("="*60)
        logger.info("   AUTONOMOUS TRADING SESSION")
        logger.info("="*60)
        
        results = {
            "trades": [],
            "total_pnl": 0,
            "win_rate": 0,
            "max_drawdown": 0,
        }
        
        # Reset state
        self.current_position = 0
        self.daily_pnl = 0
        
        # Walk through data
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            
            # Make decision
            decision = self.make_decision(current_data)
            
            # Execute
            if decision.action in ["LONG", "CLOSE"]:
                exec_result = self.execute_decision(decision, current_data)
                
                if exec_result.get("executed"):
                    results["trades"].append(exec_result)
        
        # Close any open position
        if self.current_position == 1:
            close_decision = TradeDecision(
                action="CLOSE",
                confidence=0.5,
                position_size=0,
                stop_loss=0,
                take_profit=0,
                reason="Session end",
            )
            exec_result = self.execute_decision(close_decision, data)
            if exec_result.get("executed"):
                results["trades"].append(exec_result)
        
        # Calculate stats
        if results["trades"]:
            pnls = [t.get("pnl", 0) for t in results["trades"] if "pnl" in t]
            if pnls:
                results["total_pnl"] = sum(pnls)
                results["win_rate"] = len([p for p in pnls if p > 0]) / len(pnls)
                
                cumulative = np.cumsum(pnls)
                peak = np.maximum.accumulate(cumulative)
                drawdown = peak - cumulative
                results["max_drawdown"] = max(drawdown) if len(drawdown) > 0 else 0
        
        logger.info("="*60)
        logger.info("   SESSION COMPLETE")
        logger.info(f"   Trades: {len(results['trades'])}")
        logger.info(f"   P&L: ${results['total_pnl']:.2f}")
        logger.info(f"   Win Rate: {results['win_rate']:.1%}")
        logger.info("="*60)
        
        return results
    
    def check_and_evolve(self):
        """ตรวจสอบและ evolve AI ถ้าจำเป็น"""
        
        # Check retraining
        should_retrain, reason = self.auto_trainer.should_retrain()
        if should_retrain:
            logger.info(f"Retraining triggered: {reason}")
            # Would trigger actual retraining here
        
        # Check evolution
        new_version = self.evolution_engine.auto_evolve()
        if new_version:
            logger.info(f"Evolved to {new_version.version_id}")
        
        # Check shadow promotion
        if self.shadow_trader.shadow_strategy:
            should_promote, reason = self.shadow_trader.should_promote_shadow()
            if should_promote:
                self.shadow_trader.promote_shadow_to_production()
    
    def get_status(self) -> Dict:
        """ดึงสถานะระบบทั้งหมด"""
        
        return {
            "capital": self.capital,
            "daily_pnl": self.daily_pnl,
            "current_position": self.current_position,
            "consecutive_losses": self.consecutive_losses,
            "daily_trades": self.daily_trades,
            "modules": {
                "learner": self.learner.get_learning_stats(),
                "position_optimizer": self.position_optimizer.get_stats(),
                "evolution": self.evolution_engine.get_status(),
                "knowledge_base": self.knowledge_base.get_stats(),
            }
        }


def create_autonomous_ai(capital: float = 10000.0) -> AutonomousAI:
    """สร้าง AutonomousAI"""
    return AutonomousAI(capital=capital)


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print()
    print("="*60)
    print("   AUTONOMOUS AI TRADING SYSTEM TEST")
    print("="*60)
    print()
    
    np.random.seed(42)
    
    # Create sample data
    n = 500
    dates = pd.date_range("2024-01-01", periods=n, freq="H")
    prices = 2000 + np.cumsum(np.random.randn(n) * 5)
    
    data = pd.DataFrame({
        "datetime": dates,
        "open": prices - np.random.rand(n) * 5,
        "high": prices + np.random.rand(n) * 10,
        "low": prices - np.random.rand(n) * 10,
        "close": prices,
        "volume": np.random.randint(1000, 10000, n),
    })
    
    # Create AI
    ai = create_autonomous_ai(capital=10000)
    
    # Run session
    results = ai.run_autonomous_session(data)
    
    # Status
    print("\nFinal Status:")
    status = ai.get_status()
    print(f"  Daily P&L: ${status['daily_pnl']:.2f}")
    print(f"  Trades today: {status['daily_trades']}")
    print(f"  Active version: {status['modules']['evolution'].get('active_version', 'default')}")
