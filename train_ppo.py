"""
PPO Training Script (v2.0 - Walk-Forward)
==========================================
Train PPO agent with walk-forward validation

Usage:
    python train_ppo.py                    # Quick training (3 folds)
    python train_ppo.py --folds 10         # Full walk-forward (10 folds)
    python train_ppo.py --episodes 50      # Custom episodes per fold
"""

import os
import sys
import argparse
import gc
from datetime import datetime
from loguru import logger
import torch

# Configure logger
logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")

# Add project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ai_agent.ppo_walk_forward import PPOAgentWalkForward, TradingEnvironment
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="PPO Walk-Forward Training")
    parser.add_argument('--data', type=str, default='data/training/GOLD_H1.csv')
    parser.add_argument('--folds', type=int, default=10, help='Number of walk-forward folds')
    parser.add_argument('--episodes', type=int, default=100, help='Episodes per fold')
    parser.add_argument('--window', type=int, default=20000, help='Training window size')
    parser.add_argument('--resume', action='store_true', help='Resume from best_wf.pt')
    parser.add_argument('--start-fold', type=int, default=0, help='Start from this fold (0-indexed)')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("  PPO WALK-FORWARD TRAINING (v2.0)")
    logger.info("=" * 60)
    logger.info(f"Data: {args.data}")
    logger.info(f"Folds: {args.folds}")
    logger.info(f"Episodes/Fold: {args.episodes}")
    logger.info(f"Window: {args.window} bars")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    # Load data
    df = pd.read_csv(args.data)
    logger.info(f"Loaded {len(df)} rows")
    
    # Create agent
    agent = PPOAgentWalkForward(state_dim=11)
    
    # Resume from checkpoint if requested
    if args.resume:
        if agent.load("best_wf"):
            logger.info("✅ Resumed from best_wf.pt")
        else:
            logger.warning("⚠️ No checkpoint found, starting fresh")
    
    # Walk-forward training
    fold_size = len(df) // (args.folds + 1)
    best_reward = float('-inf')
    
    for fold in range(args.start_fold, args.folds):
        train_start = fold * fold_size
        train_end = train_start + args.window
        
        if train_end >= len(df):
            break
            
        train_data = df.iloc[train_start:train_end].copy()
        logger.info(f"\n📊 Fold {fold+1}/{args.folds}: rows {train_start} to {train_end}")
        
        # Create environment
        env = TradingEnvironment(train_data)
        
        # Train for episodes using built-in optimized method
        fold_rewards = []
        for ep in range(args.episodes):
            # Use train_episode which handles store_transition + update properly
            result = agent.train_episode(env)
            fold_rewards.append(result['reward'])
            
            # Log progress every 10 episodes
            if (ep + 1) % 10 == 0:
                avg_reward = sum(fold_rewards[-10:]) / min(10, len(fold_rewards))
                logger.info(f"  Episode {ep+1}: Avg Reward = {avg_reward:.2f}")
        
        # Memory cleanup after each fold (not every 10 eps - less overhead)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        avg_fold_reward = sum(fold_rewards) / len(fold_rewards)
        logger.info(f"Fold {fold+1} complete: Avg Reward = {avg_fold_reward:.2f}")
        
        if avg_fold_reward > best_reward:
            best_reward = avg_fold_reward
            agent.save("best_wf")
            logger.info(f"📁 New best model saved!")
    
    # Copy best to final
    import shutil
    if os.path.exists("ai_agent/models/best_wf.pt"):
        shutil.copy("ai_agent/models/best_wf.pt", "ai_agent/models/final.pt")
        logger.info("📁 Copied best_wf.pt → final.pt")
    
    elapsed = (datetime.now() - start_time).total_seconds() / 60
    
    logger.info("\n" + "=" * 60)
    logger.info("  TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Time: {elapsed:.1f} minutes")
    logger.info(f"Best Reward: {best_reward:.2f}")
    logger.info(f"Model: ai_agent/models/final.pt")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

