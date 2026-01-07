"""
Strategy Evolution Module
==========================
ใช้ Genetic Algorithm สร้างและพัฒนากลยุทธ์การเทรดอัตโนมัติ

Features:
1. Strategy DNA - เข้ารหัสกลยุทธ์เป็น genes
2. Population Evolution - วิวัฒนาการกลยุทธ์หลายตัว
3. Fitness Evaluation - ประเมินผลจากการ backtest
4. Crossover & Mutation - สร้างกลยุทธ์ใหม่จากกลยุทธ์ที่ดี
5. Self-Discovery - ค้นพบกลยุทธ์ใหม่อัตโนมัติ
"""

import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from dataclasses import dataclass, asdict, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from copy import deepcopy
from loguru import logger


@dataclass
class StrategyGene:
    """
    DNA ของกลยุทธ์การเทรด
    
    แต่ละ gene คือ parameter หนึ่งตัว
    """
    
    # Entry conditions
    min_confidence: float = 0.70  # 0.5 - 0.95
    min_trend_strength: float = 0.005  # 0.0 - 0.03
    max_volatility: float = 0.025  # 0.01 - 0.05
    rsi_oversold: float = 30.0  # 20 - 40
    rsi_overbought: float = 70.0  # 60 - 80
    
    # Position sizing
    base_position_pct: float = 0.02  # 0.01 - 0.05
    volatility_scale: float = 1.0  # 0.5 - 2.0
    
    # Risk management  
    stop_loss_atr: float = 1.5  # 0.5 - 3.0
    take_profit_atr: float = 3.0  # 1.5 - 6.0
    max_holding_bars: int = 48  # 12 - 96
    
    # Regime filters
    allow_ranging: bool = True
    ranging_multiplier: float = 0.5  # 0.2 - 1.0
    
    # Advanced
    require_macd_confirm: bool = True
    require_bb_position: bool = False
    min_rr_ratio: float = 2.0  # 1.5 - 4.0
    
    def to_array(self) -> np.ndarray:
        """แปลงเป็น array สำหรับ genetic operations"""
        return np.array([
            self.min_confidence,
            self.min_trend_strength,
            self.max_volatility,
            self.rsi_oversold,
            self.rsi_overbought,
            self.base_position_pct,
            self.volatility_scale,
            self.stop_loss_atr,
            self.take_profit_atr,
            self.max_holding_bars,
            float(self.allow_ranging),
            self.ranging_multiplier,
            float(self.require_macd_confirm),
            float(self.require_bb_position),
            self.min_rr_ratio,
        ])
    
    @staticmethod
    def from_array(arr: np.ndarray) -> 'StrategyGene':
        """สร้างจาก array"""
        return StrategyGene(
            min_confidence=np.clip(arr[0], 0.5, 0.95),
            min_trend_strength=np.clip(arr[1], 0.0, 0.03),
            max_volatility=np.clip(arr[2], 0.01, 0.05),
            rsi_oversold=np.clip(arr[3], 20, 40),
            rsi_overbought=np.clip(arr[4], 60, 80),
            base_position_pct=np.clip(arr[5], 0.01, 0.05),
            volatility_scale=np.clip(arr[6], 0.5, 2.0),
            stop_loss_atr=np.clip(arr[7], 0.5, 3.0),
            take_profit_atr=np.clip(arr[8], 1.5, 6.0),
            max_holding_bars=int(np.clip(arr[9], 12, 96)),
            allow_ranging=arr[10] > 0.5,
            ranging_multiplier=np.clip(arr[11], 0.2, 1.0),
            require_macd_confirm=arr[12] > 0.5,
            require_bb_position=arr[13] > 0.5,
            min_rr_ratio=np.clip(arr[14], 1.5, 4.0),
        )
    
    def mutate(self, mutation_rate: float = 0.1) -> 'StrategyGene':
        """Mutation - สุ่มปรับค่า"""
        arr = self.to_array()
        
        # Define mutation ranges for each gene
        mutation_scales = np.array([
            0.05,   # min_confidence
            0.005,  # min_trend_strength
            0.01,   # max_volatility
            5.0,    # rsi_oversold
            5.0,    # rsi_overbought
            0.01,   # base_position_pct
            0.2,    # volatility_scale
            0.3,    # stop_loss_atr
            0.5,    # take_profit_atr
            12.0,   # max_holding_bars
            0.5,    # allow_ranging (flip)
            0.1,    # ranging_multiplier
            0.5,    # require_macd_confirm (flip)
            0.5,    # require_bb_position (flip)
            0.3,    # min_rr_ratio
        ])
        
        # Apply mutations
        for i in range(len(arr)):
            if np.random.random() < mutation_rate:
                arr[i] += np.random.randn() * mutation_scales[i]
        
        return StrategyGene.from_array(arr)
    
    def crossover(self, other: 'StrategyGene') -> Tuple['StrategyGene', 'StrategyGene']:
        """Crossover - ผสมกับ gene อื่น"""
        arr1 = self.to_array()
        arr2 = other.to_array()
        
        # Two-point crossover
        n = len(arr1)
        p1, p2 = sorted(np.random.choice(n, 2, replace=False))
        
        child1 = np.concatenate([arr1[:p1], arr2[p1:p2], arr1[p2:]])
        child2 = np.concatenate([arr2[:p1], arr1[p1:p2], arr2[p2:]])
        
        return StrategyGene.from_array(child1), StrategyGene.from_array(child2)


@dataclass
class StrategyIndividual:
    """บุคคลใน population (กลยุทธ์หนึ่งตัว)"""
    id: str
    genes: StrategyGene
    fitness: float = 0.0
    generation: int = 0
    
    # Performance metrics
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_pnl: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "genes": asdict(self.genes),
            "fitness": self.fitness,
            "generation": self.generation,
            "metrics": {
                "total_trades": self.total_trades,
                "win_rate": self.win_rate,
                "profit_factor": self.profit_factor,
                "sharpe_ratio": self.sharpe_ratio,
                "max_drawdown": self.max_drawdown,
                "total_pnl": self.total_pnl,
            }
        }


class StrategyEvolution:
    """
    Genetic Algorithm สำหรับวิวัฒนาการกลยุทธ์
    
    ความสามารถ:
    1. สร้างและจัดการ population ของกลยุทธ์
    2. ประเมิน fitness ด้วย backtesting
    3. Selection, Crossover, Mutation
    4. บันทึกกลยุทธ์ที่ดีที่สุด
    """
    
    def __init__(
        self,
        population_size: int = 20,
        elite_size: int = 4,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        data_dir: str = "ai_agent/data",
    ):
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.data_dir = data_dir
        
        os.makedirs(data_dir, exist_ok=True)
        
        # Population
        self.population: List[StrategyIndividual] = []
        self.generation = 0
        
        # Best strategies ever found
        self.hall_of_fame: List[StrategyIndividual] = []
        self.max_hall_of_fame = 10
        
        # Evolution history
        self.history: List[Dict] = []
        
        # Load existing
        self._load()
        
        logger.info(f"StrategyEvolution initialized with population size {population_size}")
    
    def initialize_population(self):
        """สร้าง population เริ่มต้น"""
        
        self.population = []
        
        for i in range(self.population_size):
            # Random genes with some variation
            genes = StrategyGene(
                min_confidence=np.random.uniform(0.6, 0.85),
                min_trend_strength=np.random.uniform(0.003, 0.015),
                max_volatility=np.random.uniform(0.015, 0.035),
                rsi_oversold=np.random.uniform(25, 35),
                rsi_overbought=np.random.uniform(65, 75),
                base_position_pct=np.random.uniform(0.015, 0.035),
                volatility_scale=np.random.uniform(0.7, 1.3),
                stop_loss_atr=np.random.uniform(1.0, 2.5),
                take_profit_atr=np.random.uniform(2.0, 5.0),
                max_holding_bars=np.random.randint(24, 72),
                allow_ranging=np.random.random() > 0.5,
                ranging_multiplier=np.random.uniform(0.3, 0.7),
                require_macd_confirm=np.random.random() > 0.3,
                require_bb_position=np.random.random() > 0.7,
                min_rr_ratio=np.random.uniform(1.8, 3.0),
            )
            
            individual = StrategyIndividual(
                id=f"gen{self.generation}_ind{i}",
                genes=genes,
                generation=self.generation,
            )
            
            self.population.append(individual)
        
        logger.info(f"Initialized population with {len(self.population)} individuals")
    
    def evaluate_fitness(
        self,
        individual: StrategyIndividual,
        data: pd.DataFrame,
        initial_capital: float = 10000.0,
    ) -> float:
        """
        ประเมิน fitness ของกลยุทธ์ด้วย backtesting
        
        Fitness = Sharpe Ratio * Win Rate * (1 - Max Drawdown)
        """
        
        try:
            # Simple backtest
            trades = self._backtest_strategy(individual.genes, data, initial_capital)
            
            if len(trades) < 5:
                return 0.0
            
            # Calculate metrics
            pnls = [t['pnl'] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p < 0]
            
            win_rate = len(wins) / len(trades) if trades else 0
            total_pnl = sum(pnls)
            
            # Profit factor
            gross_profit = sum(wins) if wins else 0
            gross_loss = abs(sum(losses)) if losses else 1
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
            
            # Sharpe ratio (simplified)
            if len(pnls) > 1:
                returns = np.array(pnls) / initial_capital
                sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
            else:
                sharpe = 0
            
            # Max drawdown
            cumulative = np.cumsum(pnls)
            peak = np.maximum.accumulate(cumulative)
            drawdown = (peak - cumulative) / (peak + initial_capital + 1e-8)
            max_dd = np.max(drawdown) if len(drawdown) > 0 else 0
            
            # Update individual metrics
            individual.total_trades = len(trades)
            individual.win_rate = win_rate
            individual.profit_factor = profit_factor
            individual.sharpe_ratio = max(0, sharpe)
            individual.max_drawdown = max_dd
            individual.total_pnl = total_pnl
            
            # Fitness formula (reward good metrics, penalize drawdown)
            fitness = (
                individual.sharpe_ratio * 0.3 +
                win_rate * 0.3 +
                min(profit_factor / 3, 1) * 0.2 +
                (1 - max_dd) * 0.2
            )
            
            # Bonus for profitable strategies
            if total_pnl > 0:
                fitness += 0.1
            
            individual.fitness = max(0, fitness)
            
            return individual.fitness
            
        except Exception as e:
            logger.warning(f"Fitness evaluation failed: {e}")
            return 0.0
    
    def _backtest_strategy(
        self,
        genes: StrategyGene,
        data: pd.DataFrame,
        initial_capital: float,
    ) -> List[Dict]:
        """Simple backtest ของกลยุทธ์"""
        
        if len(data) < 100:
            return []
        
        trades = []
        position = 0
        entry_price = 0
        entry_bar = 0
        capital = initial_capital
        
        # Prepare indicators
        df = data.copy()
        df['returns'] = df['close'].pct_change()
        df['volatility'] = df['returns'].rolling(20).std()
        df['trend'] = df['close'].pct_change(20)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        df['high_low'] = df['high'] - df['low']
        df['high_close'] = abs(df['high'] - df['close'].shift())
        df['low_close'] = abs(df['low'] - df['close'].shift())
        df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
        df['atr'] = df['tr'].rolling(14).mean()
        
        df = df.fillna(0)
        
        for i in range(50, len(df) - 1):
            row = df.iloc[i]
            price = row['close']
            
            if position == 0:
                # Entry conditions
                confidence = np.random.uniform(0.6, 0.9)  # Simulated
                
                should_enter = (
                    confidence >= genes.min_confidence and
                    abs(row['trend']) >= genes.min_trend_strength and
                    row['volatility'] <= genes.max_volatility and
                    row['rsi'] > genes.rsi_oversold and
                    row['rsi'] < genes.rsi_overbought
                )
                
                # Regime filter
                is_ranging = abs(row['trend']) < 0.005
                if is_ranging and not genes.allow_ranging:
                    should_enter = False
                
                if should_enter and row['trend'] > 0:  # Only LONG
                    position = 1
                    entry_price = price
                    entry_bar = i
                    
            elif position == 1:
                # Exit conditions
                bars_held = i - entry_bar
                atr = row['atr']
                
                # Stop loss / Take profit
                stop_loss = entry_price - (atr * genes.stop_loss_atr)
                take_profit = entry_price + (atr * genes.take_profit_atr)
                
                should_exit = (
                    price <= stop_loss or
                    price >= take_profit or
                    bars_held >= genes.max_holding_bars
                )
                
                if should_exit:
                    pnl_pct = (price - entry_price) / entry_price
                    position_size = capital * genes.base_position_pct
                    pnl = position_size * pnl_pct
                    
                    trades.append({
                        'entry_bar': entry_bar,
                        'exit_bar': i,
                        'entry_price': entry_price,
                        'exit_price': price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'bars_held': bars_held,
                    })
                    
                    capital += pnl
                    position = 0
        
        return trades
    
    def evaluate_population(self, data: pd.DataFrame):
        """ประเมิน fitness ของทุกคนใน population"""
        
        for individual in self.population:
            self.evaluate_fitness(individual, data)
        
        # Sort by fitness
        self.population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Update hall of fame
        for ind in self.population[:self.elite_size]:
            if ind.fitness > 0.1:  # Minimum threshold
                self._add_to_hall_of_fame(ind)
        
        # Log best
        best = self.population[0]
        logger.info(
            f"Gen {self.generation}: Best fitness={best.fitness:.4f}, "
            f"WR={best.win_rate:.1%}, PF={best.profit_factor:.2f}"
        )
    
    def _add_to_hall_of_fame(self, individual: StrategyIndividual):
        """เพิ่มกลยุทธ์ที่ดีเข้า Hall of Fame"""
        
        # Check if already in hall of fame
        for hof in self.hall_of_fame:
            if hof.id == individual.id:
                return
        
        # Add and sort
        self.hall_of_fame.append(deepcopy(individual))
        self.hall_of_fame.sort(key=lambda x: x.fitness, reverse=True)
        
        # Keep only top N
        self.hall_of_fame = self.hall_of_fame[:self.max_hall_of_fame]
    
    def select_parents(self) -> List[StrategyIndividual]:
        """เลือก parents สำหรับสร้าง generation ถัดไป (Tournament Selection)"""
        
        parents = []
        tournament_size = 3
        
        for _ in range(self.population_size):
            # Random tournament
            tournament = np.random.choice(
                self.population, 
                size=min(tournament_size, len(self.population)),
                replace=False
            )
            winner = max(tournament, key=lambda x: x.fitness)
            parents.append(winner)
        
        return parents
    
    def evolve(self, data: pd.DataFrame) -> Dict:
        """
        ทำ Evolution 1 generation
        
        1. Evaluate fitness
        2. Select parents
        3. Crossover
        4. Mutation
        5. Create new population
        """
        
        # Evaluate current population
        self.evaluate_population(data)
        
        # Keep elites
        elites = [deepcopy(ind) for ind in self.population[:self.elite_size]]
        
        # Select parents
        parents = self.select_parents()
        
        # Create new population
        new_population = []
        
        # Add elites
        for elite in elites:
            elite.generation = self.generation + 1
            new_population.append(elite)
        
        # Create children
        while len(new_population) < self.population_size:
            # Select two parents
            p1, p2 = np.random.choice(parents, 2, replace=False)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1_genes, child2_genes = p1.genes.crossover(p2.genes)
            else:
                child1_genes = deepcopy(p1.genes)
                child2_genes = deepcopy(p2.genes)
            
            # Mutation
            child1_genes = child1_genes.mutate(self.mutation_rate)
            child2_genes = child2_genes.mutate(self.mutation_rate)
            
            # Create individuals
            for genes in [child1_genes, child2_genes]:
                if len(new_population) < self.population_size:
                    new_population.append(StrategyIndividual(
                        id=f"gen{self.generation + 1}_ind{len(new_population)}",
                        genes=genes,
                        generation=self.generation + 1,
                    ))
        
        # Update population
        self.population = new_population
        self.generation += 1
        
        # Record history
        stats = self._get_population_stats()
        self.history.append(stats)
        
        # Save
        self._save()
        
        return stats
    
    def _get_population_stats(self) -> Dict:
        """ดึงสถิติของ population"""
        
        fitnesses = [ind.fitness for ind in self.population]
        
        return {
            "generation": self.generation,
            "best_fitness": max(fitnesses) if fitnesses else 0,
            "avg_fitness": np.mean(fitnesses) if fitnesses else 0,
            "min_fitness": min(fitnesses) if fitnesses else 0,
            "best_win_rate": self.population[0].win_rate if self.population else 0,
            "best_pf": self.population[0].profit_factor if self.population else 0,
        }
    
    def get_best_strategy(self) -> Optional[StrategyIndividual]:
        """ดึงกลยุทธ์ที่ดีที่สุด"""
        
        if self.hall_of_fame:
            return self.hall_of_fame[0]
        elif self.population:
            return max(self.population, key=lambda x: x.fitness)
        return None
    
    def run_evolution(
        self,
        data: pd.DataFrame,
        n_generations: int = 10,
        early_stop_generations: int = 5,
    ) -> StrategyIndividual:
        """
        รัน Evolution หลาย generations
        
        Returns:
            Best strategy found
        """
        
        if not self.population:
            self.initialize_population()
        
        best_fitness = 0
        no_improvement = 0
        
        logger.info(f"Starting evolution for {n_generations} generations...")
        
        for gen in range(n_generations):
            stats = self.evolve(data)
            
            if stats['best_fitness'] > best_fitness:
                best_fitness = stats['best_fitness']
                no_improvement = 0
            else:
                no_improvement += 1
            
            logger.info(
                f"Generation {self.generation}: "
                f"Best={stats['best_fitness']:.4f}, "
                f"Avg={stats['avg_fitness']:.4f}"
            )
            
            # Early stopping
            if no_improvement >= early_stop_generations:
                logger.info(f"Early stopping: no improvement for {no_improvement} generations")
                break
        
        best = self.get_best_strategy()
        
        if best:
            logger.info(
                f"Evolution complete. Best strategy: "
                f"Fitness={best.fitness:.4f}, WR={best.win_rate:.1%}, "
                f"PF={best.profit_factor:.2f}"
            )
        
        return best
    
    def _save(self):
        """บันทึก population และ hall of fame"""
        
        state = {
            "generation": self.generation,
            "population": [ind.to_dict() for ind in self.population],
            "hall_of_fame": [ind.to_dict() for ind in self.hall_of_fame],
            "history": self.history,
        }
        
        path = os.path.join(self.data_dir, "strategy_evolution.json")
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลด state"""
        
        path = os.path.join(self.data_dir, "strategy_evolution.json")
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                self.generation = state.get("generation", 0)
                self.history = state.get("history", [])
                
                # Rebuild population
                for ind_dict in state.get("population", []):
                    genes = StrategyGene(**ind_dict["genes"])
                    ind = StrategyIndividual(
                        id=ind_dict["id"],
                        genes=genes,
                        fitness=ind_dict.get("fitness", 0),
                        generation=ind_dict.get("generation", 0),
                    )
                    metrics = ind_dict.get("metrics", {})
                    ind.win_rate = metrics.get("win_rate", 0)
                    ind.profit_factor = metrics.get("profit_factor", 0)
                    self.population.append(ind)
                
                # Rebuild hall of fame
                for ind_dict in state.get("hall_of_fame", []):
                    genes = StrategyGene(**ind_dict["genes"])
                    ind = StrategyIndividual(
                        id=ind_dict["id"],
                        genes=genes,
                        fitness=ind_dict.get("fitness", 0),
                        generation=ind_dict.get("generation", 0),
                    )
                    metrics = ind_dict.get("metrics", {})
                    ind.win_rate = metrics.get("win_rate", 0)
                    ind.profit_factor = metrics.get("profit_factor", 0)
                    self.hall_of_fame.append(ind)
                
                logger.info(f"Loaded evolution state: Gen {self.generation}, {len(self.population)} individuals")
                
            except Exception as e:
                logger.warning(f"Failed to load evolution state: {e}")


def create_strategy_evolution() -> StrategyEvolution:
    """สร้าง StrategyEvolution"""
    return StrategyEvolution()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   STRATEGY EVOLUTION TEST")
    print("="*60)
    
    np.random.seed(42)
    
    # Create sample data
    n = 1000
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
    
    # Run evolution
    evolution = create_strategy_evolution()
    evolution.initialize_population()
    
    print("\nRunning evolution...")
    best = evolution.run_evolution(data, n_generations=5)
    
    if best:
        print(f"\nBest Strategy:")
        print(f"  Fitness: {best.fitness:.4f}")
        print(f"  Win Rate: {best.win_rate:.1%}")
        print(f"  Profit Factor: {best.profit_factor:.2f}")
        print(f"  Max Drawdown: {best.max_drawdown:.1%}")
        print(f"\nGenes:")
        print(f"  Min Confidence: {best.genes.min_confidence:.2f}")
        print(f"  Stop Loss ATR: {best.genes.stop_loss_atr:.1f}")
        print(f"  Take Profit ATR: {best.genes.take_profit_atr:.1f}")
