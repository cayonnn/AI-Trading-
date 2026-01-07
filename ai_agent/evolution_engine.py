"""
Evolution Engine Module
========================
พัฒนา AI versions ใหม่อัตโนมัติ

Features:
1. Version Management - จัดการ AI versions
2. Continuous Evolution - พัฒนาต่อเนื่อง
3. Performance Tracking - ติดตามผลแต่ละ version
4. Automatic Selection - เลือก version ที่ดีที่สุด
"""

import numpy as np
import json
import os
import shutil
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Tuple, Any
from loguru import logger


@dataclass
class AIVersion:
    """Version ของ AI"""
    version_id: str
    created_at: datetime
    parent_version: Optional[str] = None
    
    # Configuration
    config: Dict[str, Any] = field(default_factory=dict)
    
    # Performance
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    sharpe_ratio: float = 0.0
    total_pnl: float = 0.0
    
    # Status
    status: str = "testing"  # 'testing', 'active', 'archived', 'deprecated'
    
    # Evolution metadata
    generation: int = 0
    mutations_applied: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        d = asdict(self)
        d['created_at'] = self.created_at.isoformat()
        return d
    
    @staticmethod
    def from_dict(d: Dict) -> 'AIVersion':
        d['created_at'] = datetime.fromisoformat(d['created_at'])
        return AIVersion(**d)


class EvolutionEngine:
    """
    AI Evolution Engine
    
    ความสามารถ:
    1. สร้าง versions ใหม่จาก version ที่ดี
    2. ทดสอบและเปรียบเทียบ
    3. เลือก version ที่ดีที่สุดอัตโนมัติ
    4. Archive versions เก่า
    """
    
    def __init__(
        self,
        model_dir: str = "ai_agent/models",
        data_dir: str = "ai_agent/data",
        max_versions: int = 10,
    ):
        self.model_dir = model_dir
        self.data_dir = data_dir
        self.max_versions = max_versions
        
        os.makedirs(model_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        
        # Versions
        self.versions: Dict[str, AIVersion] = {}
        self.active_version: Optional[str] = None
        
        # Evolution settings
        self.mutation_rate = 0.15
        self.evolution_threshold = 0.1  # 10% improvement needed
        
        # Evolution history
        self.evolution_history: List[Dict] = []
        
        self._load()
        
        # Create initial version if none exists
        if not self.versions:
            self._create_initial_version()
        
        logger.info(f"EvolutionEngine initialized with {len(self.versions)} versions")
    
    def _create_initial_version(self):
        """สร้าง version แรก"""
        
        version = AIVersion(
            version_id="v1.0.0",
            created_at=datetime.now(),
            config={
                'min_confidence': 0.70,
                'max_volatility': 0.025,
                'position_pct': 0.02,
                'stop_atr': 1.5,
                'target_atr': 3.0,
            },
            status="active",
            generation=0,
        )
        
        self.versions[version.version_id] = version
        self.active_version = version.version_id
        
        self._save()
    
    def evolve_from(
        self,
        parent_version_id: str,
        mutations: List[str] = None,
    ) -> AIVersion:
        """
        สร้าง version ใหม่จาก parent
        
        Args:
            parent_version_id: Version ต้นทาง
            mutations: รายการ mutations ที่จะใช้
            
        Returns:
            New AIVersion
        """
        
        if parent_version_id not in self.versions:
            raise ValueError(f"Parent version {parent_version_id} not found")
        
        parent = self.versions[parent_version_id]
        
        # Parse version number
        parts = parent_version_id.replace('v', '').split('.')
        major, minor, patch = int(parts[0]), int(parts[1]), int(parts[2])
        
        # Increment version
        new_version_id = f"v{major}.{minor}.{patch + 1}"
        
        # Apply mutations to config
        new_config = self._apply_mutations(parent.config, mutations)
        
        # Create new version
        new_version = AIVersion(
            version_id=new_version_id,
            created_at=datetime.now(),
            parent_version=parent_version_id,
            config=new_config,
            status="testing",
            generation=parent.generation + 1,
            mutations_applied=mutations or [],
        )
        
        self.versions[new_version_id] = new_version
        
        # Copy model files
        self._copy_model_files(parent_version_id, new_version_id)
        
        logger.info(f"Evolved {parent_version_id} -> {new_version_id}")
        
        self._save()
        return new_version
    
    def _apply_mutations(
        self,
        config: Dict,
        mutations: List[str] = None,
    ) -> Dict:
        """Apply mutations to config"""
        
        new_config = config.copy()
        
        if mutations is None:
            mutations = []
            
            # Random mutations
            if np.random.rand() < self.mutation_rate:
                mutations.append('confidence')
            if np.random.rand() < self.mutation_rate:
                mutations.append('position')
            if np.random.rand() < self.mutation_rate:
                mutations.append('stop_loss')
        
        for mutation in mutations:
            if mutation == 'confidence':
                new_config['min_confidence'] = np.clip(
                    new_config.get('min_confidence', 0.7) + np.random.randn() * 0.05,
                    0.5, 0.95
                )
            elif mutation == 'position':
                new_config['position_pct'] = np.clip(
                    new_config.get('position_pct', 0.02) + np.random.randn() * 0.005,
                    0.01, 0.05
                )
            elif mutation == 'stop_loss':
                new_config['stop_atr'] = np.clip(
                    new_config.get('stop_atr', 1.5) + np.random.randn() * 0.3,
                    0.5, 3.0
                )
            elif mutation == 'target':
                new_config['target_atr'] = np.clip(
                    new_config.get('target_atr', 3.0) + np.random.randn() * 0.5,
                    1.5, 6.0
                )
        
        return new_config
    
    def _copy_model_files(self, src_version: str, dst_version: str):
        """Copy model files to new version"""
        
        src_path = os.path.join(self.model_dir, f"{src_version}.pt")
        dst_path = os.path.join(self.model_dir, f"{dst_version}.pt")
        
        if os.path.exists(src_path):
            shutil.copy(src_path, dst_path)
    
    def update_performance(
        self,
        version_id: str,
        trade_result: Dict,
    ):
        """อัพเดต performance ของ version"""
        
        if version_id not in self.versions:
            return
        
        version = self.versions[version_id]
        
        pnl = trade_result.get('pnl', 0)
        is_win = pnl > 0
        
        version.total_trades += 1
        version.total_pnl += pnl
        
        # Update win rate (moving average)
        alpha = 0.1
        version.win_rate = version.win_rate * (1 - alpha) + (1 if is_win else 0) * alpha
        
        self._save()
    
    def compare_versions(
        self,
        version_a: str,
        version_b: str,
    ) -> Dict[str, Any]:
        """เปรียบเทียบ 2 versions"""
        
        if version_a not in self.versions or version_b not in self.versions:
            return {"error": "Version not found"}
        
        va = self.versions[version_a]
        vb = self.versions[version_b]
        
        return {
            version_a: {
                'trades': va.total_trades,
                'win_rate': va.win_rate,
                'total_pnl': va.total_pnl,
            },
            version_b: {
                'trades': vb.total_trades,
                'win_rate': vb.win_rate,
                'total_pnl': vb.total_pnl,
            },
            'comparison': {
                'winner': version_a if va.total_pnl > vb.total_pnl else version_b,
                'pnl_diff': abs(va.total_pnl - vb.total_pnl),
                'win_rate_diff': va.win_rate - vb.win_rate,
            }
        }
    
    def should_promote(self, version_id: str) -> Tuple[bool, str]:
        """ตรวจสอบว่าควรเลื่อน version นี้เป็น active หรือไม่"""
        
        if version_id not in self.versions:
            return False, "Version not found"
        
        version = self.versions[version_id]
        
        if version.total_trades < 20:
            return False, f"Not enough trades ({version.total_trades}/20)"
        
        if not self.active_version:
            return True, "No active version"
        
        active = self.versions[self.active_version]
        
        # Compare with active version
        if version.win_rate > active.win_rate * (1 + self.evolution_threshold):
            return True, f"Win rate {version.win_rate:.1%} > {active.win_rate:.1%}"
        
        if version.total_pnl > active.total_pnl * (1 + self.evolution_threshold):
            return True, f"P&L ${version.total_pnl:.2f} > ${active.total_pnl:.2f}"
        
        return False, "Not significantly better"
    
    def promote_version(self, version_id: str) -> bool:
        """เลื่อน version เป็น active"""
        
        if version_id not in self.versions:
            return False
        
        # Archive old active
        if self.active_version:
            old = self.versions[self.active_version]
            old.status = "archived"
        
        # Promote new
        self.versions[version_id].status = "active"
        self.active_version = version_id
        
        self.evolution_history.append({
            'timestamp': datetime.now().isoformat(),
            'promoted': version_id,
            'previous': self.active_version,
        })
        
        logger.info(f"Promoted {version_id} to active")
        
        self._save()
        return True
    
    def auto_evolve(self) -> Optional[AIVersion]:
        """
        วิวัฒนาการอัตโนมัติ
        
        1. ตรวจสอบ active version
        2. สร้าง version ใหม่ถ้าต้องการ
        3. Return version ใหม่
        """
        
        if not self.active_version:
            return None
        
        active = self.versions[self.active_version]
        
        # Check if active is underperforming
        if active.total_trades >= 50 and active.win_rate < 0.45:
            logger.info("Active version underperforming, evolving...")
            
            # Try mutations that might help
            mutations = ['confidence', 'stop_loss']
            
            return self.evolve_from(self.active_version, mutations)
        
        # Periodic evolution
        if active.total_trades > 0 and active.total_trades % 100 == 0:
            logger.info("Periodic evolution...")
            return self.evolve_from(self.active_version)
        
        return None
    
    def get_active_config(self) -> Dict[str, Any]:
        """ดึง config ของ active version"""
        
        if not self.active_version:
            return {}
        
        return self.versions[self.active_version].config
    
    def cleanup_old_versions(self):
        """ลบ versions เก่าที่ไม่ได้ใช้"""
        
        if len(self.versions) <= self.max_versions:
            return
        
        # Sort by performance
        sorted_versions = sorted(
            self.versions.values(),
            key=lambda v: (v.status == 'active', v.total_pnl),
            reverse=True
        )
        
        # Keep top N
        to_keep = {v.version_id for v in sorted_versions[:self.max_versions]}
        
        # Remove others
        to_remove = [vid for vid in self.versions if vid not in to_keep]
        
        for vid in to_remove:
            del self.versions[vid]
            
            # Remove model file
            model_path = os.path.join(self.model_dir, f"{vid}.pt")
            if os.path.exists(model_path):
                os.remove(model_path)
        
        logger.info(f"Cleaned up {len(to_remove)} old versions")
        
        self._save()
    
    def get_status(self) -> Dict[str, Any]:
        """ดึงสถานะ"""
        
        return {
            'total_versions': len(self.versions),
            'active_version': self.active_version,
            'active_config': self.get_active_config(),
            'active_performance': {
                'trades': self.versions[self.active_version].total_trades,
                'win_rate': self.versions[self.active_version].win_rate,
                'pnl': self.versions[self.active_version].total_pnl,
            } if self.active_version else {},
            'versions': [
                {
                    'id': v.version_id,
                    'status': v.status,
                    'generation': v.generation,
                    'trades': v.total_trades,
                }
                for v in self.versions.values()
            ]
        }
    
    def _save(self):
        """บันทึก state"""
        
        state = {
            'versions': {k: v.to_dict() for k, v in self.versions.items()},
            'active_version': self.active_version,
            'evolution_history': self.evolution_history[-100:],
        }
        
        path = os.path.join(self.data_dir, 'evolution_engine.json')
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def _load(self):
        """โหลด state"""
        
        path = os.path.join(self.data_dir, 'evolution_engine.json')
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    state = json.load(f)
                
                for k, v in state.get('versions', {}).items():
                    self.versions[k] = AIVersion.from_dict(v)
                
                self.active_version = state.get('active_version')
                self.evolution_history = state.get('evolution_history', [])
                
            except Exception as e:
                logger.warning(f"Failed to load evolution engine: {e}")


def create_evolution_engine() -> EvolutionEngine:
    """สร้าง EvolutionEngine"""
    return EvolutionEngine()


if __name__ == "__main__":
    import sys
    logger.remove()
    logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")
    
    print("="*60)
    print("   EVOLUTION ENGINE TEST")
    print("="*60)
    
    np.random.seed(42)
    
    engine = create_evolution_engine()
    
    # Simulate trades for active version
    print("\nSimulating trades...")
    for i in range(30):
        engine.update_performance(
            engine.active_version,
            {'pnl': np.random.randn() * 50}
        )
    
    # Evolve
    print("\nEvolving new version...")
    new_version = engine.evolve_from(
        engine.active_version,
        mutations=['confidence', 'position']
    )
    print(f"  Created: {new_version.version_id}")
    print(f"  Config: {new_version.config}")
    
    # Simulate trades for new version (better performance)
    for i in range(25):
        engine.update_performance(
            new_version.version_id,
            {'pnl': np.random.randn() * 50 + 20}  # Bias toward profit
        )
    
    # Check promotion
    should, reason = engine.should_promote(new_version.version_id)
    print(f"\nShould promote: {should} ({reason})")
    
    # Status
    print("\nStatus:")
    status = engine.get_status()
    print(f"  Active: {status['active_version']}")
    print(f"  Versions: {status['total_versions']}")
    for v in status['versions']:
        print(f"    {v['id']}: {v['status']} (gen {v['generation']}, {v['trades']} trades)")
