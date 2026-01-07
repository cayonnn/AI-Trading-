"""
Configuration Manager - Production Grade
========================================
Centralized configuration management with:
- YAML config loading
- Environment variable support
- Validation
- Type safety
- Hot reload capability
"""

import os
import yaml
from typing import Any, Dict, Optional
from pathlib import Path
from loguru import logger
import sys
from dataclasses import dataclass

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss} | {level:8} | {message}")


@dataclass
class TradingConfig:
    """Trading configuration"""
    symbol: str
    timeframe: str
    min_confidence: float
    min_mtf_alignment: float
    max_position_size: float
    max_daily_trades: int
    max_open_positions: int
    default_position_size: float = 0.01
    stop_loss_default: float = 0.01
    risk_reward_min: float = 2.0


@dataclass
class RiskConfig:
    """Risk management configuration"""
    max_daily_loss_pct: float
    max_weekly_loss_pct: float
    max_drawdown_pct: float
    max_correlation: float
    max_consecutive_losses: int
    pause_duration_hours: int


@dataclass
class DatabaseConfig:
    """Database configuration"""
    enabled: bool
    type: str
    sqlite_path: Optional[str] = None
    pg_host: Optional[str] = None
    pg_port: Optional[int] = None
    pg_database: Optional[str] = None


class ConfigManager:
    """
    Production-grade configuration manager

    Features:
    - Load from YAML
    - Override with environment variables
    - Validation
    - Type conversion
    - Singleton pattern
    """

    _instance = None
    _config: Dict[str, Any] = {}
    _config_path: Path = None

    def __new__(cls):
        """Singleton pattern"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize config manager"""
        if not self._config:
            self.load_config()

    def load_config(self, config_path: str = "config.yaml") -> None:
        """
        Load configuration from YAML file

        Args:
            config_path: Path to config file
        """
        self._config_path = Path(config_path)

        if not self._config_path.exists():
            logger.error(f"Config file not found: {config_path}")
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        try:
            with open(self._config_path, 'r', encoding='utf-8') as f:
                self._config = yaml.safe_load(f)

            logger.info(f"Configuration loaded from: {config_path}")

            # Apply environment variable overrides
            self._apply_env_overrides()

            # Validate configuration
            self._validate_config()

            logger.info("Configuration validated successfully")

        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise

    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides"""

        # Database credentials
        if os.getenv('DB_USER'):
            self._config['database']['postgresql']['user'] = os.getenv('DB_USER')

        if os.getenv('DB_PASSWORD'):
            self._config['database']['postgresql']['password'] = os.getenv('DB_PASSWORD')

        # MT5 credentials
        if os.getenv('MT5_LOGIN'):
            self._config['data_sources']['primary']['mt5']['login'] = os.getenv('MT5_LOGIN')

        if os.getenv('MT5_PASSWORD'):
            self._config['data_sources']['primary']['mt5']['password'] = os.getenv('MT5_PASSWORD')

        if os.getenv('MT5_SERVER'):
            self._config['data_sources']['primary']['mt5']['server'] = os.getenv('MT5_SERVER')

        # API Keys
        if os.getenv('NEWSAPI_KEY'):
            self._config['data_sources']['api_keys']['news_api'] = os.getenv('NEWSAPI_KEY')

        if os.getenv('TWITTER_API_KEY'):
            self._config['data_sources']['api_keys']['twitter_api'] = os.getenv('TWITTER_API_KEY')

        if os.getenv('FRED_API_KEY'):
            self._config['data_sources']['api_keys']['fred_api'] = os.getenv('FRED_API_KEY')

        # Telegram
        if os.getenv('TELEGRAM_BOT_TOKEN'):
            self._config['alerts']['telegram']['bot_token'] = os.getenv('TELEGRAM_BOT_TOKEN')

        if os.getenv('TELEGRAM_CHAT_ID'):
            self._config['alerts']['telegram']['chat_id'] = os.getenv('TELEGRAM_CHAT_ID')

        # Environment override
        if os.getenv('ENVIRONMENT'):
            self._config['system']['environment'] = os.getenv('ENVIRONMENT')

        logger.debug("Environment variable overrides applied")

    def _validate_config(self) -> None:
        """Validate configuration values"""

        # Validate trading parameters
        trading = self._config.get('trading', {})

        if not 0 < trading.get('min_confidence', 0) <= 1:
            raise ValueError("min_confidence must be between 0 and 1")

        if not 0 < trading.get('min_mtf_alignment', 0) <= 1:
            raise ValueError("min_mtf_alignment must be between 0 and 1")

        if trading.get('max_daily_trades', 0) <= 0:
            raise ValueError("max_daily_trades must be positive")

        # Validate risk parameters
        risk = self._config.get('risk', {})

        if not 0 < risk.get('max_drawdown_pct', 0) <= 1:
            raise ValueError("max_drawdown_pct must be between 0 and 1")

        if risk.get('max_consecutive_losses', 0) <= 0:
            raise ValueError("max_consecutive_losses must be positive")

        # Validate timeframes
        mtf = self._config.get('multi_timeframe', {})
        if mtf.get('enabled'):
            total_weight = sum(tf.get('weight', 0) for tf in mtf.get('timeframes', {}).values())
            if not 0.99 <= total_weight <= 1.01:  # Allow small floating point errors
                logger.warning(f"Timeframe weights sum to {total_weight:.3f}, should be 1.0")

        logger.debug("Configuration validation passed")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            key: Configuration key (e.g., 'trading.min_confidence')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self._config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value

    def get_trading_config(self) -> TradingConfig:
        """Get trading configuration as dataclass"""
        trading = self._config.get('trading', {})
        risk = self._config.get('risk', {})
        signals = self._config.get('signals', {})

        return TradingConfig(
            symbol=trading.get('symbol', 'XAUUSD'),
            timeframe=trading.get('timeframe', 'H1'),
            min_confidence=signals.get('min_confidence', 0.85),
            min_mtf_alignment=trading.get('min_mtf_alignment', 0.60),
            max_position_size=trading.get('max_position_size', 1.0),
            max_daily_trades=trading.get('max_daily_trades', 5),
            max_open_positions=trading.get('max_open_positions', 3),
            default_position_size=risk.get('position_sizing', {}).get('default_size', 0.01),
            stop_loss_default=signals.get('exit', {}).get('stop_loss_atr_multiplier', 0.01),
            risk_reward_min=signals.get('exit', {}).get('take_profit_ratio', 2.0)
        )

    def get_risk_config(self) -> RiskConfig:
        """Get risk configuration as dataclass"""
        risk = self._config.get('risk', {})

        return RiskConfig(
            max_daily_loss_pct=risk.get('max_daily_loss_pct', 0.05),
            max_weekly_loss_pct=risk.get('max_weekly_loss_pct', 0.10),
            max_drawdown_pct=risk.get('max_drawdown_pct', 0.15),
            max_correlation=risk.get('max_correlation', 0.70),
            max_consecutive_losses=risk.get('max_consecutive_losses', 3),
            pause_duration_hours=risk.get('pause_duration_hours', 24)
        )

    def get_database_config(self) -> DatabaseConfig:
        """Get database configuration as dataclass"""
        db = self._config.get('database', {})

        return DatabaseConfig(
            enabled=db.get('enabled', True),
            type=db.get('type', 'sqlite'),
            sqlite_path=db.get('sqlite', {}).get('path'),
            pg_host=db.get('postgresql', {}).get('host'),
            pg_port=db.get('postgresql', {}).get('port'),
            pg_database=db.get('postgresql', {}).get('database')
        )

    def is_production(self) -> bool:
        """Check if running in production environment"""
        env = self.get('system.environment', 'development')
        return env.lower() == 'production'

    def is_paper_trading(self) -> bool:
        """Check if paper trading mode is enabled"""
        return self.get('paper_trading.enabled', False)

    def reload(self) -> None:
        """Reload configuration from file"""
        logger.info("Reloading configuration...")
        self._config = {}
        self.load_config(str(self._config_path))
        logger.info("Configuration reloaded successfully")

    def save(self, path: Optional[str] = None) -> None:
        """
        Save current configuration to file

        Args:
            path: Path to save config (default: original path)
        """
        save_path = Path(path) if path else self._config_path

        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self._config, f, default_flow_style=False, allow_unicode=True)

            logger.info(f"Configuration saved to: {save_path}")

        except Exception as e:
            logger.error(f"Error saving config: {e}")
            raise

    def __repr__(self) -> str:
        """String representation"""
        env = self.get('system.environment', 'unknown')
        version = self.get('system.version', 'unknown')
        return f"<ConfigManager: {env} v{version}>"


def demo_config_manager():
    """Demo configuration manager"""

    logger.info("")
    logger.info("="*80)
    logger.info("  CONFIGURATION MANAGER DEMO")
    logger.info("="*80)
    logger.info("")

    # Initialize config manager
    config = ConfigManager()

    # Display system info
    logger.info("SYSTEM CONFIGURATION:")
    logger.info(f"  Name: {config.get('system.name')}")
    logger.info(f"  Version: {config.get('system.version')}")
    logger.info(f"  Environment: {config.get('system.environment')}")
    logger.info(f"  Production Mode: {config.is_production()}")
    logger.info(f"  Paper Trading: {config.is_paper_trading()}")
    logger.info("")

    # Get trading config
    logger.info("TRADING CONFIGURATION:")
    trading_config = config.get_trading_config()
    logger.info(f"  Symbol: {trading_config.symbol}")
    logger.info(f"  Timeframe: {trading_config.timeframe}")
    logger.info(f"  Min Confidence: {trading_config.min_confidence:.0%}")
    logger.info(f"  Min MTF Alignment: {trading_config.min_mtf_alignment:.0%}")
    logger.info(f"  Max Position Size: {trading_config.max_position_size:.0%}")
    logger.info(f"  Max Daily Trades: {trading_config.max_daily_trades}")
    logger.info("")

    # Get risk config
    logger.info("RISK MANAGEMENT:")
    risk_config = config.get_risk_config()
    logger.info(f"  Max Daily Loss: {risk_config.max_daily_loss_pct:.0%}")
    logger.info(f"  Max Drawdown: {risk_config.max_drawdown_pct:.0%}")
    logger.info(f"  Max Consecutive Losses: {risk_config.max_consecutive_losses}")
    logger.info(f"  Pause Duration: {risk_config.pause_duration_hours}h")
    logger.info("")

    # Get database config
    logger.info("DATABASE:")
    db_config = config.get_database_config()
    logger.info(f"  Enabled: {db_config.enabled}")
    logger.info(f"  Type: {db_config.type}")
    if db_config.type == 'sqlite':
        logger.info(f"  Path: {db_config.sqlite_path}")
    logger.info("")

    # Get specific values
    logger.info("FEATURE FLAGS:")
    logger.info(f"  MTF Analysis: {config.get('multi_timeframe.enabled')}")
    logger.info(f"  Regime Detection: {config.get('regime.enabled')}")
    logger.info(f"  Sentiment Analysis: {config.get('sentiment.enabled')}")
    logger.info(f"  Alternative Data: {config.get('alternative_data.enabled')}")
    logger.info(f"  Ensemble: {config.get('ensemble.enabled')}")
    logger.info(f"  ML Models: {config.get('ml.enabled')}")
    logger.info("")

    logger.info("="*80)
    logger.info("CONFIGURATION MANAGER READY!")
    logger.info("="*80)
    logger.info("")
    logger.info("USAGE:")
    logger.info("  from config_manager import ConfigManager")
    logger.info("  config = ConfigManager()")
    logger.info("  min_conf = config.get('trading.min_confidence')")
    logger.info("  trading_config = config.get_trading_config()")
    logger.info("")


if __name__ == "__main__":
    demo_config_manager()
