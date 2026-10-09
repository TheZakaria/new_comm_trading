from dataclasses import dataclass
from typing import Optional
@dataclass
class FXTradingConfig:
    MODEL_NAME: str = "tcn"
    INPUT_CHUNK_LENGTH: int = 64
    OUTPUT_CHUNK_LENGTH: int = 1
    N_EPOCHS: int = 3
    TRAIN_BATCH_SIZE: int = 1024
    EVAL_BATCH_SIZE: int = 128

    MASTER_FX_FILE: str = "dataset/corn_futures.csv"
    MASTER_NEWS_FILE: Optional[str] = "dataset/news/usdbrl-news.csv"
    TRAIN_WINDOW: str = "1M"
    VAL_WINDOW: str = "1M"
    TEST_WINDOW: str = "1M"
    WINDOW_STEP: str = "3M"

    MIN_FIRST_HOUR_VALUES: int = 0
    
    WALLET_A: float = 10000.0
    WALLET_B: float = 10000.0
    BET_SIZING: str = "fixed"
    ENABLE_TRANSACTION_COSTS: bool = False
    OUTPUT_DIR: str = "results/usd-cnh"
    NEWS_HOLD_MINUTES: int = 3
    ALLOW_NEWS_OVERLAP: bool = False
    SENTIMENT_SOURCE: str = "label"
    KELLY_WINDOW_DAYS: int = None
    MIN_TRADES_FOR_FULL_KELLY: int = None
    MIN_KELLY_FRACTION: float = 0.005
    THRESHOLD: float = 0.0
    FAST_MA_WINDOW: int = 10
    SLOW_MA_WINDOW: int = 30
    SEED: int = 59