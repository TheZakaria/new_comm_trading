import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.preprocessing import MinMaxScaler
from dataclasses import dataclass
from typing import List, Any, Optional

@dataclass
class ProcessedData:
    """Container for all processed and scaled data."""
    # Darts format data (for Darts models)
    darts_train_scaled: Any
    darts_val_scaled: Any
    darts_test_scaled: Any
    darts_scaler: Any  # Darts Scaler for inverse transform

    # llm format data (for Chronos/Toto models)
    llm_train_scaled: np.ndarray
    llm_val_scaled: np.ndarray
    llm_test_scaled: np.ndarray
    llm_scaler: Any  # MinMaxScaler for inverse transform

    # Test metadata (common to all models)
    test_fx_timestamps: List
    test_bid_prices: List[float]
    test_ask_prices: List[float]
    test_news_timestamps: List
    test_news_sentiments: List[float]
    test_mid_prices: List[float]  # Unscaled test mid prices for evaluation


class DataProcessor:
    def __init__(
        self, 
        fx_trading_config, 
        train_df: Optional[pd.DataFrame] = None, 
        val_df: Optional[pd.DataFrame] = None, 
        test_df: Optional[pd.DataFrame] = None,
        news_train_df: Optional[pd.DataFrame] = None,
        news_test_df: Optional[pd.DataFrame] = None
    ):
        self.fx_trading_config = fx_trading_config
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df
        self.news_train_df = news_train_df
        self.news_test_df = news_test_df

    def _standardize_date_col(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles empty dataframes and standardizes 'timestamp' vs 'date' column naming."""
        if df is None or df.empty:
            return df
        
        # Unify column naming to 'date'
        if 'timestamp' in df.columns and 'date' not in df.columns:
            df.rename(columns={'timestamp': 'date'}, inplace=True)
            
        df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True)
        df.sort_values("date", inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def load_fx_data(self):
        """Loads the time series data either from memory or disk."""
        if self.train_df is not None:
            # 100% In-Memory Path
            fx_data_train = self.train_df.copy()
            fx_data_val = self.val_df.copy()
            fx_data_test = self.test_df.copy()
        else:
            # FIX: Restored the legacy fallback path just in case
            fx_data_train = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TRAIN)
            fx_data_val = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_VAL)
            fx_data_test = pd.read_csv(self.fx_trading_config.FX_DATA_PATH_TEST)

        return (
            self._standardize_date_col(fx_data_train),
            self._standardize_date_col(fx_data_val),
            self._standardize_date_col(fx_data_test)
        )

    def load_news_data(self):
        """Loads the news data either from memory or disk."""
        if self.news_train_df is not None and self.news_test_df is not None:
            news_data_train = self.news_train_df.copy()
            news_data_test = self.news_test_df.copy()
        elif hasattr(self.fx_trading_config, 'NEWS_DATA_PATH_TRAIN') and self.fx_trading_config.NEWS_DATA_PATH_TRAIN:
            news_data_train = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TRAIN)
            news_data_test = pd.read_csv(self.fx_trading_config.NEWS_DATA_PATH_TEST)
        else:
            news_data_train = pd.DataFrame()
            news_data_test = pd.DataFrame()

        return (
            self._standardize_date_col(news_data_train),
            self._standardize_date_col(news_data_test)
        )

    def _aggregate_news_by_minute(self, news_df, sentiment_col):
        if news_df.empty or sentiment_col not in news_df.columns:
            return news_df

        df = news_df.copy()
        df["date_minute"] = df["date"].dt.floor("min")

        def pick_majority_with_tie_zero(x):
            counts = x.value_counts()
            if counts.empty: return 0
            max_count = counts.max()
            tied = counts[counts == max_count].index.tolist()
            return tied[0] if len(tied) == 1 else 0

        agg = (
            df.groupby("date_minute", as_index=False)[sentiment_col]
              .agg(pick_majority_with_tie_zero)
              .rename(columns={"date_minute": "date"})
        )
        agg.sort_values("date", inplace=True)
        agg.reset_index(drop=True, inplace=True)
        return agg

    def split_and_scale_data(self):
        """Split and scale data for all models."""
        input_chunk_length = self.fx_trading_config.INPUT_CHUNK_LENGTH
        fx_data_train, fx_data_val, fx_data_test = self.load_fx_data()
        news_data_train, news_data_test = self.load_news_data()
        
        price_col = "close"
        sentiment_col = self.fx_trading_config.SENTIMENT_SOURCE
        
        news_data_train = self._aggregate_news_by_minute(news_data_train, sentiment_col)
        news_data_test = self._aggregate_news_by_minute(news_data_test, sentiment_col)

        fx_timestamps = fx_data_test["date"].tolist()
        bid_prices = fx_data_test[price_col].tolist()
        ask_prices = fx_data_test[price_col].tolist()
        
        # Handle cases where news data might be empty
        news_timestamps = news_data_test["date"].tolist() if not news_data_test.empty else []
        news_sentiments = news_data_test[sentiment_col].tolist() if not news_data_test.empty else []

        # --- Prepare Darts format data ---
        darts_train = TimeSeries.from_dataframe(fx_data_train, value_cols=[price_col])
        darts_val = TimeSeries.from_dataframe(fx_data_val, value_cols=[price_col])
        darts_test = TimeSeries.from_dataframe(fx_data_test, value_cols=[price_col])

        darts_scaler = Scaler()
        darts_train_scaled = darts_scaler.fit_transform(darts_train)
        darts_val_scaled = darts_scaler.transform(darts_val)
        darts_test_scaled = darts_scaler.transform(darts_test)

        # --- Prepare llm format data ---
        llm_train = fx_data_train[price_col].values.reshape(-1, 1).astype(np.float32)
        llm_val = fx_data_val[price_col].values.reshape(-1, 1).astype(np.float32)
        llm_test = fx_data_test[price_col].values.reshape(-1, 1).astype(np.float32)

        llm_scaler = MinMaxScaler(feature_range=(0, 1))
        llm_train_scaled = llm_scaler.fit_transform(llm_train)
        llm_val_scaled = llm_scaler.transform(llm_val)
        llm_test_scaled = llm_scaler.transform(llm_test)

        # --- Prepare test metadata with input_chunk_length offset ---
        test_fx_timestamps = fx_timestamps[input_chunk_length:]
        test_bid_prices = bid_prices[input_chunk_length:]
        test_ask_prices = ask_prices[input_chunk_length:]
        test_mid_prices = fx_data_test[price_col].values[input_chunk_length:].tolist()

        return ProcessedData(
            darts_train_scaled=darts_train_scaled,
            darts_val_scaled=darts_val_scaled,
            darts_test_scaled=darts_test_scaled,
            darts_scaler=darts_scaler,
            llm_train_scaled=llm_train_scaled,
            llm_val_scaled=llm_val_scaled,
            llm_test_scaled=llm_test_scaled,
            llm_scaler=llm_scaler,
            test_fx_timestamps=test_fx_timestamps,
            test_bid_prices=test_bid_prices,
            test_ask_prices=test_ask_prices,
            test_news_timestamps=news_timestamps,
            test_news_sentiments=news_sentiments,
            test_mid_prices=test_mid_prices,
        )