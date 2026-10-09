import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import copy
import json
import argparse
import random
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Union
from collections import defaultdict

from utils import FXTradingConfig
from data_processing import DataProcessor
from models import DartsFinancialForecastingModel, ChronosFinancialForecastingModel, TotoFinancialForecastingModel
from strategies import TradingStrategy
from metrics import ModelEvalMetrics

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def generate_sliding_bounds(master_fx_path: str, train_window: str, val_window: str, test_window: str, step_window: str):
    df = pd.read_csv(master_fx_path, usecols=['date'])
    timestamps = pd.to_datetime(df['date'], utc=True)
    
    min_ts, max_ts = timestamps.min(), timestamps.max()
    current_start = pd.Timestamp(year=min_ts.year, month=min_ts.month, day=1, tz="UTC")
    
    train_offset = pd.DateOffset(months=int(train_window.replace("M", "")))
    val_offset = pd.DateOffset(months=int(val_window.replace("M", "")))
    test_offset = pd.DateOffset(months=int(test_window.replace("M", "")))
    step_offset = pd.DateOffset(months=int(step_window.replace("M", "")))
    
    folds = []
    fold_idx = 0
    
    while True:
        train_end = current_start + train_offset
        val_end = train_end + val_offset
        test_end = val_end + test_offset
        if test_end > max_ts + pd.Timedelta(days=1): break
            
        folds.append({
            "period_id": f"period_{fold_idx:02d}_{val_end.strftime('%Y%m%d')}",
            "train_start": current_start,
            "train_end": train_end,
            "val_end": val_end,
            "test_end": test_end
        })
        current_start += step_offset
        fold_idx += 1
    return folds


def compute_hourly_earnings_table(all_trade_records: list, strategy_filter: str = "model_driven") -> pd.DataFrame:
    df = pd.DataFrame(all_trade_records)
    if df.empty or strategy_filter not in df['strategy'].values:
        return pd.DataFrame()

    df = df[df["strategy"] == strategy_filter]
    total_abs_pnl = df["profit"].sum()

    hourly_rows = []
    for hour, grp in df.groupby("hour", sort=True):
        pnls = grp["profit"].to_numpy(dtype=float)
        tot_pnl = float(np.sum(pnls))
        mean_pnl = float(np.mean(pnls))
        std_pnl = float(np.std(pnls, ddof=1)) if len(pnls) > 1 else 0.0
        
        hourly_rows.append({
            "Hour": f"{int(hour):02d}:00",
            "Trades": len(pnls),
            "Win Rate (%)": round(float(np.mean(pnls > 0) * 100.0), 2),
            "Total PnL": round(tot_pnl, 6),
            "Mean PnL": round(mean_pnl, 6),
            "Sharpe": round((mean_pnl / std_pnl), 4) if std_pnl > 1e-12 else 0.0,
            "PnL Share (%)": round((tot_pnl / abs(total_abs_pnl)) * 100.0, 2) if abs(total_abs_pnl) > 1e-12 else 0.0,
        })
    return pd.DataFrame(hourly_rows)

def group_data_by_date(
    fx_timestamps,
    news_timestamps,
    true_values: List[float],
    predicted_values: Union[List[float], Dict[str, List[float]]],
    bid_prices: List[float],
    ask_prices: List[float],
    news_sentiments: List[float],
    min_first_hour_values: int = 0,
):
    chunked_values = defaultdict(lambda: {
        "fx_timestamps": [],
        "news_timestamps": [],
        "true_values": [],
        "predicted_values": [] if isinstance(predicted_values, list) else {},
        "bid_prices": [],
        "ask_prices": [],
        "news_sentiments": [],
    })

    n = len(true_values)
    for i in range(n):
        fx_timestamp = fx_timestamps[i]

        # =====================================================================
        # Stop trading entirely once a specific date is reached.
        # =====================================================================
        stop_date = pd.Timestamp(year=2026, month=4, day=1, tz="UTC").date()
        if fx_timestamp.date() >= stop_date:
            continue

        # =====================================================================
        # Set trading duration to only happen between 9:00 AM - 11:00 AM
        # =====================================================================
        if not (9 <= fx_timestamp.hour < 11):
            continue
        
        true_val = true_values[i]
        bid_price = bid_prices[i]
        ask_price = ask_prices[i]
        date_key = fx_timestamp.date()

        day_bucket = chunked_values[date_key]
        day_bucket["fx_timestamps"].append(fx_timestamp)
        day_bucket["true_values"].append(true_val)
        day_bucket["bid_prices"].append(bid_price)
        day_bucket["ask_prices"].append(ask_price)

        if isinstance(predicted_values, list):
            day_bucket["predicted_values"].append(predicted_values[i])
        elif isinstance(predicted_values, dict):
            if not isinstance(day_bucket["predicted_values"], dict):
                day_bucket["predicted_values"] = {}
            for model_name, preds in predicted_values.items():
                if model_name not in day_bucket["predicted_values"]:
                    day_bucket["predicted_values"][model_name] = []
                day_bucket["predicted_values"][model_name].append(preds[i])

    for news_timestamp, news_sentiment in zip(news_timestamps, news_sentiments):
        date_key = news_timestamp.date()
        if date_key not in chunked_values:
            continue
        chunked_values[date_key]["news_timestamps"].append(news_timestamp)
        chunked_values[date_key]["news_sentiments"].append(news_sentiment)

    print("\n--- First Hour (09:00 - 09:59) Sparsity Check ---")
    for date_key in sorted(chunked_values.keys()):
        day_bucket = chunked_values[date_key]
        total_day_values = len(day_bucket["fx_timestamps"])
        first_hour_count = sum(1 for ts in day_bucket["fx_timestamps"] if ts.hour == 9)

        if first_hour_count < min_first_hour_values:
            valid_fx_indices = [idx for idx, ts in enumerate(day_bucket["fx_timestamps"]) if ts.hour < 10]
            day_bucket["fx_timestamps"] = [day_bucket["fx_timestamps"][idx] for idx in valid_fx_indices]
            day_bucket["true_values"] = [day_bucket["true_values"][idx] for idx in valid_fx_indices]
            day_bucket["bid_prices"] = [day_bucket["bid_prices"][idx] for idx in valid_fx_indices]
            day_bucket["ask_prices"] = [day_bucket["ask_prices"][idx] for idx in valid_fx_indices]

            if isinstance(day_bucket["predicted_values"], list):
                day_bucket["predicted_values"] = [day_bucket["predicted_values"][idx] for idx in valid_fx_indices]
            elif isinstance(day_bucket["predicted_values"], dict):
                for model_name in day_bucket["predicted_values"]:
                    day_bucket["predicted_values"][model_name] = [
                        day_bucket["predicted_values"][model_name][idx] for idx in valid_fx_indices
                    ]

            valid_news_indices = [idx for idx, ts in enumerate(day_bucket["news_timestamps"]) if ts.hour < 10]
            day_bucket["news_timestamps"] = [day_bucket["news_timestamps"][idx] for idx in valid_news_indices]
            day_bucket["news_sentiments"] = [day_bucket["news_sentiments"][idx] for idx in valid_news_indices]

            print(f"[{date_key}] Found {first_hour_count} values (req >= {min_first_hour_values}) -> STOPPING AT 09:59 (traded {len(valid_fx_indices)}/{total_day_values})")
        else:
            print(f"[{date_key}] Found {first_hour_count} values (req >= {min_first_hour_values}) -> CONTINUING FULL DAY ({total_day_values} values)")
    print("-------------------------------------------------\n")
    return chunked_values

def run_ml_based_trading_strategies(fx_trading_config, train_df, val_df, test_df, news_train_df, news_test_df):
    """Run trading strategy for a single configured fold."""
    set_seed(fx_trading_config.SEED)
    torch.set_float32_matmul_precision("high")

    dataProcessor = DataProcessor(
        fx_trading_config, 
        train_df=train_df, 
        val_df=val_df, 
        test_df=test_df, 
        news_train_df=news_train_df, 
        news_test_df=news_test_df
    )
    processed_data = dataProcessor.split_and_scale_data()

    test_fx_timestamps = processed_data.test_fx_timestamps
    test_bid_prices = processed_data.test_bid_prices
    test_ask_prices = processed_data.test_ask_prices
    test_news_timestamps = processed_data.test_news_timestamps
    test_news_sentiments = processed_data.test_news_sentiments
    true_values = processed_data.test_mid_prices

    if fx_trading_config.MODEL_NAME == 'toto':
        predictor = TotoFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values = predictor.generate_predictions(processed_data.llm_test_scaled)
    elif fx_trading_config.MODEL_NAME == 'chronos':
        predictor = ChronosFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values = predictor.generate_predictions(processed_data.llm_test_scaled)
    elif fx_trading_config.MODEL_NAME == 'ensemble':
        predictor1 = DartsFinancialForecastingModel(fx_trading_config, processed_data.darts_scaler)
        predictor1.train(processed_data.darts_train_scaled, processed_data.darts_val_scaled)
        predicted_values1 = predictor1.generate_predictions(processed_data.darts_test_scaled)

        predictor2 = ChronosFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values2 = predictor2.generate_predictions(processed_data.llm_test_scaled)

        predictor3 = TotoFinancialForecastingModel(fx_trading_config, processed_data.llm_scaler)
        predicted_values3 = predictor3.generate_predictions(processed_data.llm_test_scaled)

        predicted_values = {
            **predicted_values1,
            "chronos": predicted_values2,
            "toto": predicted_values3
        }
    else:
        predictor = DartsFinancialForecastingModel(fx_trading_config, processed_data.darts_scaler)
        predictor.train(processed_data.darts_train_scaled, processed_data.darts_val_scaled)
        predicted_values = predictor.generate_predictions(processed_data.darts_test_scaled)

    metrics = ModelEvalMetrics()
    prediction_errors = None
    if fx_trading_config.MODEL_NAME != 'ensemble':
        prediction_errors = metrics.calculate_prediction_errors(true_values, predicted_values)
        metrics.rmse = prediction_errors.get("rmse", 0) # ensure metric is saved

    chunked_values = group_data_by_date(
        test_fx_timestamps, test_news_timestamps, true_values, predicted_values,
        test_bid_prices, test_ask_prices, test_news_sentiments,
        min_first_hour_values=getattr(fx_trading_config, "MIN_FIRST_HOUR_VALUES", 0),
    )

    is_ensemble_model = fx_trading_config.MODEL_NAME == 'ensemble'
    trading_strategy = TradingStrategy(
        fx_trading_config.WALLET_A, fx_trading_config.WALLET_B, fx_trading_config.NEWS_HOLD_MINUTES,
        fx_trading_config.BET_SIZING, fx_trading_config.ENABLE_TRANSACTION_COSTS,
        fx_trading_config.ALLOW_NEWS_OVERLAP, kelly_window_days=fx_trading_config.KELLY_WINDOW_DAYS,
        min_trades_for_full_kelly=fx_trading_config.MIN_TRADES_FOR_FULL_KELLY,
        min_kelly_fraction=fx_trading_config.MIN_KELLY_FRACTION, threshold=fx_trading_config.THRESHOLD,
        fast_ma_window=fx_trading_config.FAST_MA_WINDOW, slow_ma_window=fx_trading_config.SLOW_MA_WINDOW,
    )
    
    # Optional: Attach config namespace if you implemented timestamp logic in trading_strategy
    trading_strategy.config = fx_trading_config 

    for date_key, values in sorted(chunked_values.items()):
        trading_strategy.advance_kelly_day()
        if len(values['fx_timestamps']) > 0:
            if is_ensemble_model:
                trading_strategy.simulate_trading_with_strategies(
                    values['fx_timestamps'], values['true_values'], values['true_values'],
                    values['bid_prices'], values['ask_prices'], values['news_timestamps'],
                    values['news_sentiments'], strategy_names=['mean_reversion', 'trend'],
                )
                trading_strategy.simulate_trading_with_ensemble_strategy(
                    values['fx_timestamps'], values['true_values'], values['predicted_values'],
                    values['bid_prices'], values['ask_prices'], seed=fx_trading_config.SEED,
                )
            else:
                trading_strategy.simulate_trading_with_strategies(
                    values['fx_timestamps'], values['true_values'], values['predicted_values'],
                    values['bid_prices'], values['ask_prices'], values['news_timestamps'], values['news_sentiments'],
                )

    # Export standard JSON for backward compatibility
    profit_per_trade_data = {
        "mean_reversion": trading_strategy.pnl["mean_reversion"],
        "trend": trading_strategy.pnl["trend"],
        "ma_crossover": trading_strategy.pnl["ma_crossover"],
        "news_sentiment": trading_strategy.pnl["news_sentiment"]
    }
    if is_ensemble_model:
        profit_per_trade_data["ensemble"] = trading_strategy.pnl["ensemble"]
    else:
        profit_per_trade_data["model_driven"] = trading_strategy.pnl["model_driven"]

    trades_output_path = os.path.join(fx_trading_config.OUTPUT_DIR, f'{fx_trading_config.MODEL_NAME}_profit_per_trade.json')
    with open(trades_output_path, "w") as f:
        json.dump(profit_per_trade_data, f, indent=4)

    return metrics, trading_strategy

# --- The Main Orchestrator ---
def run_sliding_window_pipeline(args):
    root_dir = os.path.dirname(os.path.abspath(__file__))
    base_out = os.path.join(root_dir, args.output_dir)

    folds = generate_sliding_bounds(args.master_fx_file, args.train_window, args.val_window, args.test_window, args.step_window)

    print(f"Loading master dataset {args.master_fx_file} into memory...")
    master_df = pd.read_csv(args.master_fx_file)
    master_df['date'] = pd.to_datetime(master_df['date'], utc=True)
    
    master_news_df = None
    if args.master_news_file:
        master_news_df = pd.read_csv(args.master_news_file)
        master_news_df['date'] = pd.to_datetime(master_news_df['date'], utc=True)

    period_summaries = []
    all_timestamped_trades = []

    for fold in folds:
        print(f"\n" + "="*60)
        print(f"Slicing Data for {fold['period_id']}")
        print("="*60)
        
        fold_dir = os.path.join(base_out, args.model_name, fold["period_id"])
        os.makedirs(fold_dir, exist_ok=True)

        t_start, t_end = fold["train_start"], fold["train_end"]
        v_end, test_end = fold["val_end"], fold["test_end"]

        # 1. Slice in memory
        train_df = master_df[(master_df['date'] >= t_start) & (master_df['date'] < t_end)]
        val_df = master_df[(master_df['date'] >= t_end) & (master_df['date'] < v_end)]
        test_df = master_df[(master_df['date'] >= v_end) & (master_df['date'] < test_end)]

        if train_df.empty or val_df.empty or test_df.empty:
            print(f"Skipping {fold['period_id']} (insufficient data).")
            continue

        news_train, news_test = None, None
        if master_news_df is not None:
            news_train = master_news_df[(master_news_df['date'] >= t_start) & (master_news_df['date'] < v_end)]
            news_test = master_news_df[(master_news_df['date'] >= v_end) & (master_news_df['date'] < test_end)]

        # 2. Build Config for this fold
        config = FXTradingConfig()
        config.INPUT_CHUNK_LENGTH = args.input_chunk_length
        config.OUTPUT_CHUNK_LENGTH = args.output_chunk_length
        config.N_EPOCHS = args.n_epochs
        config.MIN_FIRST_HOUR_VALUES = args.min_first_hour_values
        config.TRAIN_BATCH_SIZE = args.train_batch_size
        config.EVAL_BATCH_SIZE = args.eval_batch_size
        config.WALLET_A = args.wallet_a
        config.WALLET_B = args.wallet_b
        config.BET_SIZING = args.bet_sizing
        config.ENABLE_TRANSACTION_COSTS = args.enable_transaction_costs
        config.NEWS_HOLD_MINUTES = args.news_hold_minutes
        config.ALLOW_NEWS_OVERLAP = args.allow_news_overlap
        config.SENTIMENT_SOURCE = args.sentiment_source
        config.SEED = args.seed
        config.KELLY_WINDOW_DAYS = args.kelly_window_days
        config.MIN_TRADES_FOR_FULL_KELLY = args.min_trades_for_full_kelly
        config.MIN_KELLY_FRACTION = args.min_kelly_fraction
        config.THRESHOLD = args.threshold
        config.FAST_MA_WINDOW = args.fast_ma_window
        config.SLOW_MA_WINDOW = args.slow_ma_window
        config.MODEL_NAME = args.model_name
        
        # Inject split specifics
        config.OUTPUT_DIR = fold_dir
        config.period_id = fold["period_id"]

        eval_metrics, trading_strategy = run_ml_based_trading_strategies(
            config, 
            train_df=train_df, 
            val_df=val_df, 
            test_df=test_df, 
            news_train_df=news_train, 
            news_test_df=news_test
        )

        # 4. Collect logic for reporting
        if hasattr(trading_strategy, 'trade_records'):
            ts_json_path = os.path.join(config.OUTPUT_DIR, f"{config.MODEL_NAME}_timestamped_trades.json")
            with open(ts_json_path, "w") as f:
                json.dump(trading_strategy.trade_records, f, indent=2)

            strat_key = "ensemble" if args.model_name == "ensemble" else "model_driven"
            model_trades = trading_strategy.trade_records.get(strat_key, [])
            for strat_name, records in trading_strategy.trade_records.items():
                for r in records:
                    all_timestamped_trades.append({"strategy": strat_name, **r})
            pnls = [t["profit"] for t in model_trades]
        else:
            strat_key = "ensemble" if args.model_name == "ensemble" else "model_driven"
            pnls = trading_strategy.pnl.get(strat_key, [])

        period_summaries.append({
            "Period": config.period_id,
            "Trades": len(pnls),
            "Win Rate (%)": round(float(np.mean(np.array(pnls) > 0) * 100.0), 2) if pnls else 0.0,
            "Total PnL": round(float(np.sum(pnls)), 6) if pnls else 0.0,
            "RMSE": getattr(eval_metrics, "rmse", None)
        })

    print("\n" + "=" * 80)
    print("SLIDING WINDOW SUMMARY")
    print("=" * 80)
    periods_df = pd.DataFrame(period_summaries)
    print(periods_df.to_string(index=False))

    if all_timestamped_trades:
        print("\n" + "=" * 80)
        print("HOURLY EARNINGS BREAKDOWN (model_driven)")
        print("=" * 80)
        strat_key = "ensemble" if args.model_name == "ensemble" else "model_driven"
        hourly_df = compute_hourly_earnings_table(all_timestamped_trades, strat_key)
        if not hourly_df.empty:
            print(hourly_df.to_string(index=False))
            hourly_df.to_csv(f"{args.output_dir}/{args.model_name}_hourly_earnings.csv", index=False)
        pd.DataFrame(all_timestamped_trades).to_csv(f"{args.output_dir}/{args.model_name}_all_timestamped_trades.csv", index=False)

    periods_df.to_csv(f"{args.output_dir}/{args.model_name}_sliding_periods_summary.csv", index=False)
    if all_timestamped_trades:
        df_trades = pd.DataFrame(all_timestamped_trades)
        
        # 1. Daily / Cumulative PnL Plot Across Testing Timeline with Fold Dividers
        if not df_trades.empty and "date" in df_trades.columns:
            df_trades['timestamp'] = pd.to_datetime(df_trades['date'])
            df_trades['day'] = df_trades['timestamp'].dt.date
            daily_pnl = df_trades.groupby(['day', 'strategy'])['profit'].sum().unstack(fill_value=0)
            cumulative_pnl = daily_pnl.cumsum()

            plt.figure(figsize=(14, 6))
            for strat in cumulative_pnl.columns:
                plt.plot(cumulative_pnl.index, cumulative_pnl[strat], label=f"{strat} (Cumulative)", linewidth=2)

            # Add vertical lines for each fold's test period start (val_end)
            for fold in folds:
                fold_start_date = pd.to_datetime(fold["val_end"]).date()
                if cumulative_pnl.index.min() <= fold_start_date <= cumulative_pnl.index.max():
                    plt.axvline(
                        x=fold_start_date, 
                        color="gray", 
                        linestyle="--", 
                        alpha=0.3,
                        linewidth=1
                    )

            plt.title(f"Cumulative PnL Over Time with Fold Boundaries ({args.model_name.upper()})")
            plt.xlabel("Date")
            plt.ylabel("Cumulative Profit")
            plt.legend()
            plt.grid(True, linestyle="--", alpha=0.4)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_cumulative_pnl_over_time.png"))
            plt.close()

        # 2. Aggregated PnL per Hour Plot
        strat_key = "ensemble" if args.model_name == "ensemble" else f"model_driven"
        hourly_df = compute_hourly_earnings_table(all_timestamped_trades, strat_key)
        if not hourly_df.empty:
            plt.figure(figsize=(10, 5))
            plt.bar(hourly_df["Hour"], hourly_df["Total PnL"], color="skyblue", edgecolor="navy", alpha=0.8)
            plt.axhline(0, color="gray", linestyle="--", linewidth=1)
            plt.title(f"Aggregated Total PnL per Hour - {args.model_name.upper()} ({strat_key})")
            plt.xlabel("Hour of Day (UTC)")
            plt.ylabel("Total Profit")
            plt.grid(axis="y", linestyle="--", alpha=0.6)
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, f"{args.model_name}_hourly_pnl_bar.png"))
            plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility.")
    parser.add_argument("--wallet_a", type=float, default=1000000.0)
    parser.add_argument("--wallet_b", type=float, default=1000000.0)
    parser.add_argument("--model_name", type=str, choices=["arima", "nbeats", "nhits", "tcn", "toto", "chronos", "ensemble"], default="tcn")
    parser.add_argument("--input_chunk_length", type=int, default=64)
    parser.add_argument("--output_chunk_length", type=int, default=1)
    parser.add_argument("--n_epochs", type=int, default=50)
    parser.add_argument("--train_batch_size", type=int, default=96000)
    parser.add_argument("--eval_batch_size", type=int, default=96000)
    
    # Master file config
    parser.add_argument("--master_fx_file", type=str, required=True, help="Path to single master CSV")
    parser.add_argument("--master_news_file", type=str, default=None)
    parser.add_argument("--train_window", type=str, default="1M")
    parser.add_argument("--val_window", type=str, default="1M")
    parser.add_argument("--test_window", type=str, default="1M")
    parser.add_argument("--step_window", type=str, default="1M")
    
    # Strategy Configs
    parser.add_argument("--bet_sizing", type=str, choices=["active_kelly", "passive_kelly", "fixed"], default="fixed")
    parser.add_argument("--enable_transaction_costs", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/usd-cny-2023")
    parser.add_argument("--news_hold_minutes", type=int, default=-1)
    parser.add_argument("--sentiment_source", type=str, default="label")
    parser.add_argument("--allow_news_overlap", action="store_true")
    parser.add_argument("--kelly_window_days", type=int, default=None)
    parser.add_argument("--min_trades_for_full_kelly", type=int, default=None)
    parser.add_argument("--min_kelly_fraction", type=float, default=0.005)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--fast_ma_window", type=int, default=10)
    parser.add_argument("--slow_ma_window", type=int, default=30)
    parser.add_argument("--min_first_hour_values", type=int, default=0)

    args = parser.parse_args()

    run_sliding_window_pipeline(args)