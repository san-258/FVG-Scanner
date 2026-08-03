# NASDAQ 100 FVG SCANNER - Forked from san-258/FVG-Scanner
# Adds: 50-EMA trend gate + enforced volume gate
# Daily Timeframe

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

NASDAQ_100 = [
    'AAPL', 'MSFT', 'AMZN', 'NVDA', 'META', 'GOOGL', 'GOOG', 'TSLA', 'AVGO', 'COST',
    'NFLX', 'ASML', 'AMD', 'PEP', 'ADBE', 'CSCO', 'TMUS', 'LIN', 'TXN', 'CMCSA',
    'QCOM', 'INTU', 'AMGN', 'HON', 'ISRG', 'AMAT', 'BKNG', 'ARM', 'VRTX', 'ADP',
    'PANW', 'SBUX', 'MU', 'GILD', 'ADI', 'INTC', 'LRCX', 'REGN', 'MDLZ', 'MELI',
    'SNPS', 'KLAC', 'CDNS', 'PYPL', 'CRWD', 'MAR', 'PDD', 'MRVL', 'CEG', 'FTNT',
    'CSX', 'ADSK', 'ORLY', 'DASH', 'ABNB', 'NXPI', 'ROP', 'WDAY', 'MNST', 'PCAR',
    'CPRT', 'TTD', 'AEP', 'CHTR', 'PAYX', 'FAST', 'ODFL', 'ROST', 'KDP', 'EA',
    'BKR', 'CTSH', 'VRSK', 'KHC', 'GEHC', 'DDOG', 'EXC', 'LULU', 'XEL', 'CCEP',
    'TEAM', 'IDXX', 'ZS', 'CSGP', 'TTWO', 'FANG', 'ON', 'CDW', 'MDB',
    'DXCM', 'GFS', 'WBD', 'BIIB', 'ILMN', 'MRNA', 'ALGN', 'SMCI', 'DLTR'
]

MIN_FVG_STRENGTH_PCT = 50
MIN_VOLUME_RATIO = 1.3
TREND_EMA = 50

def fetch_data(ticker, period='1y', interval='1d'):
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty or len(df) < 60:
            return None
        return df
    except Exception:
        return None

def calculate_indicators(df):
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    df[f'EMA{TREND_EMA}'] = df['Close'].ewm(span=TREND_EMA, adjust=False).mean()
    df['EMA_Slope'] = df[f'EMA{TREND_EMA}'].diff(5)

    return df

def detect_high_strength_fvgs(df, min_strength_pct=MIN_FVG_STRENGTH_PCT):
    bull_fvg_condition = df['Low'] > df['High'].shift(2)

    df['Bull_FVG_Top'] = np.where(bull_fvg_condition, df['Low'], np.nan)
    df['Bull_FVG_Bottom'] = np.where(bull_fvg_condition, df['High'].shift(2), np.nan)
    df['Bull_FVG_Height'] = df['Bull_FVG_Top'] - df['Bull_FVG_Bottom']
    df['Bull_FVG_50%'] = (df['Bull_FVG_Top'] + df['Bull_FVG_Bottom']) / 2

    df['Bull_FVG_Strength_%'] = (df['Bull_FVG_Height'] / df['ATR']) * 100
    df['High_Strength_FVG'] = df['Bull_FVG_Strength_%'] >= min_strength_pct

    return df

def is_fvg_unmitigated(df, fvg_idx, fvg_top):
    candles_after_fvg = df.loc[fvg_idx:].iloc[1:]
    if len(candles_after_fvg) == 0:
        return True
    return not (candles_after_fvg['Low'] <= fvg_top).any()

def scan_fvg_50_percent_approach(df, ticker):
    latest = df.iloc[-1]
    recent = df.iloc[-10:]

    # Trend gate: price above rising 50 EMA
    ema = latest[f'EMA{TREND_EMA}']
    ema_slope = latest['EMA_Slope']
    if pd.isna(ema) or latest['Close'] < ema or pd.isna(ema_slope) or ema_slope <= 0:
        return None

    high_strength_fvgs = recent[recent['High_Strength_FVG'] == True]
    if len(high_strength_fvgs) == 0:
        return None

    for fvg_idx in reversed(high_strength_fvgs.index):
        fvg_candle = df.loc[fvg_idx]

        fvg_top = fvg_candle['Bull_FVG_Top']
        fvg_bottom = fvg_candle['Bull_FVG_Bottom']
        fvg_50_percent = fvg_candle['Bull_FVG_50%']
        fvg_strength = fvg_candle['Bull_FVG_Strength_%']
        fvg_height = fvg_candle['Bull_FVG_Height']

        fvg_date = fvg_idx.strftime('%Y-%m-%d')
        fvg_date_display = fvg_idx.strftime('%b %d, %Y')

        current_price = latest['Close']
        days_since_fvg = len(df.loc[fvg_idx:]) - 1

        if not is_fvg_unmitigated(df, fvg_idx, fvg_top):
            continue

        fvg_age_valid = 3 <= days_since_fvg <= 10

        candles_after_fvg = df.loc[fvg_idx:].iloc[1:]
        if len(candles_after_fvg) == 0:
            continue

        moved_above_fvg = candles_after_fvg['High'].max() > fvg_top * 1.02
        above_fvg = current_price > fvg_top

        recent_high = candles_after_fvg['High'].max()
        recent_high_date = candles_after_fvg.loc[candles_after_fvg['High'] == recent_high].index[0]
        recent_high_date_display = recent_high_date.strftime('%b %d, %Y')

        is_retracing = current_price < recent_high * 0.98

        distance_to_fvg_top = current_price - fvg_top
        distance_to_fvg_top_pct = (distance_to_fvg_top / current_price) * 100
        approaching_fvg = distance_to_fvg_top_pct <= 10.0

        # Volume gate: enforced (was only a label in the original)
        volume_confirmed = fvg_candle['Volume_Ratio'] > MIN_VOLUME_RATIO

        if (fvg_age_valid and moved_above_fvg and above_fvg and
                is_retracing and approaching_fvg and volume_confirmed):

            distance_to_50 = abs(current_price - fvg_50_percent)
            distance_to_50_pct = (distance_to_50 / fvg_50_percent) * 100

            entry = fvg_50_percent
            stop_loss = fvg_bottom * 0.995
            target = recent_high

            risk = entry - stop_loss
            reward = target - entry
            risk_reward = reward / risk if risk > 0 else 0

            if risk_reward < 1.5:
                continue

            atr = latest['ATR']
            position_risk_dollars = 100
            shares = int(position_risk_dollars / risk) if risk > 0 else 0

            ema_dist_pct = ((current_price - ema) / ema) * 100

            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
                'Above_EMA50_%': round(ema_dist_pct, 2),
                'FVG_Formed_Date': fvg_date,
                'FVG_Formed_Display': fvg_date_display,
                'Days_Since_FVG': days_since_fvg,
                'FVG_Top': round(fvg_top, 2),
                'FVG_50%': round(fvg_50_percent, 2),
                'FVG_Bottom': round(fvg_bottom, 2),
                'FVG_Height_$': round(fvg_height, 2),
                'FVG_Strength_%': round(fvg_strength, 1),
                'Entry': round(entry, 2),
                'Stop_Loss': round(stop_loss, 2),
                'Target': round(target, 2),
                'Risk_$': round(risk, 2),
                'Reward_$': round(reward, 2),
                'R:R': round(risk_reward, 2),
                'Shares': shares,
                'Distance_to_FVG_Top_%': round(distance_to_fvg_top_pct, 2),
                'Distance_to_Entry_%': round(distance_to_50_pct, 2),
                'Recent_High': round(recent_high, 2),
                'Recent_High_Date': recent_high_date_display,
                'Volume_on_FVG': round(fvg_candle['Volume_Ratio'], 2),
                'ATR': round(atr, 2),
                'FVG_Status': 'UNMITIGATED',
                'Setup_Quality': 'EXCELLENT' if fvg_strength > 60 and distance_to_fvg_top_pct < 3.0 else 'GOOD'
            }

    return None

def run_fvg_scanner():
    print("=" * 100)
    print("NASDAQ 100 FVG SCANNER - Daily Timeframe (Forked: +Trend +Volume gates)")
    print(f"Filters: FVG >={MIN_FVG_STRENGTH_PCT}% ATR | Vol >{MIN_VOLUME_RATIO}x | Price > rising {TREND_EMA}EMA | R:R >=1.5")
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)

    signals = []
    for i, ticker in enumerate(NASDAQ_100, 1):
        print(f"Scanning {i}/{len(NASDAQ_100)}: {ticker}...", end='\r')

        df = fetch_data(ticker, period='1y', interval='1d')
        if df is None:
            continue

        df = calculate_indicators(df)
        df = detect_high_strength_fvgs(df)

        signal = scan_fvg_50_percent_approach(df, ticker)
        if signal:
            signals.append(signal)

    print("\n" + "=" * 100)
    print(f"SCAN COMPLETE - Found {len(signals)} SETUPS")
    print("=" * 100)

    if not signals:
        print("No setups found today.")
        return pd.DataFrame()

    results_df = pd.DataFrame(signals).sort_values('Distance_to_FVG_Top_%')

    summary_cols = [
        'Ticker', 'Current_Price', 'Above_EMA50_%', 'FVG_Formed_Date', 'Days_Since_FVG',
        'FVG_Top', 'FVG_50%', 'FVG_Bottom', 'Entry', 'Stop_Loss', 'Target', 'R:R',
        'Distance_to_FVG_Top_%', 'Volume_on_FVG'
    ]
    print("\nSUMMARY TABLE")
    print("-" * 100)
    print(results_df[summary_cols].to_string(index=False))

    print("\n" + "=" * 100)
    print("DETAILED BREAKDOWN")
    print("=" * 100)
    for _, row in results_df.iterrows():
        print(f"\n{row['Ticker']} - {row['Setup_Quality']} ({row['FVG_Status']})")
        print(f"  Price: ${row['Current_Price']}  |  {row['Above_EMA50_%']:+.2f}% vs 50EMA  |  {row['Distance_to_FVG_Top_%']:.2f}% above FVG top")
        print(f"  FVG:   ${row['FVG_Bottom']} - ${row['FVG_Top']}  (50%: ${row['FVG_50%']})  formed {row['FVG_Formed_Display']} ({row['Days_Since_FVG']}d ago)")
        print(f"  Strength: {row['FVG_Strength_%']}% ATR  |  Vol on FVG: {row['Volume_on_FVG']}x")
        print(f"  PLAN:  Entry ${row['Entry']}  Stop ${row['Stop_Loss']}  Target ${row['Target']}  R:R {row['R:R']}:1  ({row['Shares']} shares / $100 risk)")

    print(f"\nTotal: {len(signals)}  |  Avg R:R: {results_df['R:R'].mean():.2f}:1  |  Avg Strength: {results_df['FVG_Strength_%'].mean():.1f}% ATR")
    return results_df

if __name__ == "__main__":
    results = run_fvg_scanner()
    if len(results) > 0:
        filename = f"fvg_setups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
