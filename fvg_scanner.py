# NASDAQ 100 FVG SCANNER - Optimized Version
# Scans for UNMITIGATED FVGs with realistic distance criteria
# Daily Timeframe - Created for Sandip

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# NASDAQ 100 STOCK LIST
# ============================================================================

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
    'DXCM', 'GFS', 'WBD', 'BIIB', 'ILMN',  'MRNA', 'ALGN', 'SMCI', 'DLTR'
]

# ============================================================================
# FUNCTIONS
# ============================================================================

def fetch_data(ticker, period='6mo', interval='1d'):
    """Fetch historical DAILY data"""
    try:
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        if df.empty or len(df) < 50:
            return None
        return df
    except:
        return None

def calculate_indicators(df):
    """Calculate ATR and basic indicators"""
    
    # ATR for FVG strength measurement
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()
    
    # Volume
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df

def detect_high_strength_fvgs(df, min_strength_pct=50):
    """Detect HIGH-STRENGTH Bullish FVGs (>50% of ATR)"""
    
    # Bullish FVG: Current candle's Low > High from 2 candles ago
    bull_fvg_condition = df['Low'] > df['High'].shift(2)
    
    df['Bull_FVG_Top'] = np.where(bull_fvg_condition, df['Low'], np.nan)
    df['Bull_FVG_Bottom'] = np.where(bull_fvg_condition, df['High'].shift(2), np.nan)
    df['Bull_FVG_Height'] = df['Bull_FVG_Top'] - df['Bull_FVG_Bottom']
    df['Bull_FVG_50%'] = (df['Bull_FVG_Top'] + df['Bull_FVG_Bottom']) / 2
    
    # Calculate FVG strength as % of ATR
    df['Bull_FVG_Strength_%'] = (df['Bull_FVG_Height'] / df['ATR']) * 100
    
    # Mark HIGH-STRENGTH FVGs only
    df['High_Strength_FVG'] = df['Bull_FVG_Strength_%'] >= min_strength_pct
    
    return df

def is_fvg_unmitigated(df, fvg_idx, fvg_bottom, fvg_top):
    """Check if FVG is UNMITIGATED"""
    
    candles_after_fvg = df.loc[fvg_idx:].iloc[1:]
    
    if len(candles_after_fvg) == 0:
        return True
    
    # Check if any candle's LOW touched or entered the FVG zone
    touched_fvg = (candles_after_fvg['Low'] <= fvg_top).any()
    
    return not touched_fvg

def scan_fvg_50_percent_approach(df, ticker):
    """
    OPTIMIZED UNMITIGATED FVG setup scanner
    """
    
    latest = df.iloc[-1]
    recent = df.iloc[-10:]
    
    # Find HIGH-STRENGTH FVGs in recent history
    high_strength_fvgs = recent[recent['High_Strength_FVG'] == True]
    
    if len(high_strength_fvgs) == 0:
        return None
    
    # Check each FVG (most recent first)
    for fvg_idx in reversed(high_strength_fvgs.index):
        fvg_candle = df.loc[fvg_idx]
        
        fvg_top = fvg_candle['Bull_FVG_Top']
        fvg_bottom = fvg_candle['Bull_FVG_Bottom']
        fvg_50_percent = fvg_candle['Bull_FVG_50%']
        fvg_strength = fvg_candle['Bull_FVG_Strength_%']
        fvg_height = fvg_candle['Bull_FVG_Height']
        
        # FVG formation date
        fvg_date = fvg_idx.strftime('%Y-%m-%d')
        fvg_date_display = fvg_idx.strftime('%b %d, %Y')
        
        current_price = latest['Close']
        days_since_fvg = len(df.loc[fvg_idx:]) - 1
        
        # CRITICAL CHECK: Is FVG UNMITIGATED?
        is_unmitigated = is_fvg_unmitigated(df, fvg_idx, fvg_bottom, fvg_top)
        
        if not is_unmitigated:
            continue
        
        # OPTIMIZED FILTERS
        
        # 1. FVG must be recent (3-10 days old)
        fvg_age_valid = 3 <= days_since_fvg <= 10
        
        # 2. Price moved above FVG after formation
        candles_after_fvg = df.loc[fvg_idx:].iloc[1:]
        if len(candles_after_fvg) == 0:
            continue
        
        moved_above_fvg = candles_after_fvg['High'].max() > fvg_top * 1.02
        
        # 3. Price NOW above FVG
        above_fvg = current_price > fvg_top
        
        # 4. Price is retracing (came down from recent high)
        recent_high = candles_after_fvg['High'].max()
        recent_high_date = candles_after_fvg.loc[candles_after_fvg['High'] == recent_high].index[0]
        recent_high_date_display = recent_high_date.strftime('%b %d, %Y')
        
        is_retracing = current_price < recent_high * 0.98
        
        # 5. RELAXED: Price approaching FVG (within 10% instead of 3%)
        distance_to_fvg_top = current_price - fvg_top
        distance_to_fvg_top_pct = (distance_to_fvg_top / current_price) * 100
        approaching_fvg = distance_to_fvg_top_pct <= 10.0  # CHANGED FROM 3.0 to 10.0
        
        # 6. Volume confirmation
        volume_confirmed = fvg_candle['Volume_Ratio'] > 1.3
        
        # CHECK ALL CONDITIONS
        if (fvg_age_valid and moved_above_fvg and above_fvg and 
            is_retracing and approaching_fvg):
            
            # Calculate distance to 50% level
            distance_to_50 = abs(current_price - fvg_50_percent)
            distance_to_50_pct = (distance_to_50 / fvg_50_percent) * 100
            
            # TRADE SETUP
            entry = fvg_50_percent
            stop_loss = fvg_bottom * 0.995
            target = recent_high
            
            risk = entry - stop_loss
            reward = target - entry
            risk_reward = reward / risk if risk > 0 else 0
            
            # Only return setups with decent R:R
            if risk_reward < 1.5:
                continue
            
            # Position sizing
            atr = latest['ATR']
            position_risk_dollars = 100
            shares = int(position_risk_dollars / risk) if risk > 0 else 0
            
            return {
                'Ticker': ticker,
                'Current_Price': round(current_price, 2),
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
                'Setup_Quality': 'EXCELLENT' if volume_confirmed and fvg_strength > 60 and distance_to_fvg_top_pct < 3.0 else 'GOOD'
            }
    
    return None

# ============================================================================
# MAIN SCANNER
# ============================================================================

def run_fvg_scanner():
    """FVG Scanner - UNMITIGATED FVGs with Optimized Filters"""
    
    print("=" * 100)
    print("NASDAQ 100 FVG SCANNER - Daily Timeframe")
    print("Scanning for UNMITIGATED HIGH-STRENGTH FVGs (Optimized Filters)")
    print(f"Scan Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 100)
    print()
    
    signals = []
    
    for i, ticker in enumerate(NASDAQ_100, 1):
        print(f"Scanning {i}/{len(NASDAQ_100)}: {ticker}...", end='\r')
        
        df = fetch_data(ticker, period='6mo', interval='1d')
        if df is None:
            continue
        
        df = calculate_indicators(df)
        df = detect_high_strength_fvgs(df, min_strength_pct=50)
        
        signal = scan_fvg_50_percent_approach(df, ticker)
        if signal:
            signals.append(signal)
    
    print("\n" + "=" * 100)
    print(f"SCAN COMPLETE - Found {len(signals)} UNMITIGATED FVG SETUPS")
    print("=" * 100)
    print()
    
    if len(signals) > 0:
        results_df = pd.DataFrame(signals)
        results_df = results_df.sort_values('Distance_to_FVG_Top_%', ascending=True)
        
        # Display results
        print("\n" + "=" * 100)
        print("UNMITIGATED FVG SETUPS - SUMMARY TABLE")
        print("=" * 100)
        print()
        
        summary_cols = [
            'Ticker', 'Current_Price', 'FVG_Formed_Date', 'Days_Since_FVG',
            'FVG_Top', 'FVG_50%', 'FVG_Bottom', 'Entry', 'Stop_Loss', 'Target', 'R:R', 'Distance_to_FVG_Top_%'
        ]
        print(results_df[summary_cols].to_string(index=False))
        
        # Detailed breakdown
        print("\n" + "=" * 100)
        print("DETAILED BREAKDOWN")
        print("=" * 100)
        
        for idx, row in results_df.iterrows():
            print(f"\n{'='*100}")
            print(f"TICKER: {row['Ticker']} - {row['Setup_Quality']} SETUP ({row['FVG_Status']})")
            print(f"{'='*100}")
            print(f"CURRENT PRICE: ${row['Current_Price']}")
            print(f"DISTANCE TO FVG TOP: {row['Distance_to_FVG_Top_%']:.2f}% (Price retracing from above)")
            print()
            
            print(f"FVG FORMATION DETAILS:")
            print(f"  Date Formed:  {row['FVG_Formed_Display']} ({row['Days_Since_FVG']} days ago)")
            print(f"  FVG Top:      ${row['FVG_Top']:.2f}  <- Price never touched this level")
            print(f"  FVG 50%:      ${row['FVG_50%']:.2f}  <- ENTRY LEVEL (Consequent Encroachment)")
            print(f"  FVG Bottom:   ${row['FVG_Bottom']:.2f}")
            print(f"  FVG Height:   ${row['FVG_Height_$']:.2f}")
            print(f"  Strength:     {row['FVG_Strength_%']:.1f}% of ATR (HIGH-STRENGTH)")
            print(f"  Status:       {row['FVG_Status']}")
            print()
            
            print(f"TRADE PLAN:")
            print(f"  Entry:        ${row['Entry']:.2f}  (50% FVG level)")
            print(f"  Stop Loss:    ${row['Stop_Loss']:.2f}  (Below FVG)")
            print(f"  Target:       ${row['Target']:.2f}  (Recent high)")
            print(f"  Risk:         ${row['Risk_$']:.2f} per share")
            print(f"  Reward:       ${row['Reward_$']:.2f} per share")
            print(f"  R:R Ratio:    {row['R:R']:.2f}:1")
            print(f"  Position:     {row['Shares']} shares ($100 risk)")
        
        # Summary
        print("\n" + "=" * 100)
        print("SUMMARY")
        print("=" * 100)
        print(f"Total Setups: {len(signals)}")
        print(f"Average R:R: {results_df['R:R'].mean():.2f}:1")
        print(f"Average FVG Strength: {results_df['FVG_Strength_%'].mean():.1f}% of ATR")
        
        return results_df
    else:
        print("No setups found today.")
        return pd.DataFrame()

# ============================================================================
# EXECUTE
# ============================================================================

if __name__ == "__main__":
    results = run_fvg_scanner()
    
    if len(results) > 0:
        filename = f"fvg_setups_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        results.to_csv(filename, index=False)
        print(f"\nResults saved to: {filename}")
