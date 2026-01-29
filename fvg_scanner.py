
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION & CONSTANTS
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
    'DXCM', 'GFS', 'WBD', 'BIIB', 'ILMN', 'MRNA', 'ALGN', 'SMCI', 'DLTR'
]

# ============================================================================
# DATA FUNCTIONS
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
    df['H-L'] = df['High'] - df['Low']
    df['H-PC'] = abs(df['High'] - df['Close'].shift(1))
    df['L-PC'] = abs(df['Low'] - df['Close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    return df

def detect_high_strength_fvgs(df, min_strength_pct=50):
    """Detect HIGH-STRENGTH Bullish FVGs (>50% of ATR)"""
    bull_fvg_condition = df['Low'] > df['High'].shift(2)

    df['Bull_FVG_Top'] = np.where(bull_fvg_condition, df['Low'], np.nan)
    df['Bull_FVG_Bottom'] = np.where(bull_fvg_condition, df['High'].shift(2), np.nan)
    df['Bull_FVG_Height'] = df['Bull_FVG_Top'] - df['Bull_FVG_Bottom']
    df['Bull_FVG_50%'] = (df['Bull_FVG_Top'] + df['Bull_FVG_Bottom']) / 2
    df['Bull_FVG_Strength_%'] = (df['Bull_FVG_Height'] / df['ATR']) * 100
    df['High_Strength_FVG'] = df['Bull_FVG_Strength_%'] >= min_strength_pct
    return df

def scan_fvg_50_percent_approach(df, ticker):
    """Main Scanner Logic"""
    latest = df.iloc[-1]
    recent = df.iloc[-10:]
    high_strength_fvgs = recent[recent['High_Strength_FVG'] == True]

    if len(high_strength_fvgs) == 0: return None

    fvg_idx = high_strength_fvgs.index[-1]
    fvg_candle = df.loc[fvg_idx]

    fvg_top = fvg_candle['Bull_FVG_Top']
    fvg_bottom = fvg_candle['Bull_FVG_Bottom']
    fvg_50_percent = fvg_candle['Bull_FVG_50%']
    fvg_strength = fvg_candle['Bull_FVG_Strength_%']
    fvg_height = fvg_candle['Bull_FVG_Height']

    current_price = latest['Close']
    days_since_fvg = len(df.loc[fvg_idx:]) - 1

    fvg_age_valid = 3 <= days_since_fvg <= 10
    candles_after_fvg = df.loc[fvg_idx:].iloc[1:]
    never_filled = (candles_after_fvg['Close'] > fvg_bottom).all()
    moved_above_fvg = candles_after_fvg['High'].max() > fvg_top
    in_retracement_zone = (fvg_50_percent <= current_price <= fvg_top * 1.01)
    recent_high = candles_after_fvg['High'].max()
    is_retracing = current_price < recent_high * 0.98
    volume_confirmed = fvg_candle['Volume_Ratio'] > 1.3
    not_at_bottom = current_price > (fvg_bottom + (fvg_height * 0.3))

    if (fvg_age_valid and never_filled and moved_above_fvg and
        in_retracement_zone and is_retracing and not_at_bottom):

        distance_to_50 = abs(current_price - fvg_50_percent)
        distance_to_50_pct = (distance_to_50 / fvg_50_percent) * 100
        entry = fvg_50_percent
        stop_loss = fvg_bottom * 0.995
        target = recent_high
        risk = entry - stop_loss
        reward = target - entry
        risk_reward = reward / risk if risk > 0 else 0

        if risk_reward < 1.5: return None

        atr = latest['ATR']
        position_risk_dollars = 100
        shares = int(position_risk_dollars / risk) if risk > 0 else 0

        return {
            'Ticker': ticker,
            'Current_Price': round(current_price, 2),
            'Entry': round(entry, 2),
            'Stop_Loss': round(stop_loss, 2),
            'Target': round(target, 2),
            'R:R': round(risk_reward, 2),
            'Shares': shares,
            'FVG_Strength_%': round(fvg_strength, 1),
            'Distance_to_50%_pct': round(distance_to_50_pct, 2),
            'Setup_Quality': 'EXCELLENT' if volume_confirmed and fvg_strength > 60 and distance_to_50_pct < 1.0 else 'GOOD'
        }
    return None

# ============================================================================
# ALERTING
# ============================================================================

def send_alert(results_df):
    """Send email alert if setups are found"""
    email_user = os.environ.get('EMAIL_USER')
    email_password = os.environ.get('EMAIL_PASSWORD')
    email_recipient = os.environ.get('EMAIL_RECIPIENT', email_user) # Default to self

    if not email_user or not email_password:
        print("No email credentials found in environment variables. Skipping alert.")
        return

    print("Preparing email alert...")

    msg = MIMEMultipart()
    msg['From'] = email_user
    msg['To'] = email_recipient
    msg['Subject'] = f"FVG Alert: {len(results_df)} Setups Found - {datetime.now().strftime('%Y-%m-%d')}"

    # HTML Body
    html = """
    <html>
      <head>
        <style>
          table { border-collapse: collapse; width: 100%; }
          th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
          th { background-color: #4CAF50; color: white; }
          tr:nth-child(even) { background-color: #f2f2f2; }
        </style>
      </head>
      <body>
        <h2>High Quality FVG Setups Found</h2>
    """

    html += results_df.to_html(index=False)
    html += "</body></html>"

    msg.attach(MIMEText(html, 'html'))

    try:
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(email_user, email_password)
        server.send_message(msg)
        server.quit()
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"Failed to send email: {e}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"Starting Scan: {datetime.now()}")
    signals = []
    for ticker in NASDAQ_100:
        try:
            df = fetch_data(ticker)
            if df is not None:
                df = calculate_indicators(df)
                df = detect_high_strength_fvgs(df)
                signal = scan_fvg_50_percent_approach(df, ticker)
                if signal:
                    signals.append(signal)
        except Exception as e:
            continue

    if signals:
        results_df = pd.DataFrame(signals)
        results_df = results_df.sort_values('Distance_to_50%_pct')
        print(f"Found {len(signals)} setups. Sending alert.")

        # Print to console for logs
        print(results_df[['Ticker', 'Entry', 'Stop_Loss', 'Target', 'R:R']].to_string(index=False))

        # Send Email
        send_alert(results_df)
    else:
        print("No setups found. Staying silent.")

if __name__ == "__main__":
    main()
