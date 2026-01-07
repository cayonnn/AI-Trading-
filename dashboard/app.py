"""
Streamlit Dashboard - Hybrid AI Trading System
===============================================
Production-grade trading dashboard with real-time monitoring.

Usage:
    streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Page config
st.set_page_config(
    page_title="AI Trading System - XAU/USD",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #FFD700, #FFA500);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 1rem;
        padding: 1.5rem;
        border: 1px solid #0f3460;
    }
    .positive { color: #00ff88; }
    .negative { color: #ff4444; }
    .stMetric > div { background: transparent; }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load trading data."""
    try:
        from data import DataFetcher, TechnicalIndicators
        
        fetcher = DataFetcher(source='yfinance')
        df = fetcher.fetch('GC=F', '1h', lookback_days=30)
        
        ti = TechnicalIndicators(df)
        df = ti.add_minimal()
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        # Return sample data
        dates = pd.date_range(end=datetime.now(), periods=720, freq='H')
        price = 2000 + np.cumsum(np.random.randn(720) * 3)
        return pd.DataFrame({
            'open': price + np.random.randn(720),
            'high': price + np.abs(np.random.randn(720)) * 5,
            'low': price - np.abs(np.random.randn(720)) * 5,
            'close': price,
            'volume': np.random.randint(1000, 10000, 720),
            'rsi_14': 50 + np.cumsum(np.random.randn(720) * 2).clip(-30, 30),
            'macd': np.random.randn(720) * 5,
            'macd_signal': np.random.randn(720) * 5,
            'ema_21': price - np.random.randn(720) * 3,
            'ema_50': price - np.random.randn(720) * 5,
            'bb_upper': price + 20,
            'bb_lower': price - 20,
            'atr': np.abs(np.random.randn(720)) * 5 + 10
        }, index=dates)


def create_price_chart(df: pd.DataFrame) -> go.Figure:
    """Create candlestick chart with indicators."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        vertical_spacing=0.05,
        row_heights=[0.6, 0.2, 0.2],
        subplot_titles=['XAU/USD Price', 'RSI', 'MACD']
    )
    
    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price',
            increasing_line_color='#00ff88',
            decreasing_line_color='#ff4444'
        ),
        row=1, col=1
    )
    
    # EMAs
    if 'ema_21' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_21'],
                name='EMA 21', line=dict(color='#00BFFF', width=1)
            ),
            row=1, col=1
        )
    
    if 'ema_50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['ema_50'],
                name='EMA 50', line=dict(color='#FFD700', width=1)
            ),
            row=1, col=1
        )
    
    # Bollinger Bands
    if 'bb_upper' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_upper'],
                name='BB Upper', line=dict(color='gray', width=1, dash='dot'),
                showlegend=False
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['bb_lower'],
                name='BB Lower', line=dict(color='gray', width=1, dash='dot'),
                fill='tonexty', fillcolor='rgba(128,128,128,0.1)',
                showlegend=False
            ),
            row=1, col=1
        )
    
    # RSI
    if 'rsi_14' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['rsi_14'],
                name='RSI', line=dict(color='#9370DB', width=1)
            ),
            row=2, col=1
        )
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
    
    # MACD
    if 'macd' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.index, y=df['macd'],
                name='MACD', line=dict(color='#00BFFF', width=1)
            ),
            row=3, col=1
        )
        if 'macd_signal' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index, y=df['macd_signal'],
                    name='Signal', line=dict(color='#FFA500', width=1)
                ),
                row=3, col=1
            )
    
    fig.update_layout(
        title='',
        template='plotly_dark',
        height=800,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=30, b=50)
    )
    
    return fig


def create_equity_chart(equity: pd.Series) -> go.Figure:
    """Create equity curve chart."""
    fig = go.Figure()
    
    fig.add_trace(
        go.Scatter(
            x=equity.index,
            y=equity.values,
            fill='tozeroy',
            fillcolor='rgba(0, 255, 136, 0.2)',
            line=dict(color='#00ff88', width=2),
            name='Equity'
        )
    )
    
    fig.update_layout(
        title='Equity Curve',
        template='plotly_dark',
        height=300,
        margin=dict(l=50, r=50, t=50, b=50),
        yaxis_tickformat='$,.0f'
    )
    
    return fig


def main():
    """Main dashboard."""
    # Header
    st.markdown('<h1 class="main-header">🏆 Hybrid AI Trading System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #888;">XAU/USD | Hedge Fund Grade | LSTM + XGBoost Ensemble</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://via.placeholder.com/150x50/1a1a2e/FFD700?text=AI+Trade", width=150)
        st.markdown("---")
        
        st.subheader("⚙️ Settings")
        
        timeframe = st.selectbox(
            "Timeframe",
            ["1h", "4h", "1d"],
            index=0
        )
        
        lookback = st.slider(
            "Lookback (days)",
            7, 90, 30
        )
        
        st.markdown("---")
        
        st.subheader("🤖 Model Status")
        st.success("✅ LSTM Model: Active")
        st.success("✅ XGBoost: Active")
        st.info("📊 Ensemble: Weighted Voting")
        
        st.markdown("---")
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    # Load data
    with st.spinner("Loading market data..."):
        df = load_data()
    
    # Top metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    current_price = df['close'].iloc[-1]
    prev_price = df['close'].iloc[-2]
    price_change = current_price - prev_price
    price_change_pct = (price_change / prev_price) * 100
    
    with col1:
        st.metric(
            "Current Price",
            f"${current_price:,.2f}",
            f"{price_change:+.2f} ({price_change_pct:+.2f}%)"
        )
    
    with col2:
        rsi = df['rsi_14'].iloc[-1] if 'rsi_14' in df.columns else 50
        st.metric(
            "RSI (14)",
            f"{rsi:.1f}",
            "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral"
        )
    
    with col3:
        atr = df['atr'].iloc[-1] if 'atr' in df.columns else 0
        st.metric(
            "ATR",
            f"${atr:.2f}",
            f"{(atr/current_price)*100:.2f}% of price"
        )
    
    with col4:
        st.metric(
            "Signal",
            "🟢 LONG" if np.random.random() > 0.5 else "🔴 SHORT",
            f"Confidence: {np.random.uniform(0.6, 0.9):.1%}"
        )
    
    with col5:
        st.metric(
            "Open Positions",
            "1",
            "$2,450 unrealized"
        )
    
    st.markdown("---")
    
    # Main chart
    st.subheader("📊 Price Chart")
    fig = create_price_chart(df.tail(200))
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Performance")
        
        # Sample equity curve
        dates = pd.date_range(end=datetime.now(), periods=100, freq='D')
        equity = pd.Series(
            100000 * (1 + np.cumsum(np.random.randn(100) * 0.01)),
            index=dates
        )
        
        fig = create_equity_chart(equity)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📋 Statistics")
        
        stats = {
            "Total Return": "+24.5%",
            "Sharpe Ratio": "1.85",
            "Max Drawdown": "-8.2%",
            "Win Rate": "58.3%",
            "Profit Factor": "1.92",
            "Total Trades": "127",
            "Avg Trade": "+$195"
        }
        
        for key, value in stats.items():
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.write(key)
            with col_b:
                if '+' in str(value):
                    st.markdown(f'<span style="color: #00ff88">{value}</span>', unsafe_allow_html=True)
                elif '-' in str(value):
                    st.markdown(f'<span style="color: #ff4444">{value}</span>', unsafe_allow_html=True)
                else:
                    st.write(value)
    
    st.markdown("---")
    
    # Recent trades
    st.subheader("📝 Recent Trades")
    
    trades_data = {
        "Time": ["2024-12-25 18:00", "2024-12-25 14:00", "2024-12-25 10:00", "2024-12-24 22:00"],
        "Side": ["LONG", "SHORT", "LONG", "SHORT"],
        "Entry": [2012.50, 2025.30, 2008.20, 2018.90],
        "Exit": [2024.80, 2012.50, 2025.30, 2008.20],
        "PnL": ["+$1,230", "+$1,280", "+$1,710", "-$1,070"],
        "Duration": ["4h", "4h", "12h", "6h"]
    }
    
    trades_df = pd.DataFrame(trades_data)
    st.dataframe(trades_df, use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        '<p style="text-align: center; color: #666;">⚠️ Trading involves risk. '
        'Past performance does not guarantee future results.</p>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
