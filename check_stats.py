"""Quick stats check"""
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from ai_agent.trade_memory import TradeMemory

m = TradeMemory()
s = m.get_performance_stats()

print("\n" + "="*50)
print("BACKTEST RESULTS")
print("="*50)
print(f"Total Trades: {s.get('total_trades', 0)}")
print(f"Win Rate: {s.get('win_rate', 0):.1%}")
print(f"Avg Win: ${s.get('average_win', 0):.2f}")
print(f"Avg Loss: ${s.get('average_loss', 0):.2f}")
print(f"Total P&L: ${s.get('total_pnl', 0):.2f}")
print(f"Avg R Multiple: {s.get('average_r_multiple', 0):.2f}")
print(f"Best R: {s.get('best_r', 0):.2f}")
print(f"Worst R: {s.get('worst_r', 0):.2f}")

# Profit Factor
avg_win = s.get('average_win', 0)
avg_loss = abs(s.get('average_loss', 1))
win_rate = s.get('win_rate', 0)
if avg_loss > 0:
    profit_factor = avg_win / avg_loss
    expectancy = (win_rate * avg_win) + ((1-win_rate) * s.get('average_loss', 0))
    print(f"\nProfit Factor: {profit_factor:.2f}")
    print(f"Expectancy per Trade: ${expectancy:.2f}")

print("\n" + "="*50)
print("Strategy Performance:")
print("="*50)
df = m.get_strategy_performance()
if not df.empty:
    for _, row in df.iterrows():
        print(f"  {row['strategy']}: {row['trades']} trades, {row['win_rate']:.0%} win, ${row['total_pnl']:.2f}")
