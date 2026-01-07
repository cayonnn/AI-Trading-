import sqlite3

# Test MasterBrain save
print("Testing MasterBrain save...")
from ai_agent.master_brain import MasterBrain
brain = MasterBrain()
brain._save_model()
print("✅ MasterBrain saved!")

# Check trade_memory.db
try:
    c1 = sqlite3.connect('trade_memory.db')
    tables1 = c1.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("\ntrade_memory.db tables:", tables1)
    
    for table in tables1:
        count = c1.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"  - {table[0]}: {count} rows")
    c1.close()
except Exception as e:
    print(f"trade_memory.db error: {e}")

# Check trading_data.db
try:
    c2 = sqlite3.connect('trading_data.db')
    tables2 = c2.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    print("\ntrading_data.db tables:", tables2)
    
    for table in tables2:
        count = c2.execute(f"SELECT COUNT(*) FROM {table[0]}").fetchone()[0]
        print(f"  - {table[0]}: {count} rows")
    c2.close()
except Exception as e:
    print(f"trading_data.db error: {e}")
