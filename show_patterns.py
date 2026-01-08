import json

with open('ai_agent/data/mined_patterns.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

print('='*70)
print('   GOOD PATTERNS (Win Rate > 55%)')
print('='*70)

for cat, patterns in data.items():
    for p in patterns:
        if p['is_good']:
            print(f"   [OK] {p['name']:<30} WR: {p['win_rate']:.1%} ({p['occurrences']} trades)")

print()
print('='*70)
print('   BAD PATTERNS (Win Rate < 40%) - AVOID!')
print('='*70)

for cat, patterns in data.items():
    for p in patterns:
        if p['is_bad']:
            print(f"   [X] {p['name']:<30} WR: {p['win_rate']:.1%} ({p['occurrences']} trades)")

# Summary
good_count = sum(1 for cat in data.values() for p in cat if p['is_good'])
bad_count = sum(1 for cat in data.values() for p in cat if p['is_bad'])
total = sum(len(cat) for cat in data.values())

print()
print('='*70)
print(f"   TOTAL: {total} patterns")
print(f"   GOOD: {good_count} patterns (WR > 55%)")
print(f"   BAD: {bad_count} patterns (WR < 40%)")
print('='*70)
