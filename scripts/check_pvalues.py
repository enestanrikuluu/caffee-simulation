import re
from pathlib import Path

project_root = Path(__file__).parent.parent

with open(project_root / 'results' / 'distribution_fitting_report.txt', 'r') as f:
    content = f.read()

# Find all K-S p-values
ks_pvalues = []
sections = content.split('VARIABLE:')

for section in sections[1:]:  # Skip first empty section
    lines = section.split('\n')
    var_name = lines[0].strip()
    
    for line in lines:
        if 'K-S p-val' in line:
            # Get the table that follows
            idx = lines.index(line)
            for data_line in lines[idx+2:idx+8]:  # Check next few lines
                if data_line.strip() and not data_line.startswith('-'):
                    parts = data_line.split()
                    if len(parts) >= 5:
                        try:
                            pval = float(parts[4])
                            ks_pvalues.append((var_name, parts[1], pval))
                            break
                        except:
                            pass
            break

# Count passes
total = len(ks_pvalues)
passes = sum(1 for _, _, p in ks_pvalues if p > 0.05)

print(f'K-S Test Results: {passes}/{total} passed (p > 0.05)')
print(f'Pass rate: {100*passes/total:.1f}%\n')

print('P-values by variable (best fit only):')
for var, dist, pval in sorted(ks_pvalues, key=lambda x: x[2], reverse=True):
    status = 'PASS' if pval > 0.05 else 'FAIL'
    print(f'  {pval:.4f} [{status}] - {var} ({dist})')

print(f'\n\nP-value range: {min(p for _,_,p in ks_pvalues):.4f} to {max(p for _,_,p in ks_pvalues):.4f}')
print(f'Middle range (0.05-0.50): {sum(1 for _,_,p in ks_pvalues if 0.05 < p < 0.50)} out of {total}')
