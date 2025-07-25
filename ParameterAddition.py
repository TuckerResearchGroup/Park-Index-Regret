


import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

# --- Helper to normalize GEOID20 to str ---
def to_str_geoid(x):
    try:
        return str(int(x))
    except:
        return str(x)

# --- Settings ---
STEPS_BUDGET = 3000
OUTPUT_CSV = 'Sensitivity_Q_Columns.csv'

# --- Working directory ---
os.chdir(r'C:/Users/gabeg/OneDrive/Documents/H-CORE REU/Models/V2')


# --- Load data ---
base = pd.ExcelFile('datasheet1.xlsx')
svi  = pd.ExcelFile('disaggregated_by_index_normalized.xlsx')

# Parks and costs
sets = base.parse('Sets')
K = sets['ParkID'].dropna().astype(str).tolist()
kdf = base.parse('K')
e_k, v_k, cost = {}, {}, {}
for _, r in kdf.iterrows():
    pid = str(r['ParkID'])
    e_k[pid] = int(r.get('existing',0))
    v_k[pid] = float(r.get('vkl',0.0))
    cost[pid] = 0.0 if e_k[pid] else float(r['price'])

# Population
ldf = base.parse('L')
ldf['GEOID20'] = ldf['GEOID20'].apply(to_str_geoid)
n = {r['GEOID20']: float(r['total_pop']) for _, r in ldf.iterrows()}

# Primary location indicator
skl = {}
ldf = ldf.copy()
primary_col = next((c for c in ldf.columns if 'primary' in c.lower()), None)
L_geo = []
if primary_col:
    for _, r in ldf.iterrows():
        loc = r['GEOID20']
        p = str(r[primary_col]).strip()
        L_geo.append(loc)
        skl.setdefault((p, loc), 1)

# SVI columns
svi_df = svi.parse('SVI_Disaggregated_Normalized')
svi_df['GEOID20'] = svi_df['GEOID20'].apply(to_str_geoid)
Q_cols = [c for c in svi_df.columns if c != 'GEOID20']
L = svi_df['GEOID20'].tolist()

# Budgets
total_cost = sum(cost[pid] for pid in K if any(skl.get((pid,l),0) for l in L))
budgets = np.linspace(0, total_cost, STEPS_BUDGET+1, dtype=int)[1:]

# Results
records = []

for k in range(1, len(Q_cols)+1):
    cols = Q_cols[:k]
    # build cumulative Q
    q_vals = {}
    for _, r in svi_df.iterrows():
        loc = r['GEOID20']
        q_vals[loc] = sum(r[col] for col in cols)
    # normalize by total
    s = sum(q_vals.values())
    if s > 0:
        for loc in q_vals:
            q_vals[loc] /= s
    # compute p_hat for each park
    p_hat = {}
    for pid in K:
        p_hat[pid] = sum(q_vals.get(loc,0)*n.get(loc,0)*(1-v_k.get(pid,0))
                         for loc in L if skl.get((pid,loc),0)==1)
    # for each budget solve knapsack
    for b in budgets:
        m = gp.Model(f'knap_k{k}_b{b}')
        y = m.addVars(K, vtype=GRB.BINARY)
        for pid in K:
            if e_k[pid]: m.addConstr(y[pid]==1)
        m.addConstr(gp.quicksum(cost[pid]*y[pid] for pid in K) <= b)
        m.setObjective(gp.quicksum(p_hat[pid]*y[pid] for pid in K), GRB.MAXIMIZE)
        m.Params.OutputFlag = 0; m.optimize()
        for pid in K:
            records.append({
                'k': k,
                'budget': b,
                'park': pid,
                'p_hat': p_hat[pid],
                'chosen': int(y[pid].X > 0.5)
            })

# Save
df_out = pd.DataFrame(records)
df_out.to_csv(OUTPUT_CSV, index=False)
print(f'Saved {OUTPUT_CSV}')
