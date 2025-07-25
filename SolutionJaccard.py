"""
sensitivity_similarity.py

Perform sensitivity analysis on similarity of knapsack solutions as budget increases.
Computes Jaccard similarity of selected park sets between successive budgets for each subindex,
and averages across indices. Saves results to CSV and optionally plots the trend.
"""
import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# --- Working directory ---
os.chdir(r'C:/Users/gabeg/OneDrive/Documents/H-CORE REU/Models/V2')

# --- Load data ---
xls = pd.ExcelFile('datasheet1.xlsx')

# Park IDs and parameters
sets_df = xls.parse('Sets')
K = sets_df['ParkID'].dropna().astype(str).tolist()
k_df = xls.parse('K')
e_k, v_k, c = {}, {}, {}
for _, r in k_df.iterrows():
    pid = str(r['ParkID'])
    e_k[pid] = int(r.get('existing', 0))
    v_k[pid] = float(r.get('vkl', 0.0))
    price = float(r['price'])
    c[pid] = 0.0 if e_k[pid] else price

# Population by geography
l_df = xls.parse('L')
def to_str_geoid(x):
    try:
        return str(int(x))
    except:
        return str(x)
l_df['GEOID20'] = l_df['GEOID20'].apply(to_str_geoid)
n = {r['GEOID20']: float(r['total_pop']) for _, r in l_df.iterrows()}

# Load SVI disaggregated normalized
svi_xls = pd.ExcelFile('disaggregated_by_index_normalized.xlsx')
svi_df = svi_xls.parse('SVI_Disaggregated_Normalized')
svi_df['GEOID20'] = svi_df['GEOID20'].apply(to_str_geoid)
I = [col for col in svi_df.columns if col != 'GEOID20']
L = svi_df['GEOID20'].tolist()

# Primary-park indicator
skl = {(pid, loc): 0 for pid in K for loc in L}
primary_col = next((col for col in l_df.columns if 'primary' in col.lower()), None)
if primary_col:
    for _, r in l_df.iterrows():
        loc = r['GEOID20']
        p = str(r.get(primary_col, '')).strip()
        if loc in L and p in K:
            skl[(p, loc)] = 1

# Utility q
q = {}
for _, r in svi_df.iterrows():
    loc = r['GEOID20']
    for ip in I:
        q[(ip, loc)] = float(r[ip])

# Budget settings
elem_cands = [pid for pid in K if any(skl[(pid, loc)] for loc in L)]
total_cost = sum(c[pid] for pid in elem_cands)
max_budget = int(total_cost)
# Define budgets to test (exclude 0)
num_steps = 1000
all_budgets = np.linspace(0, max_budget, num_steps, dtype=int)
budgets = all_budgets[1:]  # drop 0 budget

num_steps = 100
budgets = np.linspace(0, max_budget, num_steps, dtype=int)

# Precompute p_hat for all (i,pid)
p_hat = {(ip, pid): sum(q.get((ip, loc), 0.0) * n.get(loc, 0.0) * (1 - v_k.get(pid,0.0))
            for loc in L if skl.get((pid, loc),0) == 1)
         for ip in I for pid in K}

# Solve knapsack for each index at each budget and record selections
solutions = {}
for b in budgets:
    sol_i = {}
    for ip in I:
        model = gp.Model()
        y = model.addVars(K, vtype=GRB.BINARY, name='y')
        for pid in K:
            if e_k[pid]: model.addConstr(y[pid] == 1)
        model.addConstr(gp.quicksum(c[pid]*y[pid] for pid in K) <= b)
        model.setObjective(gp.quicksum(p_hat[(ip, pid)]*y[pid] for pid in K), GRB.MAXIMIZE)
        model.Params.OutputFlag = 0
        model.optimize()
        # record selected non-existing parks
        sel = {pid for pid in K if not e_k[pid] and y[pid].X > 0.5}
        sol_i[ip] = sel
    solutions[b] = sol_i

# Compute similarity (Jaccard) between successive budgets
records = []
prev = None
for b in budgets:
    curr = solutions[b]
    if prev is None:
        sim = {ip: 1.0 for ip in I}
    else:
        sim = {}
        for ip in I:
            A = prev[ip]
            B = curr[ip]
            union = A.union(B)
            sim[ip] = len(A.intersection(B)) / len(union) if union else np.nan
    avg_sim = np.nanmean(list(sim.values()))
    rec = {'budget': b, 'avg_similarity': avg_sim}
    rec.update(sim)
    records.append(rec)
    prev = curr

# Save results
df_sim = pd.DataFrame(records)
df_sim.to_csv('SolutionSimilaritySensitivity.csv', index=False)
print('Similarity sensitivity saved to SolutionSimilaritySensitivity.csv')

# Plot average similarity trend
plt.figure(figsize=(12,6))  # wider figure
plt.plot(df_sim['budget'], df_sim['avg_similarity'], marker='o', linestyle='-')
plt.xlabel('Budget')
plt.ylabel('Avg. Jaccard Similarity to Prev Budget')
plt.title('Solution Similarity Across Budgets')
plt.grid(True)
plt.show()
