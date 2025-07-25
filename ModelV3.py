
"""
Created on Wed Jul 2 2025

@author: Gabriel Griffin
"""


# regret_sensitivity_model.py
# Created on Wed Jul 2 2025
# Author: Gabe Griffin

import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

os.chdir('C:/Users/gabeg/OneDrive/Documents/H-CORE REU/Models/V2')

file_path = 'datasheet1.xlsx'
xls = pd.ExcelFile(file_path)

# --- Sets and Indices ---
sets_df = xls.parse('Sets')
I = sets_df['I'].dropna().astype(str).tolist()
K = sets_df['ParkID'].dropna().astype(str).tolist()

# --- Resident data (L sheet) ---
l_df = xls.parse('L')
L = l_df['GEOID20'].dropna().astype(str).tolist()
n = {str(r['GEOID20']): float(r['total_pop']) for _, r in l_df.iterrows()}

# Build q[(index, location)] from *_rank columns
q = {}
for col in l_df.columns:
    if col.lower().endswith('_rank'):
        ip = col[:-5]
        for _, r in l_df.iterrows():
            q[(ip, str(r['GEOID20']))] = float(r[col])

# --- Park data (K sheet) ---
k_df = xls.parse('K')
e_k, v_k, c = {}, {}, {}
price_col = 'price'
for _, r in k_df.iterrows():
    pid = str(r['ParkID'])
    e_k[pid] = int(r.get('existing', 0))
    v_k[pid] = float(r.get('vkl', 0.0))
    price = float(r[price_col])
    c[pid] = 0.0 if e_k[pid] else price

# --- Primary-park indicator s_{k,ℓ} from L sheet ---
primary_cols = [col for col in l_df.columns if 'primary' in col.lower()]
primary_col = primary_cols[0]
skl = {(k, loc): 0 for k in K for loc in L}
for _, r in l_df.iterrows():
    loc = str(r['GEOID20'])
    p = str(r.get(primary_col, '')).strip()
    if p in K:
        skl[(p, loc)] = 1

# --- Base-case Budget: 25% of total pos-candidate cost ---
positive_cands = [pid for pid in K if any(skl[(pid, loc)] for loc in L)]
total_pos_cost = sum(c[pid] for pid in positive_cands)
b = total_pos_cost * 0.25

# --- Compute p_hat[i,k] for Stage 1 ---
p_hat = {}
for ip in I:
    for pid in K:
        p_hat[(ip, pid)] = sum(
            q.get((ip, loc), 0) * n.get(loc, 0) * (1 - v_k[pid])
            for loc in L if skl[(pid, loc)] == 1
        )

# --- Stage 1: one knapsack per index for base-case ---
pre_y = {(pid, i): 0 for pid in K for i in I}
for i in I:
    m1 = gp.Model(f'knap_{i}')
    y_i = m1.addVars(K, vtype=GRB.BINARY, name='y')
    for pid in K:
        if e_k[pid]:
            m1.addConstr(y_i[pid] == 1)
    m1.addConstr(gp.quicksum(c[pid] * y_i[pid] for pid in K) <= b)
    m1.setObjective(
        gp.quicksum(p_hat[(i, pid)] * y_i[pid] for pid in K),
        GRB.MAXIMIZE
    )
    m1.Params.OutputFlag = 0
    m1.optimize()
    for pid in K:
        pre_y[(pid, i)] = int(y_i[pid].X > 0.5)

# --- Compute p_pre[i',i] for Stage 2 regret model (base-case reference) ---
p_pre = {}
for ip in I:
    for i in I:
        p_pre[(ip, i)] = sum(pre_y[(pid, i)] * p_hat[(ip, pid)] for pid in K)

# --- Stage 2: minimize max regret for base-case ---
m2 = gp.Model('regret_sum')
R = m2.addVars(I, lb=0.0, name='R')
T = m2.addVar(lb=0.0, name='T')
m2.setObjective(T, GRB.MINIMIZE)
for i in I:
    m2.addConstr(T >= R[i])
for i in I:
    for ip in I:
        diff = p_pre[(ip, ip)] - p_pre[(ip, i)]
        m2.addConstr(R[i] >= diff, name=f'Regret_{i}_{ip}')
m2.optimize()

# Report base-case results
print(f'=== Base-case (budget = {b:.0f}) ===')
for i in I:
    cost_i = sum(c[pid] * pre_y[(pid, i)] for pid in K)
    print(f'Index {i}: cost = {cost_i:.2f}, regret = {R[i].X:.2f}')
best = min(I, key=lambda i: R[i].X)
print(f'Chosen index = {best}\n')
print('Decision vectors (candidate parks only) – Base-case:')
for i in I:
    sel = [pid for pid in K if (not e_k[pid] and pre_y[(pid, i)] == 1)]
    print(f'  {i} (n={len(sel)}): {sel}')

# --- Sensitivity Analysis: budgets from 0 up to max_budget ---
max_budget = 8000000
fractions = np.linspace(0, max_budget, 5000)
results = []

for frac in fractions:
    b_s = int(np.ceil(frac))

    # Stage 1: knapsack for each index at budget b_s
    pre_y_s = {(pid, i): 0 for pid in K for i in I}
    for i in I:
        m1 = gp.Model(f'sens_knap_{i}_{frac:.2f}')
        y_i = m1.addVars(K, vtype=GRB.BINARY)
        for pid in K:
            if e_k[pid]:
                m1.addConstr(y_i[pid] == 1)
        m1.addConstr(gp.quicksum(c[pid] * y_i[pid] for pid in K) <= b_s)
        m1.setObjective(
            gp.quicksum(p_hat[(i, pid)] * y_i[pid] for pid in K),
            GRB.MAXIMIZE
        )
        m1.Params.OutputFlag = 0
        m1.optimize()
        for pid in K:
            pre_y_s[(pid, i)] = int(y_i[pid].X > 0.5)

    # --- New: compute reference utilities p_ref for this budget ---
    p_ref = {ip: sum(pre_y_s[(pid, ip)] * p_hat[(ip, pid)] for pid in K) for ip in I}

    # Stage 2: minimize max regret at budget b_s using p_ref
    m2_s = gp.Model(f'sens_regret_{frac:.2f}')
    R_s = m2_s.addVars(I, lb=0.0, name='R_s')
    T_s = m2_s.addVar(lb=0.0, name='T_s')
    m2_s.setObjective(T_s, GRB.MINIMIZE)
    for i in I:
        m2_s.addConstr(T_s >= R_s[i])
    for i in I:
        for ip in I:
            util_i = sum(pre_y_s[(pid, i)] * p_hat[(ip, pid)] for pid in K)
            m2_s.addConstr(R_s[i] >= p_ref[ip] - util_i)
    m2_s.Params.OutputFlag = 0
    m2_s.optimize()

    # Collect results
    regret_vals = {i: R_s[i].X for i in I}
    total_regret = sum(regret_vals.values())
    min_regret = min(regret_vals.values())
    tol = 1e-5
    min_indices = [i for i, val in regret_vals.items() if abs(val - min_regret) < tol]
    if len(min_indices) == 1:
        chosen_s = min_indices[0]
    elif len(min_indices) == len(I):
        chosen_s = '/'.join(sorted(I))
    else:
        chosen_s = '/'.join(sorted(min_indices))

    choices_s = {i: [pid for pid in K if pre_y_s[(pid, i)] == 1 and e_k[pid] == 0] for i in I}
    choices_s_str = {f'choices_{i}': ';'.join(choices_s[i]) for i in I}
    costs_s = {i: sum(c[pid] * pre_y_s[(pid, i)] for pid in K) for i in I}
    num_choices = {f'num_choices_{i}': len(choices_s[i]) for i in I}

    regret_matrix = {}
    for i in I:
        for ip in I:
            if i == ip:
                regret_matrix[f'Regret_{i}_assumed_{ip}'] = 0.0
            else:
                best_ip = p_ref[ip]
                util_ip_i = sum(pre_y_s[(pid, i)] * p_hat[(ip, pid)] for pid in K)
                regret_matrix[f'Regret_{i}_assumed_{ip}'] = best_ip - util_ip_i

    results.append({
        'budget': b_s,
        'max_regret': T_s.X,
        'chosen_index': chosen_s,
        **{f'regret_{i}': regret_vals[i] for i in I},
        **num_choices,
        **choices_s_str,
        **regret_matrix,
        **costs_s
    })

df = pd.DataFrame(results)

# Only compare TPL and EJI for who is worst hurt by SVI’s selection
nonself_cols = [f'Regret_SVI_assumed_{i}' for i in ['TPL', 'EJI']]
df['SVI_worst_nonself'] = df[nonself_cols].idxmax(axis=1)

print('Who is most hurt by SVI’s park selection (excluding SVI itself):')
print(df['SVI_worst_nonself'].value_counts())

mask = df[nonself_cols].max(axis=1) > 0
df_nonzero = df[mask]
print('\nFor budgets where someone is actually hurt (regret > 0):')
print(df_nonzero['SVI_worst_nonself'].value_counts())

df_sens = pd.DataFrame(results)
df_sens.to_csv('TwoStateSensitivityAnalysis.csv', index=False)
print(df_sens)
print('=== Sensitivity Analysis (max budget = 2.7M) ===')
print(df_sens.to_string(index=False))

print(f'Total parks in K: {len(K)}')

# Re-load park data to count candidates
k_df = xls.parse('K')
e_k, v_k, c = {}, {}, {}
for _, r in k_df.iterrows():
    pid = str(r['ParkID'])
    e_k[pid] = int(r.get('existing', 0))
    v_k[pid] = float(r.get('vkl', 0.0))
    price = float(r[price_col])
    c[pid] = 0.0 if e_k[pid] else price

num_candidates = sum(1 for pid in K if c.get(pid, 0) > 0)
print(f'Candidate parks (cost > 0): {num_candidates}')
