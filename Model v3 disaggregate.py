
"""

SVI_Disaggregated_Normalized
Created on Wed Jul 2 2025

@author: Gabriel Griffin
"""




# Model v3 disaggregate: Use SVI subindices as I and rebuild q from Excel
import os
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
import matplotlib.pyplot as plt

# --- Working directory ---
os.chdir('C:/Users/gabeg/OneDrive/Documents/H-CORE REU/Models/V2')

# --- Load original data ---
xls = pd.ExcelFile('datasheet1.xlsx')

# --- Park IDs and parameters ---
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

# --- Population by geography ---
l_df = xls.parse('L')
# Harmonize GEOID20 field to string without decimals
def to_str_geoid(x):
    try:
        return str(int(x))
    except:
        return str(x)
l_df['GEOID20'] = l_df['GEOID20'].apply(to_str_geoid)
n = {r['GEOID20']: float(r['total_pop']) for _, r in l_df.iterrows()}

# --- Load SVI disaggregated sheet ---
svi_xls = pd.ExcelFile('disaggregated_by_index_normalized.xlsx')
svi_df = svi_xls.parse('SVI_Disaggregated_Normalized')  # updated sheet name
# Harmonize GEOID20 formatting
svi_df['GEOID20'] = svi_df['GEOID20'].apply(to_str_geoid)
# Define indices and locations
I = [col for col in svi_df.columns if col != 'GEOID20']
L = svi_df['GEOID20'].tolist()

# --- Primary-park indicator s_{k,ℓ}: rebuild for SVI locations ---
skl = {(pid, loc): 0 for pid in K for loc in L}
primary_col = next((col for col in l_df.columns if 'primary' in col.lower()), None)
print(f"Detected primary column: {primary_col}")
if primary_col:
    for _, r in l_df.iterrows():
        loc = r['GEOID20']
        p = str(r.get(primary_col, '')).strip()
        if loc in L and p in K:
            skl[(p, loc)] = 1
num_skl = sum(skl.values())
print(f"Total skl (park,location) pairs marked primary: {num_skl}")

# --- Build utility q[(i,l)] from SVI columns ---
q = {}
for _, r in svi_df.iterrows():
    loc = r['GEOID20']
    for ip in I:
        q[(ip, loc)] = float(r[ip])
print(f"Total q entries: {len(q)}")

# --- Compute p_hat[i,k] for Stage 1 ---
p_hat = {}
for ip in I:
    for pid in K:
        total = 0.0
        for loc in L:
            if skl.get((pid, loc), 0) == 1:
                total += q.get((ip, loc), 0.0) * n.get(loc, 0.0) * (1.0 - v_k.get(pid, 0.0))
        p_hat[(ip, pid)] = total
# Debug p_hat
nonzero_phat = sum(1 for v in p_hat.values() if v > 0)
print(f"Nonzero p_hat entries: {nonzero_phat} of {len(p_hat)}")
for idx, ((ip, pid), val) in enumerate(p_hat.items()):
    if idx < 5:
        print(f"p_hat[{ip},{pid}] = {val:.2f}")

# --- Base-case budget: 25% of total cost for candidate parks ---
positive_cands = [pid for pid in K if any(skl[(pid, loc)] for loc in L)]
total_pos_cost = sum(c[pid] for pid in positive_cands)
b = total_pos_cost * 0.25
print(f"Base-case budget b = {b:.2f}")




# --- Stage 1: knapsack per index (base-case) ---
pre_y = {(pid, i): 0 for pid in K for i in I}
for i in I:
    m = gp.Model(f'knap_{i}')
    y = m.addVars(K, vtype=GRB.BINARY)
    for pid in K:
        if e_k[pid]:
            m.addConstr(y[pid] == 1)
    m.addConstr(gp.quicksum(c[pid]*y[pid] for pid in K) <= b)
    m.setObjective(gp.quicksum(p_hat[(i, pid)]*y[pid] for pid in K), GRB.MAXIMIZE)
    m.Params.OutputFlag = 0
    m.optimize()
    for pid in K:
        pre_y[(pid, i)] = int(y[pid].X > 0.5)

# --- Compute reference utilities p_pre ---
p_pre = {(ip, i): sum(pre_y[(pid, i)]*p_hat[(ip, pid)] for pid in K)
         for ip in I for i in I}

# --- Stage 2: minimize max regret (base-case) ---
m2 = gp.Model('regret_base')
R = m2.addVars(I, lb=0)
T = m2.addVar(lb=0)
m2.setObjective(T, GRB.MINIMIZE)
for i in I:
    m2.addConstr(T >= R[i])
    for ip in I:
        diff = p_pre[(ip, ip)] - p_pre[(ip, i)]
        m2.addConstr(R[i] >= diff)
m2.Params.OutputFlag = 0
m2.optimize()

# --- Print base-case results ---
print(f'=== Base-case (budget = {b:.0f}) ===')
for i in I:
    cost_i = sum(c[pid]*pre_y[(pid, i)] for pid in K)
    print(f'Index {i}: cost = {cost_i:.2f}, regret = {R[i].X:.2f}')
best = min(I, key=lambda i: R[i].X)
print(f'Chosen index = {best}\n')

# --- Sensitivity analysis function ---
def run_sensitivity(max_budget=2_700_000, steps=500):
    fractions = np.linspace(0, max_budget, steps)
    results = []
    for frac in fractions:
        b_s = int(np.ceil(frac))
        pre_y_s = {(pid, i): 0 for pid in K for i in I}
        for i in I:
            m1 = gp.Model()
            y_i = m1.addVars(K, vtype=GRB.BINARY)
            for pid in K:
                if e_k[pid]: m1.addConstr(y_i[pid] == 1)
            m1.addConstr(gp.quicksum(c[pid]*y_i[pid] for pid in K) <= b_s)
            m1.setObjective(gp.quicksum(p_hat[(i, pid)]*y_i[pid] for pid in K), GRB.MAXIMIZE)
            m1.Params.OutputFlag = 0
            m1.optimize()
            for pid in K:
                pre_y_s[(pid, i)] = int(y_i[pid].X > 0.5)
        p_ref = {ip: sum(pre_y_s[(pid, ip)]*p_hat[(ip, pid)] for pid in K) for ip in I}
        m2_s = gp.Model()
        R_s = m2_s.addVars(I, lb=0)
        T_s = m2_s.addVar(lb=0)
        m2_s.setObjective(T_s, GRB.MINIMIZE)
        for i in I:
            m2_s.addConstr(T_s >= R_s[i])
            for ip in I:
                util = sum(pre_y_s[(pid, i)]*p_hat[(ip, pid)] for pid in K)
                m2_s.addConstr(R_s[i] >= p_ref[ip] - util)
        m2_s.Params.OutputFlag = 0
        m2_s.optimize()
        regret_vals = {i: R_s[i].X for i in I}
        min_reg = min(regret_vals.values())
        chosen = [i for i,v in regret_vals.items() if abs(v-min_reg)<1e-5]
        results.append({
            'budget': b_s,
            'max_regret': T_s.X,
            'chosen_index': '/'.join(sorted(chosen)),
            **{f'regret_{i}': v for i,v in regret_vals.items()},
            **{f'cost_{i}': sum(c[pid]*pre_y_s[(pid,i)] for pid in K) for i in I},
            **{f'num_choices_{i}': sum(pre_y_s[(pid,i)] and not e_k[pid] for pid in K) for i in I}
        })
    return pd.DataFrame(results)

# --- Execute sensitivity analysis ---
df_sens = run_sensitivity()
df_sens.to_csv('TwoStateSensitivityAnalysis_V3_Disaggregated.csv', index=False)
print('=== Sensitivity Analysis Results ===')
print(df_sens.to_string(index=False))

# --- Summary counts ---
print(f'Total parks in K: {len(K)}')
num_candidates = sum(1 for pid in K if c[pid]>0)
print(f'Candidate parks (cost > 0): {num_candidates}')
