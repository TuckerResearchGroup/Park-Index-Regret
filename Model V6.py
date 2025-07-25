import os
import numpy as np
import pandas as pd
from gurobipy import Model, GRB, quicksum

# --- CONFIGURATION ---
EXCEL_FILE = 'datasheet1.xlsx'
SOLVER_PARAMS = {'OutputFlag': 0}
budget_list = list(np.ceil(np.linspace(0, 7500000, 5000)).astype(int))

# --- LOAD DATA ---
os.chdir("C:/Users/gabeg/OneDrive/Documents/H-CORE REU/Models/V2")
xls     = pd.ExcelFile(EXCEL_FILE)
sets_df = xls.parse('Sets')
l_df    = xls.parse('L')
k_df    = xls.parse('K')

l_df['GEOID20'] = l_df['GEOID20'].astype(str)
k_df['ParkID']  = k_df['ParkID'].astype(str)
Park_all   = list(k_df['ParkID'])
exist_dict = k_df.set_index('ParkID')['existing'].fillna(0).astype(int).to_dict()
ALL_K      = [k for k in Park_all if exist_dict.get(k, 0) == 0]
I = list(sets_df['I'].dropna().astype(str))
L = list(l_df['GEOID20'])

n_l     = l_df.set_index('GEOID20')['total_pop'].fillna(0).to_dict()
c_k_all = k_df.set_index('ParkID')['price'].fillna(0).to_dict()
c_k     = {k: c_k_all.get(k, 0) for k in ALL_K}
q_il    = {i: l_df.set_index('GEOID20')[f'{i}_rank'].fillna(0).to_dict() for i in I}
primary_col = [c for c in l_df.columns if 'primary' in c.lower()][0]
s_kl = {(k, loc): 0 for k in ALL_K for loc in L}
for _, r in l_df.iterrows():
    loc, p = str(r['GEOID20']), str(r.get(primary_col, '')).strip()
    if p in ALL_K:
        s_kl[(p, loc)] = 1
v_k = k_df['vkl'].fillna(0).to_dict() if 'vkl' in k_df.columns else {k: 0 for k in ALL_K}
p_ik = {(i, k): sum(
    q_il[i].get(loc, 0) * n_l.get(loc, 0) * (1 - v_k.get(k, 0))
    for loc in L if s_kl[(k, loc)]
) for i in I for k in ALL_K}
K = [k for k in ALL_K if c_k[k] > 0]

records = []

for budget in budget_list:
    # Precompute max cardinality N for this budget
    # --- Filter parks: remove duplicates and zero‑cost parks ---
    K = list(dict.fromkeys(K))                  # deduplicate
    K = [k for k in K if c_k.get(k, 0) > 0]
         # keep only parks with cost > 0
    m_N = Model()
    for p, v in SOLVER_PARAMS.items():
        m_N.setParam(p, v)
    x_N = m_N.addVars(K, vtype=GRB.BINARY)
    m_N.setObjective(quicksum(x_N[k] for k in K), GRB.MAXIMIZE)
    m_N.addConstr(quicksum(c_k[k] * x_N[k] for k in K) <= budget)
    m_N.optimize()
    if m_N.Status == GRB.OPTIMAL:
        N = int(round(sum(x_N[k].X for k in K)))
    else:
        print(f"WARNING: Knapsack infeasible at Budget={budget}")
        N = 0
    print(f"Budget={budget}, Max cardinality N={N}")

    # Precompute o_star[i] for each leader AND record their optimal picks
    o_star = {}
    y_optimal = {}
    for i in I:
        m_tmp = Model()
        for p, v in SOLVER_PARAMS.items():
            m_tmp.setParam(p, v)
        x_tmp = m_tmp.addVars(K, vtype=GRB.BINARY)
        m_tmp.setObjective(quicksum(p_ik[(i, k)] * x_tmp[k] for k in K), GRB.MAXIMIZE)
        m_tmp.addConstr(quicksum(x_tmp[k] for k in K) <= N)
        m_tmp.optimize()
        o_star[i] = m_tmp.ObjVal if m_tmp.Status == GRB.OPTIMAL else 0
        y_optimal[i] = [k for k in K if m_tmp.Status == GRB.OPTIMAL and x_tmp[k].X > 0.5]

    # Main KKT-transformed model
    m = Model()
    for p, v in SOLVER_PARAMS.items():
        m.setParam(p, v)

    x_s = m.addVars(K, vtype=GRB.BINARY, name='x')
    y_s = m.addVars(I, K, vtype=GRB.BINARY, name='y')
    R_s = m.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name='R')

    m.addConstr(quicksum(c_k[k] * x_s[k] for k in K) <= budget, name="follower_budget")
    m.addConstrs((x_s[k] <= quicksum(y_s[i, k] for i in I) for k in K), name="link")
    m.addConstrs((quicksum(y_s[i, k] for k in K) <= N for i in I), name="leader_N")
    m.addConstrs((R_s >= o_star[i] - quicksum(p_ik[(i, k)] * x_s[k] for k in K) for i in I), name="regret")
    m.addConstrs((quicksum(p_ik[(i, k)] * x_s[k] for k in K) <= o_star[i] for i in I), name="regret_upper")

    lam_leader = m.addVars(I, lb=0.0, name='lam_leader')
    d_lam_leader = m.addVars(I, vtype=GRB.BINARY, name='d_lam_leader')

    for i in I:
        for k in K:
            m.addConstr(
                p_ik[(i, k)] * (2 * y_s[i, k] - x_s[k]) - lam_leader[i] <= 1e6 * (1 - d_lam_leader[i]),
                name=f"leader_KKT_stat_{i}_{k}")
        m.addConstr(lam_leader[i] >= 0, name=f"lam_leader_feas_{i}")
        m.addConstr(lam_leader[i] <= 1e6 * d_lam_leader[i], name=f"lam_leader_comp1_{i}")
        m.addConstr(N - quicksum(y_s[i, k] for k in K) <= 1e6 * (1 - d_lam_leader[i]), name=f"lam_leader_comp2_{i}")

    eq_obj_s = quicksum(d_lam_leader[i] for i in I)
    bigM_s = sum(o_star.values()) + 1
    m.setObjective(eq_obj_s * bigM_s + R_s, GRB.MINIMIZE)

    m.optimize()

    # --- Collect and store results ---
    output_row = {'Budget': budget, 'N': N}
    if m.Status == GRB.OPTIMAL:
        output_row['MaxRegret'] = R_s.X
        # Follower picks
        follower_picks = [k for k in K if x_s[k].X > 0.5]
        output_row["Follower"] = ";".join(follower_picks)
        # Per-leader results
        for i in I:
            # Leader's choice in equilibrium (in KKT model)
            leader_pick = [k for k in K if y_s[i, k].X > 0.5]
            output_row[f"{i}_leader"] = ";".join(leader_pick)
            # Leader's perfect foresight optimal picks
            output_row[f"{i}_optimal"] = ";".join(y_optimal[i])
            # Leader's regret
            follower_val = sum(p_ik[(i, k)] * x_s[k].X for k in K)
            output_row[f"{i}_regret"] = o_star[i] - follower_val
    else:
        output_row['MaxRegret'] = np.nan
        output_row["Follower"] = ""
        for i in I:
            output_row[f"{i}_leader"] = ""
            output_row[f"{i}_optimal"] = ""
            output_row[f"{i}_regret"] = np.nan
        print(f"WARNING: No optimal solution at Budget={budget}, N={N} (status={m.Status})")
    records.append(output_row)

# --- EXPORT RESULTS ---
res_df = pd.DataFrame(records)
res_df.to_csv("sensitivity_analysis_kkt_epec.csv", index=False)
print(res_df)
print("Sensitivity analysis results saved to sensitivity_analysis_kkt_epec.csv")

   # drop zero‑cost
print(f"Parks with cost > 0 (unique): {len(K)}")
