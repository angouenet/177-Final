from joblib import Parallel, delayed
import numpy as np
import pandas as pd
from itertools import product
import time

# === PARAMETERS ===
years = 26
initial_budget = 100_000
annual_replenishment = 15_000
discount = 0.97
B_penalty = 10_000
lambdas = {'E': 0.4, 'H': 0.3, 'T': 0.3}
D = np.linspace(1, 2, years)
total_customers = 2528
willing_pct = 0.47
cost_willing = 20
cost_unwilling = 280

state_space_e = [0, 50, 100]
state_space_h = [0, 25, 50, 75, 100]
state_space_t = [0, 25, 50, 75, 100]
state_space_b = [0, 25_000, 50_000, 75_000, 100_000, 125_000]

# === TRANSITION MATRICES ===
heat_transition = np.array([
    [0.92, 0.08, 0.00, 0.00, 0.00],
    [0.00, 0.92, 0.08, 0.00, 0.00],
    [0.00, 0.00, 0.92, 0.08, 0.00],
    [0.00, 0.00, 0.00, 0.92, 0.08],
    [0.00, 0.00, 0.00, 0.00, 1.00],
])
transp_transition = np.array([
    [0.90, 0.10, 0.00, 0.00, 0.00],
    [0.00, 0.90, 0.10, 0.00, 0.00],
    [0.00, 0.00, 0.90, 0.10, 0.00],
    [0.00, 0.00, 0.00, 0.90, 0.10],
    [0.00, 0.00, 0.00, 0.00, 1.00],
])
tiers = ['GB', 'GP', 'C50', 'C100']
tier_renewable_pct = np.array([24.3, 33.0, 50.0, 100.0])
electricity_transition = np.array([
    [0.884, 0.080, 0.036, 0.000],
    [0.009, 0.875, 0.080, 0.036],
    [0.002, 0.009, 0.873, 0.116],
    [0.000, 0.003, 0.037, 0.960]
])

# === HELPERS ===
def project(s, space):
    return min(space, key=lambda x: abs(x - s))

def project_state(s):
    e, h, t, b = s
    return (project(e, state_space_e),
            project(h, state_space_h),
            project(t, state_space_t),
            project(b, state_space_b))

def upgrade_customers(distr, budget):
    if budget <= 0: return distr
    tier_counts = (distr * total_customers).astype(int)
    willing = int(willing_pct * total_customers)
    unwilling = total_customers - willing
    willing_upgrades = min(willing, budget // cost_willing)
    budget -= willing_upgrades * cost_willing
    unwilling_upgrades = min(unwilling, budget // cost_unwilling)
    for _ in range(willing_upgrades + unwilling_upgrades):
        for i in range(len(tiers)-1):
            if tier_counts[i] > 0:
                tier_counts[i] -= 1
                tier_counts[i+1] += 1
                break
    return tier_counts / total_customers

def transition(s, a):
    e, h, t, b = project_state(s)
    b_e, b_h, b_t = a[0]*5_000, a[1]*5_000, a[2]*5_000
    spent = b_e + b_h + b_t
    elec_dist = np.array([1, 0, 0, 0]) if e == 0 else np.array([0, 0.1, 0.7, 0.2]) if e == 50 else np.array([0, 0, 0, 1])
    elec_upd = upgrade_customers(elec_dist, b_e)
    elec_next = elec_upd @ electricity_transition
    e_next = np.dot(elec_next, tier_renewable_pct)
    h_idx = state_space_h.index(h)
    h_next = np.dot(heat_transition[h_idx], state_space_h)
    t_idx = state_space_t.index(t)
    t_next = np.dot(transp_transition[t_idx], state_space_t)
    b_next = min(125_000, max(0, b - spent) + annual_replenishment)
    return {project_state((e_next, h_next, t_next, b_next)): 1.0}

def reward(s_prev, s_curr, t):
    sp = project_state(s_prev)
    sc = project_state(s_curr)
    return sum(lambdas[k] * (sc[i] - sp[i]) for i, k in enumerate('EHT')) * D[t]

def penalty(s, t):
    return -B_penalty if project_state(s)[0] < 100 and t >= 5 else 0

# === VALUE ITERATION ===
V = [{} for _ in range(years + 1)]
policy = [{} for _ in range(years)]
state_space = list(product(state_space_e, state_space_h, state_space_t, state_space_b))
for s in state_space:
    V[years][s] = 0

def best_action(s, t):
    best_q = -np.inf
    best_a = (0,0,0)
    max_units = min(20, s[3] // 5_000)
    actions = [
        (e, h, max(0, max_units - e - h))
        for e in range(0, max_units+1)
        for h in range(0, max_units+1-e)
        if (e+h+(max_units-e-h)) >= 1
    ]
    for a in actions:
        s_next = next(iter(transition(s, a)))
        r = reward(s, s_next, t)
        p = penalty(s_next, t)
        q = r + p + discount * V[t+1][s_next]
        if q > best_q:
            best_q = q
            best_a = a
    return s, best_q, best_a

print("Starting parallel value iteration...")
start = time.time()
for t in reversed(range(years)):
    results = Parallel(n_jobs=-1)(delayed(best_action)(s, t) for s in state_space)
    for s, val, act in results:
        V[t][s] = val
        policy[t][s] = act
    print(f"Finished year {t+1} / {years}...")

print("Total time: %.2f minutes" % ((time.time() - start) / 60))