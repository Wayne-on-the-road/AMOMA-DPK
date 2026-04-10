import os.path
from multiprocessing import Pool, cpu_count
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from time import time, strftime, localtime


from pymoo.indicators.hv import HV
import glob
import os

random.seed(42)
np.random.seed(42)

DEBUG = False

#─────────────── helpers ───────────────


def bootstrap_nicv_stability_kmeans(X, k, base_seed, B=6, sample_frac=0.6, replace=False):
    nicv_vals = []
    n = X.shape[0]
    m = max(2, int(sample_frac * n))
    for b in range(B):
        np.random.seed(base_seed + 10000 * b)
        idx = np.random.choice(n, size=m, replace=replace)
        X_sub = X[idx]
        km = KMeans(k, n_init=10, random_state=base_seed + 10000 * b).fit(X_sub)
        centroids = km.cluster_centers_
        nicv_vals.append(float(calculate_nicv(X, centroids)))
    nicv_vals = np.array(nicv_vals, dtype=float)
    mu = float(np.mean(nicv_vals))
    sd = float(np.std(nicv_vals, ddof=0))
    return float((sd / (mu + 1e-12)))

def bootstrap_nicv_stability_dp(
    X,
    k,
    epsilon_seq,
    base_seed,
    B=6,               # number of bootstrap repeats
    sample_frac=0.6,    # use 80% subsample (often better than true bootstrap for stability)
    replace=False,      # False=subsample; True=bootstrap with replacement
):
    """
    Data-perturbation stability of NICV for a fixed epsilon schedule.

    Returns a score to Minimize: bstab = CV(NICV) where CV = std/mean.
    We train on resampled data but always evaluate NICV on full X_train
    so values are comparable across resamples.
    """
    nicv_vals = []

    n = X.shape[0]
    m = max(2, int(sample_frac * n))

    for b in range(B):
        np.random.seed(base_seed + 10000 * b)
        random.seed(base_seed + 10000 * b)

        idx = np.random.choice(n, size=m, replace=replace)
        X_sub = X[idx]

        init_centroids = select_initial_centroids(X_sub, k)
        centroids = dp_kmeans_train_with_schedule(X_sub, k, init_centroids, epsilon_seq)

        # evaluate NICV on full training set for comparability
        nicv_b = float(calculate_nicv(X, centroids))
        nicv_vals.append(nicv_b)

    nicv_vals = np.array(nicv_vals, dtype=float)
    mu = float(np.mean(nicv_vals))
    sd = float(np.std(nicv_vals, ddof=0))
    cv = sd / (mu + 1e-12)

    return float(cv)



def build_fitness_from_metrics(metrics: dict, objective_spec: list):
    """
    metrics: dict with keys like 'acc', 'eps', 'stab', 'nicv', 'ari', 'nmi'
    objective_spec: list of tuples (metric_key, direction)
        direction: 'min' or 'max'
    Returns: np.array fitness for minimization (NSGA).
    """
    vals = []
    for key, direction in objective_spec:
        v = float(metrics[key])
        if direction == 'max':
            vals.append(-v)   # convert to minimization
        elif direction == 'min':
            vals.append(v)
        else:
            raise ValueError(f"direction must be 'min' or 'max', got {direction}")
    return np.array(vals, dtype=float)

def objective_labels(objective_spec):
    # e.g. [('acc','max'), ('eps','min'), ('stab','max')] -> ['acc(max)','eps(min)','stab(max)']
    return [f"{k}({d})" for (k, d) in objective_spec]


def attach_fitness_columns(row, fitness, objective_spec, max_obj=5):
    """
    Adds columns:
      obj_1, f_1, obj_2, f_2, ...
    fitness is already minimization-form (after max->-).
    """
    for i in range(max_obj):
        if i < len(objective_spec):
            key, direction = objective_spec[i]
            row[f"obj_{i+1}"] = f"{key}:{direction}"
            row[f"f_{i+1}"] = float(fitness[i])
        else:
            row[f"obj_{i+1}"] = ""
            row[f"f_{i+1}"] = np.nan
    return row

def ga_objective_name(objective_spec, idx=0):
    key, direction = objective_spec[idx]
    return f"{key}:{direction}"

def dp_kmeans_train_with_schedule(X_train, k, init_centroids, epsilon_seq):
    centroids = init_centroids.copy()
    d = X_train.shape[1]
    for eps_t in epsilon_seq:
        clusters = assign_clusters(X_train, centroids, k)
        centroids = dp_kmeans_update_centroid(
            clusters,
            old_centroids=centroids,
            epsilon_t=eps_t,
            d=d
        )
    return centroids




def calculate_nicv(data, centroids, print_sse=False):
    """
    Compute NICV = (1/N) * sum_i sum_{x in C_i} ||x - c_i||^2,
    and optionally print the total SSE over all clusters.
    """
    # 1) Assign points to clusters
    k = len(centroids)
    clusters = {i: [] for i in range(k)}
    for x in data:
        j = np.argmin([np.linalg.norm(x - c) for c in centroids])
        clusters[j].append(x)

    # 2) Accumulate squared error
    total_se = 0.0
    total_pts = 0
    for i, pts in clusters.items():
        ci = centroids[i]
        for x in pts:
            total_se += np.linalg.norm(x - ci) ** 2
            total_pts += 1


    # 4) Return normalized variance
    return total_se / total_pts if total_pts > 0 else float('inf')

def strict_normalize(epsilon_seq, epsilon_total, epsilon_m, max_iters=40, tol=1e-7):
    seq = epsilon_seq.copy()
    for _ in range(max_iters):
        # 1) scale to sum == epsilon_total
        seq = np.clip(seq, epsilon_m, None)
        total = seq.sum()
        if abs(total - epsilon_total) < tol:
            break
        seq = seq / total * epsilon_total

        # 2) floor every coordinate at epsilon_m

    return seq

def enforce_budget(epsilon_seq, epsilon_total, epsilon_m, max_iters=40):
    seq = epsilon_seq.copy()
    for _ in range(max_iters):
        seq = np.clip(seq, epsilon_m, None)
        total = seq.sum()
        if total <= epsilon_total:
            break
        # scale back so sum == epsilon_total
        seq = seq / total * epsilon_total
        # floor any entries that fell below epsilon_m

    return seq


def compute_rho_from_centroids(centroids, data_range=1.0):
    """
    centroids: array of shape (k, d), already in the same normalized space
    data_range: r if you scaled into [-r, r]; here r=1
    """
    # infinity‐norm of each centroid
    rho_vals = np.max(np.abs(centroids), axis=1) / data_range
    return float(np.mean(rho_vals))


# calculate minimum privacy budget
def calculate_minimum_privacy_budget(data, k, rho=0.225):
    """
    Eq.(18) from GAPBAS: εₘ = sqrt( 200·k³·d·(1+d)²·(1+ρ²) / N² )
    Default ρ=0.225 as per paper’s empirical setting.
    """
    N, d = data.shape
    numerator = 200 * (k ** 3) * d * (1 + d) ** 2
    denominator = N ** 2
    epsilon_m = np.sqrt(numerator * (1 + rho ** 2) / denominator)
    return max(epsilon_m, 0.01)



def generate_individual_moma(epsilon_m, epsilon_total, T):

    # average budget per iteration
    avg = epsilon_total / T
    # sample each entry in [εₘ, 2·avg] or [εₘ, avg*1.5], whichever you prefer
    upper = max(epsilon_m, epsilon_total)
    lower = 0
    epsilon_seq = np.random.uniform(lower, upper, T)
    # sample each εₜ independently, then enforce total‐≤‐ε
    # epsilon_seq = np.random.uniform(epsilon_m, epsilon_total, size=T)
    # print(epsilon_seq)
    epsilon_seq = enforce_budget(epsilon_seq, epsilon_total, epsilon_m)
    # print(epsilon_seq)

    return epsilon_seq


# Function to perform DP K-means centroid update with Laplace noise

def dp_kmeans_update_centroid(clusters, old_centroids, epsilon_t, d):
    """
    clusters:  list of length k of arrays of shape (n_i, d)
    old_centroids: array of shape (k, d) — used if a cluster is empty
    epsilon_t: float, the privacy budget for this iteration
    d: dimension of data
    """
    delta_f = d + 1
    scale = delta_f / epsilon_t
    new_centroids = []

    for i, cluster in enumerate(clusters):
        if len(cluster) == 0:
            # keep the previous centroid if cluster is empty
            new_centroids.append(old_centroids[i])
        else:
            # true sum & count
            S_i = np.sum(cluster, axis=0)  # shape (d,)
            n_i = len(cluster)  # scalar

            # one Laplace draw per coordinate of (S_i, n_i)
            noise = np.random.laplace(0, scale, size=d + 1)
            S_noisy = S_i + noise[:d]
            n_noisy = max(n_i + noise[d], 1e-6)  # avoid zero

            new_centroids.append(S_noisy / n_noisy)

    return np.vstack(new_centroids)  # shape (k, d)



# Function to select initial centroids using K-means++ initialization
def select_initial_centroids(data, k):
    centroids = [data[np.random.choice(range(data.shape[0]))]]
    for _ in range(1, k):
        min_distances = np.array([min(np.linalg.norm(x - c) for c in centroids) for x in data])
        probabilities = min_distances ** 2 / np.sum(min_distances ** 2)
        new_centroid = data[np.random.choice(range(data.shape[0]), p=probabilities)]
        centroids.append(new_centroid)
    return np.array(centroids)


# Function to assign clusters based on the closest centroid
def assign_clusters(data, centroids, k):
    clusters = [[] for _ in range(k)]
    for point in data:
        distances = [np.linalg.norm(point - centroid) for centroid in centroids]
        closest_centroid = np.argmin(distances)
        clusters[closest_centroid].append(point)
    return clusters


def tournament_selection(population, scores, tournament_size=3):
    n = len(population)
    k = min(tournament_size, n)
    idxs = np.random.choice(n, size=k, replace=False)
    best_local = idxs[np.argmax(scores[idxs])]
    return population[best_local]

def tournament_selection_index(scores, tournament_size=3):
    n = len(scores)
    k = min(tournament_size, n)
    idxs = np.random.choice(n, size=k, replace=False)
    return int(idxs[np.argmax(scores[idxs])])


# Two-point crossover
def crossover(parent1, parent2, epsilon_m, epsilon_total, T ):
    if np.random.rand() < Pc:
        point1 = np.random.randint(0, T)
        point2 = np.random.randint(point1, T)
        child1 = np.concatenate([parent1[:point1], parent2[point1:point2], parent1[point2:]])
        child2 = np.concatenate([parent2[:point1], parent1[point1:point2], parent2[point2:]])

        # Strict normalization after crossover
        child1 = strict_normalize(child1, epsilon_total, epsilon_m)
        child2 = strict_normalize(child2, epsilon_total, epsilon_m)

        return child1, child2
    else:
        return parent1.copy(), parent2.copy()


# Mutation
def mutation(individual, epsilon_m, epsilon_total, T):
    individual = individual.copy()
    if np.random.rand() < Pm:
        index = np.random.randint(0, T)
        individual[index] = np.random.uniform(epsilon_m, epsilon_total)

    # Strict normalization after mutation
    individual = strict_normalize(individual, epsilon_total, epsilon_m)

    return individual



# Main genetic algorithm

def genetic_algorithm(
    Nm, G, epsilon_m, epsilon_total, T,
    X, k,
    base_seed=42,
    objective_spec=None,
    ga_opt_index=0,          # which objective in objective_spec GA maximizes/minimizes via fitness
):
    """
    Single-objective GA baseline aligned with evaluate_triobjective + objective_spec.

    - objective_spec defines the multi-objective fitness vector used by MOMA/NSGA.
    - GA optimizes ONE dimension of that vector (ga_opt_index).
      Since fitness vector is minimization-form, GA should MAXIMIZE (-fitness[m]).

    Returns:
      best_seq, best_generation, best_score, best_fit_vec, best_extras
    """
    if objective_spec is None:
        objective_spec = [('nicv', 'min'), ('bstab', 'min'), ('eps', 'min')]

    # init population (cap-based)
    population = np.array([
        generate_individual_moma(epsilon_m, epsilon_total, T)
        for _ in range(Nm)
    ])

    best_individual_overall = None
    best_score_overall = float('-inf')   # GA maximizes a scalar score
    best_generation_overall = -1
    best_fit_overall = None
    best_extras_overall = None

    obj_label = ga_objective_name(objective_spec, ga_opt_index)

    for generation in range(G):
        # evaluate scalar score for each individual based on objective_spec
        fit_list = []
        extras_list = []
        scores = np.zeros(len(population), dtype=float)

        for i, individual in enumerate(population):
            fit_vec, extras = evaluate_triobjective(
                individual, X,
                k=k,
                base_seed=base_seed + 1000*generation + i,     # reduce evaluation tie effects
                objective_spec=objective_spec
            )
            fit_list.append(fit_vec)
            extras_list.append(extras)

            # fitness is minimization-form; GA should maximize (-fit_vec[ga_opt_index])
            scores[i] = -float(fit_vec[ga_opt_index])
        # best in this generation
        best_idx = int(np.argmax(scores))
        best_score_gen = float(scores[best_idx])

        if best_score_gen > best_score_overall:
            best_score_overall = best_score_gen
            best_individual_overall = population[best_idx].copy()
            best_generation_overall = generation + 1
            best_fit_overall = fit_list[best_idx]
            best_extras_overall = extras_list[best_idx]

        # ---- aligned printing (shows raw metrics like MOMA) ----
        ex = extras_list[best_idx]
        if DEBUG:
            print(
                f"GA Gen {generation + 1} (opt {obj_label}) -> "
                f"score={best_score_gen:.4f} | "
                f"NICV={ex['NICV']:.4f}, BSTAB={ex['BSTAB_NICV']:.4f}, ε={ex['eps_spent']:.4f}"
            )

        # selection + reproduction uses scores as "fitness_values"
        new_population = []
        for _ in range(Nm // 2):
            parent1 = tournament_selection(population, scores, tournament_size=3)
            parent2 = tournament_selection(population, scores, tournament_size=3)

            child1, child2 = crossover(parent1, parent2, epsilon_m, epsilon_total, T)
            new_population.append(mutation(child1, epsilon_m, epsilon_total, T))
            new_population.append(mutation(child2, epsilon_m, epsilon_total, T))

        population = np.array(new_population)

    if DEBUG:
        print(f"GA best found in generation {best_generation_overall}: {best_individual_overall}")

    # Final detailed evaluation with full stability repeats for reporting
    best_fit_vec, best_extras = evaluate_triobjective(
        best_individual_overall, X,
        k=k,
        base_seed=base_seed + 99999,
        objective_spec=objective_spec
    )

    # best_score_overall corresponds to -fit_vec[ga_opt_index] during search
    # recompute best_score from final fit for consistency
    best_score_final = -float(best_fit_vec[ga_opt_index])

    return best_individual_overall, best_generation_overall, best_score_final, best_fit_vec, best_extras


def dominates(fitness1, fitness2):
    """Check if fitness1 dominates fitness2 (both objectives are minimization)."""
    # Convert to numpy arrays if they aren't already
    fitness1 = np.array(fitness1, dtype=float)
    fitness2 = np.array(fitness2, dtype=float)

    # Check if fitness1 is better or equal in all objectives
    better_or_equal = np.all(fitness1 <= fitness2)
    # Check if fitness1 is strictly better in at least one objective
    strictly_better = np.any(fitness1 < fitness2)

    return bool(better_or_equal and strictly_better)


def non_dominated_sort(population):
    """Perform non-dominated sorting on the population."""
    if not population:
        return []

    # Initialize data structures
    fronts = [[]]  # List of fronts
    domination_counts = {}  # Number of solutions dominating each solution
    dominated_solutions = {}  # Solutions dominated by each solution

    # Initialize for each solution
    for p in population:
        p_id = id(p)
        domination_counts[p_id] = 0
        dominated_solutions[p_id] = []

        # Compare with all other solutions
        for q in population:
            if p is not q:
                q_id = id(q)
                if dominates(p['fitness'], q['fitness']):
                    dominated_solutions[p_id].append(q)
                elif dominates(q['fitness'], p['fitness']):
                    domination_counts[p_id] += 1

    # Assign first front
    for p in population:
        if domination_counts[id(p)] == 0:
            p['rank'] = 0
            fronts[0].append(p)

    # Generate remaining fronts
    i = 0
    while i < len(fronts) and fronts[i]:
        next_front = []
        for p in fronts[i]:
            for q in dominated_solutions[id(p)]:
                q_id = id(q)
                domination_counts[q_id] -= 1
                if domination_counts[q_id] == 0:
                    q['rank'] = i + 1
                    next_front.append(q)
        i += 1
        if next_front:
            fronts.append(next_front)

    return fronts


def calculate_crowding_distance(front):
    """Calculate crowding distance for a front (supports any #objectives)."""
    if len(front) <= 2:
        for individual in front:
            individual['crowding_distance'] = float('inf')
        return

    for individual in front:
        individual['crowding_distance'] = 0.0

    M = len(front[0]['fitness'])  # number of objectives

    for m in range(M):
        front.sort(key=lambda x: float(x['fitness'][m]))

        front[0]['crowding_distance'] = float('inf')
        front[-1]['crowding_distance'] = float('inf')

        f_max = float(front[-1]['fitness'][m])
        f_min = float(front[0]['fitness'][m])
        if f_max == f_min:
            continue

        for i in range(1, len(front) - 1):
            front[i]['crowding_distance'] += (
                float(front[i + 1]['fitness'][m]) - float(front[i - 1]['fitness'][m])
            ) / (f_max - f_min)

def evaluate_triobjective(epsilon_seq, X, k, base_seed, objective_spec=None):
    """
    Full-dataset tri-objective evaluation.

    Default objective order:
      [('nicv', 'min'), ('bstab', 'min'), ('eps', 'min')]
    """
    if objective_spec is None:
        objective_spec = [('nicv', 'min'), ('bstab', 'min'), ('eps', 'min')]

    np.random.seed(base_seed)
    random.seed(base_seed)

    init_centroids = select_initial_centroids(X, k)
    centroids = dp_kmeans_train_with_schedule(X, k, init_centroids, epsilon_seq)

    nicv_val = float(calculate_nicv(X, centroids))
    eps_spent = float(np.sum(epsilon_seq))

    bstab = bootstrap_nicv_stability_dp(
        X=X,
        k=k,
        epsilon_seq=epsilon_seq,
        base_seed=base_seed,
        B=6,
        sample_frac=0.6,
        replace=False
    )

    metrics = {
        "nicv": nicv_val,
        "bstab": bstab,
        "eps": eps_spent,
    }

    fitness = build_fitness_from_metrics(metrics, objective_spec)

    extras = {
        "NICV": float(nicv_val),
        "BSTAB_NICV": float(bstab),
        "eps_spent": float(eps_spent),
    }
    extras["objective_spec"] = objective_spec
    return fitness, extras


def mutate_eps(epsilon_seq, epsilon_m, epsilon_total, T, Pm):
    """Polynomial‐like mutation on the eps vector + re-normalize."""
    if np.random.rand() < Pm:
        i = np.random.randint(0, T)
        delta = np.random.normal(0, (epsilon_total / T) * 0.1)  # 10% scale
        epsilon_seq[i] = np.clip(epsilon_seq[i] + delta, epsilon_m, epsilon_total)
        # enforce lower bound & renormalize
        epsilon_seq = enforce_budget(epsilon_seq, epsilon_total, epsilon_m)
    return epsilon_seq


def mate_eps(p1, p2, epsilon_m, epsilon_total, T, Pc):
    """Blend‐style crossover on two eps vectors + re-normalize each child."""
    if np.random.rand() < Pc:
        alpha = 0.5
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return (
            enforce_budget(c1, epsilon_total, epsilon_m),
            enforce_budget(c2, epsilon_total, epsilon_m)
        )
    else:
        return p1.copy(), p2.copy()


# ──────────────────────────────────────────────────────────────────────────────
# APMA-style adaptive memory + adaptive operators (minimal patch)
# ──────────────────────────────────────────────────────────────────────────────

class AdaptiveMemory:
    def __init__(self, H=5):
        self.H = H
        self.memory_F = [0.5] * H
        self.memory_CR = [0.5] * H
        self.index = 0

    def sample_parameters(self):
        m_F = self.memory_F[self.index]
        m_CR = self.memory_CR[self.index]
        F = float(np.clip(random.gauss(m_F, 0.1), 0.0, 1.0))
        CR = float(np.clip(random.gauss(m_CR, 0.1), 0.0, 1.0))
        return F, CR

    def update_memory(self, successful_F, successful_CR):
        if len(successful_F) > 0 and np.sum(successful_F) > 0:
            self.memory_F[self.index] = float(
                np.sum(np.square(successful_F)) / (np.sum(successful_F) + 1e-12)
            )
        if len(successful_CR) > 0:
            self.memory_CR[self.index] = float(np.mean(successful_CR))

        self.index = (self.index + 1) % self.H


def adaptive_crossover_eps(parent1, parent2, CR, epsilon_m, epsilon_total, T):
    """
    APMA-style gene-wise crossover for epsilon schedules.
    Child gene comes from parent1 with prob CR, else parent2.
    Then repair by enforce_budget().
    """
    child1 = np.array([
        parent1[t] if np.random.rand() < CR else parent2[t]
        for t in range(T)
    ], dtype=float)

    child2 = np.array([
        parent2[t] if np.random.rand() < CR else parent1[t]
        for t in range(T)
    ], dtype=float)

    child1 = enforce_budget(child1, epsilon_total, epsilon_m)
    child2 = enforce_budget(child2, epsilon_total, epsilon_m)
    return child1, child2


def adaptive_mutation_eps(individual, F, epsilon_m, epsilon_total, T):
    """
    APMA-style mutation for epsilon schedules.
    Each coordinate mutates with probability F, then repair.
    """
    child = np.array(individual, dtype=float).copy()

    # mutation scale tied to average budget per round
    sigma = max((epsilon_total / max(T, 1)) * 0.1, 1e-8)

    for t in range(T):
        if np.random.rand() < F:
            child[t] = child[t] + np.random.normal(0.0, sigma)

    child = enforce_budget(child, epsilon_total, epsilon_m)
    return child

def local_search_on_eps(eps, data, initial_centroids, k,
                        epsilon_m, epsilon_total, T,
                        max_iters=3, step_frac=0.1, increase=True):
    """
    Try up to max_iters small transfers between two rounds:
    - pick i≠j, transfer delta from eps[i]→eps[j], renormalize,

    """
    scaler = SCALER

    for _ in range(max_iters):
        i = np.random.randint(0, T)
        delta = (epsilon_total / T) * step_frac
        cand = eps.copy()
        if increase:
            cand[i] = cand[i] + scaler*delta
        else:
            cand[i] = cand[i] - delta
        cand = enforce_budget(cand, epsilon_total, epsilon_m)

    return cand


def moma_ga(X, k,
            epsilon_m, epsilon_total, T,
            Nm, G, Pm, Pc,
            local_search,
            base_seed,
            objective_spec=None):
    """
    Returns:
      pareto_front: list of individuals with fields:
         'eps', 'fitness', 'extras', 'rank', 'crowding_distance'
      epsilon_seqs: list of eps schedules on Pareto front
    """

    # init population
    pop = [generate_individual_moma(epsilon_m, epsilon_total, T) for _ in range(Nm)]
    population = []
    for eps in pop:
        fit, extras = evaluate_triobjective(
            eps, X,
            k=k,
            base_seed=base_seed,
            objective_spec=objective_spec
        )
        population.append({'eps': eps, 'fitness': fit, 'extras': extras})

    for gen in range(G):
        children_eps = []
        for _ in range(Nm // 2):
            p1, p2 = random.sample(population, 2)

            c1, c2 = mate_eps(p1['eps'], p2['eps'],
                              epsilon_m, epsilon_total, T, Pc)

            c1 = mutate_eps(c1, epsilon_m, epsilon_total, T, Pm)
            c2 = mutate_eps(c2, epsilon_m, epsilon_total, T, Pm)

            if local_search:
                c1 = local_search_on_eps(c1, X, None, k,  # initial_centroids not needed now
                                         epsilon_m, epsilon_total, T, increase=True)
                c2 = local_search_on_eps(c2, X, None, k,
                                         epsilon_m, epsilon_total, T, increase=False)

            children_eps.extend([c1, c2])

        # evaluate offspring
        children = []
        for eps in children_eps:
            fit, extras = evaluate_triobjective(
                eps, X,
                k=k,
                base_seed=base_seed + 10000 + gen,
                objective_spec=objective_spec
            )
            children.append({'eps': eps, 'fitness': fit, 'extras': extras})

        # selection
        combined = population + children
        fronts = non_dominated_sort(combined)
        newpop = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(newpop) + len(front) <= Nm:
                newpop.extend(front)
            else:
                front.sort(key=lambda ind: ind['crowding_distance'], reverse=True)
                need = Nm - len(newpop)
                newpop.extend(front[:need])
                break
        population = newpop

        # reporting
        spec = objective_spec if objective_spec is not None else [('nicv', 'min'), ('bstab', 'min'), ('eps', 'min')]
        obj_names = objective_labels(spec)

        if DEBUG:
            best_by_obj = []
            for m in range(len(spec)):
                best_by_obj.append(min(population, key=lambda ind: ind['fitness'][m]))

            print(f"Gen {gen + 1}:")
            for m, best_ind in enumerate(best_by_obj):
                ex = best_ind['extras']

                print(
                    f"  best {obj_names[m]} -> "
                    f"NICV={ex['NICV']:.4f}, BSTAB={ex['BSTAB_NICV']:.4f}, ε={ex['eps_spent']:.4f}"
                )

    pareto_front = non_dominated_sort(population)[0]
    epsilon_seqs = [ind['eps'] for ind in pareto_front]
    return pareto_front, epsilon_seqs

def amoma_ga(X, k,
            epsilon_m, epsilon_total, T,
            Nm, G, Pm, Pc,
            local_search,
            base_seed,
            objective_spec=None, HL=None):
    """
    Returns:
      pareto_front: list of individuals with fields:
         'eps', 'fitness', 'extras', 'rank', 'crowding_distance'
      epsilon_seqs: list of eps schedules on Pareto front

    Minimal-change APMA-style version:
      - adaptive memory for F and CR
      - adaptive crossover
      - adaptive mutation
      - existing NSGA-II survivor selection unchanged
      - local_search flag still controls NSGA-II vs MOMA
    """

    # init population
    pop = [generate_individual_moma(epsilon_m, epsilon_total, T) for _ in range(Nm)]
    population = []
    for eps in pop:
        fit, extras = evaluate_triobjective(
            eps, X,
            k=k,
            base_seed=base_seed,
            objective_spec=objective_spec
        )
        population.append({'eps': eps, 'fitness': fit, 'extras': extras})

    # NEW: adaptive memory
    adaptive_memory = AdaptiveMemory(H=HL)

    for gen in range(G):
        children = []
        successful_F = []
        successful_CR = []

        for pair_idx in range(Nm // 2):
            # parent selection: keep  current simple random sampling
            p1, p2 = random.sample(population, 2)

            # NEW: sample adaptive parameters
            F, CR = adaptive_memory.sample_parameters()

            # NEW: adaptive crossover + adaptive mutation
            c1, c2 = adaptive_crossover_eps(
                p1['eps'], p2['eps'], CR,
                epsilon_m, epsilon_total, T
            )
            c1 = adaptive_mutation_eps(c1, F, epsilon_m, epsilon_total, T)
            c2 = adaptive_mutation_eps(c2, F, epsilon_m, epsilon_total, T)

            # keep  existing MOMA switch
            if local_search:
                c1 = local_search_on_eps(
                    c1, X, None, k,
                    epsilon_m, epsilon_total, T, increase=True
                )
                c2 = local_search_on_eps(
                    c2, X, None, k,
                    epsilon_m, epsilon_total, T, increase=False
                )

            # evaluate offspring
            fit1, extras1 = evaluate_triobjective(
                c1, X,
                k=k,
                base_seed=base_seed + 10000 + gen * 100 + 2 * pair_idx,

                objective_spec=objective_spec
            )
            fit2, extras2 = evaluate_triobjective(
                c2, X,
                k=k,
                base_seed=base_seed + 10000 + gen * 100 + 2 * pair_idx + 1,

                objective_spec=objective_spec
            )

            child1 = {'eps': c1, 'fitness': fit1, 'extras': extras1}
            child2 = {'eps': c2, 'fitness': fit2, 'extras': extras2}
            children.extend([child1, child2])

            # NEW: success rule for adaptive memory
            # successful if a child dominates at least one parent
            if dominates(fit1, p1['fitness']) or dominates(fit1, p2['fitness']):
                successful_F.append(F)
                successful_CR.append(CR)

            if dominates(fit2, p1['fitness']) or dominates(fit2, p2['fitness']):
                successful_F.append(F)
                successful_CR.append(CR)

        # NEW: update adaptive memory once per generation
        adaptive_memory.update_memory(successful_F, successful_CR)

        # selection (unchanged)
        combined = population + children
        fronts = non_dominated_sort(combined)
        newpop = []
        for front in fronts:
            calculate_crowding_distance(front)
            if len(newpop) + len(front) <= Nm:
                newpop.extend(front)
            else:
                front.sort(key=lambda ind: ind['crowding_distance'], reverse=True)
                need = Nm - len(newpop)
                newpop.extend(front[:need])
                break
        population = newpop

        # reporting (unchanged)
        spec = objective_spec if objective_spec is not None else [('nicv', 'min'), ('bstab', 'min'), ('eps', 'min')]
        obj_names = objective_labels(spec)
        if DEBUG:
            best_by_obj = []
            for m in range(len(spec)):
                best_by_obj.append(min(population, key=lambda ind: ind['fitness'][m]))

            print(f"Gen {gen + 1}:")
            for m, best_ind in enumerate(best_by_obj):
                ex = best_ind['extras']
                print(
                    f"  best {obj_names[m]} -> "
                    f"NICV={ex['NICV']:.4f}, BSTAB={ex['BSTAB_NICV']:.4f}, ε={ex['eps_spent']:.4f}"
                )

    pareto_front = non_dominated_sort(population)[0]
    epsilon_seqs = [ind['eps'] for ind in pareto_front]
    return pareto_front, epsilon_seqs




def run_experiments(
    epsilon_totals,
    norm_mode='auto',
    seed_list=[],
    result_path="./results",
    eps_min=None,
    objective_spec=None,
    HL=None, k=None):
    # hv_no_ls   = np.zeros((len(seed_list), len(epsilon_totals)))
    # hv_with_ls = np.zeros((len(seed_list), len(epsilon_totals)))

    for seed_id, seed in enumerate(seed_list):
        if USE_SEED:
            np.random.seed(seed)
            random.seed(seed)

        print(f"\n=== Trial {seed_id+1}/{len(seed_list)} ===")
        # current_seed_path = os.path.join(result_path, f"seed_{seed}")
        # os.makedirs(current_seed_path, exist_ok=True)

        hv_rows = []
        front_rows = []


        for idx, eps_total in enumerate(epsilon_totals):
            # ── εₘ, T ───────────────────────────────────────────────
            T = min(7, int(eps_total / eps_min)) if eps_total / eps_min > 2 else 2
            # print(f"εₘ={eps_min:.4f}, T={T}")

            # ── Baselines (fit on TRAIN, evaluate on TEST) ──────────
            # 1) Non-private KMeans baseline
            km = KMeans(k, n_init=10, random_state=seed).fit(X)
            cent_km = km.cluster_centers_

            nicv_km = float(calculate_nicv(X, cent_km))
            bstab_km = bootstrap_nicv_stability_kmeans(
                X=X, k=k, base_seed=seed, B=6, sample_frac=0.6, replace=False
            )


            # 2) DP-KMeans with uniform schedule ( baseline)
            init_centroids = select_initial_centroids(X, k)
            uniform_sched = np.full(T, eps_total / T, dtype=float)
            uniform_sched = enforce_budget(uniform_sched, eps_total, eps_min)
            cent_dp = dp_kmeans_train_with_schedule(X, k, init_centroids, uniform_sched)

            nicv_dp = float(calculate_nicv(X, cent_dp))
            bstab_dp = bootstrap_nicv_stability_dp(
                X=X, k=k, epsilon_seq=uniform_sched, base_seed=seed,
                B=6, sample_frac=0.6, replace=False
            )

            # 3) GA-DP baseline ( existing GA optimizes NICV; keep it for comparison)
            #    NOTE: GA currently uses NICV-based fitness on full data; for journal, you may later upgrade GA too.
            #    For now: obtain best_ga_seq from  GA, then evaluate it on train/test like above.


            best_ga_seq, best_ga_gen, best_ga_score, best_ga_fit, best_ga_extras = genetic_algorithm(
                Nm, G, eps_min, eps_total, T,
                X, k,
                base_seed=seed,
                objective_spec=objective_spec,
                ga_opt_index=0
            )

            # ── NSGA-II / MOMA fronts (tri-objective: ) ──
            front_moma, seqs_moma = moma_ga(
                X, k,
                eps_min, eps_total, T,
                Nm, G, Pm, Pc,
                local_search=True,
                base_seed=seed,
                objective_spec=objective_spec
            )

            front_nsga, seqs_nsga = moma_ga(
                X, k,
                eps_min, eps_total, T,
                Nm, G, Pm, Pc,
                local_search=False,
                base_seed=seed,
                objective_spec=objective_spec
            )

            front_amoma, seqs_amoma = amoma_ga(
                X, k,
                eps_min, eps_total, T,
                Nm, G, Pm, Pc,
                local_search=True,
                base_seed=seed,
                objective_spec=objective_spec,
                HL=HL)
            # ── record front points, including eps_1…eps_7 (float + NaN padding) ─────────

            methods = [
                ("KMeans", [{
                    "NICV": nicv_km,
                    "BSTAB_NICV": bstab_km,
                    "epsilon": float(eps_total),
                    "sched": [eps_total / T] * T
                }]),
                ("DP-KMeans", [{
                    "NICV": nicv_dp,
                    "BSTAB_NICV": bstab_dp,
                    "epsilon": float(np.sum(uniform_sched)),
                    "sched": list(uniform_sched)
                }]),
                ("GA-DP", [{
                    "NICV": best_ga_extras["NICV"],
                    "BSTAB_NICV": best_ga_extras["BSTAB_NICV"],
                    "epsilon": float(np.sum(best_ga_seq)),
                    "sched": list(best_ga_seq)
                }]),
                ("NSGA-II", [{
                    "NICV": float(ind["extras"]["NICV"]),
                    "BSTAB_NICV": float(ind["extras"]["BSTAB_NICV"]),
                    "epsilon": float(ind["extras"]["eps_spent"]),
                    "sched": list(seq)
                } for ind, seq in zip(front_nsga, seqs_nsga)]),
                ("MOMA", [{
                    "NICV": float(ind["extras"]["NICV"]),
                    "BSTAB_NICV": float(ind["extras"]["BSTAB_NICV"]),
                    "epsilon": float(ind["extras"]["eps_spent"]),
                    "sched": list(seq)
                } for ind, seq in zip(front_moma, seqs_moma)]),
                ("AMOMA", [{
                    "NICV": float(ind["extras"]["NICV"]),
                    "BSTAB_NICV": float(ind["extras"]["BSTAB_NICV"]),
                    "epsilon": float(ind["extras"]["eps_spent"]),
                    "sched": list(seq)
                } for ind, seq in zip(front_amoma, seqs_amoma)]),
            ]

            # ── One loop to record everything (preserves  old schema + adds new cols) ──
            for method, pts in methods:
                for pt_idx, pt in enumerate(pts):
                    sched = pt["sched"]

                    row = {
                        "trial": int(seed),
                        "eps_total_idx": int(idx + 3),
                        "eps_total": float(eps_total),
                        "method": method,
                        "point_idx": int(pt_idx),
                        "NICV": float(pt["NICV"]),
                        "BSTAB_NICV": float(pt["BSTAB_NICV"]),
                        "epsilon": float(pt["epsilon"]),
                        "epsilon_m": float(eps_min),
                        "T": int(T),
                    }
                    # ---- Build a fitness vector for THIS point using pt metrics ----
                    # Map  pt keys to the metric keys used in objective_spec

                    metrics_for_fitness = {
                        "nicv": float(pt["NICV"]) if pt["NICV"] is not None else np.nan,
                        "bstab": float(pt["BSTAB_NICV"]) if pt["BSTAB_NICV"] is not None else np.nan,
                        "eps": float(pt["epsilon"]) if pt["epsilon"] is not None else np.nan,
                    }
                    # If a metric is NaN but required in objective_spec (e.g., stab for baselines),

                    # if not np.isfinite(metrics_for_fitness.get("stab", np.nan)):
                    #     metrics_for_fitness["stab"] = 0.0

                    fitness_vec = build_fitness_from_metrics(metrics_for_fitness, objective_spec)
                    row["objective_spec"] = "|".join([f"{k}:{d}" for k, d in objective_spec])
                    row = attach_fitness_columns(row, fitness=fitness_vec, objective_spec=objective_spec, max_obj=5)


                    # eps_1..eps_7 are floats, pad with NaN (matches  trial_13_front_points.csv behavior)
                    for j in range(7):
                        row[f"eps_{j + 1}"] = float(sched[j]) if j < T else np.nan

                    front_rows.append(row)



        # write CSVs
        df_pts = pd.DataFrame(front_rows)
        df_path = os.path.join(result_path_spec, f"trial_{seed}_front_points.csv")
        df_pts.to_csv(df_path, index=False)



def run_for_seed(seed):
    return run_experiments(
        epsilon_totals=epsilon_totals,
        norm_mode=norm_mode,
        seed_list=[seed],
        result_path=result_path,
        eps_min=eps_min,
        objective_spec=objective_spec,
        HL = HL,
        k =k,
    )
def init_worker(X_, k_, Nm_, G_, Pc_, Pm_, allow_same_parents_, USE_SEED_, SCALER_,
                epsilon_totals_, norm_mode_, result_path_, eps_min_,objective_spec_,HL_):
    globals()['X'] = X_
    globals()['k'] = k_
    globals()['Nm'] = Nm_
    globals()['G'] = G_
    globals()['Pc'] = Pc_
    globals()['Pm'] = Pm_
    globals()['allow_same_parents'] = allow_same_parents_
    globals()['USE_SEED'] = USE_SEED_
    globals()['SCALER'] = SCALER_
    globals()['epsilon_totals'] = epsilon_totals_
    globals()['norm_mode'] = norm_mode_
    globals()['result_path'] = result_path_
    globals()['eps_min'] = eps_min_
    globals()['objective_spec'] = objective_spec_
    globals()['HL'] = HL_


if __name__ == '__main__':

    # ─────────────── Dataset & RNG ───────────────

    data_name = "heart" # "Diabetes" "heart"

    data_path = f"./dataset/{data_name}.csv"
    raw = pd.read_csv(data_path)

    # keep features only
    X = raw.drop(columns=raw.columns[-1]).to_numpy()
    X = MinMaxScaler(feature_range=(-1, 1)).fit_transform(X)

    # use only half of the dataset
    use_half_data = True
    half_data_seed = 42

    if use_half_data:
        rng = np.random.default_rng(half_data_seed)
        n = X.shape[0]
        keep_n = max(2, n // 2)
        keep_idx = rng.choice(n, size=keep_n, replace=False)
        X = X[keep_idx]

    print(f"Using X shape: {X.shape}")


    # ─────────────── Experiment Settings ───────────────

    #Parameters

    allow_same_parents = 0  # allow same parents in crossover
    use_data_driven_rho = False
    Nm = 20 # number of individuals in the population
    G = 50  # number of generations
    Pc = 0.9  # crossover probability
    Pm = 0.1  # mutation probability
    k = 2  # number of clusters
    global USE_SEED
    USE_SEED = True
    # SCALER = 10
    # HL = 1
    objective_specs = {
        # "spec_nicv_eps_acc": [('nicv', 'min'), ('eps', 'min'), ('acc', 'max')],
        # "specB_acc_eps_stab": [('acc', 'max'), ('eps', 'min'), ('stab', 'max')],
        # "specB_sil_eps_nicv": [('nicv', 'min'), ('eps', 'min'), ('sil', 'max')],
        # "spec_dbi_eps_nicv": [('nicv', 'min'), ('eps', 'min'), ('dbi', 'min')],
        "spec_nicv_eps_bstab": [('nicv', 'min'), ('bstab', 'min'),('eps', 'min')],
    }

    init_centroids = select_initial_centroids(X, k)
    rho_val = (compute_rho_from_centroids(init_centroids, 1.0)
                if use_data_driven_rho else 0.225)
    eps_min = calculate_minimum_privacy_budget(X, k, rho_val)

    interval = round(eps_min, 2)
    start = round(3*interval,2)
    stop = round(10*interval,2)

    epsilon_totals = []
    # epsilon_totals.extend([0.5,1,2,3,4])
    # # epsilon_totals.append(11) #,12,13,14,15,16,17,18,19,20
    for j in np.arange(start, stop + interval, interval):
        epsilon_totals.append(j)
    epsilon_totals = np.around(epsilon_totals, 2).tolist()


    seed_list = []

    for i in range(1, 21):
        seed_list.append(i)

    # Choose normalization‐bounds mode: "auto" or "manual"
    norm_mode = "auto"
    scaler_list = [1, 5, 10]
    hl_list = [5]

    for SCALER in scaler_list:
        for HL in hl_list:
            # (keep these globals if you rely on them elsewhere)
            folder_name = strftime("%Y%m%d_%H%M%S", localtime())
            folder_name = folder_name + f"-FormalTest-B6-Frac0.6-{data_name}_Nm{Nm}_G{G}_Pc{Pc}_Pm{Pm}_Scaler{SCALER}_HL{HL}_extension_v1.6"
            result_path = f'./result/{folder_name}'
            # os.makedirs(result_path, exist_ok=True)

            # with Pool(processes=min(len(seed_list), cpu_count())) as pool:
            #     results = pool.map(run_for_seed, seed_list)
            for spec_id, objective_spec in objective_specs.items():
                # put each spec into its own folder
                result_path_spec = os.path.join(result_path, spec_id)
                os.makedirs(result_path_spec, exist_ok=True)

                with Pool(
                        processes=min(len(seed_list), cpu_count()),
                        initializer=init_worker,
                        initargs=(X, k, Nm, G, Pc, Pm, allow_same_parents, USE_SEED, SCALER,
                                  epsilon_totals, norm_mode, result_path_spec, eps_min,objective_spec,HL)
                ) as pool:
                    results = pool.map(run_for_seed, seed_list)
                # # this path is for test:
                # result_path_spec = "./result/20260309_135613-BigTest-heart_Nm20_G50_Pc0.9_Pm0.1_Scaler10_extension_v1.2/specB_sil_eps_nicv"

                front_paths = glob.glob(os.path.join(result_path_spec, 'trial_*_front_points.csv'))
                df_front = pd.concat((pd.read_csv(p) for p in front_paths), ignore_index=True)

                # compute average of every numeric column, grouped by eps_total, method, point_idx
                df_front_avg = (
                    df_front
                    .groupby(['eps_total', 'method','point_idx'], as_index=False)
                    .mean(numeric_only=True)
                )
                # after df_front_avg computed
                label_cols = [c for c in df_front.columns if c.startswith("obj_")]

                df_labels = (
                    df_front.groupby(['eps_total', 'method', 'point_idx'])[label_cols]
                    .agg(lambda s: s.dropna().astype(str).loc[lambda x: x.str.strip().astype(bool)].mode().iloc[0]
                    if len(s.dropna()) else "")
                    .reset_index()
                )
                df_front_avg = df_front_avg.merge(df_labels, on=['eps_total', 'method', 'point_idx'], how='left')
                df_front_avg['trial'] = 'average'
                df_front_final = pd.concat([df_front,df_front_avg], ignore_index=True)
                df_front_final.to_csv(os.path.join(result_path_spec, 'final_front_points.csv'), index=False)

