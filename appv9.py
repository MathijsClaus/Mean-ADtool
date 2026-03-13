import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.stats import truncnorm
from scipy.optimize import root_scalar

#Helper functions
def rel_to_abs_location(a: float, b: float, t: float) -> float:
    t = float(np.clip(t, 0.0, 1.0))
    return a + t * (b - a)

def truncnorm_from_truncated_moments(a: float, b: float, mean_t: float, sd_t: float):
    mean_t = float(np.clip(mean_t, a + 1e-9, b - 1e-9))
    sd_t = float(max(sd_t, 1e-9))

    def moments_for(mu0, sigma0):
        alpha = (a - mu0) / sigma0
        beta = (b - mu0) / sigma0
        tn = truncnorm(alpha, beta, loc=mu0, scale=sigma0)
        return float(tn.mean()), float(tn.std())

    def sigma_for_mu(mu0):
        def f(sig):
            _, s = moments_for(mu0, sig)
            return s - sd_t

        lo = 1e-8 * max(1e-9, (b - a))
        hi = 10.0 * max(1e-9, (b - a))

        flo = f(lo)
        fhi = f(hi)
        if flo > 0:
            return float(lo)
        if fhi < 0:
            return float(hi)

        sol = root_scalar(f, bracket=[lo, hi], method="brentq")
        return float(sol.root)

    def g(mu0):
        sig = sigma_for_mu(mu0)
        m, _ = moments_for(mu0, sig)
        return m - mean_t

    lo = a - 2.0 * (b - a)
    hi = b + 2.0 * (b - a)

    try:
        sol = root_scalar(g, bracket=[lo, hi], method="brentq")
        mu0 = float(sol.root)
    except Exception:
        mu0 = float(mean_t)

    sigma0 = float(sigma_for_mu(mu0))
    return mu0, sigma0

def sample_demand(dist: str, n: int, a: float, b: float, rng: np.random.Generator, **kwargs) -> np.ndarray:
    if dist == "Uniform[a,b]":
        return rng.uniform(a, b, size=n)

    if dist == "TruncNormal":
        mean_abs = float(kwargs.get("mean_abs", 0.5 * (a + b)))
        sd_abs = float(kwargs.get("sd_abs", 0.15 * (b - a)))
        mu0, sigma0 = truncnorm_from_truncated_moments(a, b, mean_abs, sd_abs)
        alpha = (a - mu0) / sigma0
        beta = (b - mu0) / sigma0
        return truncnorm.rvs(alpha, beta, loc=mu0, scale=sigma0, size=n, random_state=rng)

    if dist == "Triangular":
        c = rel_to_abs_location(a, b, kwargs.get("mode_frac", 0.5))
        return rng.triangular(a, c, b, size=n)

    if dist == "Beta (scaled to [a,b])":
        aa = float(kwargs.get("alpha", 2.0))
        bb = float(kwargs.get("beta", 2.0))
        u = rng.beta(aa, bb, size=n)
        return a + (b - a) * u

    raise ValueError("Unknown distribution")

#Newsvendor cost

def C(q: float, D: np.ndarray, m: float, d: float) -> np.ndarray:
    return m * np.maximum(D - q, 0.0) + d * np.maximum(q - D, 0.0)

def exp_cost_from_pmf(q: float, x: np.ndarray, p: np.ndarray, m: float, d: float) -> float:
    return float(np.dot(p, C(q, x, m=m, d=d)))

#AD certificates

def empirical_ad_curve(samples: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(samples.reshape(1, -1) - grid.reshape(-1, 1)), axis=1)

def ad_certificate_from_pmf(x: np.ndarray, p: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return np.sum(p.reshape(1, -1) * np.abs(x.reshape(1, -1) - grid.reshape(-1, 1)), axis=1)

#Hoeffding radius

def hoeffding_radius(R: float, n: int, delta: float) -> float:
    delta = float(np.clip(delta, 1e-12, 1.0))
    return float(R * np.sqrt(np.log(2.0 / delta) / (2.0 * max(1, n))))

#Midpoint locations

def ad_knots_equidistant(a: float, b: float, mu: float, L: int):
    if L % 2 != 1 or L < 3:
        raise ValueError("L must be odd and >= 3.")
    k = (L - 1) // 2
    mu = float(np.clip(mu, a + 1e-12, b - 1e-12))

    left_all = np.linspace(a, mu, k + 2) 
    right_all = np.linspace(mu, b, k + 2) 

    left_knots = left_all[1:-1]
    right_knots = right_all[1:-1]

    cs = np.concatenate([left_knots, np.array([mu]), right_knots])
    x = np.concatenate([np.array([a]), left_knots, np.array([mu]), right_knots, np.array([b])])
    return cs.astype(float), x.astype(float)

#Confidence region methods and naive approach

def band_builder_paper(samples: np.ndarray, a: float, b: float, mode: str, eps: float):
    mu_hat = float(np.mean(samples))
    mad_hat = float(np.mean(np.abs(samples - mu_hat)))

    T_mu = (mu_hat - eps, mu_hat + eps)
    D_mu = (mad_hat - eps, mad_hat + eps)

    out = {
        "mu_hat": mu_hat,
        "mad_hat": mad_hat,
        "T_mu": T_mu,
        "D_mu": D_mu,
        "eps_used": float(eps),
        "band_style": "k-fold CV",
    }

    if mode == "Mean-AD3":
        c1_hat = 0.5 * (a + mu_hat)
        c2_hat = 0.5 * (mu_hat + b)

        ad1_hat = float(np.mean(np.abs(samples - c1_hat)))
        ad2_hat = float(np.mean(np.abs(samples - c2_hat)))

        Tc1 = (c1_hat - 0.5 * eps, c1_hat + 0.5 * eps)
        Tc2 = (c2_hat - 0.5 * eps, c2_hat + 0.5 * eps)

        Dc1 = (ad1_hat - eps, ad1_hat + eps)
        Dc2 = (ad2_hat - eps, ad2_hat + eps)

        da_hat = float(np.mean(np.abs(samples - a)))
        db_hat = float(np.mean(np.abs(samples - b)))
        Da = (da_hat - eps, da_hat + eps)
        Db = (db_hat - eps, db_hat + eps)

        out.update({
            "c1_hat": c1_hat, "c2_hat": c2_hat,
            "ad1_hat": ad1_hat, "ad2_hat": ad2_hat,
            "Tc1": Tc1, "Tc2": Tc2,
            "D_c1": Dc1, "D_c2": Dc2,
            "da_hat": da_hat, "db_hat": db_hat,
            "D_a": Da, "D_b": Db,
        })

    return out

def band_builder_hoeffding(samples: np.ndarray, a: float, b: float, mode: str, alpha: float):
    n = len(samples)
    R = b - a
    mu_hat = float(np.mean(samples))

    K = 2 if mode == "Mean-MAD" else 4
    delta_each = float(alpha) / max(1, K)

    r = hoeffding_radius(R, n, delta_each)
    T_mu = (mu_hat - r, mu_hat + r)

    mad_hat = float(np.mean(np.abs(samples - mu_hat)))
    D_mu = (mad_hat - 3 * r, mad_hat + 3 * r)

    out = {
        "mu_hat": mu_hat,
        "mad_hat": mad_hat,
        "T_mu": T_mu,
        "D_mu": D_mu,
        "band_style": "Conservative Hoeffding",
        "alpha": float(alpha),
        "K": int(K),
        "delta_each": float(delta_each),
        "Hoeffding radius": float(r),
    }

    if mode == "Mean-AD3":
        c1_hat = 0.5 * (a + mu_hat)
        c2_hat = 0.5 * (mu_hat + b)
        ad1_hat = float(np.mean(np.abs(samples - c1_hat)))
        ad2_hat = float(np.mean(np.abs(samples - c2_hat)))

        r3 = hoeffding_radius(R, n, delta_each)

        Dc1 = (ad1_hat - 2 * r3, ad1_hat + 2 * r3)
        Dc2 = (ad2_hat - 2 * r3, ad2_hat + 2 * r3)
        D_mu3 = (mad_hat - 3 * r3, mad_hat + 3 * r3)

        Tc1 = (c1_hat - 0.5 * r3, c1_hat + 0.5 * r3)
        Tc2 = (c2_hat - 0.5 * r3, c2_hat + 0.5 * r3)
        T_mu3 = (mu_hat - r3, mu_hat + r3)

        da_hat = float(np.mean(np.abs(samples - a)))
        db_hat = float(np.mean(np.abs(samples - b)))
        D_a3 = (da_hat - r3, da_hat + r3)
        D_b3 = (db_hat - r3, db_hat + r3)

        out.update({
            "c1_hat": c1_hat, "c2_hat": c2_hat,
            "ad1_hat": ad1_hat, "ad2_hat": ad2_hat,
            "Tc1": Tc1, "T_mu": T_mu3, "Tc2": Tc2,
            "D_c1": Dc1, "D_mu": D_mu3, "D_c2": Dc2,
            "da_hat": da_hat, "db_hat": db_hat,
            "D_a": D_a3, "D_b": D_b3,
        })

    return out

def naive_equalities(samples: np.ndarray, a: float, b: float, mode: str):
    mu_hat = float(np.mean(samples))
    mad_hat = float(np.mean(np.abs(samples - mu_hat)))

    tiny_x = 1e-6 * (b - a)
    tiny_y = 1e-6 * (b - a)

    out = {
        "mu_hat": mu_hat,
        "mad_hat": mad_hat,
        "T_mu": (mu_hat - tiny_x, mu_hat + tiny_x),
        "D_mu": (mad_hat, mad_hat),
        "band_style": "Naive estimates",
        "naive": True,
    }

    if mode == "Mean-AD3":
        c1_hat = 0.5 * (a + mu_hat)
        c2_hat = 0.5 * (mu_hat + b)

        ad1_hat = float(np.mean(np.abs(samples - c1_hat)))
        ad2_hat = float(np.mean(np.abs(samples - c2_hat)))

        da_hat = float(np.mean(np.abs(samples - a)))
        db_hat = float(np.mean(np.abs(samples - b)))

        out.update({
            "c1_hat": c1_hat, "c2_hat": c2_hat,
            "ad1_hat": ad1_hat, "ad2_hat": ad2_hat,
            "Tc1": (c1_hat - tiny_x, c1_hat + tiny_x),
            "Tc2": (c2_hat - tiny_x, c2_hat + tiny_x),
            "D_c1": (ad1_hat - tiny_y, ad1_hat + tiny_y),
            "D_c2": (ad2_hat - tiny_y, ad2_hat + tiny_y),
            "D_a": (da_hat - tiny_y, da_hat + tiny_y),
            "D_b": (db_hat - tiny_y, db_hat + tiny_y),
        })
        return out

    if mode.startswith("Mean-AD"):
        L = int(mode.split("AD")[1])
        cs, _x = ad_knots_equidistant(a, b, mu_hat, L=L)
        ad_hats = np.array([float(np.mean(np.abs(samples - float(c)))) for c in cs], dtype=float)

        da_hat = float(np.mean(np.abs(samples - a)))
        db_hat = float(np.mean(np.abs(samples - b)))

        T_cs = [(float(c - tiny_x), float(c + tiny_x)) for c in cs]
        D_cs = [(float(ad - tiny_y), float(ad + tiny_y)) for ad in ad_hats]

        mid = len(cs) // 2
        out.update({
            "L": int(L),
            "c_hats": cs,
            "ad_hats": ad_hats,
            "T_cs": T_cs,
            "D_cs": D_cs,
            "D_a": (da_hat - tiny_y, da_hat + tiny_y),
            "D_b": (db_hat - tiny_y, db_hat + tiny_y),
            "D_mu": (float(ad_hats[mid]), float(ad_hats[mid])),
            "mad_hat": float(ad_hats[mid]),
        })
        return out

    return out

#Worst case distributions

def wc_mean_mad_interval(a: float, b: float, l_mu: float, u_mu: float, u0: float, q: float, m: float, d: float):
    R = b - a

    def delta_max(mu_star: float):
        return 2.0 * (mu_star - a) * (b - mu_star) / R

    def probs(mu_star: float):
        delta_star = min(u0, delta_max(mu_star))
        pa = delta_star / (2.0 * (mu_star - a))
        pb = delta_star / (2.0 * (b - mu_star))
        pm = 1.0 - pa - pb
        return np.array([pa, pm, pb], dtype=float), float(delta_star)

    cand = []
    for mu_star in [l_mu, u_mu]:
        if not (a < mu_star < b):
            continue
        p, _ = probs(mu_star)
        if np.any(p < -1e-9):
            continue
        p = np.clip(p, 0.0, 1.0)
        p = p / p.sum()
        x = np.array([a, mu_star, b], dtype=float)
        val = exp_cost_from_pmf(q, x, p, m=m, d=d)
        cand.append((val, x, p))

    if not cand:
        return {"ok": False}

    cand.sort(key=lambda t: t[0], reverse=True)
    val, x, p = cand[0]
    return {"ok": True, "x": x, "p": p, "value": float(val)}

def wc_mean_mad_naive(a: float, b: float, mu_hat: float, mad_hat: float, q: float, m: float, d: float):
    if not (a < mu_hat < b):
        return {"ok": False}

    pa = mad_hat / (2.0 * (mu_hat - a))
    pb = mad_hat / (2.0 * (b - mu_hat))
    pm = 1.0 - pa - pb

    p = np.array([pa, pm, pb], dtype=float)
    if np.any(p < -1e-9):
        return {"ok": False}

    p = np.clip(p, 0.0, 1.0)
    p = p / p.sum()
    x = np.array([a, mu_hat, b], dtype=float)
    val = exp_cost_from_pmf(q, x, p, m=m, d=d)
    return {"ok": True, "x": x, "p": p, "value": float(val)}

def wc_mean_ad3_interval(a: float, b: float, l_mu: float, u_mu: float, u0: float, u1: float, u2: float, q: float, m: float, d: float):
    R = b - a

    def solve_p(mu_val: float):
        if not (a < mu_val < b):
            return None
        c1 = 0.5 * (a + mu_val)
        c2 = 0.5 * (mu_val + b)

        delta0 = min(u0, 2.0 * (mu_val - a) * (b - mu_val) / R)
        delta1 = min(u1, delta0 + 0.5 * (mu_val - a))
        delta2 = min(u2, delta0 + 0.5 * (b - mu_val))

        A = np.array([
            [1, 1, 1, 1, 1],
            [a, c1, mu_val, c2, b],
            [c1 - a, 0, mu_val - c1, c2 - c1, b - c1],
            [mu_val - a, mu_val - c1, 0, c2 - mu_val, b - mu_val],
            [c2 - a, c2 - c1, c2 - mu_val, 0, b - c2],
        ], dtype=float)
        dvec = np.array([1.0, mu_val, delta1, delta0, delta2], dtype=float)

        try:
            p = np.linalg.solve(A, dvec)
        except np.linalg.LinAlgError:
            return None

        if np.any(p < -1e-8):
            return None

        p = np.clip(p, 0.0, 1.0)
        p = p / p.sum()
        x = np.array([a, c1, mu_val, c2, b], dtype=float)
        val = exp_cost_from_pmf(q, x, p, m=m, d=d)
        return float(val), x, p

    cands = [l_mu, u_mu, q, 2.0 * q - a, 2.0 * q - b]
    cands = [float(mu) for mu in cands if (l_mu <= float(mu) <= u_mu)]
    cands = sorted(set(cands))

    best = None
    for mu_val in cands:
        out = solve_p(mu_val)
        if out is None:
            continue
        if (best is None) or (out[0] > best[0]):
            best = out

    if best is None:
        return {"ok": False}

    val, x, p = best
    return {"ok": True, "x": x, "p": p, "value": float(val)}

def wc_mean_ad3_naive(a: float, b: float, mu_hat: float, ad0_hat: float, ad1_hat: float, ad2_hat: float, q: float, m: float, d: float):
    if not (a < mu_hat < b):
        return {"ok": False}

    c1 = 0.5 * (a + mu_hat)
    c2 = 0.5 * (mu_hat + b)

    A = np.array([
        [1, 1, 1, 1, 1],
        [a, c1, mu_hat, c2, b],
        [c1 - a, 0, mu_hat - c1, c2 - c1, b - c1],
        [mu_hat - a, mu_hat - c1, 0, c2 - mu_hat, b - mu_hat],
        [c2 - a, c2 - c1, c2 - mu_hat, 0, b - c2],
    ], dtype=float)
    dvec = np.array([1.0, mu_hat, ad1_hat, ad0_hat, ad2_hat], dtype=float)

    try:
        p = np.linalg.solve(A, dvec)
    except np.linalg.LinAlgError:
        return {"ok": False}

    if np.any(p < -1e-8):
        return {"ok": False}

    p = np.clip(p, 0.0, 1.0)
    p = p / p.sum()
    x = np.array([a, c1, mu_hat, c2, b], dtype=float)
    val = exp_cost_from_pmf(q, x, p, m=m, d=d)
    return {"ok": True, "x": x, "p": p, "value": float(val)}

def wc_mean_ad_general_naive(a: float, b: float, mu_hat: float, L: int, ad_hats: np.ndarray, q: float, m: float, d: float):
    if not (a < mu_hat < b):
        return {"ok": False}

    cs, x = ad_knots_equidistant(a, b, mu_hat, L=L)
    if len(ad_hats) != len(cs):
        return {"ok": False}

    n = len(x)  # L+2
    A = np.zeros((n, n), dtype=float)
    dvec = np.zeros(n, dtype=float)

    A[0, :] = 1.0
    dvec[0] = 1.0

    A[1, :] = x
    dvec[1] = float(mu_hat)

    for i, c in enumerate(cs):
        A[2 + i, :] = np.abs(x - float(c))
        dvec[2 + i] = float(ad_hats[i])

    try:
        p = np.linalg.solve(A, dvec)
    except np.linalg.LinAlgError:
        return {"ok": False}

    if np.any(p < -1e-8):
        return {"ok": False}

    p = np.clip(p, 0.0, 1.0)
    p = p / p.sum()

    val = exp_cost_from_pmf(q, x, p, m=m, d=d)
    return {"ok": True, "x": x, "p": p, "value": float(val)}

#K fold CV 

def cv_select_C1(samples: np.ndarray, a: float, b: float, mode: str, m: float, d: float,
                 C1_candidates: list[float], q_grid: np.ndarray, seed: int = 23):
    N = len(samples)
    k = min(N, 5)

    rng = np.random.default_rng(seed)
    idx = np.arange(N)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)

    scores: dict[float, float] = {}
    infeas_detail: dict[float, int] = {}

    for C1 in C1_candidates:
        fold_losses = []
        infeas_folds = 0

        for i in range(k):
            val_idx = folds[i]
            tr_idx = np.concatenate([folds[j] for j in range(k) if j != i])
            subtrain = samples[tr_idx]
            val = samples[val_idx]

            eps = float(C1) / np.sqrt(max(1, len(subtrain)))
            dd = band_builder_paper(subtrain, a, b, mode=mode, eps=eps)

            l_mu, u_mu = dd["T_mu"]
            u0 = dd["D_mu"][1]

            best_q = None
            best_wc_val = None

            if mode == "Mean-MAD":
                for qq in q_grid:
                    wc = wc_mean_mad_interval(a, b, l_mu, u_mu, u0, float(qq), m=m, d=d)
                    if not wc.get("ok", False):
                        continue
                    if best_wc_val is None or wc["value"] < best_wc_val:
                        best_wc_val = wc["value"]
                        best_q = float(qq)
            else:
                u1 = dd["D_c1"][1]
                u2 = dd["D_c2"][1]
                for qq in q_grid:
                    wc = wc_mean_ad3_interval(a, b, l_mu, u_mu, u0, u1, u2, float(qq), m=m, d=d)
                    if not wc.get("ok", False):
                        continue
                    if best_wc_val is None or wc["value"] < best_wc_val:
                        best_wc_val = wc["value"]
                        best_q = float(qq)

            if best_q is None:
                infeas_folds += 1
                continue

            fold_losses.append(float(np.mean(C(best_q, val, m=m, d=d))))

        infeas_detail[float(C1)] = int(infeas_folds)
        scores[float(C1)] = float("inf") if (infeas_folds > 0 or len(fold_losses) == 0) else float(np.mean(fold_losses))

    finite = {c: s for c, s in scores.items() if np.isfinite(s)}
    C1_best = min(finite.keys(), key=lambda c: finite[c]) if finite else min(C1_candidates)
    return float(C1_best), scores, infeas_detail, int(k)

#True underlying distribution through MC approximation

def compute_true_situation(dist: str, dist_kwargs: dict, a: float, b: float, n_true: int, seed_true: int, mode: str):
    rng = np.random.default_rng(int(seed_true))
    D = sample_demand(dist, int(n_true), a, b, rng, **dist_kwargs)

    mu = float(np.mean(D))
    mad = float(np.mean(np.abs(D - mu)))

    out = {"mu": mu, "mad": mad}

    if mode == "Mean-AD3":
        c1 = 0.5 * (a + mu)
        c2 = 0.5 * (mu + b)
        ad1 = float(np.mean(np.abs(D - c1)))
        ad2 = float(np.mean(np.abs(D - c2)))
        out.update({"c1": c1, "c2": c2, "ad1": ad1, "ad2": ad2})
        return out

    if mode.startswith("Mean-AD"):
        L = int(mode.split("AD")[1])
        cs, _x = ad_knots_equidistant(a, b, mu, L=L)
        ads = np.array([float(np.mean(np.abs(D - float(c)))) for c in cs], dtype=float)
        out.update({"L": int(L), "cs": cs, "ads": ads})
        out["mad"] = float(ads[len(cs) // 2])
        return out

    return out

#Plots

def plot_regions_and_certs(samples: np.ndarray, a: float, b: float, dd: dict, mode: str, wc: dict,
                           show_support: bool, true_info: dict | None, true_cert_curve: np.ndarray | None, grid: np.ndarray) -> go.Figure:
    y_emp = empirical_ad_curve(samples, grid)
    fig = go.Figure()

    if true_cert_curve is not None:
        fig.add_trace(go.Scatter(x=grid, y=true_cert_curve, mode="lines",
                                 name="True underlying AD certificate.", opacity=0.25))

    fig.add_trace(go.Scatter(x=grid, y=y_emp, mode="lines", name="Empirical AD certificate.", opacity=1.0))

    if wc.get("ok", False):
        y_wc = ad_certificate_from_pmf(wc["x"], wc["p"], grid)
        fig.add_trace(go.Scatter(x=grid, y=y_wc, mode="lines", name="Worst case AD certificate."))

    def add_rect(x0, x1, y0, y1, fill="rgba(0,0,0,0.08)"):
        fig.add_shape(
            type="rect",
            x0=x0, x1=x1, y0=y0, y1=y1,
            xref="x", yref="y",
            line=dict(width=2),
            fillcolor=fill,
            layer="below",
        )

    ys = [float(y_emp.min()), float(y_emp.max()), float(dd["D_mu"][0]), float(dd["D_mu"][1])]
    if true_cert_curve is not None:
        ys += [float(np.min(true_cert_curve)), float(np.max(true_cert_curve))]

    if mode == "Mean-AD3":
        ys += [float(dd["D_c1"][0]), float(dd["D_c1"][1]), float(dd["D_c2"][0]), float(dd["D_c2"][1])]
    elif mode.startswith("Mean-AD"):
        for (y0, y1) in dd.get("D_cs", []):
            ys += [float(y0), float(y1)]

    y_min, y_max = min(ys), max(ys)
    yr = max(1e-9, y_max - y_min)

    if mode == "Mean-MAD":
        add_rect(dd["T_mu"][0], dd["T_mu"][1], dd["D_mu"][0], dd["D_mu"][1])
    elif mode == "Mean-AD3":
        add_rect(dd["Tc1"][0], dd["Tc1"][1], dd["D_c1"][0], dd["D_c1"][1])
        add_rect(dd["T_mu"][0], dd["T_mu"][1], dd["D_mu"][0], dd["D_mu"][1])
        add_rect(dd["Tc2"][0], dd["Tc2"][1], dd["D_c2"][0], dd["D_c2"][1])
    else:
        for (Tc, Dc) in zip(dd.get("T_cs", []), dd.get("D_cs", [])):
            add_rect(float(Tc[0]), float(Tc[1]), float(Dc[0]), float(Dc[1]))

    if true_info is not None:
        if mode == "Mean-MAD":
            fig.add_trace(go.Scatter(x=[true_info["mu"]], y=[true_info["mad"]],
                                     mode="markers", name="True (μ, MAD)", opacity=0.25, marker=dict(size=10)))
        elif mode == "Mean-AD3":
            xs = [true_info["c1"], true_info["mu"], true_info["c2"]]
            ys_ = [true_info["ad1"], true_info["mad"], true_info["ad2"]]
            fig.add_trace(go.Scatter(x=xs, y=ys_, mode="markers",
                                     name="True underlying values.", opacity=0.25, marker=dict(size=9)))
        else:
            xs = list(map(float, true_info.get("cs", [])))
            ys_ = list(map(float, true_info.get("ads", [])))
            fig.add_trace(go.Scatter(x=xs, y=ys_, mode="markers",
                                     name="True underlying values.", opacity=0.25, marker=dict(size=9)))

    if mode == "Mean-MAD":
        fig.add_trace(go.Scatter(x=[dd["mu_hat"]], y=[dd["mad_hat"]],
                                 mode="markers", name="Empirical (μ̂, MAD̂)", opacity=1.0, marker=dict(size=10)))
    elif mode == "Mean-AD3":
        xs = [dd["c1_hat"], dd["mu_hat"], dd["c2_hat"]]
        ys_ = [dd["ad1_hat"], dd["mad_hat"], dd["ad2_hat"]]
        fig.add_trace(go.Scatter(x=xs, y=ys_, mode="markers",
                                 name="Empirical values.", opacity=1.0, marker=dict(size=9)))
    else:
        xs = list(map(float, dd.get("c_hats", [])))
        ys_cs = [float(y0 + (y1 - y0) / 2.0) for (y0, y1) in dd.get("D_cs", [])]
        fig.add_trace(go.Scatter(x=xs, y=ys_cs, mode="markers",
                                 name="Empirical values.", opacity=1.0, marker=dict(size=9)))

    fig.update_layout(
        title=f"AD-certificates: True vs Empirical vs Worst case (regions: {dd.get('band_style','')})",
        xaxis_title="c (knot/location)",
        yaxis_title="delta(c) = E|X-c|",
        height=720,
        margin=dict(l=20, r=20, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.update_yaxes(range=[y_min - 0.10 * yr, y_max + 0.10 * yr])
    return fig

def plot_costs_true_emp_wc(dist: str, dist_kwargs: dict, a: float, b: float, m: float, d: float,
                           wc: dict, q_grid: np.ndarray, true_mc_n: int, true_mc_seed: int, empirical_samples: np.ndarray) -> go.Figure:
    rng = np.random.default_rng(int(true_mc_seed))
    D_true = sample_demand(dist, int(true_mc_n), a, b, rng, **dist_kwargs)
    true_vals = np.array([float(np.mean(C(float(q), D_true, m=m, d=d))) for q in q_grid], dtype=float)
    emp_vals = np.array([float(np.mean(C(float(q), empirical_samples, m=m, d=d))) for q in q_grid], dtype=float)

    wc_vals = None
    if wc.get("ok", False):
        x = wc["x"]
        p = wc["p"]
        wc_vals = np.array([exp_cost_from_pmf(float(q), x, p, m=m, d=d) for q in q_grid], dtype=float)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q_grid, y=true_vals, mode="lines", name="True expected cost.", opacity=0.25))
    fig.add_trace(go.Scatter(x=q_grid, y=emp_vals, mode="lines", name="Empirical expected cost.", opacity=1.0))
    if wc_vals is not None:
        fig.add_trace(go.Scatter(x=q_grid, y=wc_vals, mode="lines", name="Worst case expected cost.", opacity=1.0))

    fig.update_layout(
        title="Costs: True vs Empirical vs Worst case",
        xaxis_title="Order quantity q",
        yaxis_title="Expected cost",
        height=560,
        margin=dict(l=20, r=20, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig

#Ordering policy

def saa_policy(samples: np.ndarray, q_grid: np.ndarray, m: float, d: float) -> tuple[float, float]:
    vals = np.array([float(np.mean(C(float(q), samples, m=m, d=d))) for q in q_grid], dtype=float)
    j = int(np.argmin(vals))
    return float(q_grid[j]), float(vals[j])

def robust_policy(
    samples: np.ndarray,
    a: float,
    b: float,
    mode: str,
    band_method: str,
    m: float,
    d: float,
    q_grid: np.ndarray,
    alpha: float | None = None,
    cv_use: bool | None = None,
    C1_candidates: list[float] | None = None,
    q_grid_cv: np.ndarray | None = None,
    C1_manual: float | None = None,
    return_meta: bool = False,
):
    #Only naive estimation for Mean-AD5, 7 ...
    if mode in ["Mean-AD5", "Mean-AD7", "Mean-AD9"]:
        band_method = "Naive estimation"

    C1_used = None
    eps_used = None

    if band_method == "Naive estimation":
        dd = naive_equalities(samples, a, b, mode=mode)

    elif band_method == "k-fold cv":
        if (cv_use is False) and (C1_manual is not None):
            C1_used = float(C1_manual)
            eps_used = float(C1_used) / np.sqrt(max(1, len(samples)))
            dd = band_builder_paper(samples, a, b, mode=mode, eps=eps_used)
        else:
            C1_used = 0.1
            if cv_use and (C1_candidates is not None) and (q_grid_cv is not None):
                C1_best, _, _, _ = cv_select_C1(
                    samples=samples, a=a, b=b, mode=mode, m=m, d=d,
                    C1_candidates=C1_candidates, q_grid=q_grid_cv, seed=28
                )
                C1_used = float(C1_best)
            eps_used = float(C1_used) / np.sqrt(max(1, len(samples)))
            dd = band_builder_paper(samples, a, b, mode=mode, eps=eps_used)

    else:
        dd = band_builder_hoeffding(samples, a, b, mode=mode, alpha=float(alpha))

    best_q = None
    best_val = None

    if band_method == "Naive estimation":
        if mode == "Mean-MAD":
            for q in q_grid:
                wc = wc_mean_mad_naive(a, b, dd["mu_hat"], dd["mad_hat"], float(q), m=m, d=d)
                if not wc.get("ok", False):
                    continue
                if best_val is None or wc["value"] < best_val:
                    best_val = wc["value"]
                    best_q = float(q)

        elif mode == "Mean-AD3":
            for q in q_grid:
                wc = wc_mean_ad3_naive(
                    a, b,
                    mu_hat=dd["mu_hat"],
                    ad0_hat=dd["mad_hat"],
                    ad1_hat=dd["ad1_hat"],
                    ad2_hat=dd["ad2_hat"],
                    q=float(q), m=m, d=d
                )
                if not wc.get("ok", False):
                    continue
                if best_val is None or wc["value"] < best_val:
                    best_val = wc["value"]
                    best_q = float(q)

        else:
            L = int(mode.split("AD")[1])
            ad_hats = np.array(dd["ad_hats"], dtype=float)
            for q in q_grid:
                wc = wc_mean_ad_general_naive(a, b, dd["mu_hat"], L=L, ad_hats=ad_hats, q=float(q), m=m, d=d)
                if not wc.get("ok", False):
                    continue
                if best_val is None or wc["value"] < best_val:
                    best_val = wc["value"]
                    best_q = float(q)

    else:
        l_mu, u_mu = dd["T_mu"]
        u0 = dd["D_mu"][1]

        if mode == "Mean-MAD":
            for q in q_grid:
                wc = wc_mean_mad_interval(a, b, float(l_mu), float(u_mu), float(u0), float(q), m=m, d=d)
                if not wc.get("ok", False):
                    continue
                if best_val is None or wc["value"] < best_val:
                    best_val = wc["value"]
                    best_q = float(q)

        else:
            u1 = dd["D_c1"][1]
            u2 = dd["D_c2"][1]
            for q in q_grid:
                wc = wc_mean_ad3_interval(a, b, float(l_mu), float(u_mu), float(u0), float(u1), float(u2), float(q), m=m, d=d)
                if not wc.get("ok", False):
                    continue
                if best_val is None or wc["value"] < best_val:
                    best_val = wc["value"]
                    best_q = float(q)

    ok = best_q is not None

    if not return_meta:
        if not ok:
            return None
        return float(best_q), float(best_val)

    return {
        "ok": bool(ok),
        "q": (float(best_q) if ok else np.nan),
        "wc_val": (float(best_val) if ok else np.nan),
        "C1_used": (float(C1_used) if C1_used is not None else np.nan),
        "eps_used": (float(eps_used) if eps_used is not None else np.nan),
        "band_method": band_method,
        "mode": mode,
    }

#OOS experiment for SAA and robust methods
def oos_experiment(dist: str, dist_kwargs: dict, a: float, b: float,
                   mode: str, band_method: str,
                   N_train: int, N_test: int, trials: int,
                   m: float, d: float, q_grid: np.ndarray,
                   seed0: int,
                   alpha: float | None,
                   cv_use: bool | None,
                   C1_candidates: list[float] | None,
                   q_grid_cv: np.ndarray | None,
                   C1_manual: float | None):
    robust_costs = []
    saa_costs = []
    dropped = 0

    for t in range(int(trials)):
        rng_tr = np.random.default_rng(int(seed0) + 10_000 * t + 17)
        rng_te = np.random.default_rng(int(seed0) + 10_000 * t + 911)

        train = sample_demand(dist, int(N_train), a, b, rng_tr, **dist_kwargs)
        test = sample_demand(dist, int(N_test), a, b, rng_te, **dist_kwargs)

        q_saa, _ = saa_policy(train, q_grid=q_grid, m=m, d=d)
        cost_saa = float(np.mean(C(q_saa, test, m=m, d=d)))

        rob = robust_policy(
            samples=train, a=a, b=b, mode=mode, band_method=band_method,
            m=m, d=d, q_grid=q_grid,
            alpha=alpha if band_method == "Conservative Hoeffding" else None,
            cv_use=cv_use if band_method == "k-fold cv" else None,
            C1_candidates=C1_candidates if band_method == "k-fold cv" else None,
            q_grid_cv=q_grid_cv if (band_method == "k-fold cv" and cv_use) else None,
            C1_manual=C1_manual if band_method == "k-fold cv" else None,
            return_meta=False,
        )

        if rob is None:
            dropped += 1
            continue

        q_rob, _ = rob
        cost_rob = float(np.mean(C(q_rob, test, m=m, d=d)))

        saa_costs.append(cost_saa)
        robust_costs.append(cost_rob)

    return np.array(robust_costs, dtype=float), np.array(saa_costs, dtype=float), int(dropped)

def boxplot_compare(robust_costs: np.ndarray, saa_costs: np.ndarray, robust_label: str, saa_label: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Box(y=robust_costs, name=robust_label, boxpoints="outliers"))
    fig.add_trace(go.Box(y=saa_costs, name=saa_label, boxpoints="outliers"))
    fig.update_layout(
        title="Out-of-sample average cost over 10.000 trials",
        yaxis_title="Average cost on test data",
        height=520,
        margin=dict(l=20, r=20, t=70, b=45),
    )
    return fig

#Comparison vs SAA over different N
def make_N_grid():
    out = []
    out += list(range(2, 12, 2))        # 2,4,6,8,10
    out += list(range(15, 55, 5))       # 15,20,...,50
    out += list(range(60, 110, 10))     # 60,70,...,100
    out += list(range(120, 201, 20))    # 120,140,...,200
    return out

def ambiguity_choice(label: str):
    if label.startswith("Mean-MAD"):
        mode = "Mean-MAD"
    elif label.startswith("Mean-AD3"):
        mode = "Mean-AD3"
    elif label.startswith("Mean-AD5"):
        mode = "Mean-AD5"
    elif label.startswith("Mean-AD7"):
        mode = "Mean-AD7"
    else:
        mode = "Mean-AD9"

    if "(naive)" in label:
        band_method = "Naive estimation"
    elif "banded: k-fold cv" in label:
        band_method = "k-fold cv"
    else:
        band_method = "Conservative Hoeffding"

    if mode in ["Mean-AD5", "Mean-AD7", "Mean-AD9"]:
        band_method = "Naive estimation"

    return mode, band_method

def one_trial_multi_series(
    dist: str, dist_kwargs: dict,
    a: float, b: float,
    series_specs: list[dict],
    N_train: int, N_test: int,
    m: float, d: float,
    q_grid: np.ndarray,
    seed_train: int, seed_test: int,
):
    rng_tr = np.random.default_rng(int(seed_train))
    rng_te = np.random.default_rng(int(seed_test))

    train = sample_demand(dist, int(N_train), a, b, rng_tr, **dist_kwargs)
    test = sample_demand(dist, int(N_test), a, b, rng_te, **dist_kwargs)

    q_saa, _ = saa_policy(train, q_grid=q_grid, m=m, d=d)
    cost_saa = float(np.mean(C(q_saa, test, m=m, d=d)))

    costs = {}    
    qs = {}         
    c1_used = {}   
    eps_used = {}  

    for spec in series_specs:
        label = spec["label"]
        mode = spec["mode"]
        band_method = spec["band_method"]

        meta = robust_policy(
            samples=train, a=a, b=b, mode=mode, band_method=band_method,
            m=m, d=d, q_grid=q_grid,
            alpha=spec.get("alpha", None),
            cv_use=spec.get("cv_use", None),
            C1_candidates=spec.get("C1_candidates", None),
            q_grid_cv=spec.get("q_grid_cv", None),
            C1_manual=spec.get("C1_manual", None),
            return_meta=True,
        )

        if not meta["ok"]:
            costs[label] = np.nan
            qs[label] = np.nan
            c1_used[label] = np.nan
            eps_used[label] = np.nan
        else:
            q_rob = float(meta["q"])
            qs[label] = q_rob
            costs[label] = float(np.mean(C(q_rob, test, m=m, d=d)))
            c1_used[label] = float(meta["C1_used"])
            eps_used[label] = float(meta["eps_used"])

    return cost_saa, costs, qs, c1_used, eps_used

def sweep_improvement_vs_saa(
    dist: str, dist_kwargs: dict,
    a: float, b: float,
    series_specs: list[dict],
    N_list: list[int],
    N_test: int,
    trials: int,
    m: float, d: float,
    q_grid: np.ndarray,
    seed0: int,
    st_progress=None,
):
    labels = [s["label"] for s in series_specs]
    mean_imp = {lab: [] for lab in labels}
    p95_imp = {lab: [] for lab in labels}
    kept = {lab: [] for lab in labels}

    total_steps = len(N_list) * max(1, len(series_specs))
    step = 0

    for N_train in N_list:
        for spec in series_specs:
            robust_costs, saa_costs, dropped = oos_experiment(
                dist=dist, dist_kwargs=dist_kwargs,
                a=a, b=b,
                mode=spec["mode"], band_method=spec["band_method"],
                N_train=int(N_train),
                N_test=int(N_test),
                trials=int(trials),
                m=m, d=d,
                q_grid=q_grid,
                seed0=int(seed0),
                alpha=spec.get("alpha", None) if spec["band_method"] == "Conservative Hoeffding" else None,
                cv_use=spec.get("cv_use", None) if spec["band_method"] == "k-fold cv" else None,
                C1_candidates=spec.get("C1_candidates", None) if spec["band_method"] == "k-fold cv" else None,
                q_grid_cv=spec.get("q_grid_cv", None) if (spec["band_method"] == "k-fold cv" and spec.get("cv_use", None)) else None,
                C1_manual=spec.get("C1_manual", None) if spec["band_method"] == "k-fold cv" else None,
            )

            lab = spec["label"]
            kept[lab].append(int(len(robust_costs)))

            if len(robust_costs) == 0 or len(saa_costs) == 0:
                mean_imp[lab].append(np.nan)
                p95_imp[lab].append(np.nan)
            else:
                imp = (saa_costs - robust_costs) / saa_costs * 100.0
                mean_imp[lab].append(float(np.mean(imp)))
                p95_imp[lab].append(float(np.percentile(imp, 95)))

            step += 1
            if st_progress is not None:
                st_progress.progress(min(1.0, step / max(1, total_steps)))

    return mean_imp, p95_imp, kept

def plot_improvement_lines(N_list: list[int], series: dict[str, list[float]], title: str, ylab: str) -> go.Figure:
    fig = go.Figure()
    for label, ys in series.items():
        fig.add_trace(go.Scatter(x=N_list, y=ys, mode="lines+markers", name=label))
    fig.update_layout(
        title=title,
        xaxis_title="Training size N",
        yaxis_title=ylab,
        height=560,
        margin=dict(l=20, r=20, t=70, b=45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    fig.add_hline(y=0.0, line_dash="dot")
    return fig

def parse_candidate_text(s: str, fallback: list[float]) -> list[float]:
    try:
        vals = [float(x.strip()) for x in str(s).split(',') if x.strip() != '']
        vals = list(dict.fromkeys(vals))
        if len(vals) == 0:
            return list(fallback)
        return vals
    except Exception:
        return list(fallback)

def manual_candidate_default(cands: list[float], preferred: float = 0.1) -> float:
    if preferred in cands:
        return float(preferred)
    return float(cands[0])

#Streamlit environment
st.set_page_config(page_title="Interactive Worst Case Test Environment", layout="wide")
st.title("Mean-MAD and Mean-AD certificates and bounds")

with st.sidebar:
    st.header("Ambiguity set (method included)")
    main_choice = st.selectbox(
        "Pick ambiguity set",
        [
            "Mean-MAD (naive)",
            "Mean-MAD (banded: k-fold cv)",
            "Mean-MAD (banded: Hoeffding)",
            "Mean-AD3 (naive)",
            "Mean-AD3 (banded: k-fold cv)",
            "Mean-AD3 (banded: Hoeffding)",
            "Mean-AD5 (naive)",
            "Mean-AD7 (naive)",
            "Mean-AD9 (naive)",
        ],
    )
    mode, band_method = ambiguity_choice(main_choice)

    st.header("Support")
    a = st.number_input("Lower bound a", value=10.0)
    b = st.number_input("Upper bound b", value=50.0)
    if b <= a:
        st.error("b > a required")

    st.header("True data-generating distribution")
    dist = st.selectbox("Underlying distribution", ["Uniform[a,b]", "TruncNormal", "Triangular", "Beta (scaled to [a,b])"])

    dist_kwargs = {}
    if dist == "TruncNormal":
        mean_abs = st.slider("TruncNormal mean (absolute)", float(a), float(b), float(0.5 * (a + b)), 0.01)
        sd_abs = st.slider("TruncNormal sd (absolute)", 0.001, float(0.5 * (b - a)), float(0.15 * (b - a)), 0.001)
        dist_kwargs["mean_abs"] = float(mean_abs)
        dist_kwargs["sd_abs"] = float(sd_abs)
    elif dist == "Triangular":
        dist_kwargs["mode_frac"] = st.slider("Triangular mode (fraction in [0,1])", 0.0, 1.0, 0.5, 0.01)
    elif dist == "Beta (scaled to [a,b])":
        dist_kwargs["alpha"] = st.number_input("Beta alpha", value=2.0)
        dist_kwargs["beta"] = st.number_input("Beta beta", value=2.0)

    st.header("Observed samples")
    n = st.slider("Number of observations N", min_value=2, max_value=1500, value=50, step=1)
    seed = st.number_input("Random seed.", value=1, step=1)

    st.header("Newsvendor cost parameters")
    m = st.number_input("m (underage/shortage penalty)", value=1.0)
    d = st.number_input("d (overage/holding penalty)", value=1.0)

    st.divider()
    st.header("Grid for cost plot")
    q_grid_n = st.slider("Number of q grid points", min_value=51, max_value=2001, value=601, step=50)

    st.divider()
    st.header("MC approximation accuracy")
    true_mc_n = st.slider("MC size for true approximation", min_value=5000, max_value=500000, value=120000, step=5000)
    true_mc_seed = st.number_input("MC random seed", value=23, step=1)

    st.divider()
    st.header("AD certificate plot settings")
    cert_grid_points = st.slider("Number of grid points for certificate plot", min_value=200, max_value=4000, value=1200, step=100)

    use_cv = None
    C1_candidates = None
    C1_used = None
    q_grid_cv = None
    alpha = None

    if band_method == "k-fold cv":
        st.header("k-fold CV settings (main)")
        C1_candidates_default = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
        C1_candidates_text = st.text_input(
            "C1 candidates (comma-separated)",
            value=", ".join(map(str, C1_candidates_default)),
            key="main_c1_candidates_text",
        )
        C1_candidates = parse_candidate_text(C1_candidates_text, C1_candidates_default)
        if C1_candidates_text.strip() == "":
            st.warning("Using default C1 candidates.")
        use_cv = st.checkbox("Select C1 via k-fold CV", value=True, key="main_usecv")
        if not use_cv:
            C1_used = float(st.select_slider("Pick C1 manually", options=C1_candidates, value=manual_candidate_default(C1_candidates), key="main_c1"))
        else:
            q_grid_cv_n = st.slider("CV: q-grid size", min_value=21, max_value=301, value=61, step=10, key="main_qcvn")
            q_grid_cv = np.linspace(float(a), float(b), int(q_grid_cv_n))

    if band_method == "Conservative Hoeffding":
        st.header("Hoeffding settings (main)")
        alpha = st.slider("Overall alpha (1 - confidence)", min_value=0.01, max_value=0.20, value=0.05, step=0.01, key="main_alpha")

    st.divider()
    st.header("Out-of-sample comparison")
    do_oos = st.checkbox("Run out-of-sample performance experiment", value=False)
    if do_oos:
        oos_trials = st.slider("Number of trials", min_value=10, max_value=500, value=50, step=10)
        oos_test_n = st.selectbox("Test sample size", [10000], index=0)
        oos_q_grid_n = st.slider("OOS: q-grid size (smaller = faster)", min_value=51, max_value=1001, value=201, step=50)

    st.divider()
    st.header("Out-of-sample performance vs SAA for different N")
    do_sweep = st.checkbox("Run performance experiment (mean & 95th percentile)", value=False)

    if do_sweep:
        sweep_options = [
            "Mean-MAD (naive)",
            "Mean-MAD (banded: k-fold cv)",
            "Mean-MAD (banded: Hoeffding)",
            "Mean-AD3 (naive)",
            "Mean-AD3 (banded: k-fold cv)",
            "Mean-AD3 (banded: Hoeffding)",
            "Mean-AD5 (naive)",
            "Mean-AD7 (naive)",
            "Mean-AD9 (naive)",
        ]
        sweep_choices = st.multiselect(
            "Select series to compare (each series includes method)",
            options=sweep_options,
            default=[],
            key="sweep_choices",
        )

        sweep_trials = st.slider("Trials per N", min_value=10, max_value=500, value=50, step=10)
        sweep_test_n = st.selectbox("Sweep test size (fixed)", [10000], index=0)
        sweep_q_grid_n = st.slider("Sweep: q-grid size (smaller = faster)", min_value=31, max_value=801, value=151, step=20)

        needs_mad_cv = any("Mean-MAD (banded: k-fold cv)" == s for s in sweep_choices)
        needs_ad3_cv = any("Mean-AD3 (banded: k-fold cv)" == s for s in sweep_choices)
        needs_mad_h = any("Mean-MAD (banded: Hoeffding)" == s for s in sweep_choices)
        needs_ad3_h = any("Mean-AD3 (banded: Hoeffding)" == s for s in sweep_choices)

        sweep_mad_cv_use = None
        sweep_mad_C1_candidates = None
        sweep_mad_C1_manual = None

        sweep_ad3_cv_use = None
        sweep_ad3_C1_candidates = None
        sweep_ad3_C1_manual = None

        sweep_mad_alpha = None
        sweep_ad3_alpha = None

        if needs_mad_cv or needs_ad3_cv or needs_mad_h or needs_ad3_h:
            st.divider()
            st.subheader("Additional parameters")

        if needs_mad_cv:
            st.markdown("**Mean-MAD (banded: k-fold cv)**")
            sweep_mad_C1_candidates_default = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
            sweep_mad_C1_candidates_text = st.text_input(
                "Mean-MAD CV: C1 candidates (comma-separated)",
                value=", ".join(map(str, sweep_mad_C1_candidates_default)),
                key="sweep_mad_c1_candidates_text",
            )
            sweep_mad_C1_candidates = parse_candidate_text(sweep_mad_C1_candidates_text, sweep_mad_C1_candidates_default)
            sweep_mad_cv_use = st.checkbox("Mean-MAD CV: select C1 via k-fold CV", value=True, key="sweep_mad_usecv")
            if not sweep_mad_cv_use:
                sweep_mad_C1_manual = float(st.select_slider("Mean-MAD CV: pick C1 manually", options=sweep_mad_C1_candidates, value=manual_candidate_default(sweep_mad_C1_candidates), key="sweep_mad_c1"))

        if needs_ad3_cv:
            st.markdown("**Mean-AD3 (banded: k-fold cv)**")
            sweep_ad3_C1_candidates_default = [5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
            sweep_ad3_C1_candidates_text = st.text_input(
                "Mean-AD3 CV: C1 candidates (comma-separated)",
                value=", ".join(map(str, sweep_ad3_C1_candidates_default)),
                key="sweep_ad3_c1_candidates_text",
            )
            sweep_ad3_C1_candidates = parse_candidate_text(sweep_ad3_C1_candidates_text, sweep_ad3_C1_candidates_default)
            sweep_ad3_cv_use = st.checkbox("Mean-AD3 CV: select C1 via k-fold CV", value=True, key="sweep_ad3_usecv")
            if not sweep_ad3_cv_use:
                sweep_ad3_C1_manual = float(st.select_slider("Mean-AD3 CV: pick C1 manually", options=sweep_ad3_C1_candidates, value=manual_candidate_default(sweep_ad3_C1_candidates), key="sweep_ad3_c1"))

        if needs_mad_h:
            st.markdown("**Mean-MAD (banded: Hoeffding)**")
            sweep_mad_alpha = st.slider("Mean-MAD Hoeffding alpha", 0.01, 0.20, 0.05, 0.01, key="sweep_mad_alpha")

        if needs_ad3_h:
            st.markdown("**Mean-AD3 (banded: Hoeffding)**")
            sweep_ad3_alpha = st.slider("Mean-AD3 Hoeffding alpha", 0.01, 0.20, 0.05, 0.01, key="sweep_ad3_alpha")

#Initialize
a_f, b_f = float(a), float(b)

rng_obs = np.random.default_rng(int(seed))
samples = sample_demand(dist, int(n), a_f, b_f, rng_obs, **dist_kwargs)

true_info = compute_true_situation(dist, dist_kwargs, a_f, b_f, int(true_mc_n), int(true_mc_seed), mode=mode)

grid = np.linspace(a_f, b_f, int(cert_grid_points))

rng_true_cert = np.random.default_rng(int(true_mc_seed) + 1)
D_true_for_cert = sample_demand(dist, int(true_mc_n), a_f, b_f, rng_true_cert, **dist_kwargs)
true_cert_curve = np.array([float(np.mean(np.abs(D_true_for_cert - float(c)))) for c in grid], dtype=float)

# Build dd for top plot
if band_method == "Naive estimation":
    dd = naive_equalities(samples, a_f, b_f, mode=mode)
elif band_method == "k-fold cv":
    if use_cv is False and C1_used is not None:
        eps = float(C1_used) / np.sqrt(max(1, len(samples)))
        dd = band_builder_paper(samples, a_f, b_f, mode=mode, eps=eps)
    else:
        C1u = 0.1
        if use_cv and (C1_candidates is not None) and (q_grid_cv is not None):
            C1_best, _, _, _ = cv_select_C1(
                samples=samples, a=a_f, b=b_f, mode=mode, m=float(m), d=float(d),
                C1_candidates=C1_candidates, q_grid=q_grid_cv, seed=28
            )
            C1u = float(C1_best)
        eps = float(C1u) / np.sqrt(max(1, len(samples)))
        dd = band_builder_paper(samples, a_f, b_f, mode=mode, eps=eps)
else:
    dd = band_builder_hoeffding(samples, a_f, b_f, mode=mode, alpha=float(alpha))

q_ref = 0.5 * (a_f + b_f)

if band_method == "Naive estimation":
    if mode == "Mean-MAD":
        wc = wc_mean_mad_naive(a_f, b_f, dd["mu_hat"], dd["mad_hat"], q_ref, float(m), float(d))
    elif mode == "Mean-AD3":
        wc = wc_mean_ad3_naive(
            a_f, b_f,
            mu_hat=dd["mu_hat"],
            ad0_hat=dd["mad_hat"],
            ad1_hat=dd["ad1_hat"],
            ad2_hat=dd["ad2_hat"],
            q=q_ref, m=float(m), d=float(d)
        )
    else:
        L = int(mode.split("AD")[1])
        wc = wc_mean_ad_general_naive(
            a_f, b_f,
            mu_hat=dd["mu_hat"],
            L=L,
            ad_hats=np.array(dd["ad_hats"], dtype=float),
            q=q_ref, m=float(m), d=float(d)
        )
else:
    l_mu, u_mu = dd["T_mu"]
    u0 = dd["D_mu"][1]
    if mode == "Mean-MAD":
        wc = wc_mean_mad_interval(a_f, b_f, float(l_mu), float(u_mu), float(u0), q_ref, float(m), float(d))
    else:
        u1 = dd["D_c1"][1]
        u2 = dd["D_c2"][1]
        wc = wc_mean_ad3_interval(a_f, b_f, float(l_mu), float(u_mu), float(u0), float(u1), float(u2), q_ref, float(m), float(d))

# Plot 1: certificates
fig_regions = plot_regions_and_certs(
    samples=samples, a=a_f, b=b_f, dd=dd, mode=mode, wc=wc,
    show_support=True,
    true_info=true_info,
    true_cert_curve=true_cert_curve,
    grid=grid,
)
st.plotly_chart(fig_regions, use_container_width=True)

# Plot 2: costs
q_grid = np.linspace(a_f, b_f, int(q_grid_n))
fig_cost = plot_costs_true_emp_wc(
    dist=dist,
    dist_kwargs=dist_kwargs,
    a=a_f,
    b=b_f,
    m=float(m),
    d=float(d),
    wc=wc,
    q_grid=q_grid,
    true_mc_n=int(true_mc_n),
    true_mc_seed=int(true_mc_seed),
    empirical_samples=samples,
)
st.plotly_chart(fig_cost, use_container_width=True)

st.subheader("Consistency check for plots above")
st.write({
    "support_[a,b]": [a_f, b_f],
    "distribution": dist,
    "distribution_params_relative": dist_kwargs,
    "main_choice": main_choice,
    "decoded_mode": mode,
    "decoded_band_method": band_method,
    "N_observed": int(n),
    "MC_TRUE_size": int(true_mc_n),
})

#Boxplot OOS performance
if "do_oos" in locals() and do_oos:
    st.subheader("Out-of-sample performance vs SAA")
    q_grid_oos = np.linspace(a_f, b_f, int(oos_q_grid_n))

    alpha_pass = float(alpha) if (band_method == "Conservative Hoeffding") else None
    cv_use_pass = bool(use_cv) if (band_method == "k-fold cv") else None
    C1_candidates_pass = C1_candidates if (band_method == "k-fold cv") else None
    q_grid_cv_pass = q_grid_cv if (band_method == "k-fold cv" and cv_use_pass) else None
    C1_manual_pass = float(C1_used) if (band_method == "k-fold cv" and (use_cv is False) and (C1_used is not None)) else None

    robust_costs, saa_costs, dropped = oos_experiment(
        dist=dist, dist_kwargs=dist_kwargs,
        a=a_f, b=b_f,
        mode=mode, band_method=band_method,
        N_train=int(n),
        N_test=int(oos_test_n),
        trials=int(oos_trials),
        m=float(m), d=float(d),
        q_grid=q_grid_oos,
        seed0=int(seed),
        alpha=alpha_pass,
        cv_use=cv_use_pass,
        C1_candidates=C1_candidates_pass,
        q_grid_cv=q_grid_cv_pass,
        C1_manual=C1_manual_pass,
    )

    if len(robust_costs) == 0 or len(saa_costs) == 0:
        st.error("OOS: no feasible robust trials were found (all trials dropped). Try larger N or Naive estimation.")
    else:
        st.write({
            "trials_requested": int(oos_trials),
            "trials_kept": int(len(robust_costs)),
            "trials_dropped_due_to_infeasibility": int(dropped),
            "N_train": int(n),
            "N_test": int(oos_test_n),
            "q_grid_size": int(oos_q_grid_n),
        })
        fig_box = boxplot_compare(robust_costs, saa_costs, robust_label=f"{main_choice}", saa_label="SAA")
        st.plotly_chart(fig_box, use_container_width=True)

#OOS performance for different N
if "do_sweep" in locals() and do_sweep:
    st.subheader("Improvement sweep vs SAA")

    if len(sweep_choices) == 0:
        st.error("Select at least one series.")
    else:
        N_list = make_N_grid()

        q_grid_sweep = np.linspace(a_f, b_f, int(sweep_q_grid_n))

        series_specs = []
        for lab in sweep_choices:
            md, bm = ambiguity_choice(lab)
            spec = {"label": lab, "mode": md, "band_method": bm}

            if bm == "Conservative Hoeffding":
                if md == "Mean-MAD":
                    spec["alpha"] = sweep_mad_alpha if sweep_mad_alpha is not None else 0.05
                elif md == "Mean-AD3":
                    spec["alpha"] = sweep_ad3_alpha if sweep_ad3_alpha is not None else 0.05

            if bm == "k-fold cv":
                spec["q_grid_cv"] = q_grid_sweep

                if md == "Mean-MAD":
                    spec["cv_use"] = sweep_mad_cv_use if sweep_mad_cv_use is not None else True
                    spec["C1_candidates"] = sweep_mad_C1_candidates if sweep_mad_C1_candidates is not None else [5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
                    spec["C1_manual"] = sweep_mad_C1_manual
                elif md == "Mean-AD3":
                    spec["cv_use"] = sweep_ad3_cv_use if sweep_ad3_cv_use is not None else True
                    spec["C1_candidates"] = sweep_ad3_C1_candidates if sweep_ad3_C1_candidates is not None else [5.0, 1.0, 0.5, 0.1, 0.05, 0.01]
                    spec["C1_manual"] = sweep_ad3_C1_manual

            series_specs.append(spec)

        prog = st.progress(0.0)
        mean_imp, p95_imp, kept = sweep_improvement_vs_saa(
            dist=dist, dist_kwargs=dist_kwargs,
            a=a_f, b=b_f,
            series_specs=series_specs,
            N_list=N_list,
            N_test=int(sweep_test_n),
            trials=int(sweep_trials),
            m=float(m), d=float(d),
            q_grid=q_grid_sweep,
            seed0=int(seed),
            st_progress=prog,
        )
        prog.progress(1.0)

        fig_mean = plot_improvement_lines(
            N_list, mean_imp,
            title="Mean performance vs SAA (percentage)",
            ylab="Performance vs SAA (%)",
        )
        st.plotly_chart(fig_mean, use_container_width=True)

        fig_p95 = plot_improvement_lines(
            N_list, p95_imp,
            title="95th percentile performance vs SAA (percentage)",
            ylab="Performance vs SAA (%)",
        )
        st.plotly_chart(fig_p95, use_container_width=True)