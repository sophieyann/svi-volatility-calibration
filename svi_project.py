from __future__ import annotations

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.stats import norm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

OPTIONS_FILE = os.path.join(BASE_DIR, "data", "options_data.csv")
FUTURES_FILE = os.path.join(BASE_DIR, "data", "spx_futures.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "svi_outputs")
FIG_DIR = os.path.join(OUTPUT_DIR, "figures")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

PARAMS = ["a", "b", "rho", "m", "sigma"]
DAYS_PER_YEAR = 365
K_MIN, K_MAX = -1.0, 0.6
BLEND = 0.02
MIN_POINTS = 8
CONSTRAINT_WEIGHT = 1e4
LAMBDA_GRID = [0.0, 0.01, 0.1, 1.0, 10.0]
FORWARD_FAMILY = "ES"


# =============================================================================
# 1. Loading
# =============================================================================

def load_options(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"pricedate": "snapshot", "Expiry": "expiry","impliedvol": "iv_raw"})

    df["snapshot"] = pd.to_datetime(df["snapshot"], format="%d/%m/%Y")
    df["expiry"] = pd.to_datetime(df["expiry"], format="%d/%m/%Y")
    df["option_type"] = df["option_type"].str.strip().str.capitalize()
    df["iv"] = df["iv_raw"] / 100.0

    return df.dropna(subset=["iv"])


def load_futures(path):
    df = pd.read_csv(path)
    df = df.rename(columns={"px_last(dates=#dt)": "last",
                            "px_mid(dates=#dt)": "mid",
                            "px_bid(dates=#dt)": "bid",
                            "px_ask(dates=#dt)": "ask",
                            "fut_last_trade_dt()": "fut_expiry"})
    
    df["snapshot"] = pd.to_datetime(df["query_date"])
    df["fut_expiry"] = pd.to_datetime(df["fut_expiry"])
    return df


# def build_forward(fut, expiry, snapshots):
#     same_expiry = fut[fut["fut_expiry"] == expiry]
#     report = (same_expiry.groupby("ID").agg(last_pop=("last", lambda s: float(s.notna().mean())),
#                                             mid_pop=("mid", lambda s: float(s.notna().mean())),
#                                             last_med=("last", "median"),
#                                             mid_med=("mid", "median")).reset_index())
#     report["is_emini"] = report["ID"].str.startswith(FORWARD_FAMILY)

#     cand = report[report["is_emini"]]
#     if cand.empty:
#         cand = report.sort_values("last_pop", ascending=False)
#         print("WARNING: no E-mini contract found; falling back to "f"{cand.iloc[0]['ID']}")
        
#     ticker = cand.sort_values("last_pop", ascending=False).iloc[0]["ID"]

#     s = (fut[fut["ID"] == ticker].set_index("snapshot")["last"].reindex(pd.DatetimeIndex(sorted(set(snapshots)))))
#     # Deferred contracts do not trade every day; carry the last observed price forward rather than dropping the snapshot entirely
#     s = s.ffill().bfill()
#     return ticker, s, report

def build_forward(fut, expiry, snapshots):
    expiry = pd.Timestamp(expiry)
    target_dates = pd.DatetimeIndex(pd.to_datetime(pd.Series(snapshots).dropna().unique())).sort_values()

    same_expiry = fut.loc[fut["fut_expiry"].eq(expiry)].copy()

    if same_expiry.empty:
        raise ValueError(f"No futures contracts found with expiry "f"{expiry:%Y-%m-%d}")

    report = (same_expiry.groupby("ID", as_index=False).agg(last_pop=("last", lambda s: float(s.notna().mean())),
                                                            mid_pop=("mid", lambda s: float(s.notna().mean())),
                                                            last_med=("last", "median"),
                                                            mid_med=("mid", "median"),))

    report["is_emini"] = report["ID"].str.startswith(FORWARD_FAMILY)

    candidates = report.loc[report["is_emini"]].copy()

    if candidates.empty:
        available = report["ID"].tolist()
        raise ValueError(
            f"No E-mini S&P 500 futures contract found for "
            f"{expiry:%Y-%m-%d}. Available contracts: {available}"
        )

    ticker = (candidates.sort_values(["last_pop", "mid_pop"],ascending=False,).iloc[0]["ID"])

    raw = (
        fut.loc[fut["ID"].eq(ticker), ["snapshot", "last"]]
        .dropna(subset=["snapshot"])
        .sort_values("snapshot")
        .drop_duplicates("snapshot", keep="last")
        .set_index("snapshot")["last"]
    )

    # Use only prices observed on the same option snapshot dates.
    forward = raw.reindex(target_dates)

    missing_dates = forward.index[forward.isna()]

    if len(missing_dates) > 0:
        raise ValueError(
            f"{ticker} is missing forward prices for "
            f"{len(missing_dates)} option snapshots. "
            f"First missing dates: "
            f"{missing_dates[:5].strftime('%Y-%m-%d').tolist()}"
        )

    forward.name = "forward"

    return ticker, forward, report


# =============================================================================
# 2. Preparation
# =============================================================================

def parity_forward_shift(opt, fwd):
    t = opt.copy()
    t["F"] = t["snapshot"].map(fwd)
    t = t.dropna(subset=["F"])

    t["k"] = np.log(t["strike"] / t["F"])
    t["T"] = (t["expiry"] - t["snapshot"]).dt.days / DAYS_PER_YEAR

    piv = (t.pivot_table(
        index=["snapshot", "strike", "F", "k", "T"],
          columns="option_type",
          values="iv"
          )
          .reset_index()
          .dropna(subset=["Call", "Put"]))
    piv = piv[piv["k"].abs() <= 0.05]

    # if piv.empty:
    #     return pd.Series(dtype=float), pd.DataFrame()
    if piv.empty:
        return fwd.copy(), pd.DataFrame()
    
    iv = 0.5 * (piv["Call"] + piv["Put"])

    d1 = (np.log(piv["F"] / piv["strike"]) + 0.5 * iv**2 * piv["T"]) / (iv * np.sqrt(piv["T"]))
    vega = piv["F"] * norm.pdf(d1) * np.sqrt(piv["T"])

    piv["dF"] = (piv["Call"] - piv["Put"]) * vega

    est = piv.groupby("snapshot").agg(F_future=("F", "first"),
                                      dF=("dF", "median"),
                                      n=("dF", "size"))
    est["F_parity"] = est["F_future"] + est["dF"]
    est["pct"] = 100 * est["dF"] / est["F_future"]

    return est["F_parity"], est.reset_index()


def prepare(opt, fwd):
    """Clean and build k, T, w.  Returns (dataframe, filter_log)."""
    log = []
    df = opt.copy()

    def apply_filter(data, mask, reason):
        removed = int(mask.sum())
        remaining = len(data) - removed

        log.append({
            "filter": reason,
            "removed": removed,
            "remaining": remaining,
            })
        return data.loc[~mask].copy()

    df["F"] = df["snapshot"].map(fwd)
    df = apply_filter(df, df["F"].isna(), "no forward")
    df = apply_filter(df, (df["iv"] < 0.01) | (df["iv"] > 2.0), "iv outside [1%, 200%]")

    df["k"] = np.log(df["strike"] / df["F"])
    df["T"] = (df["expiry"] - df["snapshot"]).dt.days / DAYS_PER_YEAR
    df = apply_filter(df, df["T"] <= 0, "expiry not in future")

    # OTM selection
    is_put = df["option_type"] == "Put"
    is_call = df["option_type"] == "Call"
    keep = ((is_put & (df["k"] < -BLEND)) | (is_call & (df["k"] > BLEND))
            | (df["k"].abs() <= BLEND))
    
    df = apply_filter(df, ~keep, "in-the-money side (OTM rule)")
    df = apply_filter(df, (df["k"] < K_MIN) | (df["k"] > K_MAX),
              f"k outside [{K_MIN}, {K_MAX}]")

    df["w"] = df["iv"] ** 2 * df["T"]

    out = (df.groupby(["snapshot", "expiry", "strike"], as_index=False
                      )
                      .agg(
                          F=("F", "first"),
                          k=("k", "first"),
                          T=("T", "first"),
                          iv=("iv", "mean"),
                          quote_count=("iv", "size"),
                          ))
    out["w"] = out["iv"] ** 2 * out["T"]
    
    return out.sort_values(["snapshot", "k"]).reset_index(drop=True), pd.DataFrame(log)


# =============================================================================
# 3. SVI
# =============================================================================

def svi_w(theta, k):
    a, b, rho, m, sg = theta
    u = np.asarray(k, float) - m
    return a + b * (rho * u + np.sqrt(u * u + sg * sg))


def svi_iv(theta, k, T):
    return np.sqrt(np.maximum(svi_w(theta, k), 1e-10) / T)


def vega_weights(F, K, T, iv):
    F = np.asarray(F, float)
    K = np.asarray(K, float)
    iv = np.asarray(iv, float)

    if len(iv) == 0:
        raise ValueError("Cannot weight an empty option slice")

    d1 = (np.log(F / K) + 0.5 * iv**2 * T) / np.maximum(iv * np.sqrt(T), 1e-12)

    v = F * norm.pdf(d1) * np.sqrt(T)
    v = np.where(np.isfinite(v), v, 0.0)

    vmax = float(np.max(v))

    if vmax <= 0:
        return np.full(len(v), 1.0 / len(v))

    v = np.maximum(v, 0.01 * vmax)

    raw_weights = v**2
    return raw_weights / raw_weights.sum()


def constraint_penalty(theta, T):
    a, b, rho, _, sigma = theta

    min_variance = a + b * sigma * np.sqrt(max(1.0 - rho**2, 0.0))

    violation = max(-min_variance, 0.0)

    return CONSTRAINT_WEIGHT * violation**2


def bounds_for(k, T):
    lo, hi = float(np.min(k)), float(np.max(k))
    span = max(hi - lo, 0.1)
    return [(-2.0, 5.0), (1e-6, 4.0 / T), (-0.999, 0.999),
            (lo - span, hi + span), (1e-4, 5.0)]


def snapshot_loss(theta, k, iv, wt, T, eta=0.0, bid=None, ask=None):
    s = svi_iv(theta, k, T)
    r = s - iv
    loss = float(np.sum(wt * r * r))
    if eta > 0:
        if bid is None or ask is None:
            raise ValueError("Both bid and ask IV are required when eta > 0")
        
    if eta > 0 and bid is not None:
        lo = np.maximum(0.0, bid - s)
        hi = np.maximum(0.0, s - ask)
        loss += eta * float(np.sum(wt * (lo**2 + hi**2)))
    return loss + constraint_penalty(theta, T)


def multi_starts(k, w, T):
    k = np.asarray(k, float)
    span = max(float(k.max() - k.min()), 0.1)
    out = []
    for rho0 in (-0.85, -0.5, -0.1):
        for f in (0.05, 0.2, 0.5):
            sg0 = f * span
            b0 = min(0.1, 0.5 * 4.0 / T)
            a0 = max(float(np.min(w)) - b0 * sg0 * np.sqrt(1 - rho0**2), 1e-6)
            out.append(np.array([a0, b0, rho0, float(k[np.argmin(w)]), sg0]))
    return out


def fit_one(k, iv, wt, T, starts, eta=0.0, bid=None, ask=None):
    bnd = bounds_for(k, T)
    best, best_loss = None, np.inf
    for x0 in starts:
        x0 = np.clip(x0, [b[0] for b in bnd], [b[1] for b in bnd])
        try:
            r = minimize(snapshot_loss, x0, args=(k, iv, wt, T, eta, bid, ask),
                         method="L-BFGS-B", bounds=bnd,
                         options={"maxiter": 800, "ftol": 1e-14})
        except Exception:
            continue
        if r.fun < best_loss:
            best, best_loss = r.x, float(r.fun)
    return best if best is not None else np.full(5, np.nan)



# =============================================================================
# 4. Joint ridge solve
# =============================================================================

def make_slices(df):
    sl = []
    for snap, g in df.groupby("snapshot", sort=True):
        g = g.sort_values("k")
        T = float(g["T"].iloc[0])
        sl.append({"snapshot": snap, "k": g["k"].to_numpy(float),
                   "iv": g["iv"].to_numpy(float), "T": T,
                   "F": g["F"].to_numpy(float), "K": g["strike"].to_numpy(float),
                   "wt": vega_weights(g["F"].to_numpy(float),
                                      g["strike"].to_numpy(float), T,
                                      g["iv"].to_numpy(float))})
    return sl


def pack(sl):
    """Flatten all slices into contiguous arrays.
    """
    return {
        "k": np.concatenate([s["k"] for s in sl]),
        "iv": np.concatenate([s["iv"] for s in sl]),
        "wt": np.concatenate([s["wt"] for s in sl]),
        "T": np.concatenate([np.full(len(s["k"]), s["T"]) for s in sl]),
        "owner": np.concatenate([np.full(len(s["k"]), i, dtype=np.intp)
                                 for i, s in enumerate(sl)]),
        "T_slice": np.array([s["T"] for s in sl]),
        "n": len(sl), "m": sum(len(s["k"]) for s in sl),
    }


def joint_obj(z, P, lam, scales, obj_scale, ridge_scale):
    n = P["n"]
    th = z.reshape(n, 5) * scales
    o = P["owner"]
    a, b, rho, m, sg = (th[o, 0], th[o, 1], th[o, 2], th[o, 3], th[o, 4])

    u = P["k"] - m
    r = np.sqrt(u * u + sg * sg)
    w = a + b * (rho * u + r)
    ws = np.maximum(w, 1e-10)
    fit = np.sqrt(ws / P["T"])

    dw = np.empty((P["m"], 5))
    dw[:, 0] = 1.0
    dw[:, 1] = rho * u + r
    dw[:, 2] = b * u
    dw[:, 3] = -b * (rho + u / r)
    dw[:, 4] = b * sg / r
    dsig = dw * np.where(w > 1e-10,
                         1.0 / (2.0 * np.sqrt(ws * P["T"])), 0.0)[:, None]

    res = fit - P["iv"]
    total = float(np.sum(P["wt"] * res * res))
    coef = (2.0 * P["wt"] * res)[:, None] * dsig
    grad = np.empty((n, 5))
    for j in range(5):
        grad[:, j] = np.bincount(o, weights=coef[:, j], minlength=n)

    A, B, R, S = th[:, 0], th[:, 1], th[:, 2], th[:, 4]
    root = np.sqrt(np.maximum(1.0 - R * R, 1e-12))
    mv = A + B * S * root
    bad = mv < 0
    if np.any(bad):
        total += CONSTRAINT_WEIGHT * float(np.sum(mv[bad] ** 2))
        c = 2 * CONSTRAINT_WEIGHT * mv[bad]
        grad[bad, 0] += c
        grad[bad, 1] += c * S[bad] * root[bad]
        grad[bad, 2] += c * (-B[bad] * S[bad] * R[bad] / root[bad])
        grad[bad, 4] += c * B[bad] * root[bad]

    # slack = 4.0 / P["T_slice"] - B * (1.0 + np.abs(R))
    # bad = slack < 0
    # if np.any(bad):
    #     total += CONSTRAINT_WEIGHT * float(np.sum(slack[bad] ** 2))
    #     c = 2 * CONSTRAINT_WEIGHT * slack[bad]
    #     grad[bad, 1] += c * (-(1.0 + np.abs(R[bad])))
    #     grad[bad, 2] += c * (-B[bad] * np.sign(R[bad]))

    total *= obj_scale
    grad = (grad * obj_scale) * scales

    if lam > 0 and n > 1:
        zz = z.reshape(n, 5)
        d = np.diff(zz, axis=0)
        total += lam * ridge_scale * float(np.sum(d * d))
        c = 2.0 * lam * ridge_scale * d
        grad[1:] += c
        grad[:-1] -= c

    return total, grad.ravel()


def fit_surface(
    df,
    lam=0.0,
    warm=None,
    scales=None,
    max_iter=15000,
):
    """Fit all snapshots jointly with an optional ridge penalty.

    Parameters
    ----------
    df : pd.DataFrame
        Prepared option data for exactly one expiry.
    lam : float
        Ridge-penalty strength.
    warm : array-like or None
        Initial parameter matrix with shape (n_snapshots, 5).
    scales : array-like or None
        Fixed parameter scales. These should be calculated from the
        baseline once and reused throughout the lambda sweep.
    max_iter : int
        Maximum number of L-BFGS-B iterations.
    """

    if df.empty:
        raise ValueError("Cannot fit an empty dataset")

    if df["expiry"].nunique() != 1:
        raise ValueError(
            "fit_surface currently supports exactly one expiry at a time"
        )

    sl = make_slices(df)
    n = len(sl)

    if n == 0:
        raise ValueError("No valid snapshot slices were created")

    # -------------------------------------------------------------
    # Stage 1: obtain starting parameters
    # -------------------------------------------------------------
    if warm is None:
        th = np.full((n, 5), np.nan)

        for i, s in enumerate(sl):
            starts = multi_starts(
                s["k"],
                s["iv"] ** 2 * s["T"],
                s["T"],
            )

            th[i] = fit_one(
                s["k"],
                s["iv"],
                s["wt"],
                s["T"],
                starts,
            )
    else:
        # Do not independently refit the warm parameters.
        # Use them directly as the joint optimizer's starting point.
        th = np.asarray(warm, dtype=float).copy()

        if th.shape != (n, 5):
            raise ValueError(
                f"warm must have shape {(n, 5)}, got {th.shape}"
            )

    if not np.all(np.isfinite(th)):
        raise RuntimeError(
            "One or more initial SVI parameter vectors are nonfinite"
        )

    # -------------------------------------------------------------
    # Fix parameter scales
    # -------------------------------------------------------------
    if scales is None:
        scale_floor = np.array(
            [
                1e-3,   # a
                1e-3,   # b
                0.05,   # rho
                0.01,   # m
                0.01,   # sigma
            ],
            dtype=float,
        )

        scales = np.maximum(
            np.std(th, axis=0),
            scale_floor,
        )
    else:
        scales = np.asarray(scales, dtype=float).copy()

        if scales.shape != (5,):
            raise ValueError(
                f"scales must have shape (5,), got {scales.shape}"
            )

        if (
            not np.all(np.isfinite(scales))
            or np.any(scales <= 0)
        ):
            raise ValueError(
                "All parameter scales must be finite and positive"
            )

    packed = pack(sl)

    # Default diagnostics
    optimizer_success = True
    optimizer_message = "Joint solve not required"
    n_it = 0
    n_eval = 0
    gradient_inf_norm = np.nan

    # -------------------------------------------------------------
    # Stage 2: joint optimization
    # -------------------------------------------------------------
    if n > 1:
        # Construct separate parameter bounds for every snapshot.
        scaled_bounds = []

        for s in sl:
            slice_bounds = bounds_for(s["k"], s["T"])

            for j, (lower, upper) in enumerate(slice_bounds):
                scaled_bounds.append(
                    (
                        lower / scales[j],
                        upper / scales[j],
                    )
                )

        result = minimize(
            joint_obj,
            (th / scales).ravel(),
            args=(
                packed,
                lam,
                scales,
                1e4 / n,
                1.0 / max(n - 1, 1),
            ),
            jac=True,
            method="L-BFGS-B",
            bounds=scaled_bounds,
            options={
                "maxiter": max_iter,
                "maxfun": 2 * max_iter,
                "ftol": 1e-13,
                "gtol": 1e-9,
            },
        )

        if not np.all(np.isfinite(result.x)):
            raise RuntimeError(
                "Joint optimizer returned nonfinite parameters"
            )

        th = result.x.reshape(n, 5) * scales

        optimizer_success = bool(result.success)
        optimizer_message = str(result.message)
        n_it = int(result.nit)
        n_eval = int(result.nfev)

        if getattr(result, "jac", None) is not None:
            gradient_inf_norm = float(
                np.max(np.abs(result.jac))
            )

    # -------------------------------------------------------------
    # Parameter output
    # -------------------------------------------------------------
    params = pd.DataFrame(th, columns=PARAMS)

    params.insert(
        0,
        "snapshot",
        [s["snapshot"] for s in sl],
    )

    params["T"] = [s["T"] for s in sl]

    # -------------------------------------------------------------
    # Fitted IV and residual output
    # -------------------------------------------------------------
    rows = []

    for s, theta in zip(sl, th):
        fitted_iv = svi_iv(
            theta,
            s["k"],
            s["T"],
        )

        rows.append(
            pd.DataFrame(
                {
                    "snapshot": s["snapshot"],
                    "k": s["k"],
                    "strike": s["K"],
                    "T": s["T"],
                    "iv_mid": s["iv"],
                    "iv_fit": fitted_iv,
                    "resid": fitted_iv - s["iv"],
                    "weight": s["wt"],
                }
            )
        )

    fitted = pd.concat(rows, ignore_index=True)

    # -------------------------------------------------------------
    # Parameter-stability diagnostics
    # -------------------------------------------------------------
    scaled_path = (
        params[PARAMS].to_numpy(dtype=float)
        / scales
    )

    parameter_changes = np.diff(
        scaled_path,
        axis=0,
    )

    if len(parameter_changes) > 0:
        mean_scaled_step = float(
            np.mean(np.abs(parameter_changes))
        )

        mean_squared_scaled_step = float(
            np.mean(
                np.sum(
                    parameter_changes**2,
                    axis=1,
                )
            )
        )
    else:
        mean_scaled_step = 0.0
        mean_squared_scaled_step = 0.0

    # -------------------------------------------------------------
    # Fit diagnostics
    # -------------------------------------------------------------
    residual = fitted["resid"].to_numpy(float)
    weights = fitted["weight"].to_numpy(float)

    summary = {
        "lambda": float(lam),
        "n_snapshots": int(n),
        "iterations": int(n_it),
        "function_evaluations": int(n_eval),
        "optimizer_success": bool(optimizer_success),
        "optimizer_message": optimizer_message,
        "gradient_inf_norm": gradient_inf_norm,
        "rmse_vol_pts": float(
            np.sqrt(np.mean(residual**2)) * 100
        ),
        "wrmse_vol_pts": float(
            np.sqrt(
                np.sum(weights * residual**2)
                / np.sum(weights)
            )
            * 100
        ),
        "max_abs_err_vol_pts": float(
            np.max(np.abs(residual)) * 100
        ),
        "mean_scaled_step": mean_scaled_step,
        "mean_squared_scaled_step": (
            mean_squared_scaled_step
        ),
    }

    return {
        "params": params,
        "fitted": fitted,
        "summary": summary,
        "scales": scales,
    }


# =============================================================================
# 5. Main
# =============================================================================

def main():
    print("Loading...")
    opt = load_options(OPTIONS_FILE)
    fut = load_futures(FUTURES_FILE)
    expiry = opt["expiry"].iloc[0]
    print(f"  {len(opt):,} option rows | {opt['snapshot'].nunique()} snapshots "
          f"| {opt['expiry'].nunique()} expiry ({expiry.date()})")

    ticker, fwd, report = build_forward(fut, expiry, opt["snapshot"].unique())
    print(f"\n[FIX 1/2] contracts sharing expiry {expiry.date()}:")
    print(report.round(3).to_string(index=False))
    print(f"  -> using {ticker}, field 'last'")
    report.to_csv(f"{OUTPUT_DIR}/forward_contract_choice.csv", index=False)

    F_par, par = parity_forward_shift(opt, fwd)
    if len(par):
        print(f"  parity-implied forward shift: {par['pct'].median():+.3f}%")
        par.to_csv(f"{OUTPUT_DIR}/parity_forward.csv", index=False)
        fwd = F_par.reindex(fwd.index).fillna(fwd)

    data, flog = prepare(opt, fwd)
    print("\nCleaning:")
    print(flog.to_string(index=False))
    print(f"  -> {len(data):,} quotes, "
          f"{len(data) / data['snapshot'].nunique():.0f} per snapshot")
    data.to_csv(f"{OUTPUT_DIR}/cleaned_svi_input.csv", index=False)

    # print("\nBaseline (multi-start, no chaining):")
    # base = fit_surface(data, lam=0.0)
    # print("  " + " | ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
    #                         for k, v in base["summary"].items()))
    # warm = base["params"][PARAMS].to_numpy()

    # print("\nLambda sweep:")
    # rows, runs = [], {}
    # for lam in LAMBDA_GRID:
    #     r = fit_surface(data, lam=lam, warm=warm)
    #     rows.append(r["summary"])
    #     runs[lam] = r
    #     s = r["summary"]
    #     print(f"  lambda={lam:<7g} rmse={s['rmse_vol_pts']:.4f} "
    #           f"step={s['mean_scaled_step']:.6f} it={s['iterations']}")
    # sweep = pd.DataFrame(rows)
    # sweep.to_csv(f"{OUTPUT_DIR}/lambda_sweep.csv", index=False)

    print("\nBaseline (multi-start, no chaining):")

    base = fit_surface(
        data,
        lam=0.0,
    )

    base_summary = base["summary"]

    print(
        f"  lambda=0"
        f" | n_snapshots={base_summary['n_snapshots']}"
        f" | iterations={base_summary['iterations']}"
        f" | rmse={base_summary['rmse_vol_pts']:.4f}"
        f" | wrmse={base_summary['wrmse_vol_pts']:.4f}"
        f" | step={base_summary['mean_scaled_step']:.6f}"
        f" | success={base_summary['optimizer_success']}"
    )

    if not base_summary["optimizer_success"]:
        print(
            "  WARNING: baseline optimizer message: "
            f"{base_summary['optimizer_message']}"
        )

    # Freeze these scales for every lambda value.
    fixed_scales = base["scales"].copy()

    # Start the regularization path from the baseline parameters.
    warm = base["params"][PARAMS].to_numpy(float)

    print("\nLambda sweep:")

    # Reuse the exact baseline result for lambda=0.
    rows = [base["summary"]]
    runs = {0.0: base}

    print(
        f"  lambda={0:<7g}"
        f" wrmse={base_summary['wrmse_vol_pts']:.4f}"
        f" step={base_summary['mean_scaled_step']:.6f}"
        f" it={base_summary['iterations']}"
        f" success={base_summary['optimizer_success']}"
    )

    for lam in sorted(
        value
        for value in LAMBDA_GRID
        if value > 0
    ):
        result = fit_surface(
            data,
            lam=lam,
            warm=warm,
            scales=fixed_scales,
        )

        rows.append(result["summary"])
        runs[lam] = result

        summary = result["summary"]

        print(
            f"  lambda={lam:<7g}"
            f" wrmse={summary['wrmse_vol_pts']:.4f}"
            f" step={summary['mean_scaled_step']:.6f}"
            f" it={summary['iterations']}"
            f" success={summary['optimizer_success']}"
        )

        if not summary["optimizer_success"]:
            print(
                f"    optimizer message: "
                f"{summary['optimizer_message']}"
            )
        # else:
        #     # Continuation: use the converged previous-lambda result
        #     # as the starting point for the next lambda.
        #     warm = result["params"][PARAMS].to_numpy(float)

    sweep = (
        pd.DataFrame(rows)
        .sort_values("lambda")
        .reset_index(drop=True)
    )

    sweep.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "lambda_sweep.csv",
        ),
        index=False,
    )


    print("\nconvergence check")
    # dr = np.diff(sweep["rmse_vol_pts"].to_numpy())
    dr = np.diff(sweep["wrmse_vol_pts"].to_numpy())
    ds = np.diff(sweep["mean_scaled_step"].to_numpy())
    ok_r, ok_s = bool((dr > -1e-4).all()), bool((ds < 1e-6).all())
    print(f"  RMSE non-decreasing in lambda : {'PASS' if ok_r else 'FAIL'}")
    print(f"  step non-increasing in lambda : {'PASS' if ok_s else 'FAIL'}")
    if not (ok_r and ok_s):
        print("  -> solver has NOT converged; do not interpret the sweep.")


    # Pick lambda* (fixed): choose the largest penalty whose weighted RMSE
    # is no more than 2% worse than the unpenalised baseline.
    baseline_wrmse = float(base["summary"]["wrmse_vol_pts"])
    rmse_tolerance = 1.02 * baseline_wrmse

    eligible = sweep.loc[
        np.isfinite(sweep["wrmse_vol_pts"])
        & (sweep["wrmse_vol_pts"] <= rmse_tolerance)
    ].copy()

    if eligible.empty:
        lam_star = 0.0
    else:
        lam_star = float(eligible["lambda"].max())

    ridge = runs[lam_star]

    base_step = float(base["summary"]["mean_scaled_step"])
    ridge_step = float(ridge["summary"]["mean_scaled_step"])
    ridge_wrmse = float(ridge["summary"]["wrmse_vol_pts"])

    if ridge_step > 0:
        stability_gain = base_step / ridge_step
    else:
        stability_gain = np.inf

    rmse_change_pct = 100.0 * (
        ridge_wrmse / baseline_wrmse - 1.0
    )

    print(f"\nlambda* = {lam_star:g}")
    print(
        f"  weighted RMSE: {ridge_wrmse:.4f} vol pts "
        f"({rmse_change_pct:+.2f}% versus baseline)"
    )

    if np.isfinite(stability_gain):
        print(f"  parameter stability gain: {stability_gain:.2f}x")
    else:
        print("  parameter stability gain: infinite "
            "(ridge parameter step is zero)")

    # ---- figures ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    for ax, p in zip(axes, PARAMS):
        ax.plot(base["params"]["snapshot"], base["params"][p], lw=0.9,
                color="#a0aec0", label="unpenalised")
        ax.plot(ridge["params"]["snapshot"], ridge["params"][p], lw=1.8,
                color="#2b6cb0", label=f"ridge $\\lambda$={lam_star:g}")
        ax.set_ylabel(p)
        ax.grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)
    fig.suptitle("SVI parameter paths")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/parameter_paths.png", dpi=140)
    plt.close(fig)

    snaps = sorted(data["snapshot"].unique())
    picks = [snaps[i] for i in np.linspace(0, len(snaps) - 1, 6).astype(int)]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    pm = ridge["params"].set_index("snapshot")
    for ax, sp in zip(axes.ravel(), picks):
        g = ridge["fitted"][ridge["fitted"]["snapshot"] == sp].sort_values("k")
        kd = np.linspace(g["k"].min(), g["k"].max(), 300)
        ax.plot(g["k"], g["iv_mid"] * 100, ".", ms=3, color="#2b6cb0")
        ax.plot(kd, svi_iv(pm.loc[sp, PARAMS].to_numpy(float), kd,
                           float(pm.loc[sp, "T"])) * 100, color="#c53030")
        ax.set_title(str(pd.Timestamp(sp).date()), fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("implied vol (%)")
    fig.suptitle("Fitted SVI smiles")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/fitted_smiles.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    pos = sweep[sweep["lambda"] > 0]
    axes[0].plot(pos["lambda"], pos["wrmse_vol_pts"], "o-", color="#2b6cb0",)
    axes[0].axhline(base["summary"]["wrmse_vol_pts"], linestyle="--", color="grey", label="unpenalised baseline",)
    axes[0].set_ylabel("Weighted RMSE (vol pts)")
    axes[0].legend(fontsize=9)

    axes[1].plot(pos["lambda"], pos["mean_scaled_step"], "o-", color="#c05621")
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel(r"$\lambda$")
    axes[1].set_ylabel("mean scaled step")
    fig.suptitle("Accuracy / stability trade-off")
    fig.tight_layout()
    fig.savefig(f"{OUTPUT_DIR}/lambda_tradeoff.png", dpi=140)
    plt.close(fig)

    print(f"\nWritten to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

