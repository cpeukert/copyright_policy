# AI Copyright Policy Dashboard — v13 (One-period ↔ Infinite-horizon toggle)
# - Adds a horizon toggle to switch between the original one-period model and the
#   infinite-horizon steady-state extension (with geometric depreciation).
# - Shows *both* horizons side-by-side (tabs) so users can compare outcomes on the same scenario.
# - Keeps scenarios compatible; adds new params (lambda_dep, beta_disc) with sensible defaults.
#
# Notes
# • Infinite-horizon implements the stationary feasibility (period-balanced funding) fixed point:
#   (1-τ) r S(A*,r) = φ (λ K*)^ψ − φ (D_f^{AI}(0))^ψ,  with  A* = θ(σ) D_s + μ (K*)^α,  D_f* = λ K*.
# • For Exception/Opt-out with r=0: D_f* = D_f^{AI}(0), K* = D_f^{AI}(0)/λ.
# • Bars & line plots use *per-period* quantities for comparability. For the infinite horizon,
#   we additionally report Present Value (PV) = per-period/(1-β) in a small table.

import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import json, os, re, math
from pathlib import Path

# ---------------- Defaults ----------------
DEFAULTS = dict(
    Ds=405.0,
    B=834.0,
    mu=0.9975,
    alpha=0.0013,
    phi=1.0,
    psi=1.866,
    kappa=0.988,
    tau=0.10,       # statutory & opt-out leakage
    gamma=1.25,
    delta=-0.362,
    x_in=-0.244,
    x_out=0.372,
    r_max=0.99,
    grid_steps=701,
    N=212.0,        # in millions (mio)
    n=0.4,          # adoption share
    months=12.0,    # months aggregation
    beta=0.5,       # creators' bargaining power (opt-in Nash). Kept for scenarios.
    # Model baselines
    Df0=100.0,
    delta_f=0.30,   # share of flow lost; Df_AI0 = (1 - delta_f)*Df0
    tau_in=None,    # if None -> defaults to tau
    r_out=0.0,
    r_stat=0.15,
    r_in=0.20,
    # NEW: Infinite-horizon params
    lambda_dep=0.20,  # geometric depreciation rate λ in (0,1)
    beta_disc=0.95,   # planner discount factor β in (0,1)
)

# ---------------- Utility ----------------
EPS = 1e-12
R_EPS = 1e-3  # minimum positive max for r-sliders

def nz(x): return np.where(np.abs(x) < EPS, 0.0, x)

def clamp(v, lo, hi): return float(max(lo, min(hi, float(v))))

# ---------------- Session/defaults ----------------

def ensure_defaults():
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v)
    if st.session_state.get("tau_in") is None:
        st.session_state["tau_in"] = st.session_state.get("tau", DEFAULTS["tau"])    

# apply pending load (for scenario management)
if "_pending_load" in st.session_state:
    data = st.session_state.pop("_pending_load")
    merged = {**DEFAULTS, **data}
    for k, v in merged.items():
        st.session_state[k] = v

ensure_defaults()

# ---------------- Core primitives ----------------

def logit(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def theta_of_sigma(sigma, gamma):
    return float(np.clip(sigma, 0.0, 1.0)) ** float(gamma)

def A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df):
    return theta_of_sigma(float(sigma_tilde), float(gamma)) * float(Ds) + float(mu) * (float(Df) ** float(alpha))

def rho_of_r(r, kappa):
    r = float(r); kappa = float(kappa)
    denom = (1.0 - kappa * r)
    return kappa * (1.0 - kappa) * (1.0 - r) / (denom * denom)

def revenue_S(A, B, r, kappa):
    return (float(A) ** 2 / float(B)) * rho_of_r(r, kappa)

def CS_of(A, B, r, kappa):
    return (float(A) ** 2 / (2.0 * float(B))) * ( (kappa**2) * (1.0 - r)**2 / (1.0 - kappa*r)**2 )

def PiAI_of(A, B, r, kappa):
    return (float(A) ** 2 / float(B)) * rho_of_r(r, kappa) * (1.0 - r)

def Roy_of(A, B, r, kappa):
    return float(r) * revenue_S(A, B, r, kappa)

def delta_T(Df, Df_AI0, phi, psi):
    # Incremental creation cost above suppressed baseline
    return float(phi) * (float(Df) ** float(psi)) - float(phi) * (float(Df_AI0) ** float(psi))

# ----- One-period flow solver (existing) -----

def solve_flow_total_static(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df0, Df_AI0):
    """Solve (1-τ) r S(A(Df), r) = ΔT(Df) for Df >= Df_AI0 (one period)."""
    r = float(r)
    if r <= 0.0:
        return float(Df_AI0)

    def pool(Df):
        A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
        S = revenue_S(A, B, r, kappa)
        return (1.0 - tau_reg) * r * S

    base_cost = float(phi) * (float(Df_AI0) ** float(psi))

    def H(Df):
        return pool(Df) - (float(phi) * (float(Df) ** float(psi)) - base_cost)

    lo = float(Df_AI0)
    hi = max(float(Df0), lo + 1.0)
    H_hi = H(hi)
    iters = 0
    while H_hi > 0.0 and iters < 60:
        hi *= 2.0
        H_hi = H(hi)
        iters += 1
    if H_hi > 0.0:
        return float(hi)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        H_mid = H(mid)
        if abs(H_mid) < 1e-10 or (hi - lo) < 1e-10:
            return float(mid)
        if H_mid >= 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# ----- Infinite-horizon steady-state solver -----

def solve_K_star_dynamic(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df_AI0, lambda_dep):
    """Solve L(K) = R(K) where
         L(K) = (1-τ) r * [ (θ(σ) Ds + μ K^α)^2 / B ] * ρ(r)
         R(K) = φ λ^ψ K^ψ - φ (Df_AI0)^ψ
       Returns (K*, Df* = λK*, A*). If r<=0, uses no-royalty case.
    """
    r = float(r)
    lam = float(lambda_dep)
    if r <= 0.0:
        Df_star = float(Df_AI0)
        K_star = Df_star / max(lam, 1e-12)
        A_star = theta_of_sigma(sigma_tilde, gamma) * Ds + mu * (K_star ** alpha)
        return float(K_star), float(Df_star), float(A_star)

    def L_of_K(K):
        A = theta_of_sigma(sigma_tilde, gamma) * Ds + mu * (K ** alpha)
        return (1.0 - tau_reg) * r * ((A * A) / B) * rho_of_r(r, kappa)

    def R_of_K(K):
        return float(phi) * (lam ** float(psi)) * (K ** float(psi)) - float(phi) * (float(Df_AI0) ** float(psi))

    # bracket root for H(K)=L(K)-R(K)
    def H(K):
        return L_of_K(K) - R_of_K(K)

    # Lower bound at 0
    lo = 0.0
    hi = max(1.0, Df_AI0 / max(lam, 1e-6))
    # increase hi until H(hi) <= 0 (i.e., R >= L) to bracket sign change
    H_hi = H(hi)
    iters = 0
    while H_hi > 0.0 and iters < 80:
        hi *= 2.0
        H_hi = H(hi)
        iters += 1
    # If even at large hi we still have H>0, take hi as solution (pool dominates)
    if H_hi > 0.0:
        K_star = float(hi)
        Df_star = lam * K_star
        A_star = theta_of_sigma(sigma_tilde, gamma) * Ds + mu * (K_star ** alpha)
        return float(K_star), float(Df_star), float(A_star)
    # bisection between lo and hi where H(lo) >= 0 is not guaranteed at lo=0, so adjust
    H_lo = H(lo)
    if H_lo < 0.0:
        # move lo up until H(lo) >= 0 or close the interval
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if H(mid) >= 0.0:
                lo = mid; break
            lo = mid
        else:
            lo = 0.0
    # bisection
    for _ in range(240):
        mid = 0.5 * (lo + hi)
        H_mid = H(mid)
        if abs(H_mid) < 1e-10 or (hi - lo) < 1e-10:
            K_star = float(mid)
            Df_star = lam * K_star
            A_star = theta_of_sigma(sigma_tilde, gamma) * Ds + mu * (K_star ** alpha)
            return float(K_star), float(Df_star), float(A_star)
        if H_mid >= 0.0:
            lo = mid
        else:
            hi = mid
    K_star = float(0.5 * (lo + hi))
    Df_star = lam * K_star
    A_star = theta_of_sigma(sigma_tilde, gamma) * Ds + mu * (K_star ** alpha)
    return float(K_star), float(Df_star), float(A_star)

# ---------------- Regime helpers ----------------

def sigma_tilde_opt_in(Ds, r_in, delta, x_in):
    return float(np.clip(logit(float(Ds) * (float(r_in) + float(delta) - float(x_in))), 0.0, 1.0))

def sigma_tilde_opt_out(Ds, r_out, delta, x_out):
    sigma_out = float(np.clip(logit(float(Ds) * (float(delta) - float(r_out) - float(x_out))), 0.0, 1.0))
    return 1.0 - sigma_out

# Unified block with horizon switch

def regime_block(regime_name, sigma_tilde, r, tau_reg, params, horizon="one"):
    Ds=params["Ds"]; B=params["B"]; mu=params["mu"]; alpha=params["alpha"]
    phi=params["phi"]; psi=params["psi"]; kappa=params["kappa"]
    gamma=params["gamma"]; Df0=params["Df0"]; delta_f=params["delta_f"]
    lam=params.get("lambda_dep", 0.2); beta_disc=params.get("beta_disc", 0.95)
    Df_AI0 = (1.0 - delta_f) * Df0

    if horizon == "one":
        if r <= 0.0:
            Df = Df_AI0
        else:
            Df = solve_flow_total_static(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df0, Df_AI0)
        A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
        S = revenue_S(A, B, r, kappa)
        Admin_leak = tau_reg * r * S
        CS = CS_of(A, B, r, kappa)
        PiAI = PiAI_of(A, B, r, kappa)
        Creation_cost = delta_T(Df, Df_AI0, phi, psi)
        PiC_transfer = (1.0 - tau_reg) * r * S
        PiC_net = PiC_transfer - Creation_cost
        TotalW = CS + PiAI - Admin_leak - Creation_cost
        return dict(name=f"{regime_name} (one)", horizon="one", r=r, sigma=sigma_tilde, Df=Df, A=A, S=S,
                    CS=CS, PiAI=PiAI, Roy=r*S, Admin=Admin_leak, Creation=Creation_cost,
                    TotalW=TotalW, PiC_transfer=PiC_transfer, PiC_net=PiC_net,
                    K=None, PV_factor=None, TotalW_PV=None)

    else:  # infinite-horizon steady state
        K_star, Df_star, A_star = solve_K_star_dynamic(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df_AI0, lam)
        S = revenue_S(A_star, B, r, kappa)
        Admin_leak = tau_reg * r * S
        CS = CS_of(A_star, B, r, kappa)
        PiAI = PiAI_of(A_star, B, r, kappa)
        Creation_cost = delta_T(Df_star, Df_AI0, phi, psi)
        PiC_transfer = (1.0 - tau_reg) * r * S
        PiC_net = PiC_transfer - Creation_cost
        TotalW = CS + PiAI - Admin_leak - Creation_cost  # per-period steady-state
        PV_factor = 1.0 / max(1.0 - float(beta_disc), 1e-6)
        TotalW_PV = TotalW * PV_factor
        return dict(name=f"{regime_name} (∞)", horizon="infty", r=r, sigma=sigma_tilde, Df=Df_star, A=A_star, S=S,
                    CS=CS, PiAI=PiAI, Roy=r*S, Admin=Admin_leak, Creation=Creation_cost,
                    TotalW=TotalW, PiC_transfer=PiC_transfer, PiC_net=PiC_net,
                    K=K_star, PV_factor=PV_factor, TotalW_PV=TotalW_PV)

# Convenience wrapper by regime label

def compute_block_for_regime_at_r(regime, r, params, horizon="one"):
    Ds=params["Ds"]; tau=params["tau"]; tau_in=params["tau_in"]
    delta=params["delta"]; x_in=params["x_in"]; x_out=params["x_out"]

    if regime == "Statutory":
        sigma = 1.0; tau_reg = tau
        return regime_block("Statutory", sigma, r, tau_reg, params, horizon)
    elif regime == "Opt-in":
        sigma = sigma_tilde_opt_in(Ds, r, delta, x_in); tau_reg = tau_in
        return regime_block("Opt-in", sigma, r, tau_reg, params, horizon)
    elif regime == "Opt-out":
        sigma = sigma_tilde_opt_out(Ds, r, delta, x_out); tau_reg = tau
        return regime_block("Opt-out", sigma, r, tau_reg, params, horizon)
    elif regime == "Exception":
        sigma = 1.0; tau_reg = 0.0
        return regime_block("Exception", sigma, 0.0, tau_reg, params, horizon)
    else:
        raise ValueError("Unknown regime")

# Sweep over r on a grid (used by plots/optima)

def sweep_regime(regime, r_min, r_max, grid_steps, params, horizon="one"):
    rs = np.linspace(float(r_min), float(r_max), int(grid_steps))
    recs = []
    for rv in rs:
        blk = compute_block_for_regime_at_r(regime, float(rv), params, horizon)
        recs.append(dict(
            r=float(rv),
            TotalW=blk["TotalW"],
            CS=blk["CS"],
            PiAI=blk["PiAI"],
            PiC_net=blk["PiC_net"],
            A=blk["A"], Df=blk["Df"], S=blk["S"], Admin=blk["Admin"], Creation=blk["Creation"],
            TotalW_PV=blk.get("TotalW_PV", np.nan)
        ))
    return pd.DataFrame(recs)

def find_optima(df):
    out = {}
    for col in ["TotalW", "PiC_net", "PiAI"]:
        idx = int(df[col].idxmax())
        out[col] = dict(r=float(df.loc[idx,"r"]), value=float(df.loc[idx,col]))
    return out

# ---------------- UI ----------------
st.set_page_config(page_title="AI Copyright Policy Dashboard", layout="wide")
st.title("AI Copyright Policy Dashboard")

# ---- Scenarios sidebar ----
st.sidebar.subheader("Scenarios")
SCEN_DIR = (Path(__file__).parent / "scenarios"); SCEN_DIR.mkdir(exist_ok=True)

def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9 \-_()]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120] if name else ""

def list_scenarios():
    return sorted([p.stem for p in SCEN_DIR.glob("*.json")])

def read_scenario(name: str):
    safe = _safe_name(name); p = SCEN_DIR / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def write_scenario(name: str, data: dict):
    safe = _safe_name(name)
    if not safe: raise ValueError("Empty scenario name.")
    p = SCEN_DIR / f"{safe}.json"; p.write_text(json.dumps(data, indent=2)); return safe

def delete_scenario(name: str):
    safe = _safe_name(name); p = SCEN_DIR / f"{safe}.json"; p.exists() and p.unlink()

def collect_current_params_from_state(state):
    data = {}
    for k in DEFAULTS.keys():
        if k in state:
            v = state[k]
            try:
                data[k] = float(v) if isinstance(v, (int, float)) else v
            except Exception:
                data[k] = v
    return data

scen_list = list_scenarios()
chosen = st.sidebar.selectbox("Existing scenarios", ["(none)"] + scen_list, index=0)
colA, colB = st.sidebar.columns([1,1], gap="small")
new_name = st.sidebar.text_input("Scenario name", value=(chosen if chosen != "(none)" else ""))
overwrite = st.sidebar.checkbox("Overwrite if exists", value=False)
if colA.button("💾 Save / Overwrite"):
    nm = _safe_name(new_name)
    if not nm:
        st.sidebar.warning("Please enter a scenario name.")
    else:
        if nm in scen_list and not overwrite:
            st.sidebar.error("Scenario exists. Check 'Overwrite if exists' to overwrite.")
        else:
            data = collect_current_params_from_state(st.session_state)
            safe = write_scenario(nm, data)
            st.sidebar.success(f"Saved scenario: {safe}")
            st.rerun()
if colB.button("📥 Load"):
    if chosen == "(none)":
        st.sidebar.warning("Select a scenario to load.")
    else:
        data = read_scenario(chosen)
        if not data:
            st.sidebar.error("Failed to read scenario.")
        else:
            st.session_state["_pending_load"] = data
            st.sidebar.success(f"Loaded scenario: {chosen}")
            st.rerun()

# ---- Parameters ----
st.sidebar.divider(); st.sidebar.header("Parameters")
# Structural
st.sidebar.subheader("Demand & Data")
st.sidebar.number_input("D_s (stock of existing works)", min_value=0.0, step=1.0, format="%.1f", key="Ds")
st.sidebar.number_input("B (demand slope)", min_value=1e-6, step=1.0, format="%.1f", key="B")
st.sidebar.number_input("μ (scale of flow in A)", min_value=0.0, step=0.0001, format="%.4f", key="mu")
st.sidebar.number_input("α (diminishing returns to flow)", min_value=0.0, max_value=1.0, step=0.0001, format="%.4f", key="alpha")
st.sidebar.number_input("φ (flow cost scale)", min_value=0.0, step=0.01, format="%.2f", key="phi")
st.sidebar.number_input("ψ (flow cost curvature >1)", min_value=1.01, step=0.001, format="%.3f", key="psi")
st.sidebar.number_input("κ (pass-through)", min_value=0.01, max_value=1.00, step=0.001, format="%.3f", key="kappa")
st.sidebar.slider("τ (admin leakage, STAT & opt-out)", 0.0, 0.99, key="tau")
st.sidebar.slider("γ (coverage penalty on stock)", 1.0, 3.0, key="gamma")
# Infinite-horizon
st.sidebar.subheader("Infinite horizon (steady state)")
st.sidebar.slider("λ (depreciation rate)", 0.0, 0.99, key="lambda_dep")
st.sidebar.slider("β (planner discount factor)", 0.0, 0.999, key="beta_disc")

st.sidebar.subheader("Creators: harms & action costs")
st.sidebar.slider("δ (perceived net effect of AI on creators)", -1.0, 1.0, step=0.001, format="%.3f", key="delta")
st.sidebar.number_input("x_in (opt-in action cost)", min_value=-1.0, max_value=1.0, step=0.001, format="%.3f", key="x_in")
st.sidebar.number_input("x_out (opt-out action cost)", min_value=-1.0, max_value=1.0, step=0.001, format="%.3f", key="x_out")

st.sidebar.subheader("Flow baselines")
st.sidebar.number_input("D_f^0 (flow without AI)", min_value=0.0, step=0.1, format="%.2f", key="Df0")
st.sidebar.number_input("δ_f (share of flow lost; D_f^{AI}(0)=(1−δ_f)·D_f^0)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="delta_f")

st.sidebar.subheader("Admin leakage in Opt-in")
st.sidebar.slider("τ_in (admin leakage in Opt-in)", 0.0, 0.99, key="tau_in")

st.sidebar.subheader("Bargaining (opt-in)")
st.sidebar.slider("β_N (creators' bargaining power)", 0.0, 1.0,
                  value=float(st.session_state.get("beta", DEFAULTS["beta"])), step=0.01, key="beta")

st.sidebar.subheader("Search & aggregation")
# Slider bounds for r search
r_max_raw = float(st.session_state.get("r_max", DEFAULTS["r_max"]))
r_max_eff = max(R_EPS, r_max_raw)
st.sidebar.slider("Max r to search over", 0.0, 0.99, key="r_max")
st.sidebar.slider("Grid steps for r search", 101, 3001, 701, 100, key="grid_steps")
st.sidebar.number_input("Adult population (mio)", min_value=1.0, step=1.0, format="%.0f", key="N")
st.sidebar.number_input("AI adoption share", 0.0, 1.0, step=0.01, format="%.2f", key="n")
st.sidebar.number_input("Months", min_value=1.0, step=1.0, format="%.0f", key="months")
scale_factor = float(st.session_state["N"] * st.session_state["n"] * (st.session_state["months"]))
st.sidebar.text(f"Scale factor (mio·person·months): {scale_factor:,.1f}")

# Horizon toggle (and dual view)
st.sidebar.divider()
st.sidebar.header("Horizon view")
view_mode = st.sidebar.radio("Select horizon for main plots", ["One period", "Infinite horizon (steady state)"])
show_both = st.sidebar.checkbox("Show both horizons side-by-side", value=True)

# r sliders (snapshot bars)
colR1, colR2, colR3 = st.columns(3)
with colR1:
    st.subheader("Statutory license r")
    r_stat_init = clamp(st.session_state.get("r_stat", 0.15), 0.0, r_max_eff)
    r_stat = st.slider("r_stat", 0.0, r_max_eff, r_stat_init, min(r_max_eff, 0.01), key="r_stat")
with colR2:
    st.subheader("Opt-in negotiated r")
    r_in_init = clamp(st.session_state.get("r_in", 0.20), 0.0, r_max_eff)
    r_in = st.slider("r_in", 0.0, r_max_eff, r_in_init, min(r_max_eff, 0.01), key="r_in")
with colR3:
    st.subheader("Opt-out royalty r_out (typically 0)")
    r_out_init = clamp(st.session_state.get("r_out", 0.0), 0.0, r_max_eff)
    r_out = st.slider("r_out", 0.0, r_max_eff, r_out_init, min(r_max_eff, 0.01), key="r_out")

# ---------- Compute blocks for BOTH horizons ----------
params = st.session_state

# One-period blocks
blk_exc_one  = compute_block_for_regime_at_r("Exception", 0.0, params, horizon="one")
blk_stat_one = compute_block_for_regime_at_r("Statutory", r_stat, params, horizon="one")
sigma_in_one = sigma_tilde_opt_in(params["Ds"], r_in, params["delta"], params["x_in"])  # for display
blk_in_one   = compute_block_for_regime_at_r("Opt-in", r_in, params, horizon="one")
sigma_out_one = sigma_tilde_opt_out(params["Ds"], r_out, params["delta"], params["x_out"])  # display
blk_out_one  = compute_block_for_regime_at_r("Opt-out", r_out, params, horizon="one")

# Infinite-horizon blocks (steady state)
blk_exc_inf  = compute_block_for_regime_at_r("Exception", 0.0, params, horizon="infty")
blk_stat_inf = compute_block_for_regime_at_r("Statutory", r_stat, params, horizon="infty")
blk_in_inf   = compute_block_for_regime_at_r("Opt-in", r_in, params, horizon="infty")
blk_out_inf  = compute_block_for_regime_at_r("Opt-out", r_out, params, horizon="infty")

# ---------- Opt-in optima on current r-grid (both horizons) ----------
for horizon_label, horizon_key in [("one", "one"), ("∞", "infty")]:
    if horizon_key == "one":
        df_optin_one = sweep_regime("Opt-in", 0.0, r_max_eff, int(params["grid_steps"]), params, horizon="one")
        # Creators' net-max, AI-profit-max, Nash β
        r_in_star_C_one  = float(df_optin_one.loc[df_optin_one["PiC_net"].idxmax(), "r"]) if not df_optin_one.empty else 0.0
        r_in_star_AI_one = float(df_optin_one.loc[df_optin_one["PiAI"].idxmax(),    "r"]) if not df_optin_one.empty else 0.0
        beta_val = float(params.get("beta", DEFAULTS["beta"]))
        # Nash bargaining on grid
        mask = (df_optin_one["PiC_net"]>0) & (df_optin_one["PiAI"]>0)
        if mask.any():
            vals = beta_val*np.log(df_optin_one.loc[mask, "PiC_net"]) + (1.0-beta_val)*np.log(df_optin_one.loc[mask, "PiAI"]) 
            idx = int(np.nanargmax(vals))
            r_in_star_B_one = float(df_optin_one.loc[vals.index[idx], "r"]) if len(vals.index)>idx else r_in
        else:
            r_in_star_B_one = float(df_optin_one.loc[df_optin_one["PiC_net"].idxmax(), "r"]) if not df_optin_one.empty else r_in
    else:
        df_optin_inf = sweep_regime("Opt-in", 0.0, r_max_eff, int(params["grid_steps"]), params, horizon="infty")
        r_in_star_C_inf  = float(df_optin_inf.loc[df_optin_inf["PiC_net"].idxmax(), "r"]) if not df_optin_inf.empty else 0.0
        r_in_star_AI_inf = float(df_optin_inf.loc[df_optin_inf["PiAI"].idxmax(),    "r"]) if not df_optin_inf.empty else 0.0
        beta_val = float(params.get("beta", DEFAULTS["beta"]))
        mask = (df_optin_inf["PiC_net"]>0) & (df_optin_inf["PiAI"]>0)
        if mask.any():
            vals = beta_val*np.log(df_optin_inf.loc[mask, "PiC_net"]) + (1.0-beta_val)*np.log(df_optin_inf.loc[mask, "PiAI"]) 
            idx = int(np.nanargmax(vals))
            r_in_star_B_inf = float(df_optin_inf.loc[vals.index[idx], "r"]) if len(vals.index)>idx else r_in
        else:
            r_in_star_B_inf = float(df_optin_inf.loc[df_optin_inf["PiC_net"].idxmax(), "r"]) if not df_optin_inf.empty else r_in

st.markdown(
    f"**Opt-in optima (one period):**  "
    f"r*_C = `{r_in_star_C_one:.3f}`,  "
    f"r*_AI = `{r_in_star_AI_one:.3f}`,  "
    f"r*_B(β={float(params.get('beta', DEFAULTS['beta'])):.2f}) = `{r_in_star_B_one:.3f}`"
)
st.markdown(
    f"**Opt-in optima (∞ steady state):**  "
    f"r*_C = `{r_in_star_C_inf:.3f}`,  "
    f"r*_AI = `{r_in_star_AI_inf:.3f}`,  "
    f"r*_B(β={float(params.get('beta', DEFAULTS['beta'])):.2f}) = `{r_in_star_B_inf:.3f}`"
)

# ---------------- Bars (scaled) ----------------
st.header("Key quantities (σ̃ and baselines — one period)")
Df_AI0 = (1.0 - params["delta_f"]) * params["Df0"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("D_f^{AI}(0) (suppressed baseline)", f"{Df_AI0:,.3f}")
c2.metric("σ̃_in (at r_in)", f"{sigma_in_one:,.3f}")
c3.metric("σ̃_out (at r_out)", f"{sigma_out_one:,.3f}")
c4.metric("θ(σ) for opt-in", f"{theta_of_sigma(sigma_in_one, params['gamma']):,.3f}")

# Build bar dataframes for each horizon

def build_bars(blocks, scale_factor, beta_disc=None):
    rows = []
    for blk in blocks:
        label = blk["name"]
        rows.append(dict(
            Regime=label,
            r=blk["r"],
            CS=blk["CS"] * scale_factor,
            PiAI=blk["PiAI"] * scale_factor,
            PiC=blk["PiC_transfer"] * scale_factor,
            Admin=-blk["Admin"] * scale_factor,
            Creation=-blk["Creation"] * scale_factor,
            TotalWelfare=blk["TotalW"] * scale_factor,
            DeltaPiC=blk["PiC_net"] * scale_factor,
            PV_total=(blk.get("TotalW_PV") * scale_factor) if blk.get("TotalW_PV") is not None else np.nan
        ))
    return pd.DataFrame(rows)

bars_one = build_bars([blk_stat_one, blk_in_one, blk_exc_one, blk_out_one], scale_factor)
bars_inf = build_bars([blk_stat_inf, blk_in_inf, blk_exc_inf, blk_out_inf], scale_factor, beta_disc=params["beta_disc"]) 

# Tabs to compare horizons
if show_both:
    tab1, tab2 = st.tabs(["One period", "Infinite horizon (steady state)"])
else:
    tab1 = st.container(); tab2 = None

with tab1:
    st.header("Welfare decomposition — One period (scaled)")
    bar = (
        alt.Chart(bars_one)
        .transform_fold(["CS", "PiAI", "PiC", "Admin", "Creation"], as_=["Component", "Value"])
        .mark_bar()
        .encode(
            x=alt.X("Regime:N", title="Regime"),
            y=alt.Y("sum(Value):Q", title="Scaled amounts"),
            color=alt.Color("Component:N", scale=alt.Scale(scheme="tableau10")),
            tooltip=["Regime:N", "Component:N", alt.Tooltip("sum(Value):Q", format=",.2f")]
        ).properties(height=420)
    )
    line_total = (
        alt.Chart(bars_one)
        .mark_point(size=90, filled=True, color="black")
        .encode(x="Regime:N", y=alt.Y("TotalWelfare:Q", title=""), tooltip=["Regime:N", alt.Tooltip("TotalWelfare:Q", format=",.2f")])
    )
    st.altair_chart(bar + line_total, use_container_width=True)
    st.caption("Πᶜ bar shows transfer-only ((1−τ)·r·S). Transfers do not enter Total Welfare. Admin and Creation (ΔT) are real costs.")
    st.subheader("Creators — incremental net (ΔΠᶜ = F − ΔT), scaled")
    st.dataframe(bars_one[["Regime","DeltaPiC"]])
    st.subheader("Detailed values (unscaled)")
    df_detail_one = pd.DataFrame([
        dict(Regime=b["name"], sigma=b["sigma"], A=b["A"], Df=b["Df"], S=b["S"], Roy=b["Roy"],
             CS=b["CS"], PiAI=b["PiAI"], Admin=b["Admin"], Creation=b["Creation"], TotalW=b["TotalW"],
             PiC_transfer=b["PiC_transfer"], PiC_net=b["PiC_net"]) for b in [blk_stat_one, blk_in_one, blk_exc_one, blk_out_one]
    ])
    st.dataframe(df_detail_one)

if tab2 is not None:
    with tab2:
        st.header("Welfare decomposition — Infinite horizon (per-period steady state, scaled)")
        bar = (
            alt.Chart(bars_inf)
            .transform_fold(["CS", "PiAI", "PiC", "Admin", "Creation"], as_=["Component", "Value"])
            .mark_bar()
            .encode(
                x=alt.X("Regime:N", title="Regime"),
                y=alt.Y("sum(Value):Q", title="Scaled amounts"),
                color=alt.Color("Component:N", scale=alt.Scale(scheme="tableau10")),
                tooltip=["Regime:N", "Component:N", alt.Tooltip("sum(Value):Q", format=",.2f")]
            ).properties(height=420)
        )
        line_total = (
            alt.Chart(bars_inf)
            .mark_point(size=90, filled=True, color="black")
            .encode(x="Regime:N", y=alt.Y("TotalWelfare:Q", title=""), tooltip=["Regime:N", alt.Tooltip("TotalWelfare:Q", format=",.2f")])
        )
        st.altair_chart(bar + line_total, use_container_width=True)
        st.caption("Infinite-horizon bars show **steady-state per-period** values. Present value (PV) = per-period/(1−β) is in the table below.")
        st.subheader("Creators — incremental net (ΔΠᶜ), scaled")
        st.dataframe(bars_inf[["Regime","DeltaPiC"]])
        st.subheader("Present values (scaled)")
        pv_tbl = bars_inf[["Regime","PV_total"]].rename(columns={"PV_total":"Total Welfare PV"})
        st.dataframe(pv_tbl)
        st.subheader("Detailed values (unscaled per period)")
        df_detail_inf = pd.DataFrame([
            dict(Regime=b["name"], sigma=b["sigma"], A=b["A"], Df=b["Df"], K=b.get("K"), S=b["S"], Roy=b["Roy"],
                 CS=b["CS"], PiAI=b["PiAI"], Admin=b["Admin"], Creation=b["Creation"], TotalW=b["TotalW"],
                 PiC_transfer=b["PiC_transfer"], PiC_net=b["PiC_net"]) for b in [blk_stat_inf, blk_in_inf, blk_exc_inf, blk_out_inf]
        ])
        st.dataframe(df_detail_inf)

# ---------------- Optimize royalties by regime (current horizon) ----------------
active_horizon = "infty" if view_mode.startswith("Infinite") else "one"
regime_choice = st.selectbox("Regime to sweep (active horizon)", ["Statutory", "Opt-in", "Opt-out"], index=0)
with st.spinner("Sweeping r grid for selected regime..."):
    df_sweep = sweep_regime(regime_choice, 0.0, r_max_eff, int(params["grid_steps"]), params, horizon=active_horizon)
    opt = find_optima(df_sweep)
colw, colc, colf = st.columns(3)
colw.metric(f"{regime_choice} — r* (Welfare)", f"{opt['TotalW']['r']:.3f}")
colc.metric(f"{regime_choice} — r* (Creators' net)", f"{opt['PiC_net']['r']:.3f}")
colf.metric(f"{regime_choice} — r* (AI profit)", f"{opt['PiAI']['r']:.3f}")

# Summary table: optima across regimes (active horizon)
sum_rows = []
for reg in ["Statutory", "Opt-in", "Opt-out"]:
    dft = sweep_regime(reg, 0.0, r_max_eff, int(params["grid_steps"]), params, horizon=active_horizon)
    o = find_optima(dft)
    sum_rows.append(dict(
        Regime=reg,
        r_star_W=o["TotalW"]["r"],
        r_star_C=o["PiC_net"]["r"],
        r_star_AI=o["PiAI"]["r"],
        W_at_r=o["TotalW"]["value"],
        PiCnet_at_r=o["PiC_net"]["value"],
        PiAI_at_r=o["PiAI"]["value"]
    ))
st.subheader(f"Optimal r by regime (unscaled, grid search) — {'Infinite horizon' if active_horizon=='infty' else 'One period'}")
st.dataframe(pd.DataFrame(sum_rows))

# Plot sweep for selected regime
plot_df = df_sweep.rename(columns={"TotalW": "Total Welfare", "PiC_net": "Creator Net", "PiAI": "AI Profit"})
long = plot_df.melt("r", value_vars=["Total Welfare", "CS", "AI Profit", "Creator Net"], var_name="Metric", value_name="Value")
color_domain = ["Total Welfare", "CS", "AI Profit", "Creator Net"]
color_range  = ["#1f77b4",       "#2ca02c", "#ff7f0e",  "#d62728"]
lines = (
    alt.Chart(long)
    .mark_line()
    .encode(
        x=alt.X("r:Q", title="Royalty r"),
        y=alt.Y("Value:Q", title="Per-unit surplus / welfare"),
        color=alt.Color("Metric:N", scale=alt.Scale(domain=color_domain, range=color_range)),
        tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("r:Q", format=".3f"), alt.Tooltip("Value:Q", format=",.4f")]
    ).properties(height=420)
)
vd = pd.DataFrame([
    {"Metric": "Total Welfare", "r": opt["TotalW"]["r"]},
    {"Metric": "AI Profit",     "r": opt["PiAI"]["r"]},
    {"Metric": "Creator Net",   "r": opt["PiC_net"]["r"]},
])
rules = (
    alt.Chart(vd)
    .mark_rule(strokeDash=[6,4], size=2)
    .encode(x="r:Q", color=alt.Color("Metric:N", scale=alt.Scale(domain=color_domain, range=color_range)),
            tooltip=[alt.Tooltip("Metric:N"), alt.Tooltip("r:Q", format=".3f")])
)
st.altair_chart(lines + rules, use_container_width=True)
st.caption("Curves are unscaled (per-unit). Dashed verticals mark argmax r for Total Welfare, AI Profit, and Creator Net (active horizon).")

st.write("— End —")
