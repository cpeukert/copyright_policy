# AI Copyright Policy Dashboard — Revised Model + Scenarios (v11 self-contained)
# Implements "Option B" (incremental creation cost ΔT), share-lost delta_f,
# tau_in for Opt-in, and persistent scenario save/load/rename/delete.
#
# To run:
#   streamlit run ai_copyright_dashboard_v11.py

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
    N=212.0,
    n=0.4,
    months=12.0,
    beta=0.5,
    # Newer params (added in later versions of the app)
    Df0=100.0,       # baseline flow without AI (chosen here as a neutral default)
    delta_f=0.30,    # share of flow lost; Df_AI0 = (1 - delta_f)*Df0
    tau_in=None,     # if None -> defaults to tau
    r_out=0.0,       # default opt-out royalty
    r_stat=0.15,     # example starting r for statutory
    r_in=0.20,       # example starting r for opt-in
)

# ---------------- Utility ----------------
EPS = 1e-12
def nz(x):
    return np.where(np.abs(x) < EPS, 0.0, x)

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

def solve_flow_total(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df0, Df_AI0):
    """Solve F(A(Df),r) = ΔT(Df) for Df >= Df_AI0 with possible expansion > Df0.
       F = (1 - tau_reg) r S(A(Df), r) where A(Df) = θ(sigma) Ds + μ Df^α.
       Uses robust bracketing + bisection; if F is huge, returns a capped high solution.
    """
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
    H_lo = H(lo)

    hi = max(float(Df0), lo + 1.0)
    H_hi = H(hi)
    iters = 0
    while H_hi > 0.0 and iters < 60:
        hi *= 2.0
        H_hi = H(hi)
        iters += 1

    if H_hi > 0.0:
        # Funding is extremely large — return hi as a practical cap
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

# ---------------- Scenarios (persisted to disk) ----------------
SCEN_DIR = (Path(__file__).parent / "scenarios")
SCEN_DIR.mkdir(exist_ok=True)

PARAM_KEYS = list(DEFAULTS.keys())

def _safe_name(name: str) -> str:
    name = (name or "").strip()
    name = re.sub(r"[^A-Za-z0-9 \-_()]+", "_", name)
    name = re.sub(r"\s+", " ", name).strip()
    return name[:120] if name else ""

def list_scenarios():
    return sorted([p.stem for p in SCEN_DIR.glob("*.json")])

def read_scenario(name: str):
    safe = _safe_name(name)
    p = SCEN_DIR / f"{safe}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None

def write_scenario(name: str, data: dict):
    safe = _safe_name(name)
    if not safe:
        raise ValueError("Empty scenario name.")
    p = SCEN_DIR / f"{safe}.json"
    p.write_text(json.dumps(data, indent=2))
    return safe

def delete_scenario(name: str):
    safe = _safe_name(name)
    p = SCEN_DIR / f"{safe}.json"
    if p.exists():
        p.unlink()

def rename_scenario(old: str, new: str):
    old_safe = _safe_name(old)
    new_safe = _safe_name(new)
    if not old_safe or not new_safe or old_safe == new_safe:
        return False
    p_old = SCEN_DIR / f"{old_safe}.json"
    p_new = SCEN_DIR / f"{new_safe}.json"
    if not p_old.exists() or p_new.exists():
        return False
    p_old.rename(p_new)
    return True

def collect_current_params_from_state(state):
    data = {}
    for k in PARAM_KEYS:
        if k in state:
            v = state[k]
            try:
                data[k] = float(v) if isinstance(v, (int, float)) else v
            except Exception:
                data[k] = v
    return data

# --- Apply pending scenario load before any widgets are created ---
if "_pending_load" in st.session_state:
    _data = st.session_state.pop("_pending_load")
    for k, v in _data.items():
        st.session_state[k] = v
    # keep tau_in defaulting to tau if not present
    if st.session_state.get("tau_in") is None:
        st.session_state["tau_in"] = st.session_state.get("tau", 0.10)

# Initialize session state
if "init" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state.setdefault(k, v if v is not None else DEFAULTS["tau"])
    if st.session_state.get("tau_in") is None:
        st.session_state["tau_in"] = st.session_state.get("tau", 0.10)
    st.session_state["init"] = True



# ---------------- UI ----------------
st.set_page_config(page_title="AI Copyright Policy Dashboard", layout="wide")
st.title("AI Copyright Policy Dashboard")

# Initialize session state
if "init" not in st.session_state:
    for k, v in DEFAULTS.items():
        st.session_state[k] = v if v is not None else DEFAULTS["tau"]
    if st.session_state.get("tau_in") is None:
        st.session_state["tau_in"] = st.session_state.get("tau", 0.10)
    st.session_state["init"] = True

# Scenarios UI
st.sidebar.subheader("Scenarios")

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
            # Defer setting session_state until the very start of next run
            st.session_state["_pending_load"] = data
            st.sidebar.success(f"Loaded scenario: {chosen}")
            st.rerun()


col1, col2 = st.sidebar.columns([1,1], gap="small")
new_name2 = st.sidebar.text_input("Rename selected to", value="")
if col1.button("✏️ Rename"):
    if chosen == "(none)":
        st.sidebar.warning("Select a scenario first.")
    else:
        if not new_name2:
            st.sidebar.warning("Enter a new name.")
        elif _safe_name(new_name2) in scen_list:
            st.sidebar.error("A scenario with that name already exists.")
        else:
            ok = rename_scenario(chosen, new_name2)
            if ok:
                st.sidebar.success("Renamed.")
                st.rerun()
            else:
                st.sidebar.error("Rename failed.")
confirm_del = st.sidebar.checkbox("Type to confirm deletion", value=False, help="Tick before deleting to avoid accidents.")
if col2.button("🗑️ Delete"):
    if chosen == "(none)":
        st.sidebar.warning("Select a scenario to delete.")
    elif not confirm_del:
        st.sidebar.error("Please tick the confirmation checkbox.")
    else:
        delete_scenario(chosen)
        st.sidebar.success(f"Deleted: {chosen}")
        st.rerun()

# Sidebar controls
st.sidebar.divider()
st.sidebar.header("Parameters")

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

st.sidebar.subheader("Creators: harms & action costs")
st.sidebar.slider("δ (perceived net effect of AI on creators)", min_value=-1.0, max_value=1.0, step=0.001, format="%.3f", key="delta")
st.sidebar.number_input("x_in (opt-in action cost)", min_value=-1.0, max_value=1.0, step=0.001, format="%.3f", key="x_in")
st.sidebar.number_input("x_out (opt-out action cost)", min_value=-1.0, max_value=1.0, step=0.001, format="%.3f", key="x_out")

st.sidebar.subheader("Flow baselines")
st.sidebar.number_input("D_f^0 (flow without AI)", min_value=0.0, step=0.1, format="%.2f", key="Df0")
st.sidebar.number_input("δ_f (share of flow lost; D_f^{AI}(0)=(1−δ_f)·D_f^0)", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="delta_f")

st.sidebar.subheader("Admin leakage in Opt-in")
st.sidebar.slider("τ_in (admin leakage in Opt-in)", 0.0, 0.99, key="tau_in")

st.sidebar.subheader("Search & aggregation")
st.sidebar.slider("Max r to search over", 0.0, 0.99, key="r_max")
st.sidebar.slider("Grid steps for r search", 101, 3001, 701, 100, key="grid_steps")
st.sidebar.number_input("Adult population (mio)", min_value=1.0, step=1.0, format="%.0f", key="N")
st.sidebar.number_input("AI adoption share", min_value=0.0, max_value=1.0, step=0.01, format="%.2f", key="n")
st.sidebar.number_input("Months", min_value=1.0, step=1.0, format="%.0f", key="months")

# Scale factor for aggregation (keep consistent with prior versions)
scale_factor = float(st.session_state["N"] * st.session_state["n"] * (st.session_state["months"]))
st.sidebar.text(f"Scale factor: {scale_factor:,.1f}")



# ---------------- Model helpers for regimes ----------------
def sigma_tilde_opt_in(Ds, r_in, delta, x_in):
    return float(np.clip(logit(float(Ds) * (float(r_in) + float(delta) - float(x_in))), 0.0, 1.0))

def sigma_tilde_opt_out(Ds, r_out, delta, x_out):
    sigma_out = float(np.clip(logit(float(Ds) * (float(delta) - float(r_out) - float(x_out))), 0.0, 1.0))
    return 1.0 - sigma_out

def compute_primitives(A, B, r, kappa):
    S = revenue_S(A, B, r, kappa)
    Roy = r * S
    CS = CS_of(A, B, r, kappa)
    PiAI = PiAI_of(A, B, r, kappa)
    return dict(A=A, S=S, Roy=Roy, CS=CS, PiAI=PiAI)

def regime_block(name, sigma_tilde, r, tau_reg, params, allow_tau_in=False):
    Ds=params["Ds"]; B=params["B"]; mu=params["mu"]; alpha=params["alpha"]
    phi=params["phi"]; psi=params["psi"]; kappa=params["kappa"]
    gamma=params["gamma"]; Df0=params["Df0"]; delta_f=params["delta_f"]
    Df_AI0 = (1.0 - delta_f) * Df0

    if r <= 0.0:  # exception-like static (ΔT = 0) when r=0
        Df = Df_AI0
    else:
        Df = solve_flow_total(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df0, Df_AI0)

    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)

    Admin_leak = tau_reg * prims["Roy"]
    Creation_cost = delta_T(Df, Df_AI0, phi, psi)  # incremental
    TotalW = prims["CS"] + prims["PiAI"] - Admin_leak - Creation_cost

    PiC_transfer = (1.0 - tau_reg) * r * prims["S"]
    PiC_net = PiC_transfer - Creation_cost

    # ↓↓↓ remove A=A here; **prims already includes "A"
    return dict(
        name=name, r=r, sigma=sigma_tilde, Df=Df,
        **prims,
        Admin=Admin_leak, Creation=Creation_cost, TotalW=TotalW,
        PiC_transfer=PiC_transfer, PiC_net=PiC_net
    )


# ---------------- Compute all regimes ----------------
Ds=st.session_state["Ds"]; B=st.session_state["B"]; mu=st.session_state["mu"]; alpha=st.session_state["alpha"]
phi=st.session_state["phi"]; psi=st.session_state["psi"]; kappa=st.session_state["kappa"]
tau=st.session_state["tau"]; gamma=st.session_state["gamma"]; delta=st.session_state["delta"]
x_in=st.session_state["x_in"]; x_out=st.session_state["x_out"]
Df0=st.session_state["Df0"]; delta_f=st.session_state["delta_f"]
tau_in=st.session_state["tau_in"]; r_max=st.session_state["r_max"]
grid_steps=int(st.session_state["grid_steps"])
N=st.session_state["N"]; n=st.session_state["n"]; months=st.session_state["months"]

Df_AI0 = (1.0 - delta_f) * Df0

# Sliders for r choices (one per funding regime)
colR1, colR2, colR3 = st.columns(3)
with colR1:
    st.subheader("Statutory license r")
    r_stat = st.slider("r_stat", 0.0, float(r_max), float(st.session_state.get("r_stat", 0.15)), 0.01, key="r_stat")
with colR2:
    st.subheader("Opt-in negotiated r")
    r_in = st.slider("r_in", 0.0, float(r_max), float(st.session_state.get("r_in", 0.20)), 0.01, key="r_in")
with colR3:
    st.subheader("Opt-out royalty r_out (typically 0)")
    r_out = st.slider("r_out", 0.0, float(r_max), float(st.session_state.get("r_out", 0.0)), 0.01, key="r_out")

# Exception
sigma_exc = 1.0
blk_exc = regime_block("Exception (r=0)", sigma_exc, 0.0, 0.0, st.session_state)

# Statutory
sigma_stat = 1.0
blk_stat = regime_block("Statutory", sigma_stat, r_stat, tau, st.session_state)

# Opt-in
sigma_in = sigma_tilde_opt_in(Ds, r_in, delta, x_in)
blk_in = regime_block("Opt-in", sigma_in, r_in, tau_in, st.session_state, allow_tau_in=True)

# Opt-out
sigma_out = sigma_tilde_opt_out(Ds, r_out, delta, x_out)
blk_out = regime_block("Opt-out", sigma_out, r_out, tau, st.session_state)

# ---------------- Display ----------------
st.header("Key quantities")
col1, col2, col3, col4 = st.columns(4)
col1.metric("D_f^{AI}(0) (suppressed baseline)", f"{Df_AI0:,.3f}")
col2.metric("σ̃_in (at r_in)", f"{sigma_in:,.3f}")
col3.metric("σ̃_out (at r_out)", f"{sigma_out:,.3f}")
col4.metric("θ(σ) for opt-in", f"{theta_of_sigma(sigma_in, gamma):,.3f}")

# Build bars DataFrame
rows = []
for blk in [blk_stat, blk_in, blk_exc, blk_out]:
    rows.append(dict(
        Regime=blk["name"],
        r=blk["r"],
        CS=blk["CS"] * scale_factor,
        PiAI=blk["PiAI"] * scale_factor,
        PiC=blk["PiC_transfer"] * scale_factor,   # transfer ONLY
        Admin=-blk["Admin"] * scale_factor,
        Creation=-blk["Creation"] * scale_factor, # incremental ΔT, neg in bar
        TotalWelfare=blk["TotalW"] * scale_factor,
        DeltaPiC=(blk["PiC_net"]) * scale_factor, # incremental net for creators
    ))
df_bars = pd.DataFrame(rows)

st.header("Welfare decomposition (scaled)")
bar = (
    alt.Chart(df_bars)
    .transform_fold(
        ["CS", "PiAI", "PiC", "Admin", "Creation"],
        as_=["Component", "Value"]
    )
    .mark_bar()
    .encode(
        x=alt.X("Regime:N", title="Regime"),
        y=alt.Y("sum(Value):Q", title="Scaled amounts"),
        color=alt.Color("Component:N", scale=alt.Scale(scheme="tableau10")),
        tooltip=["Regime:N", "Component:N", alt.Tooltip("sum(Value):Q", format=",.2f")]
    )
    .properties(height=420)
)

line_total = (
    alt.Chart(df_bars)
    .mark_point(size=90, filled=True)
    .encode(
        x="Regime:N",
        y=alt.Y("TotalWelfare:Q", title=""),
        color=alt.value("black"),
        tooltip=["Regime:N", alt.Tooltip("TotalWelfare:Q", format=",.2f")]
    )
)
st.altair_chart(bar + line_total, use_container_width=True)
st.caption("Note: Πᶜ bar shows transfer-only ((1−τ)·r·S). Transfers do not enter Total Welfare. Admin and Creation (ΔT) are real costs.")

st.subheader("Creators — incremental net (ΔΠᶜ = F − ΔT), scaled")
st.dataframe(df_bars[["Regime","DeltaPiC"]])

# Detailed table
st.subheader("Detailed values (unscaled)")
df_detail = pd.DataFrame([
    dict(Regime=blk["name"], sigma=blk["sigma"], A=blk["A"], Df=blk["Df"], S=blk["S"], Roy=blk["Roy"],
         CS=blk["CS"], PiAI=blk["PiAI"], Admin=blk["Admin"], Creation=blk["Creation"], TotalW=blk["TotalW"],
         PiC_transfer=blk["PiC_transfer"], PiC_net=blk["PiC_net"])
    for blk in [blk_stat, blk_in, blk_exc, blk_out]
])
st.dataframe(df_detail)

st.write("— End —")
