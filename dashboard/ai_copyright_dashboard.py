# Run: streamlit run ai_copyright_dashboard.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------
# HARD-CODED DEFAULTS (unchanged ones kept verbatim)
# ---------------------------------
DEFAULTS_literature = dict(
    Ds=399.0, # Subtracted Df0 from A
    B=834.0,
    mu=0.9975,
    alpha=0.0013,
    phi=1.14*1.436, # Aguiar & Waldfogel (2018) but in 2025 prices
    psi=1.001,
    kappa=0.988,
    tau=0.10,       # leakage used in statutory & opt-out
    tau_in=0.0,
    gamma=2.28,
    delta=0.0,
    x_in=0.018,
    x_out=0.010,
    r_max=0.99,
    grid_steps=1001,
    N=212.0,
    n=0.4,
    months=12.0,
    beta=0.5,
    g0=0.0142, # (0.51/(3*12)) from Peukert et al. (2025) "Unsplash"
    delta_f=0.214,    # share of flow lost (Df_AI0 = (1-delta_f) * Df0); default 0.3 lost
)

DEFAULTS_stat_r0 = dict(
    Ds=369.0, # Subtracted Df0 from A
    B=834.0,
    mu=0.9975,
    alpha=2.0,
    phi=0.5, # Aguiar & Waldfogel (2018) but in 2025 prices
    psi=1.001,
    kappa=0.988,
    tau=0.10,       # leakage used in statutory & opt-out
    tau_in=0.0,
    gamma=2.28,
    delta=0.0,
    x_in=0.018,
    x_out=0.010,
    r_max=0.99,
    grid_steps=1001,
    N=212.0,
    n=0.4,
    months=12.0,
    beta=0.5,
    g0=0.0142, # (0.51/(3*12)) from Peukert et al. (2025) "Unsplash"
    delta_f=0.214,    # share of flow lost (Df_AI0 = (1-delta_f) * Df0); default 0.3 lost
)

# --- Scenario presets ---
SCENARIOS = {
    "Literature baseline": DEFAULTS_literature,
    "STAT r>0 (illustrative)": DEFAULTS_stat_r0,
}

# --- One-time init of session state with a default scenario ---
if "init_done" not in st.session_state:
    st.session_state.update(SCENARIOS["Literature baseline"])
    st.session_state["scenario"] = "Literature baseline"
    st.session_state["init_done"] = True

# --- Helper to apply a scenario and refresh the UI ---
def apply_scenario():
    preset = SCENARIOS[st.session_state["scenario"]]
    for k, v in preset.items():
        st.session_state[k] = v
    st.rerun()



EPS = 1e-12
def nz(x):
    return np.where(np.abs(x) < EPS, EPS, x)

def logit(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

def theta_of_sigma(sigma, gamma):
    sigma = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
    return np.power(sigma, gamma)

def A_from_sigma_Df(Ds, sigma, gamma, mu, alpha, Df):
    return theta_of_sigma(sigma, gamma) * Ds + mu * (np.asarray(Df, dtype=float) ** alpha)

def pq_advalorem(A, B, r, kappa):
    denom = 1.0 - kappa * r
    p = (1.0 - kappa) * A / nz(denom)
    Q = (kappa * (1.0 - r) * A) / nz(B * denom)
    return p, Q

def revenue_S(A, B, r, kappa):
    p, Q = pq_advalorem(A, B, r, kappa)
    return p * Q

def rho_of_r(r, kappa):
    return (kappa * (1.0 - kappa) * (1.0 - r)) / ((1.0 - kappa * r) ** 2)

# ---------------- Flow solver (MODEL-CONSISTENT) ----------------
# Model: total flow Df ∈ [Df_AI0, Df0] is financed by F >= ΔT(Df) = φ Df^ψ - φ (Df_AI0)^ψ.
# No direct "stock-harm bill". Admin leakage uses regime-specific τ_reg.
def solve_flow_total(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_reg, B, kappa, r, Df0, Df_AI0):
    # No royalties -> stuck at suppressed baseline
    if r <= 0.0:
        return float(Df_AI0)

    # Define feasibility mapping at given Df (through A(Df) which affects S(A,r) and hence the pool)
    def pool(Df):
        A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
        S = revenue_S(A, B, r, kappa)
        return (1.0 - tau_reg) * r * S

    # Needed funding for Df: ΔT(Df) = φ Df^ψ - φ (Df_AI0)^ψ
    base_cost = phi * (Df_AI0 ** psi)

    # Check corners
    F_lo = pool(Df_AI0)
    need_lo = phi * (Df_AI0 ** psi) - base_cost  # == 0
    if F_lo < need_lo - 1e-14:  # always false since need_lo==0, but kept for symmetry
        return float(Df_AI0)

    F_hi = pool(Df0)
    need_hi = phi * (Df0 ** psi) - base_cost
    if F_hi >= need_hi - 1e-14:
        return float(Df0)

    # Otherwise, solve for interior root of H(Df) = pool(Df) - [φ Df^ψ - φ (Df_AI0)^ψ] = 0
    def H(Df):
        return pool(Df) - (phi * (Df ** psi) - base_cost)

    lo, hi = float(Df_AI0), float(Df0)
    H_lo, H_hi = H(lo), H(hi)

    # Bisection (monotonicity: RHS convex in Df, LHS increasing in Df through A(Df))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        H_mid = H(mid)
        if np.isnan(H_mid):
            break
        if abs(H_mid) < 1e-10 or (hi - lo) < 1e-10:
            return float(mid)
        # We expect H(lo) >= 0 (since need_lo=0) and H(hi) <= 0 if underfunded for full restoration.
        # If H_mid >= 0, move right boundary leftward to mid; else move left boundary rightward.
        if H_mid >= 0.0:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))

# ---------------- Inclusion logits ----------------
def sigma_opt_in(Ds, r_in, delta, x_in):
    return float(np.clip(logit(Ds * (r_in + delta - x_in)), 0.0, 1.0))

def sigma_tilde_opt_out(Ds, r_out, delta, x_out):
    sigma_out = float(np.clip(logit(Ds * (delta - r_out - x_out)), 0.0, 1.0))
    return 1.0 - sigma_out

# ---------------- Primitives ----------------
def compute_primitives(A, B, r, kappa):
    p, Q = pq_advalorem(A, B, r, kappa)
    S = p * Q
    CS = 0.5 * B * (Q ** 2)
    PiAI = (1.0 - r) * S
    Roy = r * S
    return dict(p=p, Q=Q, S=S, CS=CS, PiAI=PiAI, Roy=Roy)

# ---------------- Incremental creation cost ΔT ----------------
def delta_T(Df, Df_AI0, phi, psi):
    return phi * (float(Df) ** psi) - phi * (float(Df_AI0) ** psi)


# ---------------- UI ----------------
st.set_page_config(page_title="AI Copyright Policy Dashboard", layout="wide")
st.title("AI Copyright Policy Dashboard")

with st.expander("About this dashboard", expanded=True):
    st.write(
        """
    ### What this dashboard is (and isn’t)

    This dashboard accompanies a briefing prepared by Christian Peukert for the JURI Committee of the European Parliament. 
    It is an **interactive illustration** of an economic model that links copyright policy to the supply of AI training data and, ultimately, to welfare. It’s designed to help **policymakers** and advisors build intuition about the **trade-offs** across policy options.

    > **Not a quantification exercise.**  
    > The results here are **not** precise forecasts. Key parameters (how much existing data contributes to AI quality, how costly new creation is, how strongly royalties distort prices, etc.) are **extremely hard to measure empirically** and can vary by sector and over time. Treat the charts as **scenario analysis** that shows directions and sensitivities, not fixed numbers.

    ---

    ### Policy context (EU)

    AI systems learn from large corpora of copyrighted and non-copyrighted works. EU policy choices revolve around *how* those works may be used for training and *how* to finance new creation if training reduces creators’ incentives. Four stylised regimes are compared:

    - **AI exception** — all existing stock is usable for training **without compensation**. Maximises coverage of existing data, but offers **no funding** for new works.
    - **Statutory licence** — all stock is usable, and AI firms pay a **regulated royalty** that funds creators (net of admin costs). Keeps coverage high and can **restore** or **expand** new creation if the royalty is large enough.
    - **Opt-in** — works are excluded by default; rightsholders must **grant access** (often tied to a negotiated royalty). Coverage depends on participation; funding exists if royalties are paid.
    - **Opt-out** — works are included by default; rightsholders can **withdraw** (often with **no royalty** in practice). Coverage is high; funding for new creation is typically **absent**.

    ---

    ### Economic intuition of the model

    - **Value of AI to users** depends on two inputs:  
    (1) the **stock** of existing works included in training; and  
    (2) the **flow** of newly created works going forward.
    - Excluding stock lowers AI quality disproportionately when representativeness matters (captured by the **coverage penalty** γ).
    - **Royalties** (ad valorem rate *r* on sales) do two things:  
    • **distort prices/quantities** in the AI market (a cost); and  
    • create a **funding pool** (after leakage τ) that can **restore incentives** for new creation above the suppressed baseline.
    - New creation faces **convex costs** (φ, ψ). Funding must cover the **incremental** cost relative to the no-funding baseline.
    - **Welfare** = consumer surplus + AI-firm profit − admin leakage − incremental creation cost. Transfers to creators are **distributional**; they matter for incentives only insofar as they finance new works.

    **Key trade-off:** A higher *r* can **raise future value** (by financing more new works) but **reduces current surplus** (through price distortions and leakage). The welfare-maximising *r*—when it exists—balances these forces and may be zero if dynamic gains are weak or the wedge is too costly.

    ---

    ### How to use the dashboard

    - Pick a **Scenario** (preset assumptions) and adjust parameters in the **left sidebar**:
    - **γ (coverage penalty)** — how much AI quality drops when fewer works are included.  
    - **κ (pass-through / conduct)** — how strongly royalties distort prices and quantities.  
    - **τ / τ_in (admin leakage)** — how much of the royalty is lost in administration.  
    - **μ, α (returns to new data)** — how much additional new works raise AI value and with what diminishing returns.  
    - **φ, ψ (creation costs)** — level and curvature of costs for producing new works.  
    - **δ, x_in, x_out** — creators’ perceived effects and action costs shaping opt-in/opt-out participation.
    - The **top bar chart** compares **welfare and its components** across regimes under your current parameters; detailed regime plots and tables follow below.

    *Interpretation tip:* If raising *r* barely changes welfare, it usually means either (i) the revenue wedge is very costly (high κ or τ), or (ii) the dynamic payoff from more data is small (low μ or very high ψ). If welfare improves with *r*, your scenario implies that **funding new creation** outweighs **distortion/leakage** costs.

    ---

    ### Why this matters

    Different legal designs shift the balance between **immediate access** to existing works and **long-run incentives** to create new ones. This tool helps you see how those forces move when assumptions change, so you can stress-test policy ideas and understand where results are robust—and where they hinge on empirical uncertainties.

    ---

    ### Contact

    **Christian Peukert**  
    Professor of Digitization, Innovation and Intellectual Property  
    University of Lausanne – Faculty of Business and Economics (HEC)  
    CH-1015 Lausanne, Switzerland  
    Email: [christian.peukert@unil.ch](mailto:christian.peukert@unil.ch)  
    [www.christian-peukert.com](https://www.christian-peukert.com)  
    [www.digital-markets.ch](https://www.digital-markets.ch)
    """
    )




# ---- Sidebar

SCENARIOS = {
    "Literature baseline": DEFAULTS_literature,
    "STAT r>0 (illustrative)": DEFAULTS_stat_r0,
}

def seed_from(preset: dict):
    for k, v in preset.items():
        if k not in st.session_state:
            st.session_state[k] = v

if "scenario" not in st.session_state:
    st.session_state["scenario"] = "Literature baseline"
    seed_from(SCENARIOS[st.session_state["scenario"]])

# When the scenario changes, overwrite and rerun
def apply_scenario():
    preset = SCENARIOS[st.session_state["scenario"]]
    for k, v in preset.items():
        st.session_state[k] = v

st.sidebar.header("Scenarios")
st.sidebar.radio(
    "Preset",
    list(SCENARIOS.keys()),
    key="scenario",
    on_change=apply_scenario,
)

# ---- Sidebar

st.sidebar.header("Structural parameters")
Ds    = st.sidebar.slider("D_s (stock of existing works)", 0.0, 1000.0, step=1.0, format="%.1f", key="Ds")
B     = st.sidebar.slider("B (demand slope)", 1e-6, 1000.0, step=1.0, format="%.1f", key="B")
mu    = st.sidebar.slider("μ (scale of flow in A)", 0.0, 5.0, step=0.0001, format="%.4f", key="mu")
alpha = st.sidebar.slider("α (diminishing returns to flow)", 0.0, 10.0, step=0.0001, format="%.4f", key="alpha")
phi   = st.sidebar.slider("φ (flow cost scale)", 0.0, 10.0, step=0.01,  format="%.2f", key="phi")
psi   = st.sidebar.slider("ψ (flow cost curvature)", 1.001, 10.0, step=0.001, format="%.3f", key="psi")
kappa = st.sidebar.slider("κ (pass-through)", 0.01, 1.00, step=0.001, format="%.3f", key="kappa")
tau   = st.sidebar.slider("τ (admin leakage, STAT & opt-out)", 0.0, 0.99, step=0.01, key="tau")
gamma = st.sidebar.slider("γ (coverage penalty on stock)", 1.0, 10.0, step=0.01, key="gamma")

st.sidebar.header("Harms & action costs")
delta = st.sidebar.slider("δ (negative = benefit, positive = harm)", -1.0, 1.0, step=0.001, format="%.3f", key="delta")
x_in  = st.sidebar.slider("x_in (opt-in action cost)", 0.0, 1.0, step=0.001, format="%.3f", key="x_in")
x_out = st.sidebar.slider("x_out (opt-out action cost)", 0.0, 1.0, step=0.001, format="%.3f", key="x_out")

st.sidebar.header("Flow baselines")
g0     = st.sidebar.slider("g0 (stock growth per month)", 0.0000, 1.0000, step=0.001, format="%.4f", key="g0")
Df0    = g0 * Ds
delta_f = st.sidebar.slider("δ_f (suppressed share; D_f^{AI}(0)=(1-δ_f)·D_f^0)", 0.000, 1.000, step=0.001, format="%.3f", key="delta_f")
Df_AI0 = float((1.0 - delta_f) * Df0)

st.sidebar.header("Search / solution settings")
r_max      = st.sidebar.slider("Max r to search over", 0.0, 0.99, step=0.01, key="r_max")
grid_steps = st.session_state.get("grid_steps", 1001)

st.sidebar.header("Bargaining (opt-in)")
beta   = st.sidebar.slider("β (creators' bargaining power)", 0.0, 1.0, step=0.01, key="beta")
tau_in = st.sidebar.slider("τ_in (admin leakage in Opt-in)", 0.0, 0.99, step=0.01, key="tau_in")

st.sidebar.header("Aggregation")
N      = st.sidebar.number_input("Adult population (mio)", 1.0, step=1.0, format="%.0f", key="N")
n      = st.sidebar.number_input("AI adoption share", 0.0, 1.0, step=0.01, format="%.2f", key="n")
months = st.sidebar.number_input("Months", 1.0, step=1.0, format="%.0f", key="months")
scale_factor = N * n * months
st.sidebar.text(f"Scale factor: {scale_factor:,.1f}")


#if psi <= 1.0:
#    st.error("Regularity requires ψ > 1. Increase ψ.")
#    st.stop()

#x_in=delta+x_in
#x_out=delta+x_out

# -- Welfare comparison across regimes
top_cmp = st.container()


# ---------------- Statutory license ----------------
st.header("Statutory license (σ̃ = 1)")
r_grid = np.linspace(0.0, r_max, int(grid_steps))

def solve_stat_for_r(r):
    sigma_tilde = 1.0
    Df = solve_flow_total(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, Df0, Df_AI0)
    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - delta_T(Df, Df_AI0, phi, psi)
    out = dict(r=r, sigma=sigma_tilde, Df=Df, A=A, W=W); out.update(prims)
    return out

df_stat = pd.DataFrame([solve_stat_for_r(r) for r in r_grid])
idx_stat_star = int(np.nanargmax(df_stat["W"].values))
r_stat_star = float(df_stat.loc[idx_stat_star, "r"])

c0, c1, c2, c3 = st.columns(4)
with c0: st.metric("r* (STAT, max W)", f"{r_stat_star:.4f}")
with c1: st.metric("W* (scaled)", f"{(df_stat.loc[idx_stat_star, 'W'] * scale_factor):,.2f}")
with c2: st.metric("A* (unscaled)", f"{df_stat.loc[idx_stat_star, 'A']:.4f}")
with c3: st.metric("D_f* (unscaled)", f"{df_stat.loc[idx_stat_star, 'Df']:.4f}")

df_stat_scaled = df_stat.copy()
for col in ["W","CS","PiAI","Roy"]:
    df_stat_scaled[col] = df_stat_scaled[col] * scale_factor

chart_stat = (
    alt.Chart(df_stat_scaled.melt(id_vars=["r"], value_vars=["W","CS","PiAI","Roy"], var_name="Component", value_name="Value"))
    .mark_line()
    .encode(
        x=alt.X("r:Q", title="r"),
        y=alt.Y("Value:Q", title="Value (scaled)"),
        color=alt.Color("Component:N", legend=alt.Legend(title="")),
        tooltip=["Component:N", alt.Tooltip("r:Q", format=".3f"), alt.Tooltip("Value:Q", format=",.2f")]
    )
    .properties(height=340)
)
rule_star = alt.Chart(pd.DataFrame({"r":[r_stat_star]})).mark_rule(color="red", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_stat + rule_star, use_container_width=True)

# ---------------- AI exception ----------------
st.header("AI exception")
sigma_exc = 1.0; r_exc = 0.0
Df_exc = float(Df_AI0)  # suppressed baseline (no funding)
A_exc = A_from_sigma_Df(Ds, sigma_exc, gamma, mu, alpha, Df_exc)
prims_exc = compute_primitives(A_exc, B, r_exc, kappa)
W_exc = prims_exc["CS"] + prims_exc["PiAI"]  # ΔT=0 at baseline
row_exc = dict(Regime="Exception", r=r_exc, sigma=sigma_exc, Df=Df_exc, A=A_exc, W=W_exc); row_exc.update(prims_exc)
st.write(pd.DataFrame([row_exc])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# ---------------- Opt-out ----------------
st.header("Opt-out")
r_out = st.slider("r_out (opt-out royalty, typically 0)", 0.0, float(r_max), 0.0, 0.01)
sigma_tilde_out = sigma_tilde_opt_out(Ds, r_out, delta, x_out)
Df_out = solve_flow_total(Ds, sigma_tilde_out, gamma, mu, alpha, phi, psi, tau, B, kappa, r_out, Df0, Df_AI0)
A_out = A_from_sigma_Df(Ds, sigma_tilde_out, gamma, mu, alpha, Df_out)
prims_out = compute_primitives(A_out, B, r_out, kappa)
W_out = prims_out["CS"] + prims_out["PiAI"] - tau * prims_out["Roy"] - delta_T(Df_out, Df_AI0, phi, psi)
row_out = dict(Regime="Opt-out", r=r_out, sigma=sigma_tilde_out, Df=Df_out, A=A_out, W=W_out); row_out.update(prims_out)
cA, cB = st.columns(2)
with cA: st.metric("σ̃ (included share)", f"{sigma_tilde_out:.4f}")
with cB: st.metric("D_f (opt-out)", f"{Df_out:.4f}")
st.write(pd.DataFrame([row_out])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# ---------------- Opt-in (τ_in from sidebar; default = τ) ----------------
st.header("Opt-in (negotiated r)")

def solve_optin_for_r(r):
    sigma_in = sigma_opt_in(Ds, r, delta, x_in)
    Df_in = solve_flow_total(Ds, sigma_in, gamma, mu, alpha, phi, psi, tau_in, B, kappa, r, Df0, Df_AI0)
    A_in = A_from_sigma_Df(Ds, sigma_in, gamma, mu, alpha, Df_in)
    prims = compute_primitives(A_in, B, r, kappa)

    # Creators' transfer (for display)
    PiC_transfer = (1.0 - tau_in) * r * prims["S"]
    # Creators' incremental net surplus ΔΠ^C = F - ΔT
    PiC_net = PiC_transfer - delta_T(Df_in, Df_AI0, phi, psi)

    # Welfare = CS + PiAI - Admin - Creation  (model-consistent)
    W = prims["CS"] + prims["PiAI"] - tau_in * prims["Roy"] - delta_T(Df_in, Df_AI0, phi, psi)

    out = dict(r=r, sigma=sigma_in, Df=Df_in, A=A_in, W=W,
               PiC_transfer=PiC_transfer, PiC_net=PiC_net)
    out.update(prims)
    return out

df_in = pd.DataFrame([solve_optin_for_r(r) for r in r_grid])

# Use creators' NET surplus to find r^C
idx_ai = int(np.nanargmax(df_in["PiAI"].values))
idx_c  = int(np.nanargmax(df_in["PiC_net"].values))
idx_w  = int(np.nanargmax(df_in["W"].values))

# Nash bargaining objective (uses net Πᶜ too)
def nash_value(row, beta):
    PiC = row["PiC_net"]; PiAI = row["PiAI"]
    if (PiC is None) or (PiAI is None) or (PiC <= 0.0) or (PiAI <= 0.0):
        return -np.inf
    return beta * np.log(PiC) + (1.0 - beta) * np.log(PiAI)

df_in["NashVal"] = [nash_value(row, float(beta)) for _, row in df_in.iterrows()]
if np.all(~np.isfinite(df_in["NashVal"].values)):
    df_in["NashVal"] = float(beta) * df_in["PiC_net"].values + (1.0 - float(beta)) * df_in["PiAI"].values
idx_b = int(np.nanargmax(df_in["NashVal"].values))

r_in_ai = float(df_in.loc[idx_ai, "r"])
r_in_c  = float(df_in.loc[idx_c,  "r"])
r_in_w  = float(df_in.loc[idx_w,  "r"])
r_in_b  = float(df_in.loc[idx_b,  "r"])

sigma_tilde_in_r0 = sigma_opt_in(Ds, 0, delta, x_in)

c0, c1, c2, c3, c4 = st.columns(5)
with c0: st.metric("σ̃ (included share, r=0)", f"{sigma_tilde_in_r0:.4f}")
with c1: st.metric("r_in^AI (max π^AI)", f"{r_in_ai:.4f}")
with c2: st.metric("r_in^C (max Π^C_net)",  f"{r_in_c:.4f}")
with c3: st.metric("r_in^W (max W)",    f"{r_in_w:.4f}")
with c4: st.metric("r_in^B (Nash, β)",  f"{r_in_b:.4f}")

# σ^in curve & bargaining line (unchanged)
df_sigma_in = df_in[["r","sigma"]].copy()
chart_sigma_in = (
    alt.Chart(df_sigma_in)
    .mark_line()
    .encode(
        x=alt.X("r:Q", title="r_in"),
        y=alt.Y("sigma:Q", title="σ^in (included share)"),
        tooltip=[alt.Tooltip("r:Q", format=".3f"), alt.Tooltip("sigma:Q", format=".4f")]
    )
    .properties(height=260, title="Opt-in: σ^in vs r_in")
)
rule_bargain = alt.Chart(pd.DataFrame({"r":[r_in_b]})).mark_rule(color="green", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_sigma_in + rule_bargain, use_container_width=True)

# Scaled objective plot for opt-in (show creators' NET surplus)
df_in_scaled = df_in.copy()
for col in ["W","PiAI","PiC_net"]:
    df_in_scaled[col] = df_in_scaled[col] * scale_factor
chart_in = (
    alt.Chart(df_in_scaled.melt(id_vars=["r"], value_vars=["W","PiAI","PiC_net"], var_name="Objective", value_name="Value"))
    .mark_line()
    .encode(
        x=alt.X("r:Q", title="r_in"),
        y=alt.Y("Value:Q", title="Scaled objective value"),
        color=alt.Color("Objective:N", legend=alt.Legend(title="")),
        tooltip=["Objective:N", alt.Tooltip("r:Q", format=".3f"), alt.Tooltip("Value:Q", format=",.2f")]
    )
    .properties(height=340)
)
rule_b = alt.Chart(pd.DataFrame({"r":[r_in_b]})).mark_rule(color="green", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_in + rule_b, use_container_width=True)


# ---------------- Comparison bars ----------------
def regime_components(name, A, r, Df, tau_reg):
    prims = compute_primitives(A, B, r, kappa)
    Admin_leak = tau_reg * prims["Roy"]
    Creation_cost = delta_T(Df, Df_AI0, phi, psi)
    TotalW = prims["CS"] + prims["PiAI"] - Admin_leak - Creation_cost
    PiC_transfer = (1.0 - tau_reg) * r * prims["S"]
    DeltaPiC = PiC_transfer - Creation_cost
    return dict(Regime=name, r=r,
                CS=prims["CS"]*scale_factor,
                PiAI=prims["PiAI"]*scale_factor,
                PiC=PiC_transfer*scale_factor,
                Admin=-Admin_leak*scale_factor,
                Creation=-Creation_cost*scale_factor,
                TotalWelfare=TotalW*scale_factor,
                DeltaPiC=DeltaPiC*scale_factor)

row_in_r0 = df_in.iloc[0]  # r_grid starts at 0.0
rows = []
rows.append(regime_components("Exception", A_exc, r_exc, Df_exc, 0.0))
rows.append(regime_components("Opt-out (r=0)", A_out, 0, Df_out, tau))
rows.append(regime_components("Opt-in (r=0)", float(row_in_r0["A"]),0.0, float(row_in_r0["Df"]), tau_in))
rows.append(regime_components("Opt-in (AI-opt)", df_in.loc[idx_ai,"A"], df_in.loc[idx_ai,"r"], df_in.loc[idx_ai,"Df"], tau_in))
rows.append(regime_components("Opt-in (Creator-opt)", df_in.loc[idx_c,"A"], df_in.loc[idx_c,"r"], df_in.loc[idx_c,"Df"], tau_in))
rows.append(regime_components("Opt-in (W-opt)", df_in.loc[idx_w,"A"], df_in.loc[idx_w,"r"], df_in.loc[idx_w,"Df"], tau_in))
rows.append(regime_components("Opt-in (Bargain β)", df_in.loc[idx_b,"A"], df_in.loc[idx_b,"r"], df_in.loc[idx_b,"Df"], tau_in))
rows.append(regime_components("Statutory (W-opt)", df_stat.loc[idx_stat_star,"A"], df_stat.loc[idx_stat_star,"r"], df_stat.loc[idx_stat_star,"Df"], tau))

df_bars = pd.DataFrame(rows)
df_comp = df_bars.melt(id_vars=["Regime","r","TotalWelfare"],
                       value_vars=["CS","PiAI","PiC","Admin","Creation"],
                       var_name="Component", value_name="Value")

bar = (
    alt.Chart(df_comp)
    .mark_bar()
    .encode(
        x=alt.X("Regime:N", title="Policy option"),
        y=alt.Y("Value:Q", title="Component value (scaled)"),
        color=alt.Color("Component:N", legend=alt.Legend(title="Components")),
        tooltip=["Regime:N","Component:N", alt.Tooltip("Value:Q", format=",.2f")]
    )
    .properties(height=380)
)
line_total = (
    alt.Chart(df_bars)
    .mark_point(size=80, filled=True)
    .encode(
        x="Regime:N",
        y=alt.Y("TotalWelfare:Q", title=""),
        color=alt.value("black"),
        tooltip=["Regime:N", alt.Tooltip("TotalWelfare:Q", format=",.2f")]
    )
)

with top_cmp:
    st.header("Policy comparison — components and total welfare")
    st.altair_chart(bar + line_total, use_container_width=True)



# # Show creators' incremental net ΔΠ^C by regime (scaled)
# st.subheader("Creators — incremental net (ΔΠᶜ = F − ΔT), scaled")
# st.dataframe(df_bars[["Regime","DeltaPiC"]])

