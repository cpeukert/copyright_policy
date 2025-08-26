# ai_copyright_dashboard_v9.py
# Run: streamlit run ai_copyright_dashboard_v9.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------
# HARD-CODED DEFAULTS (unchanged ones kept verbatim)
# ---------------------------------
DEFAULTS = dict(
    Ds=399.0, # Subtracted Df0 from A
    B=834.0,
    mu=0.9975,
    alpha=0.0013,
    phi=1.0,
    psi=1.866,
    kappa=0.988,
    tau=0.10,       # leakage used in statutory & opt-out
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
    # --- NEW (not present before; you asked to keep existing defaults unchanged) ---
    Df0=6,        # competitive flow without AI, default: A*(0.51/(3*12)) from Peukert et al. (2025) "Unsplash"
    delta_f=0.7,    # suppressed-flow share (Df_AI0 = delta_f * Df0), default = 0.7=(1-0.3) from Peukert et al. (2025) "Unsplash"
)

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

# ---------------- UI ----------------
st.set_page_config(page_title="AI Copyright Policy Dashboard — Model-Consistent v9", layout="wide")
st.title("AI Copyright Policy Dashboard — Model-Consistent (v9)")

with st.expander("What changed vs prior version?", expanded=True):
    st.markdown("""
- **Flow financing now matches the paper**: we solve the feasibility condition  
  \\( (1-\\tau_{reg})\\,r\\,S(A,r) = \\phi D_f^{\\psi} - \\phi (D_f^{AI}(0))^{\\psi} \\)  
  with **total flow** \\(D_f\\in[D_f^{AI}(0), D_f^0]\\). No stock-harm bill.
- **Exception/Opt-out at r=0** keep \\(D_f=D_f^{AI}(0)\\) (suppressed baseline), and welfare subtracts \\(\\phi D_f^\\psi\\).
- Added **τ_in** in the sidebar (defaults to **τ**), used for opt-in’s admin leakage.
- Added **new** parameters `Df0` and `delta_f` (suppressed share) to compute the baseline flow  
  \\(D_f^{AI}(0)=\\delta_f\\,D_f^0\\). Default for `delta_f` is **0.3**, per your request.
""")

with st.expander("Model (key equations used here)", expanded=False):
    st.markdown(r"""
**Demand & value**  
- $p(Q)=A-BQ$, $B>0$.  
- $A(\sigma, D_f)=\theta(\sigma) D_s + \mu D_f^\alpha$, with $\theta(\sigma)=\sigma^\gamma$.

**Pricing (ad valorem $r$)**  
- $p=\dfrac{(1-\kappa)A}{1-\kappa r}$, \ \ $Q=\dfrac{\kappa(1-r)}{B(1-\kappa r)}A$, \ \ $S=\dfrac{A^2}{B}\rho(r)$ with $\rho(r)=\dfrac{\kappa(1-\kappa)(1-r)}{(1-\kappa r)^2}$.

**Royalty pool and flow creation**  
- Net pool: $F=(1-\tau_{\text{reg}})\,r\,S(A,r)$, with $\tau_{\text{reg}}=\tau$ (STAT/Opt-out) or $\tau_{\text{reg}}=\tau_{in}$ (Opt-in).
- Suppressed baseline (no royalties): $D_f^{AI}(0)=\delta_f\,D_f^0$ (UI parameter), $D_f\in[D_f^{AI}(0),D_f^0]$.
- Feasibility (interior): $F=\phi D_f^\psi - \phi \big(D_f^{AI}(0)\big)^\psi$.

**Welfare**  
- $W=CS+\Pi^{AI}-\tau_{\text{reg}}\,r\,S(A,r)-\phi D_f^\psi$.
""")

# ---- Sidebar
st.sidebar.header("Structural parameters")
Ds    = st.sidebar.number_input("D_s (stock of existing works)", min_value=0.0, value=float(DEFAULTS["Ds"]), step=1.0, format="%.1f")
B     = st.sidebar.number_input("B (demand slope)", min_value=1e-6, value=float(DEFAULTS["B"]), step=1.0, format="%.1f")
mu    = st.sidebar.number_input("μ (scale of flow in A)", min_value=0.0, value=float(DEFAULTS["mu"]), step=0.0001, format="%.4f")
alpha = st.sidebar.number_input("α (diminishing returns to flow)", min_value=0.0, max_value=1.0, value=float(DEFAULTS["alpha"]), step=0.0001, format="%.4f")
phi   = st.sidebar.number_input("φ (flow cost scale)", min_value=0.0, value=float(DEFAULTS["phi"]), step=0.01, format="%.2f")
psi   = st.sidebar.number_input("ψ (flow cost curvature >1)", min_value=1.01, value=float(DEFAULTS["psi"]), step=0.001, format="%.3f")
kappa = st.sidebar.number_input("κ (pass-through)", min_value=0.01, max_value=1.00, value=float(DEFAULTS["kappa"]), step=0.001, format="%.3f")
tau   = st.sidebar.slider("τ (admin leakage, STAT & opt-out)", 0.0, 0.99, float(DEFAULTS["tau"]), 0.01)
gamma = st.sidebar.slider("γ (coverage penalty on stock)", 1.0, 3.0, float(DEFAULTS["gamma"]), 0.05)

st.sidebar.header("Harms & action costs")
delta = st.sidebar.slider("δ (perceived net effect of AI on creators)", min_value=-1.0, max_value=1.0, value=float(DEFAULTS["delta"]), step=0.001, format="%.3f")
x_in  = st.sidebar.number_input("x_in (opt-in action cost)", min_value=-1.0, value=float(DEFAULTS["x_in"]), max_value=1.0, step=0.001, format="%.3f")
x_out = st.sidebar.number_input("x_out (opt-out action cost)", min_value=-1.0, value=float(DEFAULTS["x_out"]), max_value=1.0, step=0.001, format="%.3f")

st.sidebar.header("Flow baselines")
Df0 = st.sidebar.number_input("D_f^0 (flow without AI)", min_value=0.0, value=float(DEFAULTS["Df0"]), step=0.1, format="%.2f")
delta_f = st.sidebar.number_input("δ_f (suppressed share; D_f^{AI}(0)=δ_f·D_f^0)", min_value=0.0, max_value=1.0, value=float(DEFAULTS["delta_f"]), step=0.05, format="%.2f")
Df_AI0 = float(delta_f * Df0)

st.sidebar.header("Search / solution settings")
r_max = st.sidebar.slider("Max r to search over", 0.0, 0.99, float(DEFAULTS["r_max"]), 0.01)
grid_steps = st.sidebar.slider("Grid steps for r search", 101, 3001, int(DEFAULTS["grid_steps"]), 100)

st.sidebar.header("Bargaining (opt-in)")
beta = st.sidebar.slider("β (creators' bargaining power)", 0.0, 1.0, float(DEFAULTS["beta"]), 0.01)
tau_in = st.sidebar.slider("τ_in (admin leakage in Opt-in)", 0.0, 0.99, float(tau), 0.01)  # default equals τ

st.sidebar.header("Aggregation")
N = st.sidebar.number_input("Adult population (mio)", min_value=1.0, value=float(DEFAULTS["N"]), step=1.0, format="%.0f")
n = st.sidebar.number_input("AI adoption share", min_value=0.0, max_value=1.0, value=float(DEFAULTS["n"]), step=0.01, format="%.2f")
months = st.sidebar.number_input("Months", min_value=1.0, value=float(DEFAULTS["months"]), step=1.0, format="%.0f")
scale_factor = N * n * months
st.sidebar.text(f"Scale factor: {scale_factor:,.1f}")

if psi <= 1.0:
    st.error("Regularity requires ψ > 1. Increase ψ.")
    st.stop()

# ---------------- Statutory license ----------------
st.header("Statutory license (σ̃ = 1)")
r_grid = np.linspace(0.0, r_max, int(grid_steps))

def solve_stat_for_r(r):
    sigma_tilde = 1.0
    Df = solve_flow_total(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, Df0, Df_AI0)
    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - phi * (Df ** psi)
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
W_exc = prims_exc["CS"] + prims_exc["PiAI"] - phi * (Df_exc ** psi)
row_exc = dict(Regime="Exception", r=r_exc, sigma=sigma_exc, Df=Df_exc, A=A_exc, W=W_exc); row_exc.update(prims_exc)
st.write(pd.DataFrame([row_exc])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# ---------------- Opt-out ----------------
st.header("Opt-out")
r_out = st.slider("r_out (opt-out royalty, typically 0)", 0.0, float(r_max), 0.0, 0.01)
sigma_tilde_out = sigma_tilde_opt_out(Ds, r_out, delta, x_out)
Df_out = solve_flow_total(Ds, sigma_tilde_out, gamma, mu, alpha, phi, psi, tau, B, kappa, r_out, Df0, Df_AI0)
A_out = A_from_sigma_Df(Ds, sigma_tilde_out, gamma, mu, alpha, Df_out)
prims_out = compute_primitives(A_out, B, r_out, kappa)
W_out = prims_out["CS"] + prims_out["PiAI"] - tau * prims_out["Roy"] - phi * (Df_out ** psi)
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
    PiC = (1.0 - tau_in) * r * prims["S"] - phi * (Df_in ** psi)   # creators’ surplus
    W = prims["CS"] + prims["PiAI"] - tau_in * prims["Roy"] - phi * (Df_in ** psi)
    out = dict(r=r, sigma=sigma_in, Df=Df_in, A=A_in, W=W, PiC=PiC); out.update(prims)
    return out

df_in = pd.DataFrame([solve_optin_for_r(r) for r in r_grid])
idx_ai = int(np.nanargmax(df_in["PiAI"].values))
idx_c  = int(np.nanargmax(df_in["PiC"].values))
idx_w  = int(np.nanargmax(df_in["W"].values))

# Nash bargaining objective
def nash_value(row, beta):
    PiC = row["PiC"]; PiAI = row["PiAI"]
    if (PiC is None) or (PiAI is None) or (PiC <= 0.0) or (PiAI <= 0.0):
        return -np.inf
    return beta * np.log(PiC) + (1.0 - beta) * np.log(PiAI)

df_in["NashVal"] = [nash_value(row, float(beta)) for _, row in df_in.iterrows()]
if np.all(~np.isfinite(df_in["NashVal"].values)):
    df_in["NashVal"] = float(beta) * df_in["PiC"].values + (1.0 - float(beta)) * df_in["PiAI"].values
idx_b = int(np.nanargmax(df_in["NashVal"].values))

r_in_ai = float(df_in.loc[idx_ai, "r"])
r_in_c  = float(df_in.loc[idx_c,  "r"])
r_in_w  = float(df_in.loc[idx_w,  "r"])
r_in_b  = float(df_in.loc[idx_b,  "r"])

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("r_in^AI (max π^AI)", f"{r_in_ai:.4f}")
with c2: st.metric("r_in^C (max Π^C)",  f"{r_in_c:.4f}")
with c3: st.metric("r_in^W (max W)",    f"{r_in_w:.4f}")
with c4: st.metric("r_in^B (Nash, β)",  f"{r_in_b:.4f}")

# σ^in curve & bargaining line
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

# Scaled objective plot for opt-in
df_in_scaled = df_in.copy()
for col in ["W","PiAI","PiC"]:
    df_in_scaled[col] = df_in_scaled[col] * scale_factor
chart_in = (
    alt.Chart(df_in_scaled.melt(id_vars=["r"], value_vars=["W","PiAI","PiC"], var_name="Objective", value_name="Value"))
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
st.header("Policy comparison — components and total welfare")

def regime_components(name, A, r, Df, tau_reg):
    prims = compute_primitives(A, B, r, kappa)
    Admin_leak = tau_reg * prims["Roy"]
    Creation_cost = phi * (Df ** psi)
    TotalW = prims["CS"] + prims["PiAI"] - Admin_leak - Creation_cost
    return dict(Regime=name, r=r,
                CS=prims["CS"]*scale_factor,
                PiAI=prims["PiAI"]*scale_factor,
                PiC=((1.0 - tau_reg) * r * prims["S"] - Creation_cost)*scale_factor,
                Admin=-Admin_leak*scale_factor,
                Creation=-Creation_cost*scale_factor,
                TotalWelfare=TotalW*scale_factor)

rows = []
rows.append(regime_components("Exception", A_exc, r_exc, Df_exc, 0.0))
rows.append(regime_components("Opt-out", A_out, r_out, Df_out, tau))
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

st.altair_chart(bar + line_total, use_container_width=True)
