
# ai_copyright_dashboard_revised_v8.py
# Run: streamlit run ai_copyright_dashboard_revised_v8.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------
# HARD-CODED DEFAULTS (from your script)
# ---------------------------------
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
)

EPS = 1e-12
def nz(x):
    return np.where(np.abs(x) < EPS, np.nan, x)

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

# ---------------- Flow solver ----------------
def solve_flow(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau_eff, B, kappa, r, delta):
    """Solve max{F_res,0} = δ_harm Df + φ Df^ψ with A = θ(σ̃)Ds + μ Df^α and
       F_res = (1-τ_eff) r S(A,r) - δ_harm σ̃ Ds.
       Note: τ_eff can be regime-specific (we set τ_eff=0 in OPT-IN per your request)."""
    if r <= 0.0:
        return 0.0
    delta_harm = max(-delta, 0.0)
    Ds_sigma = theta_of_sigma(sigma_tilde, gamma) * Ds

    def F_res_of(Df):
        A = Ds_sigma + mu * (Df ** alpha)
        S = revenue_S(A, B, r, kappa)
        F = (1.0 - tau_eff) * r * S
        return max(F - delta_harm * (sigma_tilde * Ds), 0.0)

    F0 = F_res_of(0.0)
    if F0 <= 0.0:
        return 0.0

    def g(Df):
        return delta_harm * Df + phi * (Df ** psi) - F_res_of(Df)

    lo, hi = 0.0, 1.0
    ghi = g(hi)
    tries = 0
    while (np.isnan(ghi) or ghi < 0.0) and tries < 60:
        hi *= 2.0
        ghi = g(hi)
        tries += 1
        if hi > 1e18:
            break
    if np.isnan(ghi) or ghi < 0.0:
        return 0.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        gmid = g(mid)
        if np.isnan(gmid):
            break
        if abs(gmid) < 1e-10 or (hi - lo) < 1e-10:
            return float(mid)
        if gmid >= 0.0:
            hi = mid
        else:
            lo = mid
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
st.set_page_config(page_title="AI Copyright Policy Dashboard — Revised v8", layout="wide")
st.title("AI Copyright Policy Dashboard — Revised (v8)")

with st.expander("What changed in v8?", expanded=True):
    st.markdown("""
- **Opt-in now has _no admin leakage_**: we set \\(\\tau_{in}=0\\) in both the flow equation and welfare, as you requested.
- Added **threshold diagnostics** so you can see why \\(D_f=0\\) with the current parameters.
    """)

with st.expander("Model (key equations used here)", expanded=False):
    st.markdown(r"""
**Demand & value**  
- $p(Q)=A-BQ$, $B>0$.  
- $A(\sigma, D_f)=\theta(\sigma) D_s + \mu D_f^\alpha$, with $\theta(\sigma)=\sigma^\gamma$.

**Pricing (ad valorem $r$)**  
- $p=\dfrac{(1-\kappa)A}{1-\kappa r}$, \ \ $Q=\dfrac{\kappa(1-r)}{B(1-\kappa r)}A$, \ \ $S=\dfrac{A^2}{B}\rho(r)$ with $\rho(r)=\dfrac{\kappa(1-\kappa)(1-r)}{(1-\kappa r)^2}$.

**Royalty pool and flow creation**  
- Net pool: $F=(1-\tau_{\text{reg}})\,r\,S(A,r)$, where we use $\tau_{\text{reg}}=\tau$ for statutory/opt-out, and **$\tau_{\text{reg}}=0$ for opt-in (by design in v8)**.  
- Residual after stock-harm: $F_{\text{res}}=F-\delta_{\text{harm}}\tilde\sigma D_s$, $\delta_{\text{harm}}=\max\{-\delta,0\}$.  
- Flow: $\max\{F_{\text{res}},0\}=\delta_{\text{harm}}D_f+\phi D_f^\psi$.

**Stock inclusion**  
- Opt-in: $\sigma^{in}=\Lambda(D_s[r_{in}+\delta-x_{in}])$.  
- Opt-out: $\tilde\sigma^{out}=1-\Lambda(D_s[\delta-r_{out}-x_{out}])$.

**Welfare (baseline)**  
- $W=CS+\Pi^{AI}-\text{AdminLeak}-\phi D_f^\psi$, with $\text{AdminLeak}=\tau_{\text{reg}}\,r\,S(A,r)$.
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
tau   = st.sidebar.slider("τ (admin leakage share) — used in statutory & opt-out", 0.0, 0.99, float(DEFAULTS["tau"]), 0.01)
gamma = st.sidebar.slider("γ (coverage penalty on stock)", 1.0, 3.0, float(DEFAULTS["gamma"]), 0.05)

st.sidebar.header("Harms & action costs")
delta = st.sidebar.slider("δ (perceived net effect of AI on creators)", min_value=-1.0, max_value=1.0, value=float(DEFAULTS["delta"]), step=0.001, format="%.3f")
x_in  = st.sidebar.number_input("x_in (opt-in action cost)", min_value=-1.0, value=float(DEFAULTS["x_in"]), max_value=1.0, step=0.001, format="%.3f")
x_out = st.sidebar.number_input("x_out (opt-out action cost)", min_value=-1.0, value=float(DEFAULTS["x_out"]), max_value=1.0, step=0.001, format="%.3f")

st.sidebar.header("Search / solution settings")
r_max = st.sidebar.slider("Max r to search over", 0.0, 0.99, float(DEFAULTS["r_max"]), 0.01)
grid_steps = st.sidebar.slider("Grid steps for r search", 101, 3001, int(DEFAULTS["grid_steps"]), 100)

st.sidebar.header("Bargaining (opt-in)")
beta = st.sidebar.slider("β (creators' bargaining power)", 0.0, 1.0, float(DEFAULTS["beta"]), 0.01)

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
    Df = solve_flow(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, delta)
    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - phi * (Df ** psi)
    out = dict(r=r, sigma=sigma_tilde, Df=Df, A=A, W=W); out.update(prims)
    return out

df_stat = pd.DataFrame([solve_stat_for_r(r) for r in r_grid])
idx_stat_star = int(np.nanargmax(df_stat["W"].values))
r_stat_star = float(df_stat.loc[idx_stat_star, "r"])

# Diagnostics: stock-harm threshold for statutory (at Df=0, A=Ds, σ̃=1)
def net_pool_stat(A, r):
    return (1.0 - tau) * r * revenue_S(A, B, r, kappa)
delta_harm = max(-delta, 0.0)
stock_harm_bill_stat = delta_harm * 1.0 * Ds
r_grid_dense = np.linspace(0.0, 0.999, 4000)
max_pool_stat = max(net_pool_stat(Ds, r) for r in r_grid_dense)

c0, c1, c2, c3, c4 = st.columns(5)
with c0: st.metric("r* (max W)", f"{r_stat_star:.4f}")
with c1: st.metric("W* (scaled)", f"{(df_stat.loc[idx_stat_star, 'W'] * scale_factor):,.2f}")
with c2: st.metric("A* (unscaled)", f"{df_stat.loc[idx_stat_star, 'A']:.2f}")
with c3: st.metric("D_f* (unscaled)", f"{df_stat.loc[idx_stat_star, 'Df']:.4f}")
with c4: st.metric("Max net pool vs stock-harm", f"{max_pool_stat:,.2f} / {stock_harm_bill_stat:,.2f}")

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
rule_star = alt.Chart(pd.DataFrame({"r":[r_stat_star]})).mark_rule(color="gray", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_stat + rule_star, use_container_width=True)

# ---------------- AI exception ----------------
st.header("AI exception")
sigma_exc = 1.0; r_exc = 0.0; Df_exc = 0.0
A_exc = A_from_sigma_Df(Ds, sigma_exc, gamma, mu, alpha, Df_exc)
prims_exc = compute_primitives(A_exc, B, r_exc, kappa)
W_exc = prims_exc["CS"] + prims_exc["PiAI"]
row_exc = dict(Regime="Exception", r=r_exc, sigma=sigma_exc, Df=Df_exc, A=A_exc, W=W_exc); row_exc.update(prims_exc)
st.write(pd.DataFrame([row_exc])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# ---------------- Opt-out ----------------
st.header("Opt-out")
r_out = st.slider("r_out (opt-out royalty, typically 0)", 0.0, float(r_max), 0.0, 0.01)
sigma_tilde_out = sigma_tilde_opt_out(Ds, r_out, delta, x_out)
Df_out = solve_flow(Ds, sigma_tilde_out, gamma, mu, alpha, phi, psi, tau, B, kappa, r_out, delta)
A_out = A_from_sigma_Df(Ds, sigma_tilde_out, gamma, mu, alpha, Df_out)
prims_out = compute_primitives(A_out, B, r_out, kappa)
W_out = prims_out["CS"] + prims_out["PiAI"] - tau * prims_out["Roy"] - phi * (Df_out ** psi)
row_out = dict(Regime="Opt-out", r=r_out, sigma=sigma_tilde_out, Df=Df_out, A=A_out, W=W_out); row_out.update(prims_out)

# Diagnostics for opt-out (at Df=0, A=θ(σ̃)Ds)
A_out_zero = A_from_sigma_Df(Ds, sigma_tilde_out, gamma, mu, alpha, 0.0)
max_pool_out = max((1.0 - tau) * r * revenue_S(A_out_zero, B, r, kappa) for r in r_grid_dense)
stock_harm_bill_out = delta_harm * sigma_tilde_out * Ds

cA, cB, cC = st.columns(3)
with cA: st.metric("σ̃ (included)", f"{sigma_tilde_out:.4f}")
with cB: st.metric("Max net pool (opt-out, A at D_f=0)", f"{max_pool_out:,.2f}")
with cC: st.metric("Stock-harm bill (opt-out)", f"{stock_harm_bill_out:,.2f}")
st.write(pd.DataFrame([row_out])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# ---------------- Opt-in (τ_in = 0 by design in v8) ----------------
st.header("Opt-in (negotiated r) — **no admin leakage in v8**")

tau_in = 0.0  # <<<<<< NO LEAKAGE IN OPT-IN

def solve_optin_for_r(r):
    sigma_in = sigma_opt_in(Ds, r, delta, x_in)
    Df_in = solve_flow(Ds, sigma_in, gamma, mu, alpha, phi, psi, tau_in, B, kappa, r, delta)
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

# Bargaining (Nash product)
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

# Threshold diagnostics for opt-in (evaluate at D_f=0)
def net_pool_in_at_Df0(r):
    sigma0 = sigma_opt_in(Ds, r, delta, x_in)
    A0 = A_from_sigma_Df(Ds, sigma0, gamma, mu, alpha, 0.0)
    return (1.0 - tau_in) * r * revenue_S(A0, B, r, kappa), sigma0

max_pool_in = 0.0
sigma_at_max = None
for r in r_grid_dense:
    pool, sig0 = net_pool_in_at_Df0(r)
    if pool > max_pool_in:
        max_pool_in = pool
        sigma_at_max = sig0
stock_harm_bill_in = delta_harm * (sigma_at_max if sigma_at_max is not None else 0.0) * Ds

c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("r_in^AI (max π^AI)", f"{r_in_ai:.4f}")
with c2: st.metric("r_in^C (max Π^C)",  f"{r_in_c:.4f}")
with c3: st.metric("r_in^W (max W)",    f"{r_in_w:.4f}")
with c4: st.metric("r_in^B (Nash, β)",  f"{r_in_b:.4f}")

c5, c6 = st.columns(2)
with c5: st.metric("Max net pool at D_f=0 (opt-in, τ_in=0)", f"{max_pool_in:,.2f}")
with c6: st.metric("Stock-harm bill (uses σ at that r)",     f"{stock_harm_bill_in:,.2f}")

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

def regime_components(name, A, r, sigma_tilde, Df, tau_reg):
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
rows.append(regime_components("Exception", A_exc, r_exc, 1.0, 0.0, 0.0))
rows.append(regime_components("Opt-out", A_out, r_out, sigma_tilde_out, Df_out, tau))
rows.append(regime_components("Opt-in (AI-opt)", df_in.loc[idx_ai,"A"], df_in.loc[idx_ai,"r"], df_in.loc[idx_ai,"sigma"], df_in.loc[idx_ai,"Df"], tau_in))
rows.append(regime_components("Opt-in (Creator-opt)", df_in.loc[idx_c,"A"], df_in.loc[idx_c,"r"], df_in.loc[idx_c,"sigma"], df_in.loc[idx_c,"Df"], tau_in))
rows.append(regime_components("Opt-in (W-opt)", df_in.loc[idx_w,"A"], df_in.loc[idx_w,"r"], df_in.loc[idx_w,"sigma"], df_in.loc[idx_w,"Df"], tau_in))
rows.append(regime_components("Opt-in (Bargain β)", df_in.loc[idx_b,"A"], df_in.loc[idx_b,"r"], df_in.loc[idx_b,"sigma"], df_in.loc[idx_b,"Df"], tau_in))
rows.append(regime_components("Statutory (W-opt)", df_stat.loc[idx_stat_star,"A"], df_stat.loc[idx_stat_star,"r"], 1.0, df_stat.loc[idx_stat_star,"Df"], tau))

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

st.info(
    "With the current parameters, the **max possible net pool** is far below the **stock-harm bill** in all r-searches, "
    "so the model settles at D_f=0. To get D_f>0, either lower δ_harm (make δ less negative), reduce τ (for statutory/opt-out), "
    "reduce B, or move κ away from ~1 so κ(1-κ) is larger; any of these raise the pool relative to the obligation."
)
