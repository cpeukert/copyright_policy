
# ai_copyright_dashboard.py
# Run: streamlit run ai_copyright_dashboard.py

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

# ---------------------------------
# Numeric helpers
# ---------------------------------
EPS = 1e-12
def nz(x):
    """Guard against division by ~0; returns NaN near 0 to avoid blowups in charts."""
    return np.where(np.abs(x) < EPS, np.nan, x)

def logit(z):
    z = np.asarray(z, dtype=float)
    return 1.0 / (1.0 + np.exp(-z))

# ---------------------------------
# Core primitives from the model (aligned with Theory)
# ---------------------------------
def theta_of_sigma(sigma, gamma):
    sigma = np.clip(np.asarray(sigma, dtype=float), 0.0, 1.0)
    return np.power(sigma, gamma)

def A_from_sigma_Df(Ds, sigma, gamma, mu, alpha, Df):
    # A(D_s, D_f, σ) = θ(σ) D_s + μ D_f^α
    return theta_of_sigma(sigma, gamma) * Ds + mu * (np.asarray(Df, dtype=float) ** alpha)

def pq_advalorem(A, B, r, kappa):
    # p = ((1-κ)A) / (1-κ r),  Q = [κ(1-r)A] / [B(1-κ r)]
    denom = 1.0 - kappa * r
    p = (1.0 - kappa) * A / nz(denom)
    Q = (kappa * (1.0 - r) * A) / nz(B * denom)
    return p, Q

def revenue_S(A, B, r, kappa):
    # S(A,r) = pQ
    p, Q = pq_advalorem(A, B, r, kappa)
    return p * Q

def rho_of_r(r, kappa):
    return (kappa * (1.0 - kappa) * (1.0 - r)) / ((1.0 - kappa * r) ** 2)

# ---------------------------------
# Flow solver (baseline model, exact to Theory)
# max{F_res, 0} = δ_harm D_f + φ D_f^ψ,
# where F_res(A, r, σ̃) = (1-τ) r S(A,r) - δ_harm σ̃ D_s,
# A = θ(σ̃) D_s + μ D_f^α.
# Bisection on D_f.
# ---------------------------------
def solve_flow(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, delta):
    if r <= 0.0:
        return 0.0

    delta_harm = max(-delta, 0.0)
    Ds_sigma = theta_of_sigma(sigma_tilde, gamma) * Ds

    def F_res_of(Df):
        A = Ds_sigma + mu * (Df ** alpha)
        S = revenue_S(A, B, r, kappa)
        F = (1.0 - tau) * r * S
        return max(F - delta_harm * (sigma_tilde * Ds), 0.0)

    F0 = F_res_of(0.0)
    if F0 <= 0.0:
        return 0.0  # below stock-harm threshold

    def f(Df):
        return delta_harm * Df + phi * (Df ** psi) - F_res_of(Df)

    lo, hi = 0.0, 1.0
    fhi = f(hi)
    tries = 0
    while (np.isnan(fhi) or fhi < 0.0) and tries < 60:
        hi *= 2.0
        fhi = f(hi)
        tries += 1
        if hi > 1e18:
            break

    if np.isnan(fhi) or fhi < 0.0:
        return 0.0

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        fmid = f(mid)
        if np.isnan(fmid):
            break
        if abs(fmid) < 1e-10 or (hi - lo) < 1e-10:
            return float(mid)
        if fmid >= 0.0:
            hi = mid
        else:
            lo = mid
    return float(0.5 * (lo + hi))

# ---------------------------------
# Opt-in / Opt-out stock inclusion (exact logits from Theory)
# ---------------------------------
def sigma_opt_in(Ds, r_in, delta, x_in):
    # σ^in = Λ( Ds [ r_in + δ - x_in ] )
    return float(np.clip(logit(Ds * (r_in + delta - x_in)), 0.0, 1.0))

def sigma_tilde_opt_out(Ds, r_out, delta, x_out):
    # σ^out = Λ( Ds [ δ - r_out - x_out ] ), included share is 1 - σ^out
    sigma_out = float(np.clip(logit(Ds * (delta - r_out - x_out)), 0.0, 1.0))
    return 1.0 - sigma_out

# ---------------------------------
# Welfare primitives and decomposition
# ---------------------------------
def compute_primitives(A, B, r, kappa):
    # Returns only p,Q,S,CS,PiAI,Roy (no 'A' here to avoid duplicate dict keys)
    p, Q = pq_advalorem(A, B, r, kappa)
    S = p * Q
    CS = 0.5 * B * (Q ** 2)
    PiAI = (1.0 - r) * S
    Roy = r * S  # gross royalties
    return dict(p=p, Q=Q, S=S, CS=CS, PiAI=PiAI, Roy=Roy)

# ---------------------------------
# Streamlit UI
# ---------------------------------
st.set_page_config(page_title="AI Copyright Policy Dashboard", layout="wide")
st.title("AI Copyright Policy Dashboard")

with st.expander("Model (exact equations used here)", expanded=False):
    st.markdown(r"""
**Demand & value**  
- $p(Q)=A-BQ$, $B>0$.  
- $A(\sigma, D_f)=\theta(\sigma) D_s + \mu D_f^\alpha$, with $\theta(\sigma)=\sigma^\gamma$.

**Pricing (ad valorem royalty \(r\))**  
- $p=\dfrac{(1-\kappa)A}{1-\kappa r}$, \ \ $Q=\dfrac{\kappa(1-r)}{B(1-\kappa r)}A$.  
- $S(A,r)=pQ=\dfrac{A^2}{B}\rho(r)$ with $\rho(r)=\dfrac{\kappa(1-\kappa)(1-r)}{(1-\kappa r)^2}$.

**Royalty pool and flow creation (baseline)**  
- Net pool: $F=(1-\tau)\,r\,S(A,r)$.  
- Stock-harm compensation: $\delta_{\text{harm}}\tilde\sigma D_s$, where $\delta_{\text{harm}}=\max\{-\delta,0\}$.  
- Flow feasibility: $\max\{F-\delta_{\text{harm}}\tilde\sigma D_s,0\}=\delta_{\text{harm}} D_f + \phi D_f^\psi$.

**Stock inclusion (choice regimes)**  
- Opt-in: $\sigma^{in}=\Lambda(D_s[r_{in}+\delta-x_{in}])$.  
- Opt-out: $\tilde\sigma^{out}=1-\Lambda(D_s[\delta-r_{out}-x_{out}])$.

**Welfare (baseline)**  
- $CS=\tfrac{1}{2}B Q^2$, \ $\Pi^{AI}=(1-r)S$, \ **Admin leakage** $=\tau\, r\, S$.  
- **Total welfare**: $W=CS+\Pi^{AI}-\tau r S-\phi D_f^\psi$.
- Pecuniary harms and transfers are not directly counted in $W$ (they affect $A$ and $D_f$ only).
""")

# -----------------------------
# Sidebar parameters (exactly those needed)
# -----------------------------
st.sidebar.header("Structural parameters")
Ds    = st.sidebar.number_input("D_s (stock of existing works)", min_value=0.0, value=405.0, step=1.0, format="%.1f")
B     = st.sidebar.number_input("B (demand slope)", min_value=1e-6, value=834.0, step=1.0, format="%.1f")
mu    = st.sidebar.number_input("μ (scale of flow in A)", min_value=0.0, value=0.9975, step=0.0001, format="%.4f")
alpha = st.sidebar.number_input("α (diminishing returns to flow)", min_value=0.0, max_value=1.0, value=0.0013, step=0.0001, format="%.4f")
phi   = st.sidebar.number_input("φ (flow cost scale)", min_value=0.0, value=1.0, step=0.01, format="%.2f")
psi   = st.sidebar.number_input("ψ (flow cost curvature >1)", min_value=1.01, value=1.866, step=0.001, format="%.3f")
kappa = st.sidebar.number_input("κ (pass-through)", min_value=0.01, max_value=1.00, value=0.988, step=0.001, format="%.3f")
tau   = st.sidebar.slider("τ (admin leakage share)", 0.0, 0.99, 0.10, 0.01)
gamma = st.sidebar.slider("γ (coverage penalty on stock)", 1.0, 3.0, 1.25, 0.05)

st.sidebar.header("Harms & action costs")
delta = st.sidebar.slider("δ (perceived net effect of AI on creators)", min_value=-1.0, max_value=1.0, value=-0.362, step=0.001, format="%.3f")
x_in  = st.sidebar.number_input("x_in (opt-in action cost)", min_value=-1.0, value=-0.244, max_value=1.0, step=0.001, format="%.3f")
x_out = st.sidebar.number_input("x_out (opt-out action cost)", min_value=-1.0, value=0.372, max_value=1.0, step=0.001, format="%.3f")

st.sidebar.header("Search range")
r_max = st.sidebar.slider("Max r to search over", 0.0, 0.99, 0.99, 0.01)
grid_steps = st.sidebar.slider("Grid steps for r search", 101, 3001, 701, 100)

# Aggregation
st.sidebar.header("Aggregation")
N = st.sidebar.number_input("Adult population (mio)", min_value=1.0, value=212.0, step=1.0, format="%.0f")
n = st.sidebar.number_input("AI adoption share", min_value=0.0, max_value=1.0, value=0.4, step=0.01, format="%.2f")
months = st.sidebar.number_input("Months", min_value=1.0, value=12.0, step=1.0, format="%.0f")
scale_factor = N * n * months
st.sidebar.text(f"Scale factor: {scale_factor:,.1f}")

# Pack params for convenience
params = dict(Ds=Ds, B=B, mu=mu, alpha=alpha, phi=phi, psi=psi, kappa=kappa, tau=tau,
              gamma=gamma, delta=delta, x_in=x_in, x_out=x_out)

# Regularity
if psi <= 1.0:
    st.error("Regularity requires ψ > 1. Increase ψ.")
    st.stop()

# -----------------------------
# 1) Statutory license (σ̃ = 1): search over r
# -----------------------------
st.header("Statutory license (σ̃ = 1)")

r_grid = np.linspace(0.0, r_max, int(grid_steps))

def solve_stat_for_r(r):
    sigma_tilde = 1.0
    Df = solve_flow(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, delta)
    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - phi * (Df ** psi)
    # Build dict without duplicate keys
    out = dict(r=r, sigma=sigma_tilde, Df=Df, A=A, W=W)
    out.update(prims)
    return out

df_stat = pd.DataFrame([solve_stat_for_r(r) for r in r_grid])
idx_stat_star = int(np.nanargmax(df_stat["W"].values))
r_stat_star = float(df_stat.loc[idx_stat_star, "r"])

# UI line for chosen r
r_play = st.slider("Set r (green line) for display", min_value=0.0, max_value=float(r_max),
                   value=float(min(r_stat_star, r_max)), step=0.001, format="%.3f")

def solve_stat_single(r):
    sigma_tilde = 1.0
    Df = solve_flow(Ds, sigma_tilde, gamma, mu, alpha, phi, psi, tau, B, kappa, r, delta)
    A = A_from_sigma_Df(Ds, sigma_tilde, gamma, mu, alpha, Df)
    prims = compute_primitives(A, B, r, kappa)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - phi * (Df ** psi)
    out = dict(r=r, sigma=sigma_tilde, Df=Df, A=A, W=W)
    out.update(prims)
    return out

rec_play = solve_stat_single(r_play)

# Metrics
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("r* (max W)", f"{r_stat_star:.4f}")
with c2: st.metric("W* (scaled)", f"{(df_stat.loc[idx_stat_star, 'W'] * scale_factor):,.2f}")
with c3: st.metric("A* (unscaled)", f"{df_stat.loc[idx_stat_star, 'A']:.2f}")
with c4: st.metric("D_f* (unscaled)", f"{df_stat.loc[idx_stat_star, 'Df']:.4f}")

# Plot components (scaled)
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
rule_play = alt.Chart(pd.DataFrame({"r":[r_play]})).mark_rule(color="green", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_stat + rule_star + rule_play, use_container_width=True)

# -----------------------------
# 2) AI exception (r=0, σ̃=1, D_f=0)
# -----------------------------
st.header("AI exception")
sigma_exc = 1.0; r_exc = 0.0; Df_exc = 0.0
A_exc = A_from_sigma_Df(Ds, sigma_exc, gamma, mu, alpha, Df_exc)
prims_exc = compute_primitives(A_exc, B, r_exc, kappa)
W_exc = prims_exc["CS"] + prims_exc["PiAI"] - tau * prims_exc["Roy"]  # = CS + Π^AI at r=0
row_exc = dict(Regime="Exception", r=r_exc, sigma=sigma_exc, Df=Df_exc, A=A_exc, W=W_exc)
row_exc.update(prims_exc)
st.write(pd.DataFrame([row_exc])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# -----------------------------
# 3) Opt-out (default include, typical r=0 but allow r_out>0 variant)
# -----------------------------
st.header("Opt-out")
r_out = st.slider("r_out (opt-out royalty, typically 0)", 0.0, float(r_max), 0.0, 0.01)
sigma_tilde_out = sigma_tilde_opt_out(Ds, r_out, delta, x_out)
Df_out = solve_flow(Ds, sigma_tilde_out, gamma, mu, alpha, phi, psi, tau, B, kappa, r_out, delta)
A_out = A_from_sigma_Df(Ds, sigma_tilde_out, gamma, mu, alpha, Df_out)
prims_out = compute_primitives(A_out, B, r_out, kappa)
W_out = prims_out["CS"] + prims_out["PiAI"] - tau * prims_out["Roy"] - phi * (Df_out ** psi)
row_out = dict(Regime="Opt-out", r=r_out, sigma=sigma_tilde_out, Df=Df_out, A=A_out, W=W_out)
row_out.update(prims_out)
st.write(pd.DataFrame([row_out])[["A","Df","sigma","CS","PiAI","Roy","W"]])

# -----------------------------
# 4) Opt-in (negotiated r)
# -----------------------------
st.header("Opt-in (negotiated r)")

def solve_optin_for_r(r):
    sigma_in = sigma_opt_in(Ds, r, delta, x_in)
    Df_in = solve_flow(Ds, sigma_in, gamma, mu, alpha, phi, psi, tau, B, kappa, r, delta)
    A_in = A_from_sigma_Df(Ds, sigma_in, gamma, mu, alpha, Df_in)
    prims = compute_primitives(A_in, B, r, kappa)
    PiC = (1.0 - tau) * r * prims["S"] - phi * (Df_in ** psi)   # creators’ surplus (display)
    W = prims["CS"] + prims["PiAI"] - tau * prims["Roy"] - phi * (Df_in ** psi)
    out = dict(r=r, sigma=sigma_in, Df=Df_in, A=A_in, W=W, PiC=PiC)
    out.update(prims)  # adds p,Q,S,CS,PiAI,Roy
    return out

df_in = pd.DataFrame([solve_optin_for_r(r) for r in r_grid])

# Locate optimizers
idx_ai = int(np.nanargmax(df_in["PiAI"].values))
idx_c  = int(np.nanargmax(df_in["PiC"].values))
idx_w  = int(np.nanargmax(df_in["W"].values))

r_in_ai = float(df_in.loc[idx_ai, "r"])
r_in_c  = float(df_in.loc[idx_c, "r"])
r_in_w  = float(df_in.loc[idx_w, "r"])

c1, c2, c3 = st.columns(3)
with c1: st.metric("r_in^AI (max π^AI)", f"{r_in_ai:.4f}")
with c2: st.metric("r_in^C (max Π^C)", f"{r_in_c:.4f}")
with c3: st.metric("r_in^W (max W)", f"{r_in_w:.4f}")

# Scaled plot for opt-in objectives
df_in_scaled = df_in.copy()
for col in ["W","PiAI","PiC"]:
    df_in_scaled[col] = df_in_scaled[col] * scale_factor

chart_in = (
    alt.Chart(df_in_scaled.melt(id_vars=["r"], value_vars=["W","PiAI","PiC"],
                                var_name="Objective", value_name="Value"))
    .mark_line()
    .encode(
        x=alt.X("r:Q", title="r_in"),
        y=alt.Y("Value:Q", title="Scaled objective value"),
        color=alt.Color("Objective:N", legend=alt.Legend(title="")),
        tooltip=["Objective:N", alt.Tooltip("r:Q", format=".3f"), alt.Tooltip("Value:Q", format=",.2f")]
    )
    .properties(height=340)
)
rules_in = alt.Chart(pd.DataFrame({"r":[r_in_ai, r_in_c, r_in_w]})).mark_rule(color="green", strokeDash=[6,4]).encode(x="r:Q")
st.altair_chart(chart_in + rules_in, use_container_width=True)

# Show selected points (scaled components)
def pick_optin_row(idx, label):
    rec = df_in.loc[idx].to_dict()
    return dict(Regime=f"Opt-in ({label})", r=rec["r"], sigma=rec["sigma"], Df=rec["Df"], A=rec["A"],
                CS=rec["CS"]*scale_factor, PiAI=rec["PiAI"]*scale_factor, PiC=(rec["PiC"])*scale_factor,
                Roy=rec["Roy"]*scale_factor, W=rec["W"]*scale_factor)

optin_rows = [
    pick_optin_row(idx_ai, "AI-opt"),
    pick_optin_row(idx_c, "Creator-opt"),
    pick_optin_row(idx_w, "Welfare-opt"),
]
st.write(pd.DataFrame(optin_rows)[["Regime","r","sigma","A","Df","CS","PiAI","PiC","Roy","W"]])

# -----------------------------
# 5) Comparison bar chart across regimes
# Regimes: Exception, Opt-out, Opt-in (AI-opt / C-opt / W-opt), Statutory (W-opt)
# Components: CS, Π^AI, Π^C (display only), -Admin_leak, -Creation_cost, Total W
# -----------------------------
st.header("Policy comparison — components and total welfare")

def regime_components(name, A, r, sigma_tilde, Df):
    prims = compute_primitives(A, B, r, kappa)
    Admin_leak = tau * prims["Roy"]
    Creation_cost = phi * (Df ** psi)
    TotalW = prims["CS"] + prims["PiAI"] - Admin_leak - Creation_cost
    return dict(Regime=name, r=r,
                CS=prims["CS"]*scale_factor,
                PiAI=prims["PiAI"]*scale_factor,
                PiC=((1.0 - tau) * r * prims["S"] - Creation_cost)*scale_factor,  # creators’ surplus (display)
                Admin=-Admin_leak*scale_factor,
                Creation=-Creation_cost*scale_factor,
                TotalWelfare=TotalW*scale_factor)

rows = []
# Exception and Opt-out rows already computed above (recompute via function for consistency)
rows.append(regime_components("Exception", A_exc, r_exc, 1.0, 0.0))
rows.append(regime_components("Opt-out", A_out, r_out, sigma_tilde_out, Df_out))
# Opt-in three points
rows.append(regime_components("Opt-in (AI-opt)", df_in.loc[idx_ai,"A"], df_in.loc[idx_ai,"r"], df_in.loc[idx_ai,"sigma"], df_in.loc[idx_ai,"Df"]))
rows.append(regime_components("Opt-in (Creator-opt)", df_in.loc[idx_c,"A"], df_in.loc[idx_c,"r"], df_in.loc[idx_c,"sigma"], df_in.loc[idx_c,"Df"]))
rows.append(regime_components("Opt-in (W-opt)", df_in.loc[idx_w,"A"], df_in.loc[idx_w,"r"], df_in.loc[idx_w,"sigma"], df_in.loc[idx_w,"Df"]))
# Statutory at r*
rows.append(regime_components("Statutory license (W-opt)", df_stat.loc[idx_stat_star,"A"], df_stat.loc[idx_stat_star,"r"], 1.0, df_stat.loc[idx_stat_star,"Df"]))

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

st.caption("Bars show components (CS, Π^AI, Π^C (display), minus Admin leakage and Creation cost). Black dots mark total welfare.")
