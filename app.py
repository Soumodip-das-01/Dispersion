"""
Gas Dispersion Modelling Tool  -  Streamlit app
Run with:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

import dispersion as dm

st.set_page_config(page_title="Dispersion Model", page_icon="☁️", layout="wide")
st.title("☁️ Atmospheric Dispersion Model")
st.caption("Predict the affected zone after an accidental release of a gas/vapour "
           "using Gaussian Plume and Gaussian Puff models.")

# ------------------------------------------------------------------ SIDEBAR
with st.sidebar:
    st.header("1. Model")
    model = st.radio(
        "Release type",
        ["Continuous release (Gaussian Plume)", "Instantaneous release (Gaussian Puff)"],
    )
    is_plume = model.startswith("Continuous")

    st.header("2. Release data")
    chem = st.text_input("Chemical name", "Chlorine")
    mw = st.number_input("Molecular weight (g/mol)", 1.0, 500.0, 70.9, 0.1)
    if is_plume:
        Q_in = st.number_input("Release rate Q (g/s)", 0.001, 1e7, 100.0, 10.0)
    else:
        Q_in = st.number_input("Total mass released (kg)", 0.001, 1e6, 500.0, 10.0)
    H_stack = st.number_input("Release height (m)", 0.0, 500.0, 10.0, 1.0)
    rise = st.number_input("Plume rise (m) (0 if none)", 0.0, 500.0, 0.0, 1.0,
                           help="Added to release height to give effective height H.")
    H = H_stack + rise

    st.header("3. Meteorology")
    u = st.number_input("Wind speed at release height (m/s)", 0.5, 30.0, 4.0, 0.5)
    wind_from = st.number_input("Wind blowing FROM (° from North)", 0, 359, 270, 5,
                                help="0 = North, 90 = East, 180 = South, 270 = West")
    terrain = st.selectbox("Terrain", ["Rural", "Urban"])
    T_C = st.number_input("Ambient temperature (°C)", -50.0, 60.0, 25.0, 1.0)
    P_kPa = st.number_input("Ambient pressure (kPa)", 50.0, 120.0, 101.325, 0.5)

    mode = st.radio("Stability class", ["Choose directly", "Help me choose (Pasquill)"],
                    horizontal=True)
    if mode == "Choose directly":
        stab = st.selectbox("Pasquill–Gifford class", list(dm.STABILITY_DESCRIPTION),
                            index=3,
                            format_func=lambda s: f"{s} – {dm.STABILITY_DESCRIPTION[s]}")
    else:
        dn = st.radio("Time of day", ["Day", "Night"], horizontal=True)
        if dn == "Day":
            c = st.selectbox("Solar radiation", ["Strong", "Moderate", "Slight"])
        else:
            c = st.selectbox("Cloud cover", ["Cloudy (>=4/8)", "Clear (<=3/8)"])
        suggested = dm.pasquill_class(u, dn, c)
        st.info(f"Suggested class: **{suggested}**")
        stab = suggested.split("-")[0]  # A-B -> A, B-C -> B (conservative pick)
        stab = st.selectbox("Class to use", list(dm.STABILITY_DESCRIPTION),
                            index=list(dm.STABILITY_DESCRIPTION).index(stab))

    st.header("4. Time")
    if is_plume:
        use_time = st.checkbox("Consider plume arrival time", value=False,
                               help="If ticked, a point only sees the plume after x/u seconds.")
        t = st.number_input("Time since release began (s)", 1.0, 1e6, 600.0, 30.0,
                            disabled=not use_time)
        t_use = t if use_time else None
    else:
        t = st.number_input("Time since release (s)", 1.0, 1e6, 300.0, 30.0)
        t_use = t

    st.header("5. Hazard threshold")
    unit = st.radio("Threshold unit", ["ppm", "mg/m³"], horizontal=True)
    thr_in = st.number_input(f"Threshold ({unit}) – e.g. IDLH / ERPG-2",
                             0.0, 1e7, 10.0, 1.0)

# ------------------------------------------------------------ CONVERSIONS
T_K = T_C + 273.15
P_Pa = P_kPa * 1000
Q = Q_in if is_plume else Q_in * 1000.0           # g/s  or  g
thr_mgm3 = thr_in if unit == "mg/m³" else dm.ppm_to_mgm3(thr_in, mw, T_K, P_Pa)


def conc_mgm3(x, y, z, envelope=False, t_env=None):
    """Concentration in mg/m3 for the selected model."""
    if is_plume:
        c = dm.plume_concentration(x, y, z, Q, u, H, stab, terrain, t_use)
    elif envelope:
        c = dm.puff_envelope(x, y, z, Q, u, H, t_env, stab, terrain)
    else:
        c = dm.puff_concentration(x, y, z, Q, u, H, t, stab, terrain)
    return c * 1000.0  # g/m3 -> mg/m3


# -------------------------------------------------------------------- TABS
tab1, tab2, tab3, tab4 = st.tabs(
    ["📍 Concentration at a point", "🗺️ Contour plot", "📈 Concentration vs distance", "📘 Theory"])

# ------------------------------------------------------- TAB 1: single point
with tab1:
    st.subheader("Concentration at any distance and time")
    c1, c2, c3 = st.columns(3)
    xr = c1.number_input("Downwind distance x (m)", 1.0, 1e6, 500.0, 50.0)
    yr = c2.number_input("Crosswind distance y (m)", -1e5, 1e5, 0.0, 10.0)
    zr = c3.number_input("Height z (m)", 0.0, 1000.0, 1.5, 0.5)

    if not is_plume:
        st.write(f"Puff centre is currently at **{u * t:,.0f} m** downwind (t = {t:,.0f} s).")
    elif use_time:
        arr = xr / u
        st.write(f"Plume reaches this point after **{arr:,.0f} s**. "
                 + ("✅ Already arrived." if arr <= t else "⏳ Not yet arrived at the chosen time."))

    cm = float(conc_mgm3(xr, yr, zr))
    cp = dm.mgm3_to_ppm(cm, mw, T_K, P_Pa)
    m1, m2, m3 = st.columns(3)
    m1.metric("Concentration", f"{cm:.4g} mg/m³")
    m2.metric("Concentration", f"{cp:.4g} ppm")
    m3.metric("vs. threshold", "ABOVE ⚠️" if cm >= thr_mgm3 else "Below ✅",
              f"{cm / thr_mgm3:.2f} × threshold" if thr_mgm3 > 0 else None)

    sy, sz = dm.sigmas(xr if is_plume else max(u * t, 1), stab, terrain)
    st.caption(f"σy = {float(sy):.1f} m,  σz = {float(sz):.1f} m  "
               f"(class {stab}, {terrain.lower()}, effective height H = {H:.1f} m)")

# --------------------------------------------------------- TAB 2: contour
with tab2:
    st.subheader("Contour plot of concentration")
    cc1, cc2, cc3 = st.columns(3)
    half = cc1.number_input("Plot half-width (m)", 100.0, 100000.0, 1500.0, 100.0)
    zc = cc2.number_input("Height of plot plane z (m)", 0.0, 500.0, 1.5, 0.5)
    show_env = False
    if not is_plume:
        show_env = cc3.checkbox("Show maximum envelope up to time t", value=False,
                                help="Highest concentration reached at each location at any "
                                     "time up to t: the total affected zone.")
    cunit = st.radio("Contour units", ["mg/m³", "ppm"], horizontal=True)

    n = 220
    # centre the map downwind of the source so the cloud fills the plot
    to = np.deg2rad((wind_from + 180) % 360)
    cE, cN = 0.6 * half * np.sin(to), 0.6 * half * np.cos(to)
    Eg = np.linspace(cE - half, cE + half, n)
    Ng = np.linspace(cN - half, cN + half, n)
    EE, NN = np.meshgrid(Eg, Ng)
    X, Y = dm.to_wind_coords(EE, NN, wind_from)
    C = conc_mgm3(X, Y, zc, envelope=show_env, t_env=t)
    thr_plot = thr_mgm3
    if cunit == "ppm":
        C = dm.mgm3_to_ppm(C, mw, T_K, P_Pa)
        thr_plot = thr_in if unit == "ppm" else dm.mgm3_to_ppm(thr_mgm3, mw, T_K, P_Pa)

    cmax = C.max()
    if cmax <= 0 or not np.isfinite(cmax):
        st.warning("Concentration is zero everywhere in this window "
                   "(plume not arrived yet, or window too small). Adjust inputs.")
    else:
        levels = np.logspace(np.log10(cmax) - 5, np.log10(cmax), 16)
        fig, ax = plt.subplots(figsize=(8.5, 7))
        Cp = np.where(C > levels[0], C, np.nan)
        cs = ax.contourf(EE, NN, Cp, levels=levels, norm=LogNorm(), cmap="YlOrRd")
        cb = fig.colorbar(cs, ax=ax, pad=0.02)
        cb.set_label(f"Concentration ({cunit})")
        if cmax > thr_plot > 0:
            ax.contour(EE, NN, C, levels=[thr_plot], colors="blue", linewidths=2)
            ax.plot([], [], color="blue", lw=2, label=f"Threshold = {thr_in:g} {unit}")
        ax.plot(0, 0, "k*", ms=16, label="Release point")
        arrow_len = 0.25 * half
        ax.annotate("", xy=(arrow_len * np.sin(to), arrow_len * np.cos(to)), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", lw=2, color="k"))
        ax.text(arrow_len * 1.08 * np.sin(to), arrow_len * 1.08 * np.cos(to),
                "wind", ha="center", fontsize=9)
        ax.set_xlabel("East (m)")
        ax.set_ylabel("North (m)")
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend(loc="upper right")
        title = f"{chem} – {'plume' if is_plume else 'puff'}, class {stab}, u = {u} m/s"
        if not is_plume:
            title += f", t = {t:g} s" + (" (max envelope)" if show_env else "")
        ax.set_title(title)
        st.pyplot(fig)

        # affected-zone summary
        Cm = C if cunit == "mg/m³" else dm.ppm_to_mgm3(C, mw, T_K, P_Pa)
        mask = Cm >= thr_mgm3
        dA = (Eg[1] - Eg[0]) * (Ng[1] - Ng[0])
        st.subheader("Affected zone (concentration ≥ threshold)")
        if mask.any():
            dist = np.hypot(EE[mask], NN[mask])
            a1, a2, a3 = st.columns(3)
            a1.metric("Affected area", f"{mask.sum() * dA / 1e4:,.2f} ha")
            a2.metric("Max distance from source", f"{dist.max():,.0f} m")
            a3.metric("Peak concentration", f"{cmax:.4g} {cunit}")
        else:
            st.success("No location in this plane exceeds the threshold.")

        df = pd.DataFrame({"East_m": EE.ravel(), "North_m": NN.ravel(),
                           f"Conc_{cunit.replace('/', '_per_').replace('³', '3')}": C.ravel()})
        st.download_button("⬇️ Download grid data (CSV)", df.to_csv(index=False),
                           "dispersion_grid.csv", "text/csv")

# ------------------------------------------------------ TAB 3: profile
with tab3:
    st.subheader("Centre-line concentration vs downwind distance")
    p1, p2, p3 = st.columns(3)
    xmax = p1.number_input("Max distance (m)", 100.0, 1e6, 3000.0, 100.0)
    zp = p2.number_input("Height z (m)", 0.0, 500.0, 1.5, 0.5, key="zp")
    log_y = p3.checkbox("Log scale (y-axis)", value=True)
    xs = np.linspace(5, xmax, 600)
    Cp = conc_mgm3(xs, 0.0, zp)
    fig2, ax2 = plt.subplots(figsize=(8, 4.5))
    ax2.plot(xs, Cp, lw=2, label="Centre-line")
    if thr_mgm3 > 0:
        ax2.axhline(thr_mgm3, color="r", ls="--", label=f"Threshold {thr_in:g} {unit}")
    ax2.set_xlabel("Downwind distance (m)")
    ax2.set_ylabel("Concentration (mg/m³)")
    if log_y and Cp.max() > 0:
        ax2.set_yscale("log")
        ax2.set_ylim(max(Cp.max() * 1e-6, 1e-12), Cp.max() * 3)
    ax2.grid(alpha=0.3, which="both")
    ax2.legend()
    st.pyplot(fig2)
    above = xs[Cp >= thr_mgm3]
    if above.size:
        st.info(f"Threshold exceeded from **{above.min():,.0f} m** to **{above.max():,.0f} m** downwind.")
    else:
        st.info("Threshold is not exceeded along the centre-line.")

# ---------------------------------------------------------- TAB 4: theory
with tab4:
    st.markdown(r"""
### Gaussian plume (continuous release, steady state)
$$C(x,y,z)=\frac{Q}{2\pi u\,\sigma_y\sigma_z}\exp\!\left(-\frac{y^2}{2\sigma_y^2}\right)
\left[\exp\!\left(-\frac{(z-H)^2}{2\sigma_z^2}\right)+\exp\!\left(-\frac{(z+H)^2}{2\sigma_z^2}\right)\right]$$

### Gaussian puff (instantaneous release)
$$C(x,y,z,t)=\frac{Q}{(2\pi)^{3/2}\sigma_x\sigma_y\sigma_z}
\exp\!\left(-\frac{(x-ut)^2}{2\sigma_x^2}\right)\exp\!\left(-\frac{y^2}{2\sigma_y^2}\right)
\left[\exp\!\left(-\frac{(z-H)^2}{2\sigma_z^2}\right)+\exp\!\left(-\frac{(z+H)^2}{2\sigma_z^2}\right)\right]$$

* **Q** – release rate (g/s) or total mass (g); **u** – wind speed; **H** – effective release height  
* **σy, σz** – Briggs (1973) dispersion coefficients, function of distance and Pasquill–Gifford stability class  
* The second exponential in the bracket is the image source that models **ground reflection**  
* For the puff, σx = σy and the coefficients are evaluated at the distance travelled, *u·t*

### Assumptions & limitations
1. Flat, uniform terrain; constant wind speed and direction  
2. Passive / neutrally buoyant gas, no chemical reaction, no deposition  
3. No mixing-height (inversion lid) reflection; valid roughly for 100 m – 10 km  
4. Not valid for dense-gas releases (e.g. cold LNG, large chlorine / ammonia flashes) – use a heavy-gas model (SLAB, DEGADIS) for those  
5. Results are for screening / educational use, not for emergency decisions
""")
