"""
dispersion.py
Atmospheric dispersion models (pure NumPy, no Streamlit dependency).

Models
------
1. Gaussian PLUME  - continuous release (g/s), steady state
2. Gaussian PUFF   - instantaneous release (kg), time dependent

Dispersion coefficients: Briggs (1973) formulas for rural / urban terrain,
Pasquill-Gifford stability classes A-F.

Coordinate system used by the models
------------------------------------
x : downwind distance (m)      y : crosswind distance (m)      z : height (m)
"""
import numpy as np

R_GAS = 8.314  # J/(mol K)

STABILITY_DESCRIPTION = {
    "A": "Very unstable",
    "B": "Moderately unstable",
    "C": "Slightly unstable",
    "D": "Neutral",
    "E": "Slightly stable",
    "F": "Moderately stable",
}

# --------------------------------------------------------------------------
# Briggs dispersion coefficients  (x in metres, sigma in metres)
# --------------------------------------------------------------------------
def _rural(stab, x):
    if stab == "A":
        sy = 0.22 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.20 * x
    elif stab == "B":
        sy = 0.16 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.12 * x
    elif stab == "C":
        sy = 0.11 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.08 * x * (1 + 0.0002 * x) ** -0.5
    elif stab == "D":
        sy = 0.08 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.06 * x * (1 + 0.0015 * x) ** -0.5
    elif stab == "E":
        sy = 0.06 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.03 * x * (1 + 0.0003 * x) ** -1
    else:  # F
        sy = 0.04 * x * (1 + 0.0001 * x) ** -0.5
        sz = 0.016 * x * (1 + 0.0003 * x) ** -1
    return sy, sz


def _urban(stab, x):
    if stab in ("A", "B"):
        sy = 0.32 * x * (1 + 0.0004 * x) ** -0.5
        sz = 0.24 * x * (1 + 0.001 * x) ** 0.5
    elif stab == "C":
        sy = 0.22 * x * (1 + 0.0004 * x) ** -0.5
        sz = 0.20 * x
    elif stab == "D":
        sy = 0.16 * x * (1 + 0.0004 * x) ** -0.5
        sz = 0.14 * x * (1 + 0.0003 * x) ** -0.5
    else:  # E, F
        sy = 0.11 * x * (1 + 0.0004 * x) ** -0.5
        sz = 0.08 * x * (1 + 0.0015 * x) ** -0.5
    return sy, sz


def sigmas(x, stability="D", terrain="Rural"):
    """Return (sigma_y, sigma_z) in metres for downwind distance x (m)."""
    x = np.maximum(np.asarray(x, dtype=float), 1.0)  # avoid x = 0
    fn = _rural if terrain.lower().startswith("r") else _urban
    sy, sz = fn(stability, x)
    return np.maximum(sy, 0.5), np.maximum(sz, 0.5)


# --------------------------------------------------------------------------
# Pasquill stability class helper
# --------------------------------------------------------------------------
def pasquill_class(u, day_night, insolation_or_cloud):
    """
    Pasquill stability class from surface wind speed (m/s).
    day_night: 'Day' or 'Night'
    insolation_or_cloud:
        Day   -> 'Strong', 'Moderate', 'Slight'
        Night -> 'Cloudy (>=4/8)', 'Clear (<=3/8)'
    """
    if day_night == "Day":
        table = {
            "Strong":   ["A", "A-B", "B", "C", "C"],
            "Moderate": ["A-B", "B", "B-C", "C", "D"],
            "Slight":   ["B", "C", "C", "D", "D"],
        }
        row = table[insolation_or_cloud]
    else:
        table = {
            "Cloudy (>=4/8)": ["E", "E", "D", "D", "D"],
            "Clear (<=3/8)":  ["F", "F", "E", "D", "D"],
        }
        row = table[insolation_or_cloud]
    if u < 2:
        idx = 0
    elif u < 3:
        idx = 1
    elif u < 5:
        idx = 2
    elif u < 6:
        idx = 3
    else:
        idx = 4
    return row[idx]


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def _vertical_term(z, H, sz):
    """Gaussian vertical term with ground reflection (image source)."""
    return np.exp(-((z - H) ** 2) / (2 * sz ** 2)) + np.exp(-((z + H) ** 2) / (2 * sz ** 2))


def plume_concentration(x, y, z, Q, u, H, stability="D", terrain="Rural", t=None):
    """
    Continuous-release Gaussian plume.

    Q : emission rate (g/s)       u : wind speed (m/s)
    H : effective release height (m)
    t : (optional) time since release began (s). If given, locations the plume
        has not reached yet (x/u > t) have zero concentration.
    Returns concentration in g/m3.
    """
    x, y, z = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, z)))
    valid = x > 0
    xs = np.where(valid, x, 1.0)
    sy, sz = sigmas(xs, stability, terrain)
    C = Q / (2 * np.pi * u * sy * sz) * np.exp(-(y ** 2) / (2 * sy ** 2)) * _vertical_term(z, H, sz)
    C = np.where(valid, C, 0.0)
    if t is not None:
        C = np.where(xs / u <= t, C, 0.0)
    return C


def puff_concentration(x, y, z, Q, u, H, t, stability="D", terrain="Rural"):
    """
    Instantaneous-release Gaussian puff (centre moves with the wind).

    Q : total mass released (g)   t : time since release (s)
    sigma_x = sigma_y, evaluated at the distance travelled by the puff centre.
    Returns concentration in g/m3.
    """
    x, y, z = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (x, y, z)))
    if t <= 0:
        return np.zeros_like(x)
    xc = u * t
    sy, sz = sigmas(max(xc, 1.0), stability, terrain)
    sy, sz = float(sy), float(sz)
    sx = sy
    C = (Q / ((2 * np.pi) ** 1.5 * sx * sy * sz)
         * np.exp(-((x - xc) ** 2) / (2 * sx ** 2))
         * np.exp(-(y ** 2) / (2 * sy ** 2))
         * _vertical_term(z, H, sz))
    return C


def puff_envelope(x, y, z, Q, u, H, t_max, stability="D", terrain="Rural", n=60):
    """Maximum concentration reached at each point at any time 0 < t <= t_max."""
    out = np.zeros(np.broadcast(x, y, z).shape)
    for t in np.linspace(t_max / n, t_max, n):
        out = np.maximum(out, puff_concentration(x, y, z, Q, u, H, t, stability, terrain))
    return out


# --------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------
def to_wind_coords(E, N, wind_from_deg):
    """
    Convert East/North offsets (m) from the source to (downwind x, crosswind y).
    wind_from_deg: meteorological direction the wind blows FROM (0 = N, 90 = E).
    """
    to = np.deg2rad((wind_from_deg + 180.0) % 360.0)  # direction wind blows TO
    x = E * np.sin(to) + N * np.cos(to)
    y = E * np.cos(to) - N * np.sin(to)
    return x, y


def mgm3_to_ppm(c_mgm3, mw, temp_K=298.15, p_Pa=101325.0):
    """Convert mg/m3 to ppm (v/v) for a gas of molecular weight mw (g/mol)."""
    return c_mgm3 * (R_GAS * temp_K / p_Pa * 1000.0) / mw


def ppm_to_mgm3(c_ppm, mw, temp_K=298.15, p_Pa=101325.0):
    return c_ppm * mw / (R_GAS * temp_K / p_Pa * 1000.0)
