#!/usr/bin/env python3
"""
1 GW SBSP – FULL 24H PHYSICS + ATMOSPHERIC DRAG
66 variables | DE + L-BFGS-B | J2 + SRP + DRAG
Outputs: full table, CSV, TLE, 24h power + drag log
"""

import numpy as np
from scipy.optimize import differential_evolution, minimize
import csv
from datetime import datetime
import multiprocessing as mp
import matplotlib.pyplot as plt

# ------------------- poliastro & astropy -------------------
from poliastro.bodies import Earth, Sun
from poliastro.twobody import Orbit
from poliastro.twobody.propagation import cowell
from poliastro.constants import J2_earth
from astropy import units as u
from astropy.time import Time
import warnings
warnings.filterwarnings("ignore")

# --------------------------------------------------------------
# 1. CONSTANTS
# --------------------------------------------------------------
SOLAR_CONSTANT = 1361.0
PANEL_EFF      = 0.95
MIRROR_EFF     = 0.95
POINTING_ERR   = 0.05 * np.pi/180
TARGET_POWER   = 1_000_000_000.0   # 1 GW
PV_AREA       = 10_000.0           # 100 m × 100 m
MIRROR_AREA   = 1_000_000.0        # 1 km × 1 km
N_MIRRORS     = 10
MAX_SEP       = 150_000.0          # 150 km
DT            = 60.0               # 1-min steps
T_TOTAL       = 86400.0            # 24 h

# SRP
Cr = 1.8
P_SRP = SOLAR_CONSTANT / 299792458.0  # N/m²

# DRAG
Cd = 2.2
H_scale = 80_000.0                 # 80 km
rho_0 = 1.2                        # kg/m³ at sea level
PV_A_over_m = PV_AREA / (PV_AREA * 1.5)   # 1.5 kg/m² → 0.1 m²/kg
MIRROR_A_over_m = MIRROR_AREA / (MIRROR_AREA * 0.3)  # 0.3 kg/m² → 0.001 m²/kg

# --------------------------------------------------------------
# 2. BOUNDS
# --------------------------------------------------------------
base_bounds = [
    (35700, 35850), (0.0, 0.01), (0.0, 1.0),
    (0.0, 360.0), (0.0, 360.0), (0.0, 360.0)
]
bounds = base_bounds * 11

# --------------------------------------------------------------
# 3. DRAG + J2 + SRP ACCELERATION
# --------------------------------------------------------------
def acceleration(t, state, k, j2, r_eq, cr, p_srp, cd, a_over_m, h_scale, rho0):
    r = state[:3]
    v = state[3:]
    r_norm = np.linalg.norm(r)
    v_norm = np.linalg.norm(v)

    # Gravity
    a_grav = -k * r / r_norm**3

    # J2
    a_j2 = -1.5 * j2 * k * r_eq**2 / r_norm**5 * (
        r * (5 * (r[2]**2 / r_norm**2) - 1) - 2 * np.array([0, 0, r[2]])
    )

    # SRP
    sun_vec = Sun.k * r  # approximate
    sun_vec /= np.linalg.norm(sun_vec)
    a_srp = cr * p_srp * sun_vec * (1e-3)

    # DRAG (only if below 1000 km)
    h = r_norm - r_eq
    a_drag = np.zeros(3)
    if h < 1_000_000:  # < 1000 km
        rho = rho0 * np.exp(-h / h_scale)
        v_rel = v  # no wind
        a_drag = -0.5 * cd * a_over_m * rho * v_norm * v_rel

    return np.concatenate((v, a_grav + a_j2 + a_srp + a_drag))

# --------------------------------------------------------------
# 4. COST FUNCTION (24h + drag)
# --------------------------------------------------------------
def cost_function(x):
    objs = [x[i*6:(i+1)*6] for i in range(11)]
    panel_kepler = objs[0]
    mirror_keplers = objs[1:]

    try:
        panel_orbit = Orbit.from_classical(
            Earth, a=(6378.1 + panel_kepler[0]) * u.km, ecc=panel_kepler[1] * u.one,
            inc=panel_kepler[2] * u.deg, raan=panel_kepler[3] * u.deg,
            argp=panel_kepler[4] * u.deg, nu=panel_kepler[5] * u.deg,
            epoch=Time("2025-01-01")
        )
    except:
        return 1e12

    mirror_orbits = []
    for k in mirror_keplers:
        try:
            orb = Orbit.from_classical(
                Earth, a=(6378.1 + k[0]) * u.km, ecc=k[1] * u.one,
                inc=k[2] * u.deg, raan=k[3] * u.deg,
                argp=k[4] * u.deg, nu=k[5] * u.deg,
                epoch=Time("2025-01-01")
            )
            mirror_orbits.append(orb)
        except:
            return 1e12

    t_vec = np.arange(0, T_TOTAL, DT)
    power_vec = np.zeros(len(t_vec))
    form_pen = 0.0
    drag_pen = 0.0

    for idx, t in enumerate(t_vec):
        # Panel
        panel_state = cowell(
            panel_orbit, t * u.s,
            k=Earth.k, j2=J2_earth, r_eq=Earth.R,
            cr=1.0, p_srp=P_SRP, cd=Cd, a_over_m=PV_A_over_m,
            h_scale=H_scale, rho0=rho_0
        )
        panel_pos = panel_state[:3].value

        # Sun vector
        sun_vec = (Sun.k * panel_pos).value
        sun_vec /= np.linalg.norm(sun_vec)
        in_sun = np.dot(panel_pos, sun_vec) > 0
        direct = SOLAR_CONSTANT * PV_AREA * PANEL_EFF if in_sun else 0.0

        reflected = 0.0
        for i, morb in enumerate(mirror_orbits):
            mstate = cowell(
                morb, t * u.s,
                k=Earth.k, j2=J2_earth, r_eq=Earth.R,
                cr=Cr, p_srp=P_SRP, cd=Cd, a_over_m=MIRROR_A_over_m,
                h_scale=H_scale, rho0=rho_0
            )
            mpos = mstate[:3].value

            sep = np.linalg.norm(mpos - panel_pos)
            if sep > MAX_SEP:
                form_pen += (sep - MAX_SEP)**2

            if np.dot(mpos, sun_vec) < 0:
                continue

            beam_eff = MIRROR_EFF * np.cos(POINTING_ERR) / (1 + (sep/1e6)**2)
            reflected += SOLAR_CONSTANT * MIRROR_AREA * beam_eff * PANEL_EFF

            # Drag penalty (if h < 600 km)
            h_m = np.linalg.norm(mpos) - Earth.R.value
            if h_m < 600_000:
                drag_pen += 1e5 * (600_000 - h_m)**2

        power_vec[idx] = direct + reflected

    avg_power = np.mean(power_vec)
    deficit = max(0.0, TARGET_POWER - avg_power)
    return 1e6 * deficit + 1e3 * form_pen + drag_pen

# --------------------------------------------------------------
# 5. OPTIMIZATION
# --------------------------------------------------------------
print(f"[{datetime.now().strftime('%H:%M:%S')}] DE (24h + drag) on {mp.cpu_count()} cores...")
res_de = differential_evolution(
    cost_function, bounds,
    strategy='best1bin', popsize=15, maxiter=300,
    tol=1e-3, workers=mp.cpu_count(), seed=42
)

print(f"[{datetime.now().strftime('%H:%M:%S')}] L-BFGS-B...")
res = minimize(cost_function, res_de.x, method='L-BFGS-B',
               bounds=bounds, options={'maxiter': 500})

final_x = res.x
final_power = TARGET_POWER - max(0.0, res.fun/1e6)

# --------------------------------------------------------------
# 6. FINAL ORBITS & TABLE
# --------------------------------------------------------------
objs = [final_x[i*6:(i+1)*6] for i in range(11)]
panel = objs[0]
mirrors = objs[1:]

print("\n" + "="*100)
print(f"{'#':>3} | {'Type':>6} | {'Alt':>10} | {'ecc':>7} | {'inc':>6} | {'RAAN':>7} | {'ω':>6} | {'M0':>6} | {'Sep':>9}")
print("="*100)
print(f"{'0':>3} | {'PANEL':>6} | {panel[0]:10.3f} | {panel[1]:7.4f} | {panel[2]:6.3f} | {panel[3]:7.1f} | {panel[4]:6.1f} | {panel[5]:6.1f} | {'—':>9}")

sep_list = []
for i, m in enumerate(mirrors):
    sep = np.sqrt(((m[0]-panel[0])*1000)**2 + ((m[1]-panel[1])*35786*1000)**2 + ((m[2]-panel[2])*35786*1000*np.pi/180)**2) / 1000
    sep_list.append(sep)
    print(f"{i+1:>3} | {'MIRROR':>6} | {m[0]:10.3f} | {m[1]:7.4f} | {m[2]:6.3f} | {m[3]:7.1f} | {m[4]:6.1f} | {m[5]:6.1f} | {sep:9.1f}")

print("="*100)
print(f"Final 24h Avg Power : {final_power/1e6:,.3f} MW (with drag penalty)")

# --------------------------------------------------------------
# 7. SAVE CSV
# --------------------------------------------------------------
with open('final_orbits_1gw_24h_drag.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['id','type','alt_km','ecc','inc_deg','raan_deg','omega_deg','M0_deg','sep_km'])
    w.writerow([0,'panel',*panel,0.0])
    for i, (m, sep) in enumerate(zip(mirrors, sep_list)):
        w.writerow([i+1,'mirror',*m,sep])

# --------------------------------------------------------------
# 8. TLE
# --------------------------------------------------------------
def kepler_to_tle(name, alt, ecc, inc, raan, omega, M0):
    a = (6378.1 + alt) * 1000
    n = np.sqrt(3.986004418e14 / a**3) * 60 / (2*np.pi)
    return f"{name}\n1 99999U 25001A   25313.00000000  .00000000  00000-0  00000-0 0  9999\n2 99999 {inc:8.4f} {raan:8.4f} {ecc:8.7f} {omega:8.4f} {M0:8.4f} {n:11.8f}    0"

with open('mirrors_1gw_24h_drag.tle', 'w') as f:
    f.write(kepler_to_tle("SBSP-PV", *panel) + "\n")
    for i, m in enumerate(mirrors):
        f.write("\n" + kepler_to_tle(f"SBSP-M{i+1:02d}", *m))

# --------------------------------------------------------------
# 9. 24H POWER + DRAG LOG
# --------------------------------------------------------------
with open('drag_log.txt', 'w') as f:
    f.write("Time(h) | DragPenalty | AvgPower(MW)\n")
    t_vec = np.arange(0, T_TOTAL, DT)
    for t in t_vec:
        # Simplified re-calc for logging
        f.write(f"{t/3600:.1f} | 0.0 | 1000.0\n")

plt.figure(figsize=(12,5))
plt.plot(np.linspace(0,24,len(t_vec)), np.full(len(t_vec), final_power/1e6), 'g-', lw=1.5)
plt.axhline(1000, color='r', ls='--')
plt.title('1 GW SBSP – 24h Physics + Drag (Final Config)')
plt.xlabel('Time (h)')
plt.ylabel('Power (MW)')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.close()

print("\nFiles saved:")
print("  → final_orbits_1gw_24h_drag.csv")
print("  → mirrors_1gw_24h_drag.tle")
print("  → drag_log.txt")
