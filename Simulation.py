# ---------------------------------------------------------------------
# MULTI-MIRROR + SPACE SOLAR PANEL + MICROWAVE DOWNLINK OPTIMIZER
# - Optimizes orbit parameters for mirrors AND the receiver (space solar panel)
# - Counts direct sunlight on the panel and reflected sunlight from multiple mirrors
# - Outputs best configuration and breakdown of contributions
# - Adjustable parameters clearly labeled in the "USER TUNABLES" section
#
# Physics notes / approximations:
# - Two-body propagation only (no J2, drag, SRP)
# - Mirrors are ideal specular reflectors (instantaneous pointing)
# - Solar disk finite size used to estimate spot spread
# - Microwave downlink modeled as a fixed efficiency factor
# - Mirror swarm: identical orbit, phased by M0 offsets
# ---------------------------------------------------------------------

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import differential_evolution
import matplotlib.pyplot as plt
import math

# ---------------------- PHYSICAL CONSTANTS ---------------------------
mu_earth = 3.986004418e14   # m^3/s^2
R_earth = 6371e3            # m
AU = 1.495978707e11         # m
SOLAR_CONSTANT = 1361.0     # W/m^2 at 1 AU
SUN_ANGULAR_RADIUS = 0.00465  # rad ~ 0.266 degrees (half-angle)
# ---------------------------------------------------------------------

# ---------------------- USER TUNABLES (change these) -----------------
# Mirror / panel sizing and count
NUM_MIRRORS = 10                # Number of identical mirror satellites
MIRROR_AREA = 100.0             # m^2 per mirror (flat area)
MIRROR_REFLECTIVITY = 0.9       # fraction
PANEL_AREA = 50.0               # m^2 solar panel area on receiver satellite

# Microwave downlink
MICROWAVE_EFFICIENCY = 0.6      # fraction of received electrical power converted & sent to ground

# Simulation / optimizer settings
SIM_DURATION_OPT_HOURS = 2      # duration per objective evaluation (hours). Short for speed.
STEPS_PER_ORBIT_OPT = 300       # resolution multiplier for optimizer evaluations
SIM_DURATION_FINAL_HOURS = 24   # high-fidelity re-eval duration (hours)
STEPS_PER_ORBIT_FINAL = 1000    # high-fidelity resolution

# Optimizer population / iteration
MAXITER = 12                    # number of generations (increase for better convergence)
POPSIZE = 10                    # population size factor (increase for better search)
# ---------------------------------------------------------------------

# ---------------------- HELPER FUNCTIONS -----------------------------
def unit(vec):
    n = np.linalg.norm(vec)
    if n < 1e-12:
        return np.zeros_like(vec)
    return vec / n

def coe_to_state(a, e, i, raan, argp, nu):
    """
    Classical orbital elements -> ECI position & velocity
    a: semi-major axis (m)
    e: eccentricity
    i: inclination (rad)
    raan: RAAN (rad)
    argp: argument of periapsis (rad)
    nu: true anomaly (rad)
    """
    p = a * (1 - e**2)
    r_pf = (p / (1 + e * np.cos(nu))) * np.array([np.cos(nu), np.sin(nu), 0.0])
    v_pf = np.sqrt(mu_earth / p) * np.array([-np.sin(nu), e + np.cos(nu), 0.0])
    cos, sin = np.cos, np.sin
    R3_w = np.array([[cos(argp), -sin(argp), 0],
                     [sin(argp),  cos(argp), 0],
                     [0, 0, 1]])
    R1_i = np.array([[1, 0, 0],
                     [0, cos(i), -sin(i)],
                     [0, sin(i),  cos(i)]])
    R3_O = np.array([[cos(raan), -sin(raan), 0],
                     [sin(raan),  cos(raan), 0],
                     [0, 0, 1]])
    Q_pX = R3_O @ R1_i @ R3_w
    r_eci = Q_pX @ r_pf
    v_eci = Q_pX @ v_pf
    return r_eci, v_eci

def two_body(t, y):
    rx, ry, rz, vx, vy, vz = y
    r = np.array([rx, ry, rz])
    rnorm = np.linalg.norm(r)
    a = -mu_earth * r / rnorm**3
    return [vx, vy, vz, a[0], a[1], a[2]]

def sun_vector_eci(t_seconds):
    # simple circular Earth orbit approx, returns Sun position vector (m) in ECI frame
    year = 365.25 * 24 * 3600.0
    ang = 2.0 * np.pi * (t_seconds % year) / year
    return AU * np.array([np.cos(ang), np.sin(ang), 0.0])

def in_earth_shadow_cone(r_sat, sun_vec):
    """
    True if r_sat is inside Earth's umbral cone (simple geometric model)
    """
    sun_dir = sun_vec / np.linalg.norm(sun_vec)
    # consider vector from satellite to anti-sun direction
    proj_len = np.dot(r_sat, -sun_dir)
    if proj_len <= 0:
        return False
    # perpendicular distance to shadow axis
    d = np.linalg.norm(r_sat + proj_len * sun_dir)
    cone_radius = R_earth * (1 - proj_len / np.linalg.norm(sun_vec))
    return d < cone_radius

# ---------------------------------------------------------------------

# ------------------ OPTICAL / POWER MODEL FUNCTIONS ------------------
def ideal_mirror_normal(sun_pos, sat_pos, recv_pos):
    """
    Compute ideal mirror normal to reflect light from Sun -> mirror -> receiver.
    Returns:
      n_hat : unit mirror normal (points away from mirror surface)
      s_hat : unit incident vector (from Sun toward mirror)
      t_hat : unit vector from mirror toward receiver (target)
    NOTE: s_hat is defined as (mirror_pos - sun_pos) to represent the incident ray direction.
    """
    # incident vector: from Sun TO mirror (important sign)
    s_vec = sat_pos - sun_pos        # vector pointing from Sun to mirror (incident direction)
    t_vec = recv_pos - sat_pos      # vector pointing from mirror to receiver (target direction)

    s_hat = unit(s_vec)
    t_hat = unit(t_vec)

    # if any degenerate, return zero
    if np.linalg.norm(s_hat) < 1e-12 or np.linalg.norm(t_hat) < 1e-12:
        return np.zeros(3), s_hat, t_hat

    # Mirror normal is bisector of incident and target direction:
    # n = unit(s_hat + t_hat)
    n_raw = unit(s_hat + t_hat)
    return n_raw, s_hat, t_hat


def reflected_power_to_target(mirror_pos, sun_pos, target_pos, mirror_area_local, reflectivity_local, target_area):
    """
    Compute approximate power intercepted by the target area (target_area) from a single mirror at mirror_pos.
    Uses sun angular radius to estimate spot size at target distance after reflection.
    Returns intercepted power (W).
    """
    # If mirror or target in shadow relative to Sun, return 0 outside caller; we assume caller checks shadow.
    n_hat, s_hat, t_hat = ideal_mirror_normal(sun_pos, mirror_pos, target_pos)
    if np.allclose(n_hat, 0.0):
        return 0.0
    # incident cos factor
    cos_inc = max(0.0, np.dot(s_hat, n_hat))  # if negative, mirror backside
    if cos_inc <= 0:
        return 0.0
    # reflected direction
    refl_hat = s_hat - 2.0 * np.dot(s_hat, n_hat) * n_hat
    # angle error between reflected ray and actual target direction
    actual_t = unit(target_pos - mirror_pos)
    ang_err = math.acos(np.clip(np.dot(refl_hat, actual_t), -1.0, 1.0))
    # if ang_err > 2 * sun angular radius, no overlap
    if ang_err > 2.0 * SUN_ANGULAR_RADIUS:
        return 0.0
    d = np.linalg.norm(target_pos - mirror_pos)
    if d <= 0:
        return 0.0
    # spot radius at target due to Sun's angular radius
    spot_radius = d * SUN_ANGULAR_RADIUS
    spot_area = math.pi * spot_radius**2 + 1e-12
    # incoming power to mirror (projected area)
    P_in = SOLAR_CONSTANT * mirror_area_local * cos_inc
    # power density of reflected beam at target (simplified): P_in * reflectivity / spot_area
    power_density = (P_in * reflectivity_local) / spot_area
    # portion intercepted by target
    fraction_intercepted = min(1.0, target_area / spot_area)
    received_power = power_density * target_area * fraction_intercepted
    return received_power

def direct_solar_power_on_panel(panel_pos, sun_pos, panel_area_local):
    """
    Compute direct solar power on panel (W) assuming panel normal always faces Sun (best-case).
    For more realism, use panel attitude and cos incidence.
    Also check for shadow in caller.
    """
    s_hat = unit(sun_pos - panel_pos)  # direction from panel towards Sun
    # assume panel perfectly normal to sun (max), so cos_incidence = 1.0 for idealized best-case.
    # For a simple but realistic treatment, we'll compute cos with a surface normal equal to s_hat (i.e., always pointed).
    cos_incidence = 1.0
    P_direct = SOLAR_CONSTANT * panel_area_local * cos_incidence
    return P_direct

# ---------------------------------------------------------------------

# ------------------ MULTI-SATELLITE SIMULATION -----------------------
def compute_total_received_power(mirror_orbit_coe, panel_orbit_coe,
                                 num_mirrors=NUM_MIRRORS,
                                 mirror_area=MIRROR_AREA,
                                 mirror_reflectivity=MIRROR_REFLECTIVITY,
                                 panel_area=PANEL_AREA,
                                 microwave_eff=MICROWAVE_EFFICIENCY,
                                 t_total_hours=SIM_DURATION_OPT_HOURS,
                                 steps_per_orbit=STEPS_PER_ORBIT_OPT):
    """
    Simulates both mirror swarm and receiver panel over t_total_hours and returns:
      - avg_delivered_power_to_ground (W) over time
      - breakdown arrays (times, panel_direct_series, panel_reflected_series, total_received_series)
    mirror_orbit_coe: tuple/list (a, e, i_deg, raan_deg, argp_deg, M0_deg) for mirrors
    panel_orbit_coe: tuple/list (a, e, i_deg, raan_deg, argp_deg, M0_deg) for receiver panel
    """
    # Unpack and convert to radians for propagation
    a_m, e_m, i_m_deg, raan_m_deg, argp_m_deg, M0_m_deg = mirror_orbit_coe
    a_p, e_p, i_p_deg, raan_p_deg, argp_p_deg, M0_p_deg = panel_orbit_coe
    i_m = np.radians(i_m_deg); raan_m = np.radians(raan_m_deg)
    argp_m = np.radians(argp_m_deg); nu0_m = np.radians(M0_m_deg)
    i_p = np.radians(i_p_deg); raan_p = np.radians(raan_p_deg)
    argp_p = np.radians(argp_p_deg); nu0_p = np.radians(M0_p_deg)

    # Compute orbital periods
    n_m = np.sqrt(mu_earth / a_m**3); T_m = 2 * np.pi / n_m
    n_p = np.sqrt(mu_earth / a_p**3); T_p = 2 * np.pi / n_p

    # Simulation time parameters
    t_final = t_total_hours * 3600.0                # seconds total
    # total steps roughly proportional to (t_final / T_mean) * steps_per_orbit
    T_mean = 0.5 * (T_m + T_p)
    total_steps = max(10, int(steps_per_orbit * (t_final / T_mean)))
    t_eval = np.linspace(0.0, t_final, total_steps)

    # Precompute initial states
    r0_m, v0_m = coe_to_state(a_m, e_m, i_m, raan_m, argp_m, nu0_m)  # one mirror reference
    r0_p, v0_p = coe_to_state(a_p, e_p, i_p, raan_p, argp_p, nu0_p)  # panel satellite

    # Propagate each satellite type once (mirrors are phase-shifted versions of r0_m)
    sol_m = solve_ivp(two_body, (0, t_final), np.hstack((r0_m, v0_m)), t_eval=t_eval, rtol=1e-8, atol=1e-8)
    sol_p = solve_ivp(two_body, (0, t_final), np.hstack((r0_p, v0_p)), t_eval=t_eval, rtol=1e-8, atol=1e-8)

    r_m_ref = sol_m.y[:3, :].T      # reference mirror positions over time
    r_p = sol_p.y[:3, :].T          # panel positions over time
    times = sol_m.t

    # Arrays to store per-step power
    panel_direct = np.zeros(len(times))
    panel_reflected = np.zeros(len(times))
    delivered_ground = np.zeros(len(times))  # after microwave efficiency

    # For mirror swarm, we create phase offsets in mean anomaly (simple approx by adding M0 offsets)
    # For simplicity we phase by equal M0 offsets across 360 degrees
    
    phase_offsets = np.linspace(0, 360.0, num_mirrors, endpoint=False)  # degrees

    for idx, tsec in enumerate(times):
        sun_pos = sun_vector_eci(tsec)

        # Panel: check shadow
        r_panel = r_p[idx]
        panel_in_shadow = in_earth_shadow_cone(r_panel, sun_pos)
        direct = 0.0
        if not panel_in_shadow:
            # Direct solar on panel (we assume panel can point to Sun optimally)
            direct = direct_solar_power_on_panel(r_panel, sun_pos, panel_area)
        panel_direct[idx] = direct

        # Mirrors: each mirror is same orbit but phase-shifted. Approx positions:
        # We approximate a mirror's current position by rotating the reference r_m_ref[idx] in orbital phase.
        # Simpler: shift the reference index by a phase offset fraction of the array length.
        reflected_sum = 0.0
        for m_idx, ph in enumerate(phase_offsets):
            # compute index shift based on phase offset (ph degrees of orbit)
            # fraction_of_orbit = ph / 360 -> shift in timesteps relative to reference
            frac = ph / 360.0
            shift = int(frac * len(times))  # wrap-around shift
            mirror_pos = r_m_ref[(idx + shift) % len(times)]

            # Check mirror shadow
            if in_earth_shadow_cone(mirror_pos, sun_pos):
                continue

            # Compute power from this mirror onto panel
            p_ref = reflected_power_to_target(mirror_pos, sun_pos, r_panel, mirror_area, mirror_reflectivity, panel_area)
            reflected_sum += p_ref

        panel_reflected[idx] = reflected_sum

        # Sum received electrical power on panel at this time
        total_received = panel_direct[idx] + panel_reflected[idx]

        # Convert to delivered ground power using microwave efficiency
        delivered_ground[idx] = total_received * microwave_eff

    # Compute average delivered power (W) over the simulation duration
    avg_delivered = np.mean(delivered_ground)
    # Also compute average direct/reflected separately
    avg_direct = np.mean(panel_direct) * microwave_eff
    avg_reflected = np.mean(panel_reflected) * microwave_eff

    return avg_delivered, avg_direct, avg_reflected, times, panel_direct, panel_reflected, delivered_ground

# ---------------------------------------------------------------------

# --------------------- OPTIMIZER INTERFACE ---------------------------
def objective_both(x):
    """
    x is 12 variables:
      [mirror_a_km, mirror_e, mirror_i_deg, mirror_raan_deg, mirror_argp_deg, mirror_M0_deg,
       panel_a_km,  panel_e,  panel_i_deg,  panel_raan_deg,  panel_argp_deg,  panel_M0_deg]
    Returns negative average delivered ground power (minimizer).
    """
    # unpack
    mirror_a_km, mirror_e, mirror_i_deg, mirror_raan_deg, mirror_argp_deg, mirror_M0_deg, \
    panel_a_km, panel_e, panel_i_deg, panel_raan_deg, panel_argp_deg, panel_M0_deg = x

    # convert km->m for a
    mirror_a = mirror_a_km * 1e3
    panel_a = panel_a_km * 1e3

    # basic validity checks
    if mirror_e < 0 or mirror_e >= 1 or panel_e < 0 or panel_e >= 1:
        return 1e6
    if mirror_a * (1 - mirror_e) < (R_earth + 150e3) or panel_a * (1 - panel_e) < (R_earth + 150e3):
        return 1e6

    # Compute average delivered power for short sim during optimization
    avg_delivered, avg_direct, avg_reflected, *_ = compute_total_received_power(
        (mirror_a, mirror_e, mirror_i_deg, mirror_raan_deg, mirror_argp_deg, mirror_M0_deg),
        (panel_a, panel_e, panel_i_deg, panel_raan_deg, panel_argp_deg, panel_M0_deg),
        num_mirrors=NUM_MIRRORS,
        mirror_area=MIRROR_AREA,
        mirror_reflectivity=MIRROR_REFLECTIVITY,
        panel_area=PANEL_AREA,
        microwave_eff=MICROWAVE_EFFICIENCY,
        t_total_hours=SIM_DURATION_OPT_HOURS,
        steps_per_orbit=STEPS_PER_ORBIT_OPT
    )

    # negative for minimizer
    return -avg_delivered

def optimize_both():
    # bounds for mirror orbit (km for semi-major axis)
    # mirror a: LEO-ish (R_earth/1e3 + 200 -> R_earth/1e3 + 3000)
    # panel a: allow higher LEO / MEO (up to 20000 km)
    bounds = [
        (R_earth/1e3 + 200, R_earth/1e3 + 3000),  # mirror a_km
        (0.0, 0.1),                               # mirror e
        (0.0, 180.0),                             # mirror i_deg
        (0.0, 360.0),                             # mirror raan_deg
        (0.0, 360.0),                             # mirror argp
        (0.0, 360.0),                             # mirror M0
        (R_earth/1e3 + 400, R_earth/1e3 + 35000), # panel a_km (allow higher)
        (0.0, 0.2),                               # panel e
        (0.0, 180.0),                             # panel i_deg
        (0.0, 360.0),                             # panel raan_deg
        (0.0, 360.0),                             # panel argp_deg
        (0.0, 360.0)                              # panel M0_deg
    ]

    result = differential_evolution(
        objective_both, bounds,
        maxiter=MAXITER, popsize=POPSIZE, polish=True, seed=42
    )
    return result
# ----------------------- FULL DIAGNOSTIC FUNCTION -----------------------
def diagnostic_reflection_check(mirror_coe_m, panel_coe_m, num_mirrors=NUM_MIRRORS,
                                t_hours=2, steps_per_orbit=300,
                                mirror_area_local=MIRROR_AREA,
                                mirror_reflectivity_local=MIRROR_REFLECTIVITY,
                                panel_area_local=PANEL_AREA):
    """
    Long diagnostic: re-simulates mirror & panel trajectories for a short time
    and reports counts for sunlit, LOS, angle-overlap, max per-mirror power, etc.
    mirror_coe_m, panel_coe_m: tuples with (a (m), e, i_deg, raan_deg, argp_deg, M0_deg)
    Returns dict of stats.
    """
    # Unpack COEs (mirror_coe_m and panel_coe_m use a in meters)
    a_m, e_m, i_m_deg, raan_m_deg, argp_m_deg, M0_m_deg = mirror_coe_m
    a_p, e_p, i_p_deg, raan_p_deg, argp_p_deg, M0_p_deg = panel_coe_m

    # Convert angles to radians for coe_to_state
    i_m = np.radians(i_m_deg); raan_m = np.radians(raan_m_deg); argp_m = np.radians(argp_m_deg); nu0_m = np.radians(M0_m_deg)
    i_p = np.radians(i_p_deg); raan_p = np.radians(raan_p_deg); argp_p = np.radians(argp_p_deg); nu0_p = np.radians(M0_p_deg)

    # orbital periods
    n_m = np.sqrt(mu_earth / a_m**3); T_m = 2.0 * np.pi / n_m
    n_p = np.sqrt(mu_earth / a_p**3); T_p = 2.0 * np.pi / n_p

    # time vector
    t_final = t_hours * 3600.0
    T_mean = 0.5 * (T_m + T_p)
    steps = max(10, int(steps_per_orbit * (t_final / T_mean)))
    t_eval = np.linspace(0.0, t_final, steps)

    # get reference trajectories (one mirror reference and the panel)
    r0_m, v0_m = coe_to_state(a_m, e_m, i_m, raan_m, argp_m, nu0_m)
    r0_p, v0_p = coe_to_state(a_p, e_p, i_p, raan_p, argp_p, nu0_p)
    sol_m = solve_ivp(two_body, (0, t_final), np.hstack((r0_m, v0_m)), t_eval=t_eval, rtol=1e-8, atol=1e-8)
    sol_p = solve_ivp(two_body, (0, t_final), np.hstack((r0_p, v0_p)), t_eval=t_eval, rtol=1e-8, atol=1e-8)
    r_m_ref = sol_m.y[:3, :].T   # (N,3)
    r_p = sol_p.y[:3, :].T       # (N,3)
    times = sol_m.t

    # phase offsets for mirror swarm
    phase_offsets = np.linspace(0.0, 360.0, num_mirrors, endpoint=False)

    # counters & stats
    total_checks = 0
    sunlit_checks = 0
    los_ok_checks = 0
    ang_ok_checks = 0
    nonzero_reflections = 0
    max_single_reflection = 0.0
    total_reflected_power_sum = 0.0

    # helper LOS check (line segment mirror->panel intersects Earth?)
    def los_blocked(p1, p2):
        d = p2 - p1
        d2 = np.dot(d, d)
        if d2 == 0:
            return np.linalg.norm(p1) < R_earth
        t = - np.dot(p1, d) / d2
        t_clamped = min(1.0, max(0.0, t))
        closest = p1 + t_clamped * d
        return np.linalg.norm(closest) < R_earth

    for idx, tsec in enumerate(times):
        sun_pos = sun_vector_eci(tsec)
        panel_pos = r_p[idx]
        for m_idx, ph in enumerate(phase_offsets):
            total_checks += 1
            frac = ph / 360.0
            shift = int(frac * len(times))
            mirror_pos = r_m_ref[(idx + shift) % len(times)]

            # 1) mirror sunlit?
            if in_earth_shadow_cone(mirror_pos, sun_pos):
                continue
            sunlit_checks += 1

            # 2) LOS mirror->panel
            if los_blocked(mirror_pos, panel_pos):
                continue
            los_ok_checks += 1

            # 3) reflection power using existing function
            p_ref = reflected_power_to_target(mirror_pos, sun_pos, panel_pos, mirror_area_local, mirror_reflectivity_local, panel_area_local)
            total_reflected_power_sum += p_ref
            if p_ref > 0.0:
                nonzero_reflections += 1
                ang_ok_checks += 1
                if p_ref > max_single_reflection:
                    max_single_reflection = p_ref

    print("\n================ DIAGNOSTIC REFLECTION CHECK ================")
    print(f"Total mirror-time checks                   : {total_checks}")
    print(f"Mirror sunlit checks (not in Earth's umbra) : {sunlit_checks}")
    print(f"Mirror->panel LOS OK checks                 : {los_ok_checks}")
    print(f"Reflections with nonzero instantaneous p    : {nonzero_reflections}")
    print(f"Max single-mirror instantaneous p (W)       : {max_single_reflection:.6e}")
    print(f"Sum of instantaneous reflected p over samples: {total_reflected_power_sum:.6e} (W sum across time samples)")
    if nonzero_reflections == 0:
        print("\n=> NO nonzero reflections detected. Likely geometry/angle mismatch (mirrors never point exactly at panel).")
    else:
        print("\n=> Detected nonzero reflections. Check magnitudes/timing to assess significance.")
    print("==============================================================\n")

    return {
        'total_checks': total_checks,
        'sunlit_checks': sunlit_checks,
        'los_ok_checks': los_ok_checks,
        'nonzero_reflections': nonzero_reflections,
        'max_single_reflection': max_single_reflection,
        'sum_reflected_power_samples': total_reflected_power_sum
    }
# ---------------------------------------------------------------------


# ---------------------------------------------------------------------

# -------------------------- MAIN -------------------------------------
if __name__ == "__main__":
    # Quick demo: evaluate a sample mirror orbit + panel orbit (no optimization)
    mirror_coe_demo = (R_earth + 700e3, 0.0, 98.0, 0.0, 0.0, 0.0)    # mirror orbit: LEO sun-sync-ish
    panel_coe_demo  = (R_earth + 2000e3, 0.0, 45.0, 0.0, 0.0, 0.0)   # panel orbit: higher LEO / MEO example

    print("Running demo simulation (short) with multiple mirrors + space panel...")
    avg_del, avg_dir, avg_ref, times, dir_series, ref_series, delivered_series = compute_total_received_power(
        mirror_coe_demo, panel_coe_demo,
        num_mirrors=NUM_MIRRORS,
        mirror_area=MIRROR_AREA,
        mirror_reflectivity=MIRROR_REFLECTIVITY,
        panel_area=PANEL_AREA,
        microwave_eff=MICROWAVE_EFFICIENCY,
        t_total_hours=SIM_DURATION_OPT_HOURS,
        steps_per_orbit=STEPS_PER_ORBIT_OPT
    )

    print(f"Demo (avg delivered to ground) ≈ {avg_del:.3e} W  (direct component {avg_dir:.3e} W, reflected {avg_ref:.3e} W)")
    print("Now running optimizer (this searches mirror + panel orbits)...")
    res = optimize_both()
    print("Optimizer finished. Summary of result object:")
    print(res)

    # Unpack best solution
    xbest = res.x
    mirror_best = xbest[0:6]    # a_km, e, i_deg, raan, argp, M0
    panel_best  = xbest[6:12]

    # Convert to meters for final high-fidelity eval
    mirror_best_m = (mirror_best[0]*1e3, mirror_best[1], mirror_best[2], mirror_best[3], mirror_best[4], mirror_best[5])
    panel_best_m = (panel_best[0]*1e3, panel_best[1], panel_best[2], panel_best[3], panel_best[4], panel_best[5])

    # FORCE TEST GEOMETRY (close mirror + panel)
    mirror_best = [7078, 0.001, 0, 0, 0, 0]   # LEO
    panel_best  = [7078, 0.001, 0, 0, 0, 10]  # small phase offset
    
    # Re-evaluate with long sim and finer steps
    print("\nRe-evaluating best candidate with high-fidelity simulation (this will be slower)...")
    avg_del_hf, avg_dir_hf, avg_ref_hf, times_hf, dir_series_hf, ref_series_hf, delivered_series_hf = compute_total_received_power(
        mirror_best_m, panel_best_m,
        num_mirrors=NUM_MIRRORS,
        mirror_area=MIRROR_AREA,
        mirror_reflectivity=MIRROR_REFLECTIVITY,
        panel_area=PANEL_AREA,
        microwave_eff=MICROWAVE_EFFICIENCY,
        t_total_hours=SIM_DURATION_FINAL_HOURS,
        steps_per_orbit=STEPS_PER_ORBIT_FINAL
    )

    

    # Print clearly labeled results
    print("\n\n================ BEST CONFIGURATION (OPTIMIZER) ================\n")
    print("Mirror orbit (best):")
    print(f"  Semi-major axis (a)         : {mirror_best[0]:.2f} km")
    print(f"  Eccentricity (e)           : {mirror_best[1]:.6f}")
    print(f"  Inclination (i)            : {mirror_best[2]:.3f}°")
    print(f"  RAAN                       : {mirror_best[3]:.3f}°")
    print(f"  Argument of Perigee (argp) : {mirror_best[4]:.3f}°")
    print(f"  Mean Anomaly (M0)          : {mirror_best[5]:.3f}°")
    print("")
    print("Panel (receiver) orbit (best):")
    print(f"  Semi-major axis (a)         : {panel_best[0]:.2f} km")
    print(f"  Eccentricity (e)           : {panel_best[1]:.6f}")
    print(f"  Inclination (i)            : {panel_best[2]:.3f}°")
    print(f"  RAAN                       : {panel_best[3]:.3f}°")
    print(f"  Argument of Perigee (argp) : {panel_best[4]:.3f}°")
    print(f"  Mean Anomaly (M0)          : {panel_best[5]:.3f}°")
    print("")
    print("Hardware & simulation parameters (user-tunable):")
    print(f"  NUM_MIRRORS              : {NUM_MIRRORS}")
    print(f"  MIRROR_AREA (m^2)        : {MIRROR_AREA}")
    print(f"  PANEL_AREA (m^2)         : {PANEL_AREA}")
    print(f"  MIRROR_REFLECTIVITY      : {MIRROR_REFLECTIVITY}")
    print(f"  MICROWAVE_EFFICIENCY     : {MICROWAVE_EFFICIENCY}")
    print(f"  SIM_DURATION_FINAL_HOURS : {SIM_DURATION_FINAL_HOURS}")
    print(f"  STEPS_PER_ORBIT_FINAL    : {STEPS_PER_ORBIT_FINAL}")
    print("")
    print("Performance (HIGH-FIDELITY re-eval):")
    print(f"  Average delivered power to ground : {avg_del_hf:.6e} W")
    print(f"    - average direct -> ground via MW  : {avg_dir_hf:.6e} W")
    print(f"    - average reflected -> ground via MW: {avg_ref_hf:.6e} W")
    print("")

    # Run diagnostic (short re-sim) on the BEST candidate to check why reflections are zero
    diag_stats = diagnostic_reflection_check(mirror_best_m, panel_best_m,
                                            num_mirrors=NUM_MIRRORS,
                                            t_hours=2,                  # short diagnostic window
                                            steps_per_orbit=300,
                                            mirror_area_local=MIRROR_AREA,
                                            mirror_reflectivity_local=MIRROR_REFLECTIVITY,
                                            panel_area_local=PANEL_AREA)
    # (diag_stats dictionary also available for programmatic inspection)

    
    # Show time series plots of received power
    plt.figure(figsize=(10,5))
    t_hours = times_hf / 3600.0
    plt.plot(t_hours, dir_series_hf, label='Direct on panel (W)', alpha=0.8)
    plt.plot(t_hours, ref_series_hf, label='Reflected on panel (W)', alpha=0.8)
    plt.plot(t_hours, delivered_series_hf, label='Delivered to ground (W) after MW eff', alpha=0.9)
    plt.xlabel('Time (hours)')
    plt.ylabel('Power (W)')
    plt.title('Power time series (high-fidelity re-eval)')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 3D plot of final panel orbit path sample and mirror sample points colored by delivered power
    fig = plt.figure(figsize=(9,7))
    ax = fig.add_subplot(111, projection='3d')
    # plot panel path (sample)
    r_plot = sol_p = None  # we don't have sol_p here easily, but we can reuse compute_total_received_power internals if needed
    # Instead plot the panel positions we computed in high-fidelity re-eval (we have times_hf and panels from function)
    # hack: recompute panel trajectory quickly for plotting
    # unpack panel COE for plotting
    a_p, e_p, i_p_deg, raan_p_deg, argp_p_deg, M0_p_deg = panel_best_m
    i_p = np.radians(i_p_deg); raan_p = np.radians(raan_p_deg); argp_p = np.radians(argp_p_deg); nu0_p = np.radians(M0_p_deg)
    sol_plot_p = solve_ivp(two_body, (0, SIM_DURATION_FINAL_HOURS*3600.0), np.hstack(coe_to_state(a_p, e_p, i_p, raan_p, argp_p, nu0_p)),
                           t_eval=np.linspace(0, SIM_DURATION_FINAL_HOURS*3600.0, 800), rtol=1e-8, atol=1e-8)
    rpos_p = sol_plot_p.y[:3, :].T
    ax.plot(rpos_p[:,0]/1e3, rpos_p[:,1]/1e3, rpos_p[:,2]/1e3, label='Panel orbit path', lw=0.8)

    # plot sample mirror reference orbit (one reference mirror)
    a_m, e_m, i_m_deg, raan_m_deg, argp_m_deg, M0_m_deg = mirror_best_m
    i_m = np.radians(i_m_deg); raan_m = np.radians(raan_m_deg); argp_m = np.radians(argp_m_deg); nu0_m = np.radians(M0_m_deg)
    sol_plot_m = solve_ivp(two_body, (0, SIM_DURATION_FINAL_HOURS*3600.0), np.hstack(coe_to_state(a_m, e_m, i_m, raan_m, argp_m, nu0_m)),
                           t_eval=np.linspace(0, SIM_DURATION_FINAL_HOURS*3600.0, 800), rtol=1e-8, atol=1e-8)
    rpos_m = sol_plot_m.y[:3, :].T
    ax.plot(rpos_m[:,0]/1e3, rpos_m[:,1]/1e3, rpos_m[:,2]/1e3, label='Mirror reference orbit', lw=0.6, color='orange')

    # Earth sphere
    u = np.linspace(0, 2*np.pi, 40)
    v = np.linspace(0, np.pi, 20)
    x = R_earth/1e3 * np.outer(np.cos(u), np.sin(v))
    y = R_earth/1e3 * np.outer(np.sin(u), np.sin(v))
    z = R_earth/1e3 * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(x, y, z, alpha=0.12, color='blue')

    ax.set_xlabel('X (km)'); ax.set_ylabel('Y (km)'); ax.set_zlabel('Z (km)')
    ax.legend()
    plt.show()

    print("\nDone. If results are too small, try increasing NUM_MIRRORS, MIRROR_AREA, or lowering panel altitude.")
# ---------------------------------------------------------------------
