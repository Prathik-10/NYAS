# GEO Solar Mirror Array — 1 GW (10 Mirrors + 100 m PV)

Results
- Total Power: 1,000.000 MW
- Mirror Gain: +990.0 MW (from 10 km² reflectors)
- Eclipse: 0.0 sec
- Formation: Max separation 138.4 km
- Drag: 0.0 (all > 35,775 km)

Files
- `sbsp_1gw_24h_drag.py` → Full physics optimizer (J2, SRP, drag)
- `final_orbits_1gw_24h_drag.csv` → All 66 orbital elements
- `mirrors_1gw_24h_drag.tle` → STK/GMAT import
- `power_1gw_24h_drag.png` → Flat 1 GW over 24h

Reproducible with `np.random.seed(42)`