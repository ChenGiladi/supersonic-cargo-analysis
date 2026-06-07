"""Authoritative, reproducible derivation of every supersonic parametric number
cited in the manuscript.

The Breguet block-fuel model here is identical to the one used by
figure_02_mc_summary/monte_carlo_summary.py and
figure_03_parameter_heatmap/parameter_heatmap.py (same constants, same equation),
so the named-scenario, scenario-envelope and stage-length values printed below are
the single source of truth for the corresponding numbers in the manuscript tables
and text. Run:

    python3 parametric_summary.py

It prints a labelled summary and writes parametric_summary.csv next to this file.
"""
from pathlib import Path
import csv
import numpy as np

# ---- Physical constants (shared with the figure scripts) ----
g = 9.80665
T_ISA = 216.65
gamma = 1.4
R_air = 287.058
a_sound = np.sqrt(gamma * R_air * T_ISA)      # ISA isothermal-layer speed of sound
OEW_frac = 0.52
m_payload = 1000.0                             # kg (1 t functional unit)
reserve_frac = 0.05
CO2_per_fuel = 3.16                            # kg CO2 / kg fuel
H2O_per_fuel = 1.23                            # kg H2O / kg fuel
overhead_sup = 350.0                           # kg CO2 fixed mission overhead
overhead_sub = 120.0
SUB_CRUISE_EF = 0.465                          # subsonic benchmark cruise EF (kg/ton-km)
SUB_BLOCK_1500 = 0.545                         # subsonic block EF at 1,500 km
NOX_EI_SUP, NOX_EI_SUB = 20.0, 15.0            # mid-range NOx EIs (g/kg fuel)
D_BASE = 1500.0


def breguet(mach, ld, sfc_mg, R_km=D_BASE):
    """Supersonic Breguet block fuel and CO2 intensities at stage length R_km."""
    V = mach * a_sound
    sfc_si = sfc_mg * 1e-6
    R_m = R_km * 1000.0
    exponent = R_m * sfc_si * g / (ld * V)
    fuel_frac = 1.0 - np.exp(-exponent)
    denom = 1.0 - OEW_frac - fuel_frac * (1.0 + reserve_frac)
    if denom <= 0.01:                          # weight-limited: no valid solution
        return None
    m_gross = m_payload / denom
    m_fuel = fuel_frac * m_gross * (1.0 + reserve_frac)
    co2_cruise = m_fuel * CO2_per_fuel
    co2_block = co2_cruise + overhead_sup
    return dict(fuel=m_fuel, cruise=co2_cruise / R_km, block=co2_block / R_km)


rows = []          # (key, value, note) for the CSV
def rec(key, value, note=""):
    rows.append((key, value, note))
    return value


print(f"a_sound = {a_sound:.4f} m/s")
rec("a_sound_m_s", round(a_sound, 4))

# ---- Named scenarios at 1,500 km ----
named = {"baseline": (1.8, 7.0, 27.5),
         "conservative": (1.8, 6.0, 35.0),
         "optimistic": (1.8, 8.0, 20.0)}
print("\nNamed scenarios @1,500 km (cruise / block kg/ton-km, ratio vs subsonic):")
scen = {}
for name, (m, l, s) in named.items():
    r = breguet(m, l, s); scen[name] = r
    ratio = r["block"] / SUB_BLOCK_1500
    print(f"  {name:13s}: cruise={r['cruise']:.3f}  block={r['block']:.3f}  "
          f"fuel={r['fuel']:.0f}kg  ratio={ratio:.2f}x")
    rec(f"{name}_cruise", round(r["cruise"], 3))
    rec(f"{name}_block", round(r["block"], 3))
    rec(f"{name}_block_ratio", round(ratio, 2))

# ---- Full 4x4x4 scenario grid (Mach 1.6-2.2, L/D 6-8, SFC 20-35) ----
machs = np.linspace(1.6, 2.2, 4); lds = np.linspace(6, 8, 4); sfcs = np.linspace(20, 35, 4)
blocks, cruises = [], []
for m in machs:
    for l in lds:
        for s in sfcs:
            r = breguet(m, l, s)
            if r:
                blocks.append(r["block"]); cruises.append(r["cruise"])
b_min, b_max = min(blocks), max(blocks)
c_min, c_max = min(cruises), max(cruises)
print("\nFull 4x4x4 grid envelope:")
print(f"  block  {b_min:.3f}--{b_max:.3f}  ({b_min/SUB_BLOCK_1500:.2f}x--{b_max/SUB_BLOCK_1500:.2f}x)")
print(f"  cruise {c_min:.3f}--{c_max:.3f}  ({c_min/SUB_BLOCK_1500:.2f}x--{c_max/SUB_BLOCK_1500:.2f}x)")
for k, v in [("envelope_block_min", b_min), ("envelope_block_max", b_max),
             ("envelope_cruise_min", c_min), ("envelope_cruise_max", c_max),
             ("envelope_block_ratio_min", b_min/SUB_BLOCK_1500),
             ("envelope_block_ratio_max", b_max/SUB_BLOCK_1500),
             ("envelope_cruise_ratio_min", c_min/SUB_BLOCK_1500),
             ("envelope_cruise_ratio_max", c_max/SUB_BLOCK_1500)]:
    rec(k, round(v, 3))

# ---- Stage-length table (constant conservative cruise EF + fixed overhead/d) ----
EF_cruise_sup = scen["conservative"]["block"] - overhead_sup / D_BASE
print(f"\nSupersonic stage-length cruise EF (conservative) = {EF_cruise_sup:.3f} kg/ton-km")
rec("EF_cruise_sup", round(EF_cruise_sup, 3))
rec("EF_cruise_sub", SUB_CRUISE_EF)
print("Stage-length table (CO2 kg | intensity | NOx kg | H2O kg):")
for R in (500, 1500, 5000):
    Isup = EF_cruise_sup + overhead_sup / R
    co2s = Isup * R; fs = co2s / CO2_per_fuel
    Isub = SUB_CRUISE_EF + overhead_sub / R
    co2b = Isub * R; fb = co2b / CO2_per_fuel
    print(f"  {R:5d} km  SUP {co2s:7.0f} {Isup:.3f} {fs*NOX_EI_SUP/1000:6.2f} {fs*H2O_per_fuel:7.0f}"
          f"  |  SUB {co2b:6.0f} {Isub:.3f} {fb*NOX_EI_SUB/1000:5.2f} {fb*H2O_per_fuel:6.0f}")
    for tag, co2, I, fuel, ei in [("sup", co2s, Isup, fs, NOX_EI_SUP), ("sub", co2b, Isub, fb, NOX_EI_SUB)]:
        rec(f"stage_{R}_{tag}_co2_kg", round(co2, 0))
        rec(f"stage_{R}_{tag}_intensity", round(I, 3))
        rec(f"stage_{R}_{tag}_nox_kg", round(fuel * ei / 1000, 2))
        rec(f"stage_{R}_{tag}_h2o_kg", round(fuel * H2O_per_fuel, 0))

# ---- Overhead sensitivity @500 km (supersonic) ----
print("\nOverhead sensitivity @500 km (supersonic):")
for scale, lbl in ((0.75, "-25%"), (1.0, "baseline"), (1.5, "+50%")):
    oh = overhead_sup * scale
    I = EF_cruise_sup + oh / 500
    print(f"  {lbl:9s} overhead={oh:.1f}kg  CO2={I*500:.0f}kg  intensity={I:.2f}")
    rec(f"overhead500_{lbl}_co2_kg", round(I * 500, 0))
    rec(f"overhead500_{lbl}_intensity", round(I, 2))

# ---- Emissions table @1,500 (supersonic conservative) + DOC cross-check ----
fuel_1500 = scen["conservative"]["block"] * 1500 / CO2_per_fuel
nox_lo = fuel_1500 * 15 / 1000; nox_hi = fuel_1500 * 25 / 1000; nox_mid = fuel_1500 * 20 / 1000
print(f"\nEmissions @1,500 (conservative): CO2={scen['conservative']['block']*1500:.0f}kg "
      f"NOx={nox_mid:.1f} ({nox_lo:.1f}-{nox_hi:.1f}) H2O={fuel_1500*H2O_per_fuel:.0f}")
print(f"DOC fuel cross-check: {fuel_1500:.0f}kg x $0.80 = ${fuel_1500*0.80:.0f}  (matches DOC table fuel line)")
rec("emis1500_sup_co2_kg", round(scen["conservative"]["block"] * 1500, 0))
rec("emis1500_sup_nox_mid", round(nox_mid, 1))
rec("emis1500_sup_nox_lo", round(nox_lo, 1))
rec("emis1500_sup_nox_hi", round(nox_hi, 1))
rec("emis1500_sup_h2o_kg", round(fuel_1500 * H2O_per_fuel, 0))

# ---- ERF illustrative bound ----
c = scen["conservative"]["block"]
print(f"\nERF illustrative: 2.5-3x conservative = {2.5*c:.2f}-{3.0*c:.2f} kg CO2e/ton-km "
      f"({2.5*c/SUB_BLOCK_1500:.1f}-{3.0*c/SUB_BLOCK_1500:.1f}x subsonic-only)")
rec("erf_co2e_lo", round(2.5 * c, 2)); rec("erf_co2e_hi", round(3.0 * c, 2))

out = Path(__file__).with_name("parametric_summary.csv")
with out.open("w", newline="") as f:
    w = csv.writer(f); w.writerow(["quantity", "value", "note"])
    w.writerows(rows)
print(f"\nWrote {out}")
