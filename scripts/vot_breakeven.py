import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.ticker import FuncFormatter
from pathlib import Path

# ---- Constants ----
g = 9.80665
T_ISA = 216.65
gamma = 1.4
R_air = 287.058
a_sound = np.sqrt(gamma * R_air * T_ISA)
OEW_frac = 0.52
m_payload = 1000.0
reserve_frac = 0.05
CO2_per_fuel = 3.16

# DOC rates ($/block-hour for time-scaled terms; $/mission for fees)
SUP_MAINT, SUP_CREW, SUP_FEES, SUP_CAPITAL = 1000, 500, 350, 720
SUB_MAINT, SUB_CREW, SUB_FEES, SUB_CAPITAL = 180, 300, 150, 44
# Jet-A price: EIA Gulf Coast kerosene-type jet fuel, 2019-2023 mean in 2022 USD
# (~$0.80/kg). Base case applies no supersonic procurement premium (p_sup = 0);
# procurement/SAF premia are captured via the +-50% fuel-price sensitivity instead.
FUEL_PRICE_SUP = 0.80
FUEL_PRICE_SUB = 0.80

# ---- Breguet ----
def breguet_fuel(mach, ld, sfc_mg, R_m):
    V = mach * a_sound
    sfc_si = sfc_mg * 1e-6
    exponent = R_m * sfc_si * g / (ld * V)
    fuel_frac = 1.0 - np.exp(-exponent)
    denom = 1.0 - OEW_frac - fuel_frac * (1.0 + reserve_frac)
    if np.any(denom <= 0.01):
        return np.nan
    m_gross = m_payload / denom
    return fuel_frac * m_gross * (1.0 + reserve_frac)


def supersonic_doc(R_km, mach=1.8, ld=6.0, sfc=35, overhead_co2=350):
    """Total DOC ($) for supersonic over R_km."""
    R_m = R_km * 1000
    fuel_cruise = breguet_fuel(mach, ld, sfc, R_m)
    if np.isnan(fuel_cruise):
        return np.nan
    fuel_overhead = overhead_co2 / CO2_per_fuel
    total_fuel = fuel_cruise + fuel_overhead
    block_time = R_km / (mach * a_sound * 3.6)  # hours (V in km/h)
    # Add symmetric taxi/ground time (~0.25 h), applied equally to both modes
    block_time += 0.25
    c_fuel = total_fuel * FUEL_PRICE_SUP
    c_fixed = SUP_MAINT * block_time + SUP_CREW * block_time + SUP_FEES + SUP_CAPITAL * block_time
    return c_fuel + c_fixed


def subsonic_doc(R_km, overhead_co2=120):
    """Total DOC ($) for subsonic over R_km."""
    # Subsonic cruise EF = 0.465 kg/ton-km (from manuscript)
    cruise_co2 = 0.465 * (m_payload / 1000) * R_km
    total_co2 = cruise_co2 + overhead_co2
    total_fuel = total_co2 / CO2_per_fuel
    V_sub = 0.80 * 295.1  # Mach 0.80 at cruise altitude ~295 m/s
    block_time = R_km / (V_sub * 3.6) + 0.25  # hours with taxi
    c_fuel = total_fuel * FUEL_PRICE_SUB
    c_fixed = SUB_MAINT * block_time + SUB_CREW * block_time + SUB_FEES + SUB_CAPITAL * block_time
    return c_fuel + c_fixed


# ---- VOT break-even ----
# Step of 50 km guarantees the 1,500 and 5,000 km reference points fall exactly on
# the grid, so the annotated value matches the manuscript text.
stage_lengths = np.arange(500, 8001, 50)

# Conservative supersonic scenario
vot_conservative = []
vot_baseline_aero = []  # baseline L/D=7.0, SFC=27.5
vot_optimistic = []     # L/D=8.0, SFC=20

for R_km in stage_lengths:
    # Block times
    V_sup = 1.8 * a_sound  # m/s
    V_sub = 0.80 * a_sound
    bt_sup = R_km / (V_sup * 3.6) + 0.25  # hours
    bt_sub = R_km / (V_sub * 3.6) + 0.25
    delta_t = bt_sub - bt_sup  # time saved (hours)

    if delta_t <= 0:
        vot_conservative.append(np.nan)
        vot_baseline_aero.append(np.nan)
        vot_optimistic.append(np.nan)
        continue

    # Conservative
    doc_sup = supersonic_doc(R_km, mach=1.8, ld=6.0, sfc=35)
    doc_sub = subsonic_doc(R_km)
    delta_doc = doc_sup - doc_sub
    vot_conservative.append(delta_doc / delta_t if delta_t > 0 else np.nan)

    # Baseline aero
    doc_sup_b = supersonic_doc(R_km, mach=1.8, ld=7.0, sfc=27.5)
    vot_baseline_aero.append((doc_sup_b - doc_sub) / delta_t)

    # Optimistic
    doc_sup_o = supersonic_doc(R_km, mach=1.8, ld=8.0, sfc=20)
    vot_optimistic.append((doc_sup_o - doc_sub) / delta_t)

vot_conservative = np.array(vot_conservative)
vot_baseline_aero = np.array(vot_baseline_aero)
vot_optimistic = np.array(vot_optimistic)

# ---- Literature VOT ranges ----
vot_pharma = (300, 600)
vot_express = (100, 300)

# ---- Plot ----
# Log y-axis so the literature VOT bands (100-600) and the break-even curves
# (up to ~10,000) are both legible. Colourblind-safe (Okabe-Ito) hues plus
# distinct line styles so the three scenarios differ without relying on colour.
# Enlarged from (10, 6): the 14 pt legend moved to a dedicated strip below
# the axes (it no longer fits inside the panel without covering the curves).
fig, ax = plt.subplots(figsize=(10, 7.2))

ax.plot(stage_lengths, vot_conservative, color='#D55E00', ls='-', lw=2.2,
        label='Conservative cargo (L/D=6, SFC=35)')
ax.plot(stage_lengths, vot_baseline_aero, color='#000000', ls='--', lw=2.0,
        label='Baseline aero (L/D=7, SFC=27.5)')
ax.plot(stage_lengths, vot_optimistic, color='#0072B2', ls=':', lw=2.4,
        label='Optimistic (L/D=8, SFC=20)')

# Literature VOT bands (stronger fill + bordered upper limit)
ax.axhspan(vot_pharma[0], vot_pharma[1], alpha=0.28, color='#9467BD',
           label=f'Pharma/cold-chain VOT ({vot_pharma[0]}-{vot_pharma[1]})')
ax.axhline(vot_pharma[1], color='#9467BD', lw=0.8, ls='-', alpha=0.7)
ax.axhspan(vot_express[0], vot_express[1], alpha=0.28, color='#E69F00',
           label=f'Express/e-commerce VOT ({vot_express[0]}-{vot_express[1]})')
ax.axhline(vot_express[1], color='#E69F00', lw=0.8, ls='-', alpha=0.7)

# Reference line at 1,500 km
ax.axvline(1500, color='gray', ls=':', lw=1, alpha=0.7)
ax.text(1560, 130, '1500 km\nbaseline', fontsize=13, color='#333333',
        bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=1.5))

ax.set_yscale('log')
# Plain currency labels (100 / 1000 / 10,000) instead of 10^2/10^3 scientific notation.
# MDPI: commas only for numbers with five or more digits, so 1000 has no comma.
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:,.0f}" if y >= 10000 else f"{y:.0f}"))
ax.set_xlabel('Stage length (km)', fontsize=16)
ax.set_ylabel('Break-even VOT ($/ton-hour, log scale)', fontsize=16)
ax.tick_params(axis='both', labelsize=14)
# Conclusion as the axes title (italic), main title as the suptitle (bold) — stacked, no overlap
ax.set_title('Break-even VOT exceeds the pharma/cold-chain ceiling at every stage length',
             fontsize=13, style='italic', color='#333333', pad=8)
fig.suptitle('Value-of-Time Break-Even: Supersonic vs Subsonic Cargo',
             fontsize=17, fontweight='bold', y=0.975)
fig.legend(loc='lower center', bbox_to_anchor=(0.5, 0.0), ncol=2,
           fontsize=14, framealpha=0.9, borderpad=0.4, labelspacing=0.4,
           handlelength=1.8, handletextpad=0.6, columnspacing=1.4)
ax.set_xlim(500, 8000)
ax.set_ylim(80, max(vot_conservative[np.isfinite(vot_conservative)]) * 2.2)
ax.grid(which='both', alpha=0.3)

_halo = [pe.withStroke(linewidth=2.5, foreground='white')]

# Annotate 1500 km conservative VOT
idx_1500 = np.argmin(np.abs(stage_lengths - 1500))
ax.annotate(f'Conservative cargo: ${vot_conservative[idx_1500]:.0f}/ton-hour at 1500 km',
            xy=(1500, vot_conservative[idx_1500]),
            xytext=(2100, vot_conservative[idx_1500] * 1.7),
            arrowprops=dict(arrowstyle='->', color='#D55E00', lw=1.6),
            fontsize=14, color='#D55E00', path_effects=_halo)

# Key argument: even the OPTIMISTIC (best-case) curve stays above the market ceiling
idx_opt = np.argmin(np.abs(stage_lengths - 6000))
ax.annotate('Even the optimistic case stays\nabove the pharma/cold-chain ceiling',
            xy=(6000, vot_optimistic[idx_opt]),
            xytext=(4200, 660),
            arrowprops=dict(arrowstyle='->', color='#0072B2', lw=1.6),
            fontsize=13, color='#0072B2', ha='center', path_effects=_halo)

# Explain the conservative-curve termination (weight-limited range)
finite = np.isfinite(vot_conservative)
last_x = stage_lengths[finite][-1]
last_y = vot_conservative[finite][-1]
if last_x < stage_lengths[-1]:
    ax.annotate('weight-limited\nmax range',
                xy=(last_x, last_y), xytext=(last_x - 650, last_y * 1.28),
                arrowprops=dict(arrowstyle='->', color='#D55E00', alpha=0.85, lw=1.5),
                fontsize=13, color='#D55E00', ha='center', path_effects=_halo)

plt.tight_layout(rect=[0, 0.17, 1, 0.95])
output = Path(__file__).with_name("vot_breakeven.png")
plt.savefig(output, dpi=600, bbox_inches="tight")
print(f"Saved {output}")
print(f"\nVOT at 1,500 km:")
print(f"  Conservative: ${vot_conservative[idx_1500]:.0f}/ton-hr")
print(f"  Baseline:     ${vot_baseline_aero[idx_1500]:.0f}/ton-hr")
print(f"  Optimistic:   ${vot_optimistic[idx_1500]:.0f}/ton-hr")

# ---- Canonical DOC breakdown at 1,500 km (conservative supersonic) ----
def doc_breakdown(R_km, mode, **kw):
    if mode == "sup":
        R_m = R_km * 1000
        fuel = breguet_fuel(kw.get("mach", 1.8), kw.get("ld", 6.0),
                            kw.get("sfc", 35), R_m) + 350 / CO2_per_fuel
        bt = R_km / (1.8 * a_sound * 3.6) + 0.25
        return dict(fuel=fuel * FUEL_PRICE_SUP, maint=SUP_MAINT * bt,
                    crew=SUP_CREW * bt, fees=SUP_FEES, capital=SUP_CAPITAL * bt, bt=bt)
    else:
        cruise_co2 = 0.465 * (m_payload / 1000) * R_km
        fuel = (cruise_co2 + 120) / CO2_per_fuel
        bt = R_km / (0.80 * a_sound * 3.6) + 0.25
        return dict(fuel=fuel * FUEL_PRICE_SUB, maint=SUB_MAINT * bt,
                    crew=SUB_CREW * bt, fees=SUB_FEES, capital=SUB_CAPITAL * bt, bt=bt)

print("\n=== Canonical DOC breakdown @ 1,500 km ===")
for mode, lbl in [("sup", "Supersonic (conservative L/D=6, SFC=35)"), ("sub", "Subsonic")]:
    b = doc_breakdown(1500, mode)
    tot = b["fuel"] + b["maint"] + b["crew"] + b["fees"] + b["capital"]
    print(f"{lbl}: block={b['bt']:.2f} h | fuel ${b['fuel']:.0f}, maint ${b['maint']:.0f}, "
          f"crew ${b['crew']:.0f}, fees ${b['fees']:.0f}, capital ${b['capital']:.0f} "
          f"=> TOTAL ${tot:.0f} ({tot/1500:.2f} $/ton-km)")
bs = doc_breakdown(1500, "sup"); bb = doc_breakdown(1500, "sub")
tot_s = sum(bs[k] for k in ["fuel", "maint", "crew", "fees", "capital"])
tot_b = sum(bb[k] for k in ["fuel", "maint", "crew", "fees", "capital"])
print(f"DOC premium = ${tot_s - tot_b:.0f}/ton | ratio = {tot_s/tot_b:.2f}x | "
      f"dt = {bb['bt'] - bs['bt']:.2f} h | VOT = ${(tot_s - tot_b)/(bb['bt'] - bs['bt']):.0f}/ton-hr")

# ---- Surface-mode DOC vs stage length (print-only; reviewer B3) ----
# Per-ton mission cost components at the 1,500 km baseline (manuscript
# Table 5 / Section 2.5): time- and distance-based components (fuel,
# maintenance, crew/driver, capital) scale linearly with distance from the
# 1,500 km baseline, while the per-mission access charge (road tolls, port
# fees, track access) is fixed per mission.
SURFACE_BASE_KM = 1500.0
surface_modes = {
    'Trucking': ({'fuel': 42.0, 'maintenance': 25.0, 'driver': 60.0,
                  'capital': 25.0}, ('road tolls', 20.0)),
    'Rail':     ({'fuel': 14.0, 'maintenance': 30.0, 'crew': 35.0,
                  'capital': 15.0}, ('track access', 45.0)),
    'Shipping': ({'fuel': 2.0, 'maintenance': 5.0, 'crew': 18.0,
                  'capital': 10.0}, ('port fees', 22.0)),
}
print("\n=== Surface-mode DOC vs stage length ($/ton-km) ===")
print("(variable components linear from 1,500 km baseline; access charge fixed)")
for mode, (var_comp, (fee_name, fee)) in surface_modes.items():
    var_total = sum(var_comp.values())
    var_rate = var_total / SURFACE_BASE_KM
    cells = []
    for d_stage in (500, 1500, 5000):
        intensity = var_rate + fee / d_stage
        cells.append(f"{d_stage:>5} km: {intensity:.4f} (${intensity:.2f})")
    print(f"{mode:9s} [variable ${var_total:.0f} + {fee_name} ${fee:.0f} fixed]  "
          + " | ".join(cells))

# ---- DOC one-at-a-time (OAT) sensitivity (print-only; reviewer C3) ----
# Re-evaluates the additive DOC decomposition (conservative supersonic,
# 1,500 km, 1-ton functional unit) with one assumption-based cost input
# perturbed at a time, all others at baseline. Uses the exact rate constants
# above and unrounded block times. The subsonic DOC is held at baseline
# except in the fuel-price row (common price shifts both modes).
UTILIZATION_BASE = 3500.0   # block-h/year
CAPITAL_LIFE_N = 20         # years
CAPITAL_RESIDUAL = 0.15     # residual value fraction
CAPITAL_I_BASE = 0.08       # real cost of capital


def _annualized_capital(i, n=CAPITAL_LIFE_N, residual=CAPITAL_RESIDUAL):
    """Annualized capital charge per unit airframe price:
    (1 - residual*(1+i)^-n) * CRF(i, n)."""
    crf = i * (1 + i) ** n / ((1 + i) ** n - 1)
    return (1 - residual * (1 + i) ** (-n)) * crf


def sup_doc_oat(maint_scale=1.0, crew_scale=1.0, fees_scale=1.0,
                price_scale=1.0, i_rate=CAPITAL_I_BASE,
                utilization=UTILIZATION_BASE, fuel_scale=1.0,
                capital_scale=1.0):
    b = doc_breakdown(1500, "sup")
    capital = (b["capital"] * capital_scale * price_scale
               * (_annualized_capital(i_rate) / _annualized_capital(CAPITAL_I_BASE))
               * (UTILIZATION_BASE / utilization))
    total = (b["fuel"] * fuel_scale + b["maint"] * maint_scale
             + b["crew"] * crew_scale + b["fees"] * fees_scale + capital)
    return total / 1500.0


def sub_doc_oat(fuel_scale=1.0):
    b = doc_breakdown(1500, "sub")
    total = (b["fuel"] * fuel_scale + b["maint"] + b["crew"] + b["fees"]
             + b["capital"])
    return total / 1500.0


sub_base = sub_doc_oat()
sup_base = sup_doc_oat()
print("\n=== Supersonic DOC one-at-a-time sensitivity (1,500 km, 1-ton unit) ===")
print(f"{'Perturbed input':38s} {'Sup DOC ($/ton-km)':>20s} {'Ratio vs subsonic':>19s}")
print(f"{'None (baseline)':38s} {sup_base:>20.2f} {sup_base / sub_base:>18.2f}x")


def _oat_row(label, lo_kw, hi_kw, sub_lo=None, sub_hi=None):
    d_lo, d_hi = sup_doc_oat(**lo_kw), sup_doc_oat(**hi_kw)
    r_lo = d_lo / (sub_lo if sub_lo is not None else sub_base)
    r_hi = d_hi / (sub_hi if sub_hi is not None else sub_base)
    print(f"{label:38s} {f'{d_lo:.2f}--{d_hi:.2f}':>20s} "
          f"{f'{r_lo:.2f}--{r_hi:.2f}x':>19s}")


_oat_row("Maintenance rate +-50%", dict(maint_scale=0.5), dict(maint_scale=1.5))
_oat_row("Crew rate +-50%", dict(crew_scale=0.5), dict(crew_scale=1.5))
_oat_row("Per-mission fees +-50%", dict(fees_scale=0.5), dict(fees_scale=1.5))
_oat_row("Airframe price +-25%", dict(price_scale=0.75), dict(price_scale=1.25))
_oat_row("Real cost of capital 6--10%", dict(i_rate=0.06), dict(i_rate=0.10))
_oat_row("Fuel price +-50% (both modes)",
         dict(fuel_scale=0.5), dict(fuel_scale=1.5),
         sub_lo=sub_doc_oat(fuel_scale=0.5), sub_hi=sub_doc_oat(fuel_scale=1.5))

doc_util = sup_doc_oat(utilization=1800.0)
cap_base = doc_breakdown(1500, "sup")["capital"] / 1500.0
cap_util = cap_base * (UTILIZATION_BASE / 1800.0)
print(f"{'Utilization 1,800 h/yr':38s} {doc_util:>20.2f} {doc_util / sub_base:>18.2f}x")
print(f"  -> capital line +{(cap_util / cap_base - 1) * 100:.0f}%, "
      f"total supersonic DOC +{(doc_util / sup_base - 1) * 100:.0f}% vs baseline")

joint_25 = sup_doc_oat(maint_scale=0.75, capital_scale=0.75)
joint_50 = sup_doc_oat(maint_scale=0.50, capital_scale=0.50)
print(f"{'Maint + capital jointly -25/-50%':38s} "
      f"{f'{joint_25:.2f}--{joint_50:.2f}':>20s} "
      f"{f'{joint_25 / sub_base:.2f}--{joint_50 / sub_base:.2f}x':>19s}")
print(f"  -> joint reduction lowers supersonic DOC by "
      f"{(1 - joint_25 / sup_base) * 100:.0f}--{(1 - joint_50 / sup_base) * 100:.0f}%")
