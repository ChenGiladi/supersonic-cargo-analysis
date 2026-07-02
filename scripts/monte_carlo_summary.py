import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Larger base fonts so the 2x2 panel labels stay legible at text-column width.
plt.rcParams.update({'font.size': 18, 'axes.titlesize': 19, 'axes.labelsize': 18,
                     'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 15})

np.random.seed(42)

# ---- Physical constants ----
g = 9.80665          # m/s^2
T_ISA = 216.65       # K at 50,000 ft (ISA stratosphere)
gamma = 1.4
R_air = 287.058      # J/(kg K)
a_sound = np.sqrt(gamma * R_air * T_ISA)  # ~295.1 m/s

# ---- Mission parameters ----
R_mission = 1_500_000  # m  (1,500 km)
d_km = 1500.0
OEW_frac = 0.52        # OEW / MTOW
m_payload = 1000.0      # kg  (1 ton)
reserve_frac = 0.05     # 5% cruise reserves
CO2_per_fuel = 3.16     # kg CO2 per kg Jet-A
H2O_per_fuel = 1.23     # kg H2O per kg Jet-A
overhead_sup = 350.0    # kg CO2 fixed overhead supersonic
overhead_sub = 120.0    # kg CO2 fixed overhead subsonic

# ---- DOC rates (from manuscript Table 5) ----
# Block times derived consistently from cruise Mach (M1.8 supersonic, M0.80 subsonic)
# at the stratospheric speed of sound, plus fixed ground time (taxi/climb/descent):
SUP_BLOCK_TIME = R_mission / (1.8 * a_sound) / 3600 + 0.25  # ~1.03 h
SUB_BLOCK_TIME = R_mission / (0.80 * a_sound) / 3600 + 0.25  # ~2.02 h
SUP_MAINT_RATE = 1000   # $/h
SUP_CREW_RATE = 500     # $/h
SUP_FEES = 350           # $ per mission
SUP_CAPITAL_RATE = 720   # $/h
SUB_MAINT_RATE = 180
SUB_CREW_RATE = 300
SUB_FEES = 150
SUB_CAPITAL_RATE = 44

# ---- Breguet fuel-fraction solver ----
def breguet_block_co2(mach, ld, sfc_mg, R=R_mission):
    """Return block CO2 (kg) and block intensity (kg/ton-km) for supersonic."""
    V = mach * a_sound
    sfc_si = sfc_mg * 1e-6  # kg/(N s)
    # Correct Breguet: fuel_frac = 1 - exp(-R * SFC * g / (L/D * V))
    exponent = R * sfc_si * g / (ld * V)
    fuel_frac = 1.0 - np.exp(-exponent)

    denom = 1.0 - OEW_frac - fuel_frac * (1.0 + reserve_frac)
    denom = np.where(denom > 0.01, denom, np.nan)
    m_gross = m_payload / denom
    m_fuel = fuel_frac * m_gross * (1.0 + reserve_frac)
    co2_cruise = m_fuel * CO2_per_fuel
    co2_block = co2_cruise + overhead_sup
    intensity = co2_block / d_km  # per 1 ton over d_km
    return co2_block, intensity, m_fuel


def compute_doc_sup(fuel_kg, fuel_price, load_factor):
    """Supersonic DOC intensity ($/ton-km)."""
    c_fuel = fuel_kg * fuel_price
    c_fixed = (SUP_MAINT_RATE + SUP_CREW_RATE + SUP_CAPITAL_RATE) * SUP_BLOCK_TIME + SUP_FEES
    total = c_fuel + c_fixed
    return total / (d_km * load_factor)


def compute_doc_sub(fuel_price, load_factor):
    """Subsonic DOC intensity ($/ton-km)."""
    sub_fuel_kg = 818.0 / CO2_per_fuel  # from canonical 818 kg CO2
    c_fuel = sub_fuel_kg * fuel_price
    c_fixed = (SUB_MAINT_RATE + SUB_CREW_RATE + SUB_CAPITAL_RATE) * SUB_BLOCK_TIME + SUB_FEES
    total = c_fuel + c_fixed
    return total / (d_km * load_factor)


# ---- Monte Carlo sampling (N=10,000) ----
n = 10_000

# Aerodynamic parameters (triangular)
mach_s = np.random.triangular(1.6, 1.8, 2.2, n)
ld_s = np.random.triangular(6.0, 7.0, 8.0, n)
sfc_s = np.random.triangular(20.0, 27.5, 35.0, n)

# Fuel price (Jet-A): EIA Gulf Coast 2019-2023 mean ~$0.80/kg (2022 USD), common to
# both modes. Truncated normal: sd = 20% of mean, hard truncation at +-50% of mean.
fp_s = np.clip(np.random.normal(0.80, 0.16, n), 0.40, 1.20)

# Load factor (triangular)
lf_s = np.random.triangular(0.50, 0.70, 0.85, n)

# EI NOx (uniform)
ei_nox_sup = np.random.uniform(15, 25, n)
ei_nox_sub = np.random.uniform(8, 20, n)

# ---- Compute supersonic CO2 from real Breguet ----
co2_block_arr, intensity_arr, fuel_arr = breguet_block_co2(mach_s, ld_s, sfc_s)

valid = np.isfinite(intensity_arr)
sup_co2 = intensity_arr[valid]       # kg/ton-km
sup_co2_kg = co2_block_arr[valid]    # kg
sup_fuel = fuel_arr[valid]           # kg fuel (cruise + reserves)
fp_v = fp_s[valid]
lf_v = lf_s[valid]

# Total block fuel (cruise fuel + overhead fuel)
sup_block_fuel = sup_fuel + overhead_sup / CO2_per_fuel

# Supersonic DOC
sup_doc = np.array([compute_doc_sup(f, p, l) for f, p, l in
                    zip(sup_block_fuel, fp_v, lf_v)])

# Subsonic CO2: fixed block EF (0.545) plus a small efficiency uncertainty
# (sigma 0.015). EI draws affect NOx only, not CO2, so this spread is the
# efficiency term, not EI variability.
sub_co2 = 0.545 + np.random.normal(0, 0.015, n)
sub_co2 = np.clip(sub_co2, 0.45, 0.65)

# Subsonic DOC
sub_doc = np.array([compute_doc_sub(p, l) for p, l in zip(fp_s, lf_s)])

# ---- Tornado sensitivity for CO2 block intensity ----
def co2_at(mach=1.8, ld=7.0, sfc=27.5):
    _, bi, _ = breguet_block_co2(mach, ld, sfc)
    return float(bi)

tornado = {}
tornado['SFC'] = abs(co2_at(sfc=35) - co2_at(sfc=20))
tornado['L/D'] = abs(co2_at(ld=6.0) - co2_at(ld=8.0))
tornado['Mach'] = abs(co2_at(mach=1.6) - co2_at(mach=2.2))

total_range = sum(tornado.values())
factors = list(tornado.keys())
weights = np.array([tornado[f] / total_range for f in factors])

# ---- Print results ----
print(f"=== Monte Carlo Results (N={valid.sum()}) ===")
print(f"Supersonic CO2  5th/50th/95th: "
      f"{np.nanpercentile(sup_co2, 5):.2f} / "
      f"{np.nanpercentile(sup_co2, 50):.2f} / "
      f"{np.nanpercentile(sup_co2, 95):.2f}")
print(f"Subsonic CO2    5th/50th/95th: "
      f"{np.percentile(sub_co2, 5):.2f} / "
      f"{np.percentile(sub_co2, 50):.2f} / "
      f"{np.percentile(sub_co2, 95):.2f}")
print(f"Supersonic DOC  5th/50th/95th: "
      f"{np.nanpercentile(sup_doc, 5):.1f} / "
      f"{np.nanpercentile(sup_doc, 50):.1f} / "
      f"{np.nanpercentile(sup_doc, 95):.1f}")
print(f"Subsonic DOC    5th/50th/95th: "
      f"{np.percentile(sub_doc, 5):.1f} / "
      f"{np.percentile(sub_doc, 50):.1f} / "
      f"{np.percentile(sub_doc, 95):.1f}")
print(f"\nCO2 sensitivity (tornado): {dict(zip(factors, [f'{w:.2f}' for w in weights]))}")

# ---- Monte Carlo variance decomposition (print-only; reviewer B2) ----
# Attributes the MC spread of the sampled outputs to the input draws using
# (i) squared Spearman rank correlations, normalized to sum to one, and
# (ii) a binned first-order Sobol' cross-check on the same 10,000 draws.
# No additional sampling: uses the arrays drawn above, so figures/CSVs are
# unchanged.

def _rank(x):
    """Ranks via double argsort (Spearman rank transform)."""
    return np.argsort(np.argsort(x))


def spearman_rho(x, y):
    """Spearman rank correlation via double-argsort ranks + np.corrcoef."""
    return np.corrcoef(_rank(x), _rank(y))[0, 1]


def sobol_first_order(x, y, n_bins=50):
    """Binned first-order Sobol' index: Var(E[Y|X_i]) / Var(Y) over
    n_bins quantile bins of the input draws."""
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    idx_bin = np.clip(np.searchsorted(edges, x, side='right') - 1, 0, n_bins - 1)
    var_y = np.var(y)
    cond_means = np.array([y[idx_bin == b].mean() for b in range(n_bins)
                           if np.any(idx_bin == b)])
    counts = np.array([np.sum(idx_bin == b) for b in range(n_bins)
                       if np.any(idx_bin == b)])
    grand = y.mean()
    return float(np.sum(counts * (cond_means - grand) ** 2) / (len(y) * var_y))


mach_v = mach_s[valid]
ld_v = ld_s[valid]
sfc_v = sfc_s[valid]

print("\n=== MC variance decomposition (Spearman rho^2, normalized) ===")
co2_inputs = {'SFC': sfc_v, 'L/D': ld_v, 'Mach': mach_v}
co2_rho2 = {k: spearman_rho(v, sup_co2) ** 2 for k, v in co2_inputs.items()}
co2_raw_sum = sum(co2_rho2.values())
print(f"Supersonic CO2 intensity (raw rho^2 sum = {co2_raw_sum:.2f}):")
for k in co2_inputs:
    print(f"  {k:5s}: share = {co2_rho2[k] / co2_raw_sum:.2f} "
          f"(rho = {np.sign(spearman_rho(co2_inputs[k], sup_co2)) * np.sqrt(co2_rho2[k]):+.3f})")

doc_inputs = {'Load factor': lf_v, 'Fuel price': fp_v,
              'SFC': sfc_v, 'L/D': ld_v, 'Mach': mach_v}
doc_rho2 = {k: spearman_rho(v, sup_doc) ** 2 for k, v in doc_inputs.items()}
doc_raw_sum = sum(doc_rho2.values())
print(f"Supersonic DOC intensity (raw rho^2 sum = {doc_raw_sum:.2f}):")
for k in doc_inputs:
    print(f"  {k:11s}: share = {doc_rho2[k] / doc_raw_sum:.3f}")

print("\n=== First-order Sobol' cross-check (50 quantile bins) ===")
for k, v in co2_inputs.items():
    print(f"  {k:5s}: S1 = {sobol_first_order(v, sup_co2):.2f}")

# Low-tail characterization: the most favorable 5% of supersonic CO2 draws
tail_thresh = np.nanpercentile(sup_co2, 5)
tail = sup_co2 <= tail_thresh
joint = (sfc_v[tail] < 25) & (ld_v[tail] > 7) & (mach_v[tail] > 1.8)
print("\n=== Low-tail block (bottom 5% of supersonic CO2 draws) ===")
print(f"  n = {tail.sum()}, CO2 <= {tail_thresh:.3f} kg/ton-km")
print(f"  mean SFC  = {sfc_v[tail].mean():.1f} mg/(N s)  (mode 27.5)")
print(f"  mean L/D  = {ld_v[tail].mean():.2f}            (mode 7.0)")
print(f"  mean Mach = {mach_v[tail].mean():.2f}           (mode 1.8)")
print(f"  fraction with (SFC<25) & (L/D>7) & (Mach>1.8): {joint.mean():.2f}")

# ---- Check canonical points ----
for label, m, l, s in [("Conservative", 1.8, 6.0, 35),
                        ("Baseline", 1.8, 7.0, 27.5),
                        ("Optimistic", 1.8, 8.0, 20)]:
    _, bi, _ = breguet_block_co2(m, l, s)
    print(f"  {label} (M={m}, L/D={l}, SFC={s}): block = {bi:.3f} kg/ton-km")

# ---- Export emissions_data.csv ----
sup_nox = (sup_co2_kg / CO2_per_fuel) * ei_nox_sup[valid] / 1000
sup_h2o = (sup_co2_kg / CO2_per_fuel) * H2O_per_fuel
sub_fuel_total = 818.0 / CO2_per_fuel
sub_nox = sub_fuel_total * ei_nox_sub / 1000

csv_path = (Path(__file__).resolve().parent.parent /
            "figure_01_emissions_comparison" / "emissions_data.csv")
with open(csv_path, 'w') as f:
    f.write("Mode,CO2_total_kg,CO2_min_kg,CO2_max_kg,"
            "NOx_total_kg,NOx_min_kg,NOx_max_kg,"
            "Water_total_kg,Water_min_kg,Water_max_kg\n")
    # Supersonic marker = conservative scenario (M1.8, L/D6, SFC35), matching
    # Table 3; error bars = deterministic scenario envelope (grid min/max) for
    # CO2 and H2O, and the full EI bounds (15-25 g/kg) for NOx. Computed from the
    # same Breguet model so Figure 1 stays consistent with the table.
    cons_co2_kg = co2_at(1.8, 6.0, 35.0) * d_km          # 1881 kg
    env_lo_kg = co2_at(2.2, 8.0, 20.0) * d_km            # best grid corner
    env_hi_kg = co2_at(1.6, 6.0, 35.0) * d_km            # worst grid corner
    cons_fuel = cons_co2_kg / CO2_per_fuel
    f.write(f"Supersonic,{cons_co2_kg:.0f},{env_lo_kg:.0f},{env_hi_kg:.0f},"
            f"{cons_fuel*20/1000:.2f},{cons_fuel*15/1000:.2f},{cons_fuel*25/1000:.2f},"
            f"{cons_fuel*H2O_per_fuel:.0f},"
            f"{env_lo_kg/CO2_per_fuel*H2O_per_fuel:.0f},"
            f"{env_hi_kg/CO2_per_fuel*H2O_per_fuel:.0f}\n")
    # Subsonic: EI-driven spread
    f.write(f"Subsonic,817.5,"
            f"{817.5 * 0.95:.1f},"
            f"{817.5 * 1.05:.1f},"
            f"3.88,"
            f"{np.percentile(sub_nox, 5):.2f},"
            f"{np.percentile(sub_nox, 95):.2f},"
            f"318.2,"
            f"{318.2 * 0.95:.1f},"
            f"{318.2 * 1.05:.1f}\n")
    f.write("Trucking,120.0,96.0,144.0,0.60,0.40,0.80,46.7,37.4,56.0\n")
    f.write("Shipping,15.0,12.0,18.0,0.33,0.24,0.47,5.8,4.7,7.0\n")
    # Rail: ORR 2020-21 freight intensity 0.0265 kg CO2e/ton-km (26.5 g, blended
    # diesel+electric traction) x 1500 km = 39.75 kg; fuel = CO2/3.16 = 12.58 kg;
    # NOx = fuel x EI(40; 30-60 g/kg)/1000; H2O = 1.23 x fuel; +-20% band on CO2/H2O.
    f.write("Rail,39.8,31.8,47.7,0.50,0.38,0.76,15.5,12.4,18.6\n")
print(f"\nEmissions CSV written to {csv_path}")

# ---- Plot ----
SUB_C, SUP_C = "#0072B2", "#E69F00"  # Okabe-Ito colourblind-safe pair
fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))

# (a) CO2 distributions
ax = axes[0]
cnt_a1, _, _ = ax.hist(sub_co2, bins=50, alpha=0.75, label="Subsonic", color=SUB_C)
cnt_a2, _, _ = ax.hist(sup_co2, bins=50, alpha=0.65, label="Supersonic", color=SUP_C)
# Convention (consistent with panel b): dotted grey = subsonic reference.
# The conservative bound uses a DISTINCT dash-dot red so no style means two things.
ax.axvline(0.545, color='#444444', ls=':', lw=1.3, label="Subsonic baseline (0.545)")
ax.axvline(1.25, color='#b2182b', ls='-.', lw=1.6, label="Conservative bound (1.25)")
ax.set_xlabel("CO$_2$ intensity (kg/ton-km)")
ax.set_ylabel("Count")
# Two-line title so the enlarged 19 pt titles cannot collide across panels;
# headroom above the histograms keeps the enlarged legend clear of the bars.
ax.set_title("(a) CO$_2$ intensity\ndistribution", fontweight='bold', loc='left')
ax.set_ylim(0, max(cnt_a1.max(), cnt_a2.max()) * 1.75)
ax.legend(fontsize=13, loc='upper right', borderpad=0.25, labelspacing=0.3,
          handlelength=1.0, handletextpad=0.4, framealpha=0.9)
ax.grid(alpha=0.2, color='#cccccc')

# (b) DOC distributions (with median markers for parity with panel a)
ax = axes[1]
cnt_b1, _, _ = ax.hist(sub_doc, bins=50, alpha=0.75, label="Subsonic", color=SUB_C)
cnt_b2, _, _ = ax.hist(sup_doc, bins=50, alpha=0.65, label="Supersonic", color=SUP_C)
ax.axvline(np.nanmedian(sub_doc), color='#444444', ls=':', lw=1.3, label="Subsonic median")
ax.axvline(np.nanmedian(sup_doc), color='black', ls='--', lw=1.3, label="Supersonic median")
ax.set_xlabel("DOC intensity ($/ton-km)")
ax.set_ylabel("Count")
ax.set_title("(b) DOC intensity\ndistribution", fontweight='bold', loc='left')
ax.set_ylim(0, max(cnt_b1.max(), cnt_b2.max()) * 1.75)
ax.legend(fontsize=13, loc='upper right', borderpad=0.25, labelspacing=0.3,
          handlelength=1.0, handletextpad=0.4, framealpha=0.9)
ax.grid(alpha=0.2, color='#cccccc')

# (c) Tornado sensitivity — sorted with the largest driver (SFC) on top
ax = axes[2]
order = np.argsort(weights)  # ascending: largest lands at the top in barh
ax.barh([factors[i] for i in order], weights[order], color=SUP_C, alpha=0.8)  # match histogram fill
ax.set_xlabel("Relative CO$_2$ sensitivity\n(normalised)")
ax.set_title("(c) CO$_2$ sensitivity,\nsupersonic", fontweight='bold', loc='left')
ax.set_xlim(0, max(weights) * 1.3)
ax.grid(axis='x', alpha=0.2, color='#cccccc')

# (The 5th/50th/95th percentile summary is reported in Table 4 and is not
# duplicated here as a figure panel.)

# No figure-level title or footer: mission context (1 t, 1,500 km), sample
# size (N), and the OAT acronym all belong in the LaTeX caption so the plot
# area is not duplicated with what the caption already states.
plt.tight_layout()
pdf_path = Path(__file__).with_name("mc_uncertainty.pdf")
png_path = Path(__file__).with_name("mc_uncertainty.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
print(f"Saved (vector + raster): {pdf_path}  and  {png_path}")
