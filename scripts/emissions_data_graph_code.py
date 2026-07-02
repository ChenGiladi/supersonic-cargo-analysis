"""Figure 1 — emissions comparison.

Visual grammar:
- Marker shape encodes transport class (square = aviation, circle = surface).
- Whisker style encodes uncertainty source class (E = scenario envelope,
  L = literature bounds, F = +/-5 percent efficiency band) so the three
  range types are visually distinguishable rather than implicitly
  identical, and the figure no longer relies on the caption to warn that
  bars across modes are not directly comparable.
- An in-figure legend ("Range source") sits in the leftmost panel so the
  encoding key is visible without reading the caption.
- Numeric value labels use g-format (4 sig figs, no trailing zeros).
"""
import csv
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import numpy as np

# --- TYPOGRAPHY ---
matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
matplotlib.rcParams.update({
    'font.size': 18,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'axes.labelsize': 17,
    'legend.fontsize': 14,
})

# --- DATA LOADING ---
BASE_PATH = Path(__file__).resolve().parent
emissions_path = BASE_PATH / "emissions_data.csv"
if not emissions_path.exists():
    raise FileNotFoundError(f"Could not find {emissions_path}")

# (csv_key, total, min, max, panel title)
categories = [
    ("CO2",   "CO2_total_kg",   "CO2_min_kg",   "CO2_max_kg",   r"CO$_2$"),
    ("NOx",   "NOx_total_kg",   "NOx_min_kg",   "NOx_max_kg",   r"NO$_x$"),
    ("Water", "Water_total_kg", "Water_min_kg", "Water_max_kg", r"H$_2$O"),
]

AVIATION = {"Supersonic", "Subsonic"}
# Range-source encoding per (mode, panel):
#   E = scenario envelope (grid min/max)   -> translucent band
#   L = literature bounds (EI / intensity) -> capped whisker
#   F = ~+-5% efficiency band              -> thin dotted whisker
ENCODING = {
    ("Supersonic", "CO2"):   "E",
    ("Supersonic", "NOx"):   "L",
    ("Supersonic", "Water"): "E",
    ("Subsonic",   "CO2"):   "F",
    ("Subsonic",   "NOx"):   "L",
    ("Subsonic",   "Water"): "F",
}
# All surface modes use literature bounds in every panel
for mode in ("Trucking", "Shipping", "Rail"):
    for cat, *_ in categories:
        ENCODING[(mode, cat)] = "L"

data_store = {cat: {"val": [], "min": [], "max": []} for cat, *_ in categories}
modes = []
with emissions_path.open() as csvfile:
    for row in csv.DictReader(csvfile):
        modes.append(row["Mode"])
        for cat, total_key, min_key, max_key, _ in categories:
            data_store[cat]["val"].append(float(row[total_key]))
            data_store[cat]["min"].append(float(row[min_key]))
            data_store[cat]["max"].append(float(row[max_key]))


def fmt_value(v):
    """3 sig figs with thousands separators >=1000; adaptive precision below.

    Examples: 1881 -> '1,880'; 817.5 -> '818'; 12.8 -> '12.8'; 3.88 -> '3.88';
    0.6 -> '0.6'; 0.33 -> '0.33'.
    """
    import math
    if v <= 0:
        return f"{v}"
    if v >= 1000:
        exp = math.floor(math.log10(v))
        factor = 10 ** (exp - 2)
        rounded = round(v / factor) * factor
        return f"{rounded:,.0f}"
    if v >= 100:
        return f"{v:.0f}"
    if v >= 10:
        return f"{v:.1f}"
    if v >= 1:
        return f"{v:.2f}"
    return f"{v:.2g}"


MARKER_COLOR = "#2C3E50"      # deep slate, colourblind-safe central dot
BAND_COLOR   = "#7E8C9A"      # muted blue-grey for envelope bands
WHISKER_COL  = "#2C3E50"

# --- FIGURE LAYOUT ---
# No figure-level title: takeaway, assumptions, and the cross-mode-encoding
# caveat belong in the LaTeX caption (research-figure convention).
fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.2), sharey=False)

panel_letters = ['(a)', '(b)', '(c)']
# Aviation/Surface group separator sits between index 1 (Subsonic) and 2 (Trucking)
GROUP_SEP_X = 1.5

for idx, (cat, _, _, _, panel_title) in enumerate(categories):
    ax = axes[idx]
    vals = np.array(data_store[cat]["val"])
    mins = np.array(data_store[cat]["min"])
    maxs = np.array(data_store[cat]["max"])

    # log y-limits with explicit headroom for value labels
    ymin = max(0.05, mins.min() / 4.0)
    ymax = maxs.max() * 6.5
    ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(-0.6, len(modes) - 0.4)

    # Subtle aviation | surface group separator (drawn before markers, behind data)
    ax.axvline(GROUP_SEP_X, color='#CCCCCC', linewidth=0.7,
               linestyle=(0, (3, 3)), zorder=0)

    for i, mode in enumerate(modes):
        v, lo, hi = vals[i], mins[i], maxs[i]
        enc = ENCODING[(mode, cat)]
        marker = 's' if mode in AVIATION else 'o'

        # Uncertainty layer (under the marker)
        if enc == "E":
            # Narrower band (auditor: looked like a bar at +-0.22)
            ax.fill_between([i - 0.14, i + 0.14], [lo, lo], [hi, hi],
                            color=BAND_COLOR, alpha=0.38, linewidth=0, zorder=1)
            ax.plot([i, i], [lo, hi], color=BAND_COLOR, linewidth=1.0,
                    alpha=0.95, zorder=2)
        elif enc == "L":
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]],
                        fmt='none', ecolor=WHISKER_COL, elinewidth=1.8,
                        capsize=8, capthick=1.9, zorder=2)
        else:  # F = +-5% efficiency bracket; add small caps for visibility
            ax.plot([i, i], [lo, hi], color="#5D6D7E", linewidth=1.4,
                    linestyle=':', dash_capstyle='round', zorder=2)
            ax.plot([i - 0.09, i + 0.09], [lo, lo], color="#5D6D7E",
                    linewidth=1.1, zorder=2)
            ax.plot([i - 0.09, i + 0.09], [hi, hi], color="#5D6D7E",
                    linewidth=1.1, zorder=2)

        # Central marker
        ax.plot(i, v, marker=marker, color=MARKER_COLOR, markersize=10,
                markeredgecolor='white', markeredgewidth=1.2,
                linestyle='None', zorder=4)

        # Value label
        ax.text(i, hi * 1.6, fmt_value(v), ha='center', va='bottom',
                fontsize=14, color="#1B2631")

    ax.set_title(f"{panel_letters[idx]} {panel_title}",
                 fontsize=18, fontweight='bold', pad=10)
    if idx == 0:
        # Wrapped to two lines so the 16 pt label fits the panel height.
        ax.set_ylabel("Mission total (kg, log$_{10}$;\nindependent per panel)",
                      fontsize=16)

    ax.set_xticks(range(len(modes)))
    ax.set_xticklabels(modes, rotation=18, ha='right', fontsize=15)

    # Clean spines + light grid
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#888888')
    ax.spines['bottom'].set_color('#888888')
    ax.tick_params(axis='both', which='both', length=4, color='#888888')
    ax.grid(axis='y', which='major', linestyle='-', color='#DDDDDD',
            alpha=0.9, linewidth=0.6)
    ax.grid(axis='y', which='minor', visible=False)
    ax.set_axisbelow(True)

# --- Combined legend BENEATH all three panels (figure-level) ---
# Mode class (marker identity) and Interval type (uncertainty source) are kept
# as two distinct sub-legends so identity and range-source semantics stay
# visually separate, but both sit at the bottom of the figure so the legend
# applies to all three panels rather than living inside panel (a).
mode_handles = [
    Line2D([0], [0], marker='s', color='w', markerfacecolor=MARKER_COLOR,
           markeredgecolor='white', markersize=10, linestyle='None',
           label='Aviation'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=MARKER_COLOR,
           markeredgecolor='white', markersize=10, linestyle='None',
           label='Surface'),
]
range_handles = [
    Patch(facecolor=BAND_COLOR, alpha=0.38, edgecolor='none',
          label='Scenario envelope (supersonic)'),
    Line2D([0], [0], color=WHISKER_COL, linewidth=1.8, marker='|',
           markersize=14, markeredgewidth=1.9, linestyle='-',
           label='Literature range'),
    Line2D([0], [0], color="#5D6D7E", linewidth=1.4, marker='_',
           markersize=10, markeredgewidth=1.1, linestyle=':',
           label='±5% efficiency band'),
]

# Reserve a strip at the bottom of the figure for the two sub-legends
# (stacked as two centered rows so the enlarged 14 pt entries cannot collide).
plt.tight_layout(rect=[0, 0.20, 1, 1])

mode_leg = fig.legend(handles=mode_handles, loc='lower center',
                      bbox_to_anchor=(0.5, 0.09), ncol=3, frameon=False,
                      fontsize=14, title='Mode class:', title_fontsize=15,
                      borderpad=0.2, handletextpad=0.5, columnspacing=1.2)
mode_leg._legend_box.align = "center"
fig.add_artist(mode_leg)
range_leg = fig.legend(handles=range_handles, loc='lower center',
                       bbox_to_anchor=(0.5, 0.0), ncol=3, frameon=False,
                       fontsize=14, title='Interval type:', title_fontsize=15,
                       borderpad=0.2, handletextpad=0.5, columnspacing=1.2)
range_leg._legend_box.align = "center"
# Vector PDF is the primary output (pdflatex preferred); PNG retained
# for previews/external use.
pdf_path = BASE_PATH / 'emissions_comparison.pdf'
png_path = BASE_PATH / 'emissions_comparison.png'
plt.savefig(pdf_path, bbox_inches='tight')
plt.savefig(png_path, dpi=600, bbox_inches='tight')
print(f"Figure saved (vector + raster): {pdf_path}  and  {png_path}")


# ---------------------------------------------------------------------------
# Fixed-mission-overhead benchmark (Reviewer C5, revision round 1)
# Cross-check of the 120 kg CO2 per design-payload-ton subsonic fixed overhead
# against per-type block-fuel-versus-distance data transcribed from the
# EMEP/EEA Air Pollutant Emission Inventory Guidebook 2023, Chapter 1.A.3.a
# (Aviation), Annex 1 "master emissions calculator" (v1.5, 18 September 2024;
# (c) EUROCONTROL, year-2022 estimations). Block fuel = ICAO LTO-cycle fuel +
# climb/cruise/descent (CCD) fuel at the tabulated great-circle stage lengths.
# Print-only: no effect on the figure or CSV outputs above.
# ---------------------------------------------------------------------------
def _overhead_benchmark():
    import numpy as np
    NM_TO_KM = 1.852
    CO2_PER_KG_FUEL = 3.16          # manuscript convention (source file: 3.15)
    dist_nm = np.array([125.0, 200.0, 250.0, 500.0, 750.0])
    # type: (LTO fuel kg, CCD fuel kg at dist_nm, freighter design payload t)
    types = {
        'B777F-class (B77L-A)':    (2906.8, [2731.7, 3997.8, 4958.3, 8606.0, 12623.6], 107.0),
        'B767-300F-class (B763-A)':(1729.9, [1821.0, 2634.8, 3084.6, 5547.7,  8137.1],  57.0),
        'B747-400F-class (B744-A)':(3908.4, [3368.6, 4958.9, 5936.4, 10582.8, 15328.3], 113.0),
    }
    print("\n=== Fixed-overhead benchmark vs EMEP/EEA per-type block fuel (Reviewer C5) ===")
    per_ton = []
    for name, (lto, ccd, payload_t) in types.items():
        d_km = dist_nm * NM_TO_KM
        block = lto + np.asarray(ccd)
        slope, fixed = np.polyfit(d_km, block, 1)
        r = np.corrcoef(d_km, block)[0, 1]
        fixed_co2 = fixed * CO2_PER_KG_FUEL
        ov = fixed_co2 / payload_t
        per_ton.append(ov)
        print(f"{name}: fixed fuel {fixed:,.0f} kg, fixed CO2 {fixed_co2:,.0f} kg, "
              f"payload {payload_t:.0f} t -> {ov:.1f} kg CO2/design-payload ton (R^2={r*r:.4f})")
    print(f"Implied overhead bracket: {min(per_ton):.0f}--{max(per_ton):.0f} kg CO2/ton "
          f"vs manuscript assumption 120 kg "
          f"({'contains' if min(per_ton) <= 120.0 <= max(per_ton) else 'does NOT contain'} 120 kg)")

_overhead_benchmark()
