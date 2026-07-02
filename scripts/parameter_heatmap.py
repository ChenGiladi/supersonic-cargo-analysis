import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from pathlib import Path

# Larger base fonts so labels stay legible when the wide 3-panel figure is
# scaled to the text column.
plt.rcParams.update({'font.size': 22, 'xtick.labelsize': 20, 'ytick.labelsize': 20,
                     'axes.labelsize': 22, 'axes.titlesize': 22})

# ---- Physical constants (same as MC script) ----
g = 9.80665
T_ISA = 216.65
gamma = 1.4
R_air = 287.058
a_sound = np.sqrt(gamma * R_air * T_ISA)

R_mission = 1_500_000  # m
d_km = 1500.0
OEW_frac = 0.52
m_payload = 1000.0
reserve_frac = 0.05
CO2_per_fuel = 3.16
overhead_sup = 350.0


def breguet_block_intensity(mach, ld, sfc_mg):
    """Block CO2 intensity (kg/ton-km) from Breguet."""
    V = mach * a_sound
    sfc_si = sfc_mg * 1e-6
    exponent = R_mission * sfc_si * g / (ld * V)
    fuel_frac = 1.0 - np.exp(-exponent)
    denom = 1.0 - OEW_frac - fuel_frac * (1.0 + reserve_frac)
    denom = np.where(denom > 0.01, denom, np.nan)
    m_gross = m_payload / denom
    m_fuel = fuel_frac * m_gross * (1.0 + reserve_frac)
    co2_block = m_fuel * CO2_per_fuel + overhead_sup
    return co2_block / d_km


# ---- Grid ----
mach_vals = np.linspace(1.6, 2.2, 50)
ld_vals = np.linspace(6.0, 8.0, 50)
sfc_levels = [20, 27.5, 35]  # three SFC slices
# Two-line titles so the enlarged 22 pt titles cannot collide across panels.
sfc_labels = ['(a) SFC = 20\n(optimistic)',
              '(b) SFC = 27.5\n(baseline aero)',
              '(c) SFC = 35\n(conservative cargo)']

# Manual clabel positions per panel — placed away from the scenario markers and
# the panel edges. Only contours present in a given panel are listed.
manual_label_pos = {
    0: [(2.10, 7.85)],                 # 0.545 contour, upper-right corner of (a)
    1: [(1.68, 6.40)],                 # 1.00 contour, lower-left of (b), well away from star at (1.8, 7.0)
    2: [(2.10, 7.60), (1.65, 6.60)],   # 1.00 upper-right, 1.25 mid-left of (c); both well clear of v marker at (1.8, 6.0)
}

# constrained_layout keeps the shared colorbar outside all three panels.
fig, axes = plt.subplots(1, 3, figsize=(16, 6.2), sharey=True, constrained_layout=True)

# (marker style, colour, Mach, L/D, label, label-offset) keyed by SFC slice.
# Offset points are position-aware so top-edge / bottom-edge labels stay inside.
marker_specs = {20: ('^', '#1b7837', 1.8, 8.0, 'Optimistic', (10, -20)),
                27.5: ('*', 'black', 1.8, 7.0, 'Baseline aero', (10, 8)),
                35: ('v', '#b2182b', 1.8, 6.0, 'Conservative cargo', (0, 14))}

for idx, (sfc, label) in enumerate(zip(sfc_levels, sfc_labels)):
    ax = axes[idx]
    Mach, LD = np.meshgrid(mach_vals, ld_vals)
    Z = breguet_block_intensity(Mach, LD, sfc)

    levels = np.arange(0.4, 1.8, 0.1)
    # YlOrRd is colourblind-safe (sequential lightness ramp) and keeps the intuitive
    # "higher intensity -> darker/redder -> worse" reading.
    cf = ax.contourf(Mach, LD, Z, levels=levels, cmap='YlOrRd', extend='both')
    cs = ax.contour(Mach, LD, Z, levels=[0.545, 1.0, 1.25],
                    colors=['blue', 'black', 'red'], linewidths=[2, 1.5, 2],
                    linestyles=['--', '-', '--'])
    # White halo so the reference contours stay visible over the warm (red) fill
    cs.set_path_effects([pe.withStroke(linewidth=2.5, foreground='white', alpha=0.5)])
    # Explicit per-level labels with MANUAL positions so the rotated label text
    # never lands on top of the scenario marker / annotation in any panel.
    lbls = ax.clabel(cs, inline=True, fontsize=18,
                     fmt={0.545: '0.545', 1.0: '1.00', 1.25: '1.25'},
                     manual=manual_label_pos.get(idx))
    for t in lbls:
        t.set_bbox(dict(facecolor='white', edgecolor='none', alpha=0.8, pad=1.2))
    ax.set_ylim(5.9, 8.1)  # margin so the L/D=6.0 and L/D=8.0 markers are not clipped

    # Mark this panel's scenario point with a high-contrast white edge so it is
    # never camouflaged by the colormap and is not clipped at the axis boundary.
    if sfc in marker_specs:
        mk, mc, mx, my, mlbl, off = marker_specs[sfc]
        ax.plot(mx, my, marker=mk, color=mc, markersize=15, markeredgecolor='white',
                markeredgewidth=1.8, clip_on=False, zorder=6, linestyle='None')
        ax.annotate(mlbl, (mx, my), textcoords='offset points', xytext=off,
                    fontsize=18, fontweight='bold', color='black',
                    ha='center' if off[0] == 0 else 'left',
                    bbox=dict(facecolor='white', edgecolor='none', alpha=0.78, pad=2.0),
                    zorder=7)

    ax.set_xlabel('Mach number', fontsize=22)
    if idx == 0:
        ax.set_ylabel('L/D ratio', fontsize=22)
    ax.set_title(label, fontsize=22)
    ax.grid(alpha=0.2)

# Shared colorbar to the right of all panels (outside the data area)
cbar = fig.colorbar(cf, ax=axes, orientation='vertical', fraction=0.025, pad=0.02)
cbar.set_label('Block CO$_2$ intensity (kg/ton-km)', fontsize=22, labelpad=8)
cbar.ax.tick_params(labelsize=18)

# No figure-level title: the mission context (1,500 km, 1 ton) and the contour
# key (0.545 blue dashed / 1.00 black solid / 1.25 red dashed) live in the
# LaTeX caption, matching the convention used for Figures 1 and 2.
pdf_path = Path(__file__).with_name("parameter_heatmap.pdf")
png_path = Path(__file__).with_name("parameter_heatmap.png")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(png_path, dpi=600, bbox_inches="tight")
print(f"Saved (vector + raster): {pdf_path}  and  {png_path}")
