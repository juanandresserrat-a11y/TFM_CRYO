"""
figures.py  
Estándares tipográficos Nature / Biophysical Journal.
"""

from __future__ import annotations

import os
from collections import Counter
from typing import TYPE_CHECKING

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from scipy.ndimage import gaussian_filter
from scipy.stats import gaussian_kde

from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET

FIG_DIR = os.path.join(OUTPUT_DIR, "figuras")

_W1   = 3.46
_W15  = 5.20
_W2   = 7.09
_H_SQ = 3.10

# Paleta general
C = {
    "POPC":  "#2166ac", "POPE":  "#4393c3", "PlsPE": "#74add1",
    "POPS":  "#e08214", "CHOL":  "#525252", "SM":    "#1a9641",
    "GM1":   "#78c679",
    "lo":    "#b5451b", "ld":    "#2c6e8a",
    "head":  "#2ca02c", "tail":  "#e67e22",
}

# Paleta cálida para PIPs (destaca sobre fondos fríos)
PIP_CLR = {
    "PI":     "#d4ac0d",
    "PI3P":   "#f39c12",
    "PI4P":   "#e67e22",
    "PI5P":   "#e74c3c",
    "PI34P2": "#a04000",
    "PIP2":   "#c0392b",
    "PIP3":   "#7b241c",
}

ALL_SPECIES = [
    "POPC", "POPE", "PlsPE", "POPS", "CHOL", "SM", "GM1",
    "PI", "PI3P", "PI4P", "PI5P", "PI34P2", "PIP2", "PIP3",
]

PUB_RC = {
    "figure.facecolor":    "white",
    "axes.facecolor":      "white",
    "axes.edgecolor":      "#333333",
    "axes.linewidth":      0.75,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "axes.grid":           True,
    "grid.color":          "#e5e5e5",
    "grid.linewidth":      0.35,
    "grid.alpha":          0.7,
    "grid.linestyle":      ":",
    "xtick.direction":     "out",
    "ytick.direction":     "out",
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.minor.size":    1.5,
    "ytick.minor.size":    1.5,
    "font.family":         "sans-serif",
    "font.sans-serif":     ["Helvetica Neue", "Arial",
                            "Liberation Sans", "DejaVu Sans"],
    "font.size":           8,
    "axes.titlesize":      9,
    "axes.titleweight":    "bold",
    "axes.titlepad":       5,
    "axes.labelsize":      8,
    "axes.labelpad":       3,
    "xtick.labelsize":     7,
    "ytick.labelsize":     7,
    "legend.fontsize":     7,
    "legend.framealpha":   0.92,
    "legend.edgecolor":    "#cccccc",
    "legend.borderpad":    0.45,
    "legend.handlelength": 1.6,
    "legend.handletextpad":0.5,
    "lines.linewidth":     1.4,
    "lines.markersize":    4.5,
    "patch.linewidth":     0.6,
    "savefig.dpi":         300,
    "savefig.facecolor":   "white",
}


def _sim_fig_dir(seed):
    d = os.path.join(FIG_DIR, "simulacion%04d" % seed)
    os.makedirs(d, exist_ok=True)
    return d


# Auxiliares de datos

def _ed_profile(membrane, vol):
    """Perfil 1D de ED promediado en XY. Devuelve (z_nm, ed) mismo len."""
    nz = vol.shape[2]
    voxel_z_A = getattr(membrane, "Lz", nz * 9.0) / nz
    z_half_nm = nz * voxel_z_A / 10.0 / 2.0
    z_nm = np.linspace(-z_half_nm, z_half_nm, nz)
    ed   = vol.mean(axis=(0, 1))
    n    = min(len(z_nm), len(ed))
    return z_nm[:n], ed[:n]


def _composition(membrane):
    outer  = Counter(l.lipid_type.name for l in membrane.outer_leaflet)
    inner  = Counter(l.lipid_type.name for l in membrane.inner_leaflet)
    n_out  = max(len(membrane.outer_leaflet), 1)
    n_in   = max(len(membrane.inner_leaflet), 1)
    return ({s: outer.get(s, 0) / n_out for s in ALL_SPECIES},
            {s: inner.get(s, 0) / n_in  for s in ALL_SPECIES})


def _helfrich_spectrum(membrane):
    h    = membrane.curvature_map / 10.0
    bins = h.shape[0]
    L    = membrane.Lx / 10.0
    freq = np.fft.fftfreq(bins, d=L / bins)
    Fx, Fy = np.meshgrid(freq, freq)
    Q    = np.sqrt(Fx**2 + Fy**2)
    H_ft = np.abs(np.fft.fft2(h))**2 / bins**2
    q_fl, p_fl = Q.flatten(), H_ft.flatten()
    q_bins = np.logspace(np.log10(0.05), np.log10(1.5), 28)
    q_mid, p_mid = [], []
    for i in range(len(q_bins) - 1):
        m = (q_fl >= q_bins[i]) & (q_fl < q_bins[i + 1])
        if m.sum() > 2:
            q_mid.append(np.sqrt(q_bins[i] * q_bins[i + 1]))
            p_mid.append(np.median(p_fl[m]))
    return np.array(q_mid), np.array(p_mid)


def _thickness_dist(membrane):
    from analysis import thickness_map
    tm   = thickness_map(membrane, bins=55)
    vals = tm.flatten()
    return vals[vals > 1.0] * 10.0


def _raft_map_smooth(membrane, bins=160, sigma=2.2):
    """Mapa suavizado de fracción Lo."""
    Hr = np.zeros((bins, bins))
    Ht = np.zeros((bins, bins))
    for lip in membrane.outer_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        Ht[ix, iy] += 1
        if lip.in_raft:
            Hr[ix, iy] += 1
    mask = Ht >= 1
    with np.errstate(all="ignore"):
        raw = np.where(mask, Hr / np.where(mask, Ht, 1.0), 0.0)
    filled = np.where(mask, raw, 0.0)
    sm = gaussian_filter(filled.astype(float), sigma=sigma)
    cnt = gaussian_filter(mask.astype(float), sigma=sigma)
    # Sin umbral: rellenar TODO el espacio. Sin datos → sm≈0, cnt≈0 → resultado 0.0 (azul)
    with np.errstate(invalid="ignore"):
        result = np.clip(sm / np.maximum(cnt, 1e-9), 0, 1)
    return result

def _order_map_smooth(membrane, bins=160, sigma=1.2):
    S_sum = np.zeros((bins, bins))
    C_arr = np.zeros((bins, bins))
    for lip in membrane.outer_leaflet + membrane.inner_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        S_sum[ix, iy] += lip.order_param
        C_arr[ix, iy] += 1
    with np.errstate(all="ignore"):
        raw = np.where(C_arr > 0, S_sum / C_arr, np.nan)
    mask   = ~np.isnan(raw)
    filled = np.where(mask, raw, 0.0)
    sm     = gaussian_filter(filled.astype(float), sigma=sigma)
    cnt    = gaussian_filter(mask.astype(float),   sigma=sigma)
    with np.errstate(invalid="ignore"):
        result = np.where(cnt > 0.08, sm / cnt, np.nan)
    return result, mask


# Auxiliares gráficos

def _scalebar(ax, Lx, Ly, length_nm=10.0, color="#333333", lw=2.0,
              pos_frac=(0.05, 0.05)):
    px = Lx * pos_frac[0]
    py = Ly * pos_frac[1]
    ax.plot([px, px + length_nm], [py, py],
            color="white", lw=lw + 2.0, solid_capstyle="butt",
            alpha=0.55, zorder=9)
    ax.plot([px, px + length_nm], [py, py],
            color=color, lw=lw, solid_capstyle="butt", zorder=10)
    ax.text(px + length_nm / 2, py + Ly * 0.024,
            "%.0f nm" % length_nm,
            ha="center", fontsize=7, fontweight="bold", color=color, zorder=11)


def _styled_colorbar(fig, im, ax, label, ticks=None, ticklabels=None,
                     shrink=0.88, pad=0.02, aspect=20):
    cb = fig.colorbar(im, ax=ax, shrink=shrink, pad=pad, aspect=aspect)
    cb.set_label(label, fontsize=8, labelpad=4)
    if ticks is not None:
        cb.set_ticks(ticks)
    if ticklabels is not None:
        cb.set_ticklabels(ticklabels, fontsize=7)
    cb.ax.tick_params(labelsize=7, length=2.5, width=0.5)
    cb.outline.set_linewidth(0.5)
    return cb


def _ref_footer(fig, text):
    fig.text(0.5, 0.005, text, ha="center", fontsize=6.5,
             color="#888888", style="italic", transform=fig.transFigure)


def _info_box(ax, text, loc="upper left", fs=7.5):
    corners = {
        "upper left":  (0.025, 0.972, "left",  "top"),
        "upper right": (0.972, 0.972, "right", "top"),
        "lower left":  (0.025, 0.038, "left",  "bottom"),
        "lower right": (0.972, 0.038, "right", "bottom"),
    }
    x, y, ha, va = corners[loc]
    ax.text(x, y, text, transform=ax.transAxes,
            ha=ha, va=va, fontsize=fs, linespacing=1.45, color="#222222",
            bbox=dict(boxstyle="round,pad=0.38",
                      fc="white", ec="#bbbbbb", alpha=0.92, lw=0.6),
            zorder=15)

# Fig 1: Perfil 1D de densidad electrónica

def plot_fig1_perfil_ED(membrane, vol, stats, dpi=300):
    seed = membrane.seed
    g = membrane.geometry
    z_nm, ed_prof = _ed_profile(membrane, vol)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")

        ax.plot(ed_prof, z_nm, color="#2c3e50", lw=2.2, zorder=5,
                label="ED media del parche")
        ax.fill_betweenx(z_nm, 0.334, ed_prof,
                         where=ed_prof > 0.334, alpha=0.35,
                         color=C["head"], label="Cabezas polares")
        ax.fill_betweenx(z_nm, 0.334, ed_prof,
                         where=ed_prof < 0.334, alpha=0.25,
                         color=C["tail"], label="Núcleo hidrofóbico")
        ax.axvline(0.334, color="#888888", lw=1.0, ls=":",
                   label="Agua (0.334 $e \\cdot \\AA^{-3}$)")
        ax.axhline(g.z_outer / 10, color=C["head"], lw=1.0, ls="--", alpha=0.8,
                   label="Cabezas ext. z=%.2f nm" % (g.z_outer / 10))
        ax.axhline(g.z_inner / 10, color="#e67e22", lw=1.0, ls="--", alpha=0.8,
                   label="Cabezas int. z=%.2f nm" % (g.z_inner / 10))
        ax.axhline(0, color="#aaaaaa", lw=0.7, ls=":", alpha=0.6)

        ax.annotate("", xy=(0.51, g.z_outer / 10), xytext=(0.51, g.z_inner / 10),
                    arrowprops=dict(arrowstyle="<->", color="#555555", lw=1.2))
        ax.text(0.52, (g.z_outer + g.z_inner) / 20,
                "D$_{HH}$=%.1f Å" % g.total_thick,
                fontsize=9, color="#555555", va="center")

        ax.set_xlabel("Densidad electrónica ($e \\cdot \\AA^{-3}$)")
        ax.set_ylabel("Posición axial Z (nm)")
        ax.set_title("Perfil 1D de densidad electrónica\n"
                     "Simulación %d | Patrón dark-bright-dark cryo-ET" % seed)
        ax.legend(fontsize=7.5, loc="upper left", framealpha=0.9)
        ax.set_xlim(0.26, 0.53)
        ax.grid(True, alpha=0.3)

        fig.text(0.5, 0.01,
                 "Nagle & Tristram-Nagle BBA 2000 | Kučerka et al. Biophys. J. 2008",
                 ha="center", fontsize=7, color="#888888", style="italic")

        fig.subplots_adjust(left=0.14, right=0.96, top=0.90, bottom=0.08)
        path = os.path.join(_sim_fig_dir(seed),
                            "fig1_sim%04d_perfil_ED.png" % seed)
        fig.savefig(path, dpi=dpi, facecolor="white")
        plt.close(fig)
        print("  -> figuras/simulacion%04d/fig1_sim%04d_perfil_ED.png" % (seed, seed))
    return path

# Fig 2: Composición lipídica

def plot_fig2_composicion(membrane, dpi=300):
    seed = membrane.seed
    out_frac, in_frac = _composition(membrane)
    pip_spc = Counter(l.lipid_type.name for l in membrane.inner_leaflet
                      if l.is_pip)
    n_pips = sum(pip_spc.values())

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(_W2 + 0.2, _H_SQ + 0.9))
        gs  = gridspec.GridSpec(1, 2, figure=fig,
                                left=0.07, right=0.97,
                                top=0.88, bottom=0.11,
                                width_ratios=[1.8, 1.0], wspace=0.32)

        # Panel A: barras horizontales
        ax = fig.add_subplot(gs[0, 0])
        sp_plot = [s for s in ALL_SPECIES
                   if out_frac.get(s, 0) > 0.004 or in_frac.get(s, 0) > 0.004]
        y   = np.arange(len(sp_plot))
        bh  = 0.37
        v_o = [out_frac.get(s, 0) * 100 for s in sp_plot]
        v_i = [in_frac.get(s, 0)  * 100 for s in sp_plot]
        col_sp = [PIP_CLR.get(s, C.get(s, "#999999")) for s in sp_plot]

        bars_o = ax.barh(y + bh / 2, v_o, bh, color=col_sp, alpha=0.90,
                         edgecolor="white", lw=0.5, label="Monocapa externa")
        bars_i = ax.barh(y - bh / 2, v_i, bh, color=col_sp, alpha=0.48,
                         edgecolor="white", lw=0.5, hatch="///",
                         label="Monocapa interna")

        for bar, val in zip(bars_o, v_o):
            if val > 1.2:
                ax.text(bar.get_width() + 0.30,
                        bar.get_y() + bar.get_height() / 2,
                        "%.1f%%" % val, va="center", fontsize=6.8)
        for bar, val in zip(bars_i, v_i):
            if val > 1.2:
                ax.text(bar.get_width() + 0.30,
                        bar.get_y() + bar.get_height() / 2,
                        "%.1f%%" % val, va="center", fontsize=6.8, color="#555555")

        ax.set_yticks(y)
        ax.set_yticklabels(sp_plot, fontsize=8)
        ax.set_xlabel("Fracción molar (%)")
        ax.set_title("Composición por monocapa", loc="left", pad=4)
        ax.legend(fontsize=7, loc="lower right")
        x_max = max(max(v_o, default=[1.0]), max(v_i, default=[1.0]))
        ax.set_xlim(0, x_max * 1.22)
        ax.grid(True, axis="x", alpha=0.4)

        n_glyc = len([s for s in sp_plot if s in ["POPC","POPE","PlsPE","POPS"]])
        n_raft = len([s for s in sp_plot if s in ["CHOL","SM","GM1"]])
        for lim in [n_glyc - 0.5, n_glyc + n_raft - 0.5]:
            if 0 < lim < len(sp_plot):
                ax.axhline(lim, color="#bbbbbb", lw=0.8, ls="--", alpha=0.6)

        # Panel B: pie de PIPs + tabla
        gs_r  = gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[0, 1],
            height_ratios=[1.6, 1.0], hspace=0.08)
        ax2    = fig.add_subplot(gs_r[0])
        ax_tbl = fig.add_subplot(gs_r[1])

        if pip_spc:
            pip_lbls = list(pip_spc.keys())
            pip_vals = [pip_spc[s] for s in pip_lbls]
            pip_cols = [PIP_CLR.get(s, "#999") for s in pip_lbls]

            wedges, _, autotexts = ax2.pie(
                pip_vals, labels=None, colors=pip_cols,
                autopct="%1.1f%%", startangle=90, pctdistance=0.72,
                wedgeprops={"edgecolor": "white", "linewidth": 1.4},
                textprops={"fontsize": 7})
            for at in autotexts:
                at.set_fontsize(7)
                at.set_color("white")
                at.set_fontweight("bold")
            ax2.legend(wedges, pip_lbls, title="Especie PIP",
                       title_fontsize=7, fontsize=7, loc="center left",
                       bbox_to_anchor=(0.92, 0.5),
                       frameon=True, framealpha=0.92, edgecolor="#cccccc")
            ax2.set_title("Fosfoinosítidos (PIPs)\nmonocapa interna",
                          loc="left", pad=5)

            ax_tbl.axis("off")
            try:
                from lipid_types import LIPID_TYPES
                col_lbls_t = ["Especie", "N", "Carga", "% int."]
                rows_tbl = []
                for sp in pip_lbls:
                    lt = LIPID_TYPES.get(sp)
                    if lt:
                        rows_tbl.append([
                            sp, str(pip_spc[sp]), "%+d" % lt.charge,
                            "%.1f%%" % (100 * pip_spc[sp] /
                                        max(len(membrane.inner_leaflet), 1))])
                if rows_tbl:
                    tbl = ax_tbl.table(cellText=rows_tbl, colLabels=col_lbls_t,
                                       loc="center", cellLoc="center")
                    tbl.auto_set_font_size(False)
                    tbl.set_fontsize(7.5)
                    tbl.scale(1, 1.45)
                    for (r, _c), cell in tbl.get_celld().items():
                        cell.set_edgecolor("#d0d0d0")
                        cell.set_linewidth(0.4)
                        if r == 0:
                            cell.set_facecolor("#dadada")
                            cell.set_text_props(fontweight="bold")
                        else:
                            cell.set_facecolor("#f4f4f4" if r%2==0 else "white")
            except ImportError:
                pass
        else:
            ax2.text(0.5, 0.6, "Sin PIPs", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=10, color="#888888")
            ax2.axis("off")
            ax_tbl.axis("off")

        path = os.path.join(_sim_fig_dir(seed),
                            "fig2_sim%04d_composicion.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 3: Espectro Helfrich

def plot_fig3_helfrich(membrane, dpi=300):
    seed  = membrane.seed
    kc    = membrane.bending_modulus
    sigma = membrane.surface_tension
    q_nm, p_helf = _helfrich_spectrum(membrane)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W1 + 0.35, _H_SQ + 0.4))

        if len(q_nm) > 4:
            ax.loglog(q_nm, p_helf, "o", color="#2c6e8a",
                      ms=4.5, alpha=0.80, label="Datos simulación", zorder=5)
            mask_fit = q_nm > 0.12
            if mask_fit.sum() > 3:
                coef  = np.polyfit(np.log(q_nm[mask_fit]),
                                   np.log(p_helf[mask_fit]), 1)
                q_fit = np.linspace(0.08, 1.6, 50)
                y_fit = np.exp(np.polyval(coef, np.log(q_fit)))
                ax.loglog(q_fit, y_fit, "--", color="#b5451b", lw=1.8,
                          label="Ajuste: $q^{%.2f}$\n(Helfrich: $q^{-4}$)" % coef[0])
            if len(p_helf) > 2:
                scale = np.exp(np.log(p_helf[1]) + 4.0 * np.log(q_nm[1]))
                q_ref = np.linspace(0.08, 1.6, 50)
                ax.loglog(q_ref, scale * q_ref**(-4.0), ":",
                          color="#888888", lw=1.2, alpha=0.75,
                          label="Referencia $q^{-4}$")

        ax.axvline(0.2, color="#aaaaaa", lw=0.8, ls="--", alpha=0.65)
        ax.set_xlabel("Vector de onda $q$ (nm$^{-1}$)")
        ax.set_ylabel(r"$\langle|\hat{h}_q|^2\rangle$ (nm$^2$)")
        ax.legend(loc="upper right", fontsize=6.8)
        _info_box(ax,
            "$k_c$ = %.1f $k_BT$·nm$^2$\n"
            "$\\sigma$ = %.4f $k_BT$·nm$^{-2}$\n"
            "RMS $h$ = %.2f Å" % (kc, sigma, membrane.curvature_map.std()),
            loc="lower left")
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(_sim_fig_dir(seed),
                            "fig3_sim%04d_helfrich.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 4: Distribución de grosor D_PP

def plot_fig4_grosor(membrane, stats, dpi=300):
    seed    = membrane.seed
    thick_A = _thickness_dist(membrane)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W1 + 0.35, _H_SQ + 0.4))

        if len(thick_A) > 10:
            kde_x = np.linspace(thick_A.min() - 3, thick_A.max() + 3, 600)
            kde   = gaussian_kde(thick_A, bw_method=0.15)
            y_kde = kde(kde_x)
            ax.plot(kde_x, y_kde, color="#2c6e8a", lw=1.8)
            ax.fill_between(kde_x, 0, y_kde, alpha=0.18, color="#2c6e8a")

            from scipy.signal import find_peaks
            peaks, _ = find_peaks(y_kde,
                                  distance=int(len(kde_x) / 12),
                                  prominence=y_kde.max() * 0.05)
            colors_pk = [C["ld"], C["lo"], "#7f4f9b"]
            labels_pk = ["Ld (fluido)", "Lo (raft)", "Pico adicional"]
            for i, pk in enumerate(peaks[:3]):
                col = colors_pk[i % len(colors_pk)]
                ax.axvline(kde_x[pk], color=col, lw=1.4, ls="--", alpha=0.85,
                           label="%s: %.1f Å" % (labels_pk[i], kde_x[pk]))
                ax.annotate("%.1f Å" % kde_x[pk],
                            xy=(kde_x[pk], y_kde[pk]),
                            xytext=(kde_x[pk] + 0.8, y_kde[pk] * 1.08),
                            fontsize=7.5, color=col,
                            arrowprops=dict(arrowstyle="->", color=col, lw=0.7))
            if len(peaks) >= 2:
                delta = abs(kde_x[peaks[1]] - kde_x[peaks[0]])
                _info_box(ax,
                    "$\\Delta D_{Lo-Ld}$ = %.1f Å\n(exp: 4–8 Å)" % delta,
                    loc="upper right")

        ax.set_xlabel("Grosor $D_{PP}$ cabeza–cabeza (Å)")
        ax.set_ylabel("Densidad de probabilidad (Å$^{-1}$)")
        ax.legend(loc="upper left", fontsize=7)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(_sim_fig_dir(seed),
                            "fig4_sim%04d_grosor.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 5: Mapa de fracción Lo

def plot_fig5_mapa_raft(membrane, dpi=300):
    seed = membrane.seed
    Lx = membrane.Lx / 10.0
    Ly = membrane.Ly / 10.0
    ext = [0, Lx, 0, Ly]

    raft_map = _raft_map_smooth(membrane, bins=180, sigma=2.2)
    n_rafts = (len(getattr(membrane, "rafts_outer", [])) +
               len(getattr(membrane, "rafts_inner", [])))
    lo_frac = float(raft_map.mean())

    cmap_raft = mcolors.LinearSegmentedColormap.from_list(
        "raft_harm",
        ["#1d4e7a",   # azul acero oscuro  — Ld puro
         "#4a85b0",   # azul medio
         "#f0ede4",   # crema (transición)
         "#c9693a",   # siena cálida
         "#7a2e00"],  # terracota oscuro   — Lo puro
        N=256)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W15, _W15 * 0.95))
        ax.set_facecolor("#ebebeb")

        # Imshow directo: sin NaN, todo el espacio cubierto con color válido
        im = ax.imshow(raft_map.T, origin="lower", extent=ext,
                       cmap=cmap_raft, vmin=0, vmax=1,
                       aspect="equal", interpolation="nearest", zorder=2)

        cb = _styled_colorbar(fig, im, ax,
            label="Fracción Lo",
            ticks=[0, 0.25, 0.5, 0.75, 1.0],
            ticklabels=["0\n(Ld)", "0.25", "0.50", "0.75", "1.0\n(Lo)"],
            shrink=0.88, pad=0.02)

        # Contorno Lo/Ld
        xg = np.linspace(0, Lx, raft_map.shape[0])
        yg = np.linspace(0, Ly, raft_map.shape[1])
        ax.contour(xg, yg, raft_map.T, levels=[0.5],
                   colors=["#1a1a1a"], linewidths=1.0,
                   linestyles="--", alpha=0.80, zorder=3)

        _scalebar(ax, Lx, Ly, color="#1a1a1a")
        ax.set_xlabel("$x$ (nm)")
        ax.set_ylabel("$y$ (nm)")
        ax.set_title("Fracción Lo — monocapa externa\n– – –  límite Lo/Ld (0.50)",
                     pad=6)
        _info_box(ax,
            "%d dominio(s) Lo\nFracc. Lo = %.1f%%" % (n_rafts, lo_frac * 100),
            loc="upper left")

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(_sim_fig_dir(seed),
                            "fig5_sim%04d_mapa_raft.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 6: Mapa de Parametro de Orden

def plot_fig6_mapa_order(membrane, dpi=300):

    seed = membrane.seed
    Lx   = membrane.Lx / 10.0
    Ly   = membrane.Ly / 10.0
    ext  = [0, Lx, 0, Ly]

    order_map, _ = _order_map_smooth(membrane, bins=180, sigma=1.2)
    all_lips = membrane.outer_leaflet + membrane.inner_leaflet
    gel_s    = [l.order_param for l in all_lips if l.lipid_type.phase == "gel"]
    fluid_s  = [l.order_param for l in all_lips if l.lipid_type.phase == "fluid"]
    s_lo = np.mean(gel_s)   if gel_s   else 0.0
    s_ld = np.mean(fluid_s) if fluid_s else 0.0

    cmap_ord = mcolors.LinearSegmentedColormap.from_list(
        "order_s",
        ["#2c6e8a", "#a8cfe0", "#f0f0ea", "#5a9e6e", "#1b5e3b"],
        N=256)
    cmap_ord.set_bad(color="#f0f0f0")

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W15, _W15 * 0.95))
        ax.set_facecolor("#f0f0f0")

        im = ax.imshow(np.ma.masked_invalid(order_map).T,
                       origin="lower", extent=ext,
                       cmap=cmap_ord, vmin=0.55, vmax=0.95,
                       aspect="equal", interpolation="bilinear", zorder=2)

        _styled_colorbar(fig, im, ax,
            label="Parámetro de orden $S_{CH}$",
            ticks=[0.55, 0.65, 0.75, 0.85, 0.95],
            ticklabels=["0.55\n(Ld)", "0.65", "0.75", "0.85", "0.95\n(Lo)"],
            shrink=0.88, pad=0.02)

        xg = np.linspace(0, Lx, order_map.shape[0])
        yg = np.linspace(0, Ly, order_map.shape[1])
        om = np.where(np.isnan(order_map), 0.65, order_map)
        ax.contour(xg, yg, om.T, levels=[0.75],
                   colors=["#1a1a1a"], linewidths=0.9,
                   linestyles="--", alpha=0.75, zorder=3)

        _scalebar(ax, Lx, Ly, color="#1a1a1a")
        ax.set_xlabel("$x$ (nm)")
        ax.set_ylabel("$y$ (nm)")
        ax.set_title(
            "Parámetro de orden $S_{CH}$ — bicapa completa\n"
            "– – –  umbral Lo/Ld  ($S_{CH}$ = 0.75)", pad=6)
        _info_box(ax,
            "$S^{gel}_{CH}$ = %.3f (Lo)\n"
            "$S^{fluid}_{CH}$ = %.3f (Ld)" % (s_lo, s_ld),
            loc="upper left")

        # Inset histograma KDE (solo curvas, sin barras)
        axins = ax.inset_axes([0.72, 0.03, 0.26, 0.28])
        axins.set_facecolor("white")
        axins.patch.set_alpha(0.92)
        axins.spines["top"].set_visible(False)
        axins.spines["right"].set_visible(False)
        axins.spines["left"].set_linewidth(0.5)
        axins.spines["bottom"].set_linewidth(0.5)

        kde_x = np.linspace(0.48, 1.00, 250)
        if gel_s:
            y_kde = gaussian_kde(gel_s, bw_method=0.10)(kde_x)
            axins.fill_between(kde_x, 0, y_kde,
                               color="#1b5e3b", alpha=0.30)
            axins.plot(kde_x, y_kde, color="#1b5e3b", lw=1.3, label="gel/Lo")
        if fluid_s:
            y_kde = gaussian_kde(fluid_s, bw_method=0.10)(kde_x)
            axins.fill_between(kde_x, 0, y_kde,
                               color="#2c6e8a", alpha=0.30)
            axins.plot(kde_x, y_kde, color="#2c6e8a", lw=1.3, label="fluido/Ld")
        axins.axvline(0.75, color="#555555", lw=0.7, ls=":", alpha=0.8)
        axins.set_xlabel("$S_{CH}$", fontsize=6.5)
        axins.set_ylabel("PDF", fontsize=6.5)
        axins.tick_params(labelsize=5.5, length=2)
        axins.legend(fontsize=5.5, loc="upper left", framealpha=0.9,
                     edgecolor="#cccccc", borderpad=0.3)

        fig.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(_sim_fig_dir(seed),
                            "fig6_sim%04d_mapa_order.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 7: Distribución radial de PIPs por especie

def plot_fig7_pip_radial(membrane, dpi=300):
    """
    Barras apiladas por especie PIP vs radio desde el centroide colectivo.
    Sin curva KDE superpuesta — lectura directa y limpia.
    """
    seed    = membrane.seed
    pip_spc = Counter(l.lipid_type.name for l in membrane.inner_leaflet
                      if l.is_pip)
    n_pips  = sum(pip_spc.values())

    pip_by_sp: dict[str, list] = {}
    cx_all, cy_all = [], []
    Lx = membrane.Lx / 10
    Ly = membrane.Ly / 10
    for lip in membrane.inner_leaflet:
        if lip.is_pip:
            sp = lip.lipid_type.name
            x  = lip.head_pos[0] / 10
            y  = lip.head_pos[1] / 10
            pip_by_sp.setdefault(sp, []).append((x, y))
            cx_all.append(x)
            cy_all.append(y)

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W1 + 0.60, _H_SQ + 0.5))

        if n_pips > 5 and cx_all:
            cx_g = np.mean(cx_all)
            cy_g = np.mean(cy_all)
            r_max_bin = min(Lx, Ly) * 0.46
            r_bins = np.linspace(0, r_max_bin, 14)
            r_mid  = 0.5 * (r_bins[:-1] + r_bins[1:])
            area   = np.pi * (r_bins[1:]**2 - r_bins[:-1]**2)
            bar_w  = (r_mid[1] - r_mid[0]) * 0.78 if len(r_mid) > 1 else 1.0

            bottom = np.zeros(len(r_mid))
            for sp, pos_list in sorted(pip_by_sp.items(),
                                       key=lambda kv: -len(kv[1])):
                r_sp = np.array([np.sqrt((x - cx_g)**2 + (y - cy_g)**2)
                                 for x, y in pos_list])
                dens, _ = np.histogram(r_sp, bins=r_bins)
                dens_nm2 = dens / np.maximum(area, 1e-9)
                col = PIP_CLR.get(sp, "#999999")
                ax.bar(r_mid, dens_nm2, width=bar_w, bottom=bottom,
                       color=col, alpha=0.88, edgecolor="white", linewidth=0.4,
                       label="%s (%d)" % (sp, len(pos_list)))
                bottom += dens_nm2

            ax.set_xlabel("Radio desde el centroide (nm)")
            ax.set_ylabel("Densidad superficial (nm$^{-2}$)")
            ax.legend(loc="upper right", fontsize=7, framealpha=0.92)
            ax.set_xlim(0, r_max_bin * 1.05)
            ax.set_ylim(bottom=0)
        else:
            ax.text(0.5, 0.5, "Sin PIPs\nen esta simulación",
                    ha="center", va="center", transform=ax.transAxes,
                    fontsize=10, color="#888888")

        ax.set_title("Distribución radial de PIPs — monocapa interna", pad=5)
        fig.tight_layout(rect=[0, 0.03, 1, 1])
        path = os.path.join(_sim_fig_dir(seed),
                            "fig7_sim%04d_pip_radial.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 8: Tabla de parámetros físicos

def plot_fig8_parametros(membrane, stats, dpi=300):
    seed  = membrane.seed
    g     = membrane.geometry
    kc    = membrane.bending_modulus
    sigma = membrane.surface_tension
    all_lips = membrane.outer_leaflet + membrane.inner_leaflet
    gel_s    = [l.order_param for l in all_lips if l.lipid_type.phase == "gel"]
    fluid_s  = [l.order_param for l in all_lips if l.lipid_type.phase == "fluid"]
    n_pips   = sum(1 for l in membrane.inner_leaflet if l.is_pip)

    ROWS = [
        ("MECÁNICA",      "Módulo de flexión $k_c$",
         "%.1f" % kc,       "18 – 45",      "$k_BT$·nm$^2$",     "Chakraborty 2020"),
        ("MECÁNICA",      "Tensión superficial $\\sigma$",
         "%.4f" % sigma,    "0 – 0.05",     "$k_BT$·nm$^{-2}$",  "Pinigin 2022"),
        ("MECÁNICA",      "RMS Helfrich",
         "%.2f" % membrane.curvature_map.std(), "2 – 5", "Å",    "Sharma 2023"),
        ("GEOMETRÍA",     "Grosor total $D_{HH}$",
         "%.1f" % g.total_thick, "38 – 54", "Å",                  "Kučerka 2011"),
        ("GEOMETRÍA",     "Grosor hidrofóbico $D_C$",
         "%.1f" % g.hydro_thick, "26 – 34", "Å",                  "Kučerka 2011"),
        ("GEOMETRÍA",     "$z$ cabezas ext.",
         "%.2f" % (g.z_outer / 10), "—",    "nm",                 "—"),
        ("GEOMETRÍA",     "$z$ cabezas int.",
         "%.2f" % (g.z_inner / 10), "—",    "nm",                 "—"),
        ("ORDEN",         "$S_{CH}$ gel (Lo)",
         "%.3f" % (np.mean(gel_s)   if gel_s   else 0),
         "0.80 – 0.95",  "—",                                      "Piggot 2017"),
        ("ORDEN",         "$S_{CH}$ fluido (Ld)",
         "%.3f" % (np.mean(fluid_s) if fluid_s else 0),
         "0.55 – 0.75",  "—",                                      "Piggot 2017"),
        ("DENS. ELECTR.", "ED cabezas polares",
         "%.4f" % stats["ed_head_mean"],    "0.44 – 0.50", "e·Å$^{-3}$", "Nagle 2000"),
        ("DENS. ELECTR.", "ED núcleo Lo",
         "%.4f" % stats["ed_tail_Lo"],      "0.285 – 0.310","e·Å$^{-3}$","Nagle 2000"),
        ("DENS. ELECTR.", "ED núcleo Ld",
         "%.4f" % stats["ed_tail_Ld"],      "0.270 – 0.298","e·Å$^{-3}$","Nagle 2000"),
        ("DENS. ELECTR.", "Contraste Lo – Ld",
         "%.4f" % stats["contrast_Lo_Ld"],  "0.008 – 0.020","e·Å$^{-3}$","Chaisson 2025"),
        ("COMPOSICIÓN",   "Lípidos totales",
         "%d" % len(all_lips),              "—",           "—",    "—"),
        ("COMPOSICIÓN",   "PIPs monocapa int.",
         "%d (%.1f%%)" % (n_pips, 100 * n_pips / max(len(membrane.inner_leaflet),1)),
         "5 – 15%",      "—",                                      "Di Paolo 2006"),
        ("COMPOSICIÓN",   "Proteínas (objetos)",
         "%d" % stats["n_protein_objects"],
         "~30–50% área", "—",                                      "Singer 1972"),
        ("COMPOSICIÓN",   "Tamaño del parche",
         "%.0f × %.0f" % (membrane.Lx / 10, membrane.Ly / 10),
         "—",            "nm × nm",                                "—"),
    ]

    SEC_COLORS = {
        "MECÁNICA":      "#ddeeff",
        "GEOMETRÍA":     "#dff2e0",
        "ORDEN":         "#fdf2e0",
        "DENS. ELECTR.": "#fce8e6",
        "COMPOSICIÓN":   "#ece5f5",
    }

    with plt.rc_context(PUB_RC):
        fig, ax = plt.subplots(figsize=(_W2 + 0.5, 10.0))
        ax.axis("off")
        fig.suptitle(
            "Parámetros físicos de la membrana — Simulación %d\n"
            "Valores simulados frente a rangos experimentales publicados" % seed,
            fontsize=10, fontweight="bold", y=0.986)

        col_labels = ["Sección", "Parámetro",
                      "Simulación", "Rango exp.", "Unidad", "Referencia"]
        col_x = [0.00, 0.14, 0.44, 0.60, 0.76, 0.86]
        col_w = [0.14, 0.30, 0.16, 0.16, 0.10, 0.14]

        n_rows = len(ROWS)
        top_y  = 0.93
        row_h  = top_y / (n_rows + 1)
        hdr_y  = top_y + row_h * 0.28

        for ci, (lbl, cx) in enumerate(zip(col_labels, col_x)):
            rect = mpatches.FancyBboxPatch(
                (cx, hdr_y - row_h * 0.44), col_w[ci], row_h * 0.88,
                boxstyle="square,pad=0", linewidth=0,
                facecolor="#2c3e50", transform=ax.transAxes)
            ax.add_patch(rect)
            ax.text(cx + col_w[ci] / 2, hdr_y, lbl,
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=7.5, fontweight="bold", color="white")

        prev_sec = None
        for ri, (sec, param, val, rng, unit, ref) in enumerate(ROWS):
            y_pos = top_y - (ri + 1) * row_h
            cells = [sec if sec != prev_sec else "",
                     param, val, rng, unit, ref]
            for ci, (cell, cx, cw) in enumerate(zip(cells, col_x, col_w)):
                bg = SEC_COLORS.get(sec, "#f5f5f5") if ci == 0 else \
                     ("#f6f6f6" if ri % 2 == 0 else "white")
                rect = mpatches.FancyBboxPatch(
                    (cx, y_pos - row_h * 0.44), cw, row_h * 0.88,
                    boxstyle="square,pad=0", linewidth=0.3,
                    edgecolor="#d5d5d5", facecolor=bg,
                    transform=ax.transAxes)
                ax.add_patch(rect)
                fw = "bold" if ci == 0 and cell else "normal"
                fc = "#1a3a5c" if ci == 0 else (
                     "#8b1a1a" if ci == 2 else "#333333")
                ax.text(cx + 0.005, y_pos, cell,
                        transform=ax.transAxes,
                        ha="left", va="center",
                        fontsize=7.5 if ci <= 1 else 7.0,
                        fontweight=fw, color=fc)
            prev_sec = sec

        prev_s2 = None
        for ri, (sec, *_) in enumerate(ROWS):
            if sec != prev_s2 and ri > 0:
                y_line = top_y - ri * row_h + row_h * 0.44
                ax.plot([0, 1], [y_line, y_line],
                        color="#2c3e50", lw=0.75, alpha=0.40,
                        transform=ax.transAxes)
            prev_s2 = sec

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        handles = [mpatches.Patch(fc=v, ec="#cccccc", lw=0.5, label=k)
                   for k, v in SEC_COLORS.items()]
        ax.legend(handles=handles, loc="upper center",
                  bbox_to_anchor=(0.5, 1.005), ncol=5,
                  fontsize=7, frameon=True, edgecolor="#cccccc")
        fig.subplots_adjust(left=0.04, right=0.96, top=0.94, bottom=0.03)
        path = os.path.join(_sim_fig_dir(seed),
                            "fig8_sim%04d_parametros.png" % seed)
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        print("  -> %s" % path)
    return path

# Fig 9: Mapa integrado PIPs + balsas — sin solapamiento

def plot_fig9_mapa_pips_balsas(membrane, dpi=300):
    seed = membrane.seed
    Lx = membrane.Lx / 10.0
    Ly = membrane.Ly / 10.0
    ext = [0, Lx, 0, Ly]

    raft_map = _raft_map_smooth(membrane, bins=180, sigma=2.2)
    n_rafts = (len(getattr(membrane, "rafts_outer", [])) +
               len(getattr(membrane, "rafts_inner", [])))
    lo_frac = float(raft_map.mean())

    # PIPs por especie (monocapa interna)
    pip_by_sp = {}
    for lip in membrane.inner_leaflet:
        if lip.is_pip:
            sp = lip.lipid_type.name
            if sp not in pip_by_sp:
                pip_by_sp[sp] = ([], [])
            pip_by_sp[sp][0].append(lip.head_pos[0] / 10)
            pip_by_sp[sp][1].append(lip.head_pos[1] / 10)
    n_pips = sum(len(v[0]) for v in pip_by_sp.values())

    cmap_bg = mcolors.LinearSegmentedColormap.from_list(
        "bg_neutral",
        ["#bdd7e7",
         "#dce8e4",
         "#ece8de",   
         "#ddd0b8",   
         "#c4b090"],
        N=256)

    with plt.rc_context(PUB_RC):
        # Layout: mapa (1.0) | gap estrecho | colorbar binaria (0.028) |
        #         gap ancho  | leyenda PIPs (0.26)
        # La colorbar y la leyenda PIPs están en ejes distintos para poder
        # controlar el gap entre ambas sin que matplotlib los junte.
        fig = plt.figure(figsize=(_W2 + 1.8, _W2 * 0.82))
        gs = gridspec.GridSpec(1, 3, figure=fig,
                               left=0.07, right=0.97,
                               top=0.91, bottom=0.10,
                               width_ratios=[1.0, 0.030, 0.28], wspace=0.10)
        ax     = fig.add_subplot(gs[0, 0])
        ax_cb  = fig.add_subplot(gs[0, 1])
        ax_leg = fig.add_subplot(gs[0, 2])
        ax_leg.axis("off")
        ax.set_facecolor("#f2f2f2")

        # Colorbar binaria: solo Ld (0) y Lo (1), sin gradiente visible.
        # Se usa ListedColormap de 2 colores directamente sobre el mapa.
        # Colormap de 3 paradas: Ld (azul grisáceo) → transición (gris cálido
        # neutral) → Lo (beige tostado). Los extremos están muy saturados
        # para que el lector identifique fases de un vistazo, y la franja
        # central suaviza la frontera sin hacer el mapa completamente binario.
        cmap_lo_ld = mcolors.LinearSegmentedColormap.from_list(
            "lo_ld_pro",
            [(0.00, "#a8c8e0"),   # Ld — azul medio, saturado
             (0.35, "#ccd8d0"),   # transición hacia beige neutro
             (0.50, "#d6cfc0"),   # punto medio claramente diferenciable
             (0.65, "#c8b898"),   # transición hacia Lo
             (1.00, "#b89a68")],  # Lo — beige-ocre, saturado
            N=256)
        im = ax.imshow(raft_map.T, origin="lower", extent=ext,
                       cmap=cmap_lo_ld, vmin=0, vmax=1,
                       aspect="equal", interpolation="bilinear", zorder=2)

        # Colorbar: ticks en los extremos + punto medio, sin marcas intrusivas
        cb = fig.colorbar(im, cax=ax_cb)
        cb.set_ticks([0.0, 0.5, 1.0])
        cb.set_ticklabels(["Ld", "0.5", "Lo"], fontsize=7.5)
        cb.ax.tick_params(length=3, width=0.6, direction="out")
        cb.outline.set_linewidth(0.6)

        # Contorno Lo/Ld sobre el mapa binario
        xg = np.linspace(0, Lx, raft_map.shape[0])
        yg = np.linspace(0, Ly, raft_map.shape[1])
        ax.contour(xg, yg, raft_map.T, levels=[0.5],
                   colors=["#2c2c2c"], linewidths=1.1,
                   linestyles="--", alpha=0.80, zorder=4)

        # PIPs individuales
        pip_handles = []
        for sp, (xs, ys) in sorted(pip_by_sp.items(),
                                   key=lambda kv: -len(kv[1][0])):
            col = PIP_CLR.get(sp, "#888888")
            ax.scatter(xs, ys, s=14, c=col, marker="o",
                       edgecolors="black", linewidths=0.35,
                       alpha=0.90, zorder=6)
            pip_handles.append(
                mpatches.Patch(fc=col, ec="black", lw=0.4,
                               label="%s  (n = %d)" % (sp, len(xs))))

        # Leyenda PIPs en eje dedicado, bien separada de la colorbar
        if pip_handles:
            leg = ax_leg.legend(
                handles=pip_handles,
                title="PIPs — monocapa interna",
                title_fontsize=7.5,
                fontsize=7,
                loc="center left",
                frameon=True, framealpha=0.95,
                edgecolor="#cccccc", borderpad=0.7)
            leg.get_frame().set_linewidth(0.6)

        _scalebar(ax, Lx, Ly, color="#333333", lw=2.0, pos_frac=(0.04, 0.04))
        ax.set_xlabel("$x$ (nm)")
        ax.set_ylabel("$y$ (nm)")
        ax.set_title(
            "Distribución espacial de PIPs sobre dominios Lo/Ld\n"
            "– – –  límite Lo/Ld  |  círculos: PIPs coloreados por especie",
            pad=7)

        _info_box(ax,
            "%d dominio(s) Lo\nFracc. Lo = %.1f%%" % (n_rafts, lo_frac * 100),
            loc="upper left")
        _info_box(ax,
            "%d PIPs · %d especie(s)" % (n_pips, len(pip_by_sp)),
            loc="lower right")

        path = os.path.join(_sim_fig_dir(seed),
                            "fig9_sim%04d_mapa_pips_balsas.png" % seed)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print("  -> %s" % path)
    return path

def plot_all_figures(membrane, vol=None, labels=None, stats=None, dpi=300):
    """Genera las 9 figuras de publicación independientes."""
    from model_3d import build_physical_volume
    if vol is None or labels is None or stats is None:
        vol, labels, stats = build_physical_volume(
            membrane, bins_xy=55, bins_z=80)

    paths = {}
    paths["fig1"] = plot_fig1_perfil_ED(membrane, vol, stats, dpi=dpi)
    paths["fig2"] = plot_fig2_composicion(membrane, dpi=dpi)
    paths["fig3"] = plot_fig3_helfrich(membrane, dpi=dpi)
    paths["fig4"] = plot_fig4_grosor(membrane, stats, dpi=dpi)
    paths["fig5"] = plot_fig5_mapa_raft(membrane, dpi=dpi)
    paths["fig6"] = plot_fig6_mapa_order(membrane, dpi=dpi)
    paths["fig7"] = plot_fig7_pip_radial(membrane, dpi=dpi)
    paths["fig8"] = plot_fig8_parametros(membrane, stats, dpi=dpi)
    paths["fig9"] = plot_fig9_mapa_pips_balsas(membrane, dpi=dpi)
    return paths