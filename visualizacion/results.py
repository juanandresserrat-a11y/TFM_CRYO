"""
results.py
Generación de figuras de resultados para TFM en base a los datos recolectados.

Figuras producidas:
    R1  Caracterización biofísica (composición, parámetros elásticos, perfil ED)
    R2  Validación biofísica (tabla de benchmarks, scores, paneles de validación)
    R3  Organización lateral (mapas Lo/Ld, PIPs, parámetro de orden)
    R4  Campos de entrenamiento (12 campos para deep learning)
    R5  Calidad cryo-ET (CTF, ruido, PSD, missing wedge)
    R6  Comparativa multi-simulación y justificación de N (dinámico)

Referencias principales:
    [08]  Sharma et al. 2023 – estructura de membranas en cryo-EM
    [11]  Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos
    [12]  Peck et al. 2025 – benchmark con ground-truth para cryo-ET
    [13]  Seghiri et al. 2026 – segmentación aumentada de membranas
    [14]  Moebel et al. 2021 – deep learning en tomogramas celulares de cryo-ET
    [15]  Helfrich 1973 – elasticidad de membranas y espectro de fluctuaciones
    [16]  Pinigin 2022 – parámetros elásticos de membranas desde MD
    [17]  Chakraborty et al. 2020 – dependencia del módulo de bending con composición lipídica
    [18]  Kučerka et al. 2011 – espesores y áreas lipídicas en bicapas PC
    [20]  Piggot et al. 2017 – cálculo de parámetros de orden acil S_CH
    [23]  Glushkova et al. 2026 – variación de grosor en membranas celulares (cryo-ET)
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter, zoom
from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from analisis import analysis
from construccion.builder import BicapaCryoET
from config import OUTPUT_DIR
from simulacion.ctf_sim import add_noise, apply_ctf_2d
from simulacion.electron_density import electron_density_profile, electron_density_projection
from analisis.validation import run_all_benchmarks

RESULTS_DIR = os.path.join(OUTPUT_DIR, "resultados")

C = {
    "lo": "#b89a68",
    "ld": "#a8c8e0",
    "pip": "#c0392b",
    "chol": "#adb5bd",
    "line": "#2c2c2c",
    "grid": "#e8e8e8",
    "pass": "#2dc653",
    "close": "#f4a261",
    "fail": "#e63946",
    "neutral": "#6c757d",
    "bg": "#fafafa",
}

PUB_RC = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arial Narrow", "Helvetica", "DejaVu Sans"],
    "mathtext.default": "regular",
    "text.usetex": False,
    "font.size": 12,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 0.9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "legend.fontsize": 10,
    "legend.framealpha": 0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi": 150,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.facecolor": "white",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def _save(fig: plt.Figure, name: str, dpi: int = 300, subdir: str = "") -> str:
    """
    Guarda dos versiones de la figura:
      - PDF (con carpeta pdf/): titulos de subpanel visibles, para anexos del TFM.
      - JPEG (con carpeta img/): titulos de subpanel ocultos, solo letra del panel,
        optimizado para insertar en el documento con menor peso.
    """
    base = os.path.join(_results_dir(), subdir) if subdir else _results_dir()
    pdf_dir = os.path.join(base, "pdf")
    img_dir = os.path.join(base, "png")
    os.makedirs(pdf_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    pdf_path = os.path.join(pdf_dir, name + ".pdf")
    png_path = os.path.join(img_dir, name + ".png")

    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    titles_backup = {}
    for ax in fig.axes:
        t = ax.title
        if t.get_text():  # solo axes con titulo no vacio
            titles_backup[ax] = t.get_text()
            t.set_visible(False)

    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", facecolor="white", format="png")
    for ax, txt in titles_backup.items():
        ax.title.set_text(txt)
        ax.title.set_visible(True)

    plt.close(fig)
    print(f"  -> {os.path.relpath(pdf_path)}")
    print(f"  -> {os.path.relpath(png_path)}")
    return pdf_path


def _panel_label(ax, letter: str, x: float = -0.15, y: float = 1.06):
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        color=C["line"],
    )


def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _light_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    kw = dict(lw=0.4, color=C["grid"], zorder=0)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, **kw)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, **kw)


def _cumulative_stats(vals):
    means, sems, cvs = [], [], []
    vals = np.asarray(vals)
    for i in range(1, len(vals) + 1):
        sub = vals[:i]
        m = sub.mean()
        s = sub.std(ddof=1) if i > 1 else 0.0
        means.append(m)
        sems.append(s / np.sqrt(i))
        cvs.append(100.0 * s / m if m != 0 else 0.0)
    return np.array(means), np.array(sems), np.array(cvs)


def _plot_strip_r6(
    ax,
    values,
    x_center,
    color,
    ref_lo=None,
    ref_hi=None,
    ref_label=None,
    ylabel="",
    title="",
    annot=True,
):
    values = np.asarray(values)
    mu = values.mean()
    sd = values.std(ddof=1)
    sem = sd / np.sqrt(len(values))

    if ref_lo is not None and ref_hi is not None:
        ax.axhspan(ref_lo, ref_hi, color=C["grid"], alpha=0.5, zorder=0)
        ax.axhline(ref_lo, color=color, lw=0.8, ls=":", alpha=0.5, zorder=1)
        ax.axhline(ref_hi, color=color, lw=0.8, ls=":", alpha=0.5, zorder=1)
        if ref_label:
            ax.text(
                x_center,
                ref_hi + abs(ref_hi) * 0.02,
                ref_label,
                ha="center",
                va="bottom",
                fontsize=8,
                color=C["neutral"],
                style="italic",
            )

    ax.fill_between(
        [x_center - 0.26, x_center + 0.26], mu - sd, mu + sd, color=color, alpha=0.30, zorder=2
    )
    for sign in (+1, -1):
        ax.plot(
            [x_center - 0.26, x_center + 0.26],
            [mu + sign * sd] * 2,
            color=color,
            lw=1.0,
            ls="--",
            alpha=0.6,
            zorder=3,
        )

    ax.plot([x_center - 0.26, x_center + 0.26], [mu, mu], color=color, lw=3.0, zorder=4)

    ax.errorbar(
        x_center,
        mu,
        yerr=sem,
        fmt="none",
        color=C["line"],
        capsize=5,
        capthick=1.6,
        elinewidth=1.6,
        zorder=5,
    )

    jitter = np.linspace(-0.12, 0.12, len(values))
    for ji, vi in zip(jitter, values):
        ax.scatter(
            x_center + ji, vi, s=55, color=color, edgecolors="white", linewidths=0.8, zorder=6
        )

    if annot:
        ax.annotate(
            f"μ = {mu:.3g}\nσ = {sd:.3g}\nN = {len(values)}",
            xy=(x_center + 0.30, mu),
            xycoords="data",
            fontsize=10,
            va="center",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec=C["neutral"], lw=0.55),
            zorder=7,
        )

    ax.set_xlim(0.35, 1.85)
    ax.set_xticks([x_center])
    ax.set_xticklabels([f"N = {len(values)}"], fontsize=8.5)
    ax.set_ylabel(ylabel, labelpad=4)
    if title:
        ax.set_title(title, fontweight="bold", pad=12, loc="left")
    _despine(ax)


def plot_R1_caracterizacion(membrane: BicapaCryoET, dpi: int = 300) -> str:
    """Genera la figura R1: composición, parámetros elásticos y perfil ED."""
    seed = membrane.seed
    g = membrane.geometry

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(12, 8.5))
        fig.suptitle(
            f"Caracterización biofísica | Simulación = {seed} · "
            f"{membrane.Lx/10:.0f}×{membrane.Ly/10:.0f} nm",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            left=0.08,
            right=0.96,
            top=0.93,
            bottom=0.07,
            hspace=0.52,
            wspace=0.35,
            height_ratios=[1, 1.4],
        )

        ax_comp = fig.add_subplot(gs[0, :])  # A: fila superior, ancho completo
        ax_ed = fig.add_subplot(gs[1, 0])  # C: fila inferior izquierda
        ax_param = fig.add_subplot(gs[1, 1])  # B: fila inferior derecha

        _panel_label(ax_comp, "A", x=-0.04, y=1.08)
        comp_data = {
            "Externa": membrane.comp_outer,
            "Interna": membrane.comp_inner,
        }
        all_species = sorted(
            set(list(membrane.comp_outer.keys()) + list(membrane.comp_inner.keys()))
        )

        sp_colors = {
            "POPC": "#3a86ff",
            "POPE": "#e63946",
            "PlsPE": "#c0392b",
            "POPS": "#fb8500",
            "SM": "#2dc653",
            "CHOL": "#adb5bd",
            "GM1": "#d4a017",
            "PI": "#9b5de5",
            "PI3P": "#f39c12",
            "PI4P": "#e67e22",
            "PI5P": "#e74c3c",
            "PI34P2": "#a04000",
            "PIP2": "#c0392b",
            "PIP3": "#7b241c",
        }
        leaflets = list(comp_data.keys())
        y_pos = np.arange(len(leaflets))
        left = np.zeros(len(leaflets))
        handles = []
        for sp in all_species:
            vals = np.array([comp_data[lf].get(sp, 0) * 100 for lf in leaflets])
            col = sp_colors.get(sp, "#888888")
            bars = ax_comp.barh(
                y_pos, vals, left=left, color=col, height=0.28, edgecolor="white", linewidth=0.35
            )
            if any(v > 0 for v in vals):
                handles.append(mpatches.Patch(color=col, label=sp))
                for i, (v, lf) in enumerate(zip(vals, leaflets)):
                    if v > 3.0:
                        ax_comp.text(
                            left[i] + v / 2,
                            y_pos[i],
                            f"{v:.0f}%",
                            ha="center",
                            va="center",
                            fontsize=7.5,
                            color="white",
                            fontweight="bold",
                        )
            left += vals

        # Completar al 100% con "Fracción no clasificada" si la suma < 100%
        RESTO_COLOR = "#00796b"
        for i, lf in enumerate(leaflets):
            total = sum(comp_data[lf].get(sp, 0) * 100 for sp in all_species)
            remainder = 100.0 - total
            if remainder > 0.5:  # solo si hay fraccion significativa
                ax_comp.barh(
                    y_pos[i],
                    remainder,
                    left=left[i],
                    color=RESTO_COLOR,
                    height=0.28,
                    edgecolor="white",
                    linewidth=0.35,
                )
                if remainder > 3.0:
                    ax_comp.text(
                        left[i] + remainder / 2,
                        y_pos[i],
                        f"{remainder:.0f}%",
                        ha="center",
                        va="center",
                        fontsize=7.5,
                        color="white",
                        fontweight="bold",
                    )

        resto_patch = mpatches.Patch(color=RESTO_COLOR, label="Fracción no asignada")
        handles_full = handles + [resto_patch]

        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(leaflets, fontsize=9)
        ax_comp.set_xlabel("Fracción molar (%)", labelpad=2)
        ax_comp.set_title("Composición lipídica por monocapa", pad=5)
        ax_comp.set_xlim(0, 102)
        ax_comp.legend(
            handles=handles_full,
            loc="center",
            bbox_to_anchor=(0.5, 0.5),
            ncol=9,
            fontsize=7.5,
            frameon=True,
            fancybox=True,
            borderpad=0.55,
            labelspacing=0.2,
            columnspacing=0.8,
        )

        ax_comp.axvline(100, color=C["neutral"], lw=0.6, ls="--", alpha=0.4)

        ax_comp.set_ylim(-0.38, 1.38)
        _despine(ax_comp)

        _panel_label(ax_param, "C")
        params = {
            "kc": {
                "val": membrane.bending_modulus,
                "ref_lo": 20,
                "ref_hi": 45,
                "unit": "kBT·nm2",
                "color": C["lo"],
            },
            "sigma": {
                "val": membrane.surface_tension,
                "ref_lo": 0.0001,
                "ref_hi": 0.005,
                "unit": "kBT·nm-2",
                "color": C["ld"],
            },
            "D_PP": {
                "val": g.total_thick,
                "ref_lo": 35,
                "ref_hi": 50,
                "unit": "A (global)",
                "color": "#2a9d8f",
            },
        }
        WARN_COLOR = C["close"]  # naranja: "#f4a261"

        for i, (label, p) in enumerate(params.items()):
            v, lo, hi = p["val"], p["ref_lo"], p["ref_hi"]
            rng = hi - lo
            frac = np.clip((v - lo) / rng, 0.0, 1.0) if rng != 0 else 0.0
            out_of_range = not (lo <= v <= hi)

            ax_param.barh(
                i,
                1.0,
                color=C["grid"],
                height=0.70,
                edgecolor=C["neutral"],
                linewidth=0.4,
                alpha=0.22,
                zorder=0,
                label="Rango ref." if i == 0 else "",
            )

            bar_color = p["color"]
            ax_param.barh(
                i,
                frac,
                color=bar_color,
                height=0.22,
                edgecolor="white",
                linewidth=0.5,
                alpha=0.95,
                zorder=3,
            )

            # ── Bordes del rango de referencia ──
            ax_param.axvline(0.0, color=C["neutral"], lw=0.7, ls="-", alpha=0.35, zorder=1)

            # Marcador de desbordamiento (más prominente)
            if out_of_range:
                if v > hi:
                    ax_param.text(
                        1.05,
                        i,
                        "▶",
                        va="center",
                        ha="left",
                        fontsize=12,
                        color=WARN_COLOR,
                        fontweight="bold",
                        zorder=6,
                        clip_on=False,
                    )
                    # Línea punteada que indica cuánto se pasó
                    ax_param.plot(
                        [1.0, 1.12],
                        [i, i],
                        color=WARN_COLOR,
                        lw=1.2,
                        ls=":",
                        alpha=0.6,
                        zorder=2,
                        clip_on=False,
                    )
                else:
                    ax_param.text(
                        -0.04,
                        i,
                        "◀",
                        va="center",
                        ha="right",
                        fontsize=12,
                        color=WARN_COLOR,
                        fontweight="bold",
                        zorder=6,
                        clip_on=False,
                    )
                    ax_param.plot(
                        [-0.10, 0.0],
                        [i, i],
                        color=WARN_COLOR,
                        lw=1.2,
                        ls=":",
                        alpha=0.6,
                        zorder=2,
                        clip_on=False,
                    )

            status = "PASS" if not out_of_range else "WARN"
            status_color = C["pass"] if not out_of_range else WARN_COLOR

            val_str = f"{v:.4f}" if abs(v) < 0.1 else f"{v:.1f}"

            ax_param.text(
                1.17,
                i,
                f"{status}  {val_str} {p['unit']}",
                va="center",
                fontsize=8.5,
                color=status_color,
                fontweight="bold" if out_of_range else "normal",
                clip_on=False,
            )

        # Línea discontinua en Máx ref.
        ax_param.axvline(1.0, color=C["neutral"], lw=0.9, ls="--", alpha=0.55, zorder=1)
        ax_param.set_yticks(range(len(params)))
        ax_param.set_yticklabels(list(params.keys()), fontsize=8.5)
        # xlim extendido: la línea de Máx ref (1.0) ya NO está en el borde físico,
        # dejando espacio a la derecha para visualizar desbordamientos cómodamente.
        ax_param.set_xlim(-0.02, 1.22)
        ax_param.set_xticks([0.0, 0.5, 1.0])
        ax_param.set_xticklabels(["Mín ref.", "50 %", "Máx ref."], fontsize=8)
        ax_param.set_title("Parámetros elásticos", pad=5)
        _despine(ax_param)

        _panel_label(ax_ed, "B")
        z_cent, ed_prof = electron_density_profile(membrane, bins_z=300)
        z_nm = z_cent / 10.0
        ax_ed.plot(z_nm, ed_prof, color=C["lo"], lw=1.6, label="ED media")
        ax_ed.fill_between(
            z_nm,
            0.334,
            ed_prof,
            where=(ed_prof > 0.334),
            color=C["lo"],
            alpha=0.22,
            label="Region densa",
        )
        ax_ed.fill_between(
            z_nm,
            ed_prof,
            0.334,
            where=(ed_prof < 0.334),
            color=C["ld"],
            alpha=0.22,
            label="Region diluida",
        )
        ax_ed.axhline(0.334, color=C["neutral"], lw=0.9, ls="--", label="Agua bulk (0.334 e·A-3)")
        ax_ed.axvline(g.z_outer / 10, color=C["lo"], lw=0.8, ls=":", alpha=0.7)
        ax_ed.axvline(g.z_inner / 10, color=C["lo"], lw=0.8, ls=":", alpha=0.7)
        z_range = max(abs(z_nm.min()), abs(z_nm.max()))
        ax_ed.set_xlim(-z_range * 1.25, z_range * 1.25)
        ax_ed.set_xlabel("Posicion axial Z (nm)")
        ax_ed.set_ylabel("Densidad electronica (e·A-3)")
        ax_ed.set_title("Perfil de densidad electronica", pad=5)
        ax_ed.legend(fontsize=8, loc="upper right")
        _despine(ax_ed)

        return _save(fig, f"R1_caracterizacion_sim{seed:04d}", dpi, subdir="R1")


def plot_R2_validacion(
    membrane: BicapaCryoET, results: Optional[Dict] = None, dpi: int = 300
) -> str:
    """Genera la figura R2: tabla de benchmarks, scores y paneles de validación."""
    seed = membrane.seed
    if results is None:
        print("  Calculando benchmarks...")
        results = run_all_benchmarks(membrane)

    sch = membrane.get_sch_by_domain()
    lo_mean = sch["lo"]
    ld_mean = sch["ld"]
    chol_mean = sch["chol"]
    delta_sch = sch["delta"]

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15, 10))
        summary = results.get("summary", {"score": 0, "passed": 0, "total": 6})
        accuracy_pct = summary.get("accuracy_pct", summary["score"] * 100)
        fig.suptitle(
            f"Validación biofísica | Simulación = {seed}  "
            f"[Accuracy media: {accuracy_pct:.1f}% "
            f"· {summary['passed']}/{summary['total']} benchmarks]",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(
            3,
            3,
            figure=fig,
            left=0.07,
            right=0.96,
            top=0.88,
            bottom=0.07,
            hspace=0.55,
            wspace=0.42,
        )

        ax_tbl = fig.add_subplot(gs[0, :2])
        ax_bar = fig.add_subplot(gs[0, 2])
        ax_helf = fig.add_subplot(gs[1, 0])
        ax_sch = fig.add_subplot(gs[1, 1])
        ax_ed = fig.add_subplot(gs[1, 2])
        ax_acf = fig.add_subplot(gs[2, 0])
        ax_thick2 = fig.add_subplot(gs[2, 1])
        ax_inter = fig.add_subplot(gs[2, 2])

        _panel_label(ax_tbl, "A", x=-0.02)

        rows = [
            ("Helfrich pendiente", "slope_high_q", "helfrich", "−4 ± 0.3", "accuracy"),
            ("Grosor D_PP por fase (A)", "mean_nm", "thickness", "35–50 Å", "accuracy_diff"),
            ("ΔD Lo−Ld (Å)", "diff_A", "thickness", "3–6 Å", "accuracy_diff"),
            ("S_CH Lo", "gel_mean", "order", "0.85–0.95", "accuracy_gel"),
            ("S_CH Ld", "fluid_mean", "order", "0.60–0.75", "accuracy_fluid"),
            ("Long. corr. Lo (nm)", "xi_nm", "raft_corr", "5–25 nm", "accuracy"),
            ("Interdig. Lo > Ld", "lo_gt_ld", "interdig", "True", "accuracy"),
            (
                "ED cabeza (e·Å⁻³)",
                "ed_head_peak",
                "electron_ed",
                "0.44–0.50 e·Å⁻³",
                "accuracy_head",
            ),
            ("ED cola (e·Å⁻³)", "ed_tail", "electron_ed", "0.28–0.31 e·Å⁻³", "accuracy_tail"),
        ]

        ax_tbl.axis("off")
        col_labels = ["Parámetro", "Valor medido", "Ref. bibliográfica", ""]
        col_widths = [0.36, 0.24, 0.28, 0.09]
        x_positions = [0.02]
        for w in col_widths[:-1]:
            x_positions.append(x_positions[-1] + w)

        for j, (lbl, x) in enumerate(zip(col_labels, x_positions)):
            ax_tbl.text(
                x,
                1.18,
                lbl,
                transform=ax_tbl.transAxes,
                fontsize=8,
                fontweight="bold",
                va="top",
                color=C["line"],
            )

        ax_tbl.plot(
            [0, 1],
            [1.08, 1.08],
            color=C["line"],
            lw=0.8,
            transform=ax_tbl.transAxes,
            clip_on=False,
        )

        n = len(rows)
        for i, (label, key, bench, ref, pass_key) in enumerate(rows):
            y = 0.90 - (i + 1) * (0.90 / (n + 1))

            bench_results = results.get(bench, {})
            val = bench_results.get(key, "—")

            if key == "gel_mean":
                val = lo_mean
                bench_results = dict(bench_results)
                bench_results[pass_key] = 1.0 if 0.85 <= lo_mean <= 0.95 else 0.0
            elif key == "fluid_mean":
                val = ld_mean
                bench_results = dict(bench_results)
                bench_results[pass_key] = 1.0 if 0.60 <= ld_mean <= 0.75 else 0.0

            if key == "mean_nm" and isinstance(val, (int, float)):
                val = val * 10.0

            passed_val = bench_results.get(pass_key, False)
            if isinstance(passed_val, (int, float)):
                acc_norm = passed_val if passed_val <= 1.0 else passed_val / 100.0
            else:
                acc_norm = 1.0 if bool(passed_val) else 0.0

            if acc_norm >= 0.70:
                tier, status_col, bg_col = "PASS", C["pass"], "#f6fff8"
            elif acc_norm >= 0.40:
                tier, status_col, bg_col = "CLOSE", C["close"], "#fff8f0"
            else:
                tier, status_col, bg_col = "FAIL", C["fail"], "#fff5f5"

            if isinstance(val, float):
                if "grosor" in label.lower() or "D_PP" in label or "Delta D" in label:
                    val_str = f"{val:.1f}"
                else:
                    val_str = f"{val:.3f}"
            elif isinstance(val, bool):
                val_str = "Sí" if val else "No"
            else:
                val_str = str(val)

            ax_tbl.axhspan(y - 0.04, y + 0.04, color=bg_col, alpha=0.7, transform=ax_tbl.transAxes)
            ax_tbl.text(
                x_positions[0] + 0.01,
                y,
                label,
                transform=ax_tbl.transAxes,
                fontsize=8.5,
                va="center",
            )
            ax_tbl.text(
                x_positions[1],
                y,
                val_str,
                transform=ax_tbl.transAxes,
                fontsize=8.5,
                va="center",
                color=C["line"],
                ha="center",
            )
            ax_tbl.text(
                x_positions[2],
                y,
                ref,
                transform=ax_tbl.transAxes,
                fontsize=8,
                va="center",
                color=C["neutral"],
                ha="center",
            )
            ax_tbl.text(
                x_positions[3],
                y,
                tier,
                transform=ax_tbl.transAxes,
                fontsize=9,
                va="center",
                ha="center",
                color=status_col,
                fontweight="bold",
            )

        ax_tbl.text(
            0.5,
            -0.02,
            f"Nota: SCH excl. CHOL (CHOL intrínseco = {chol_mean:.2f}; ΔS_CH = {delta_sch:.3f}). "
            f"D_PP por fase: medición local Lo/Ld. D_PP global (R1): perfil promediado completo.",
            transform=ax_tbl.transAxes,
            fontsize=7.5,
            ha="center",
            va="top",
            color=C["neutral"],
            style="italic",
        )

        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1.05)

        _panel_label(ax_bar, "B")
        bench_names = [
            "Helfrich",
            "Grosor",
            "Orden\nS_CH",
            "Corr.\nLo",
            "Interdig.",
            "Densidad\nelec.",
        ]
        bench_keys = ["helfrich", "thickness", "order", "raft_corr", "interdig", "electron_ed"]
        scores = []
        for bk in bench_keys:
            b = results.get(bk, {})
            if "accuracy" in b and isinstance(b["accuracy"], (int, float)):
                scores.append(b["accuracy"] / 100.0)
            else:
                accs = [
                    v
                    for k, v in b.items()
                    if k.startswith("accuracy") and isinstance(v, (int, float))
                ]
                if accs:
                    scores.append(np.mean(accs) / 100.0)
                elif "pass" in b and isinstance(b["pass"], bool):
                    scores.append(1.0 if b["pass"] else 0.0)
                else:
                    scores.append(0.0)

        def _bar_col(s):
            if s >= 0.70:
                return C["pass"]
            if s >= 0.40:
                return C["close"]
            return C["fail"]

        bar_cols = [_bar_col(s) for s in scores]
        ax_bar.barh(
            range(len(scores)),
            scores,
            color=bar_cols,
            height=0.6,
            edgecolor="white",
            linewidth=0.5,
        )
        ax_bar.axvline(1.0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
        ax_bar.set_yticks(range(len(bench_names)))
        ax_bar.set_yticklabels(bench_names, fontsize=8.5)
        ax_bar.set_xlim(0, 1.2)
        ax_bar.set_xlabel("Fraccion superada")
        for i, s in enumerate(scores):
            ax_bar.text(s + 0.03, i, f"{s*100:.0f}%", va="center", fontsize=8)
        _despine(ax_bar)

        _panel_label(ax_helf, "C")
        h = results.get("helfrich", {})
        q_c = h.get("q_centers", [])
        p_m = h.get("p_mean", [])
        if len(q_c) > 3 and len(p_m) > 3:
            q_arr = np.asarray(q_c)
            p_arr = np.asarray(p_m)
            mask = (q_arr > 0) & (p_arr > 0)
            ax_helf.loglog(
                q_arr[mask], p_arr[mask], "o", ms=3.5, color=C["lo"], alpha=0.8, label="Simulación"
            )

            idx_hq = q_arr > q_arr.mean()
            if idx_hq.sum() > 2:
                q_fit = q_arr[idx_hq & mask]
                A = np.median(p_arr[idx_hq & mask] * q_fit**4)
                ax_helf.loglog(
                    q_fit, A / q_fit**4, "--", color=C["fail"], lw=1.2, label="q^{-4} Helfrich"
                )
        ax_helf.set_xlabel("q (nm-1)")
        ax_helf.set_ylabel("|hq|2 (nm2)")
        ax_helf.legend(fontsize=8)
        _despine(ax_helf)

        _panel_label(ax_sch, "D")
        bins_sch = np.linspace(0.45, 1.0, 35)
        todos = membrane.outer_leaflet + membrane.inner_leaflet
        s_lo = [l.order_param for l in todos if l.in_raft and l.lipid_type.name != "CHOL"]
        s_ld = [l.order_param for l in todos if not l.in_raft and l.lipid_type.name != "CHOL"]
        s_chol = [l.order_param for l in todos if l.lipid_type.name == "CHOL"]

        if s_lo:
            ax_sch.hist(
                s_lo,
                bins=bins_sch,
                density=True,
                color=C["lo"],
                alpha=0.70,
                label=f"Lo  μ={lo_mean:.3f}",
                edgecolor="white",
                linewidth=0.3,
            )
            ax_sch.axvline(lo_mean, color=C["lo"], lw=1.3, ls="--")
        if s_ld:
            ax_sch.hist(
                s_ld,
                bins=bins_sch,
                density=True,
                color=C["ld"],
                alpha=0.70,
                label=f"Ld  μ={ld_mean:.3f}",
                edgecolor="white",
                linewidth=0.3,
            )
            ax_sch.axvline(ld_mean, color=C["ld"], lw=1.3, ls="--")
        if s_chol:
            ax_sch.axvline(
                chol_mean, color=C["chol"], lw=1.8, ls="-.", label=f"CHOL  μ={chol_mean:.2f}"
            )
        ax_sch.axvspan(0.85, 0.95, color=C["lo"], alpha=0.15, label="Ref. Lo")
        ax_sch.axvspan(0.60, 0.75, color=C["ld"], alpha=0.15, label="Ref. Ld")
        ax_sch.text(
            0.90,
            ax_sch.get_ylim()[1] if ax_sch.get_ylim()[1] > 0 else 1,
            "Lo ref.",
            ha="center",
            va="top",
            fontsize=8,
            color=C["lo"],
            transform=ax_sch.transData,
        )
        ax_sch.set_xlabel("S_CH")
        ax_sch.set_ylabel("Densidad norm.")
        ax_sch.set_title("")
        ax_sch.legend(fontsize=8, loc="upper left")
        _despine(ax_sch)

        _panel_label(ax_ed, "E")
        z_c, ed_p = electron_density_profile(membrane, bins_z=200)
        ax_ed.plot(ed_p, z_c / 10, color=C["lo"], lw=1.5, label="Simulación")
        ax_ed.axvspan(0.44, 0.50, color=C["lo"], alpha=0.15, label="Ref. cabeza")
        ax_ed.axvspan(0.28, 0.31, color=C["ld"], alpha=0.15, label="Ref. cola")
        ax_ed.axvline(0.334, color=C["neutral"], lw=0.8, ls="--", alpha=0.6)
        ax_ed.set_xlabel("Densidad electronica (e·A-3)")
        ax_ed.set_ylabel("Z (nm)")
        ax_ed.legend(fontsize=8)
        _despine(ax_ed)

        _panel_label(ax_acf, "F")
        rc = results.get("raft_corr", {})
        r_v = rc.get("r_vals", [])
        acf = rc.get("acf", [])
        xi = rc.get("xi_nm", None)
        if len(r_v) > 2 and len(acf) > 2:
            ax_acf.plot(r_v, acf, color=C["lo"], lw=1.5)
            ax_acf.axhline(0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
            ax_acf.fill_between(r_v, 0, acf, where=np.asarray(acf) > 0, color=C["lo"], alpha=0.2)
            if xi is not None:
                ax_acf.axvline(xi, color=C["fail"], lw=1.0, ls=":", label=f"xi = {xi:.1f} nm")
                ax_acf.legend(fontsize=8)
        ax_acf.set_xlabel("Radio (nm)")
        ax_acf.set_ylabel("ACF normalizada")
        ax_acf.set_ylim(-0.25, 1.15)
        _despine(ax_acf)

        _panel_label(ax_thick2, "G")

        thick = membrane.get_thickness_by_domain()
        lo_mean_t = thick["lo"]
        ld_mean_t = thick["ld"]
        delta_t = thick["delta"]

        sup = membrane.outer_leaflet
        inf = membrane.inner_leaflet
        sup_xy = np.array([[l.head_pos[0], l.head_pos[1]] for l in sup])
        inf_xy = np.array([[l.head_pos[0], l.head_pos[1]] for l in inf])
        sup_z = np.array([l.head_pos[2] for l in sup])
        inf_z = np.array([l.head_pos[2] for l in inf])
        tree = KDTree(inf_xy)
        _, idxs = tree.query(sup_xy, k=1)
        paired = sup_z - inf_z[idxs]

        lo_vals = np.array([paired[i] for i, l in enumerate(sup) if l.in_raft])
        ld_vals = np.array([paired[i] for i, l in enumerate(sup) if not l.in_raft])

        labels_bar, means_bar, stds_bar, colors_bar = [], [], [], []
        if len(lo_vals) > 0:
            labels_bar.append("Lo")
            means_bar.append(lo_vals.mean())
            stds_bar.append(lo_vals.std())
            colors_bar.append(C["lo"])
        if len(ld_vals) > 0:
            labels_bar.append("Ld")
            means_bar.append(ld_vals.mean())
            stds_bar.append(ld_vals.std())
            colors_bar.append(C["ld"])

        x_pos = np.arange(len(labels_bar))
        bars = ax_thick2.bar(
            x_pos,
            means_bar,
            yerr=stds_bar,
            color=colors_bar,
            width=0.55,
            edgecolor="white",
            linewidth=0.5,
            capsize=4,
            error_kw=dict(ecolor=C["line"], lw=1.2, capthick=1),
        )

        for i, (m, s, lbl) in enumerate(zip(means_bar, stds_bar, labels_bar)):
            ax_thick2.text(
                i,
                m + s + 0.8,
                f"{m:.1f} ± {s:.1f} Å",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color=C["line"],
                fontweight="bold",
            )

        dpp_total = membrane.geometry.total_thick
        ax_thick2.axhline(
            dpp_total,
            color=C["fail"],
            lw=1.2,
            ls=":",
            alpha=0.8,
            label=f"D_PP total = {dpp_total:.1f} A",
        )
        ax_thick2.axhspan(35, 50, color=C["lo"], alpha=0.08, label="Ref. 35–50 Å")

        if len(labels_bar) == 2:
            ax_thick2.annotate(
                f"Δ = {delta_t:.1f} Å",
                xy=(0.5, max(means_bar) + max(stds_bar) + 2.5),
                ha="center",
                va="bottom",
                fontsize=8.5,
                fontweight="bold",
                color=C["line"],
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=C["neutral"], lw=0.6),
            )

        ax_thick2.set_xticks(x_pos)
        ax_thick2.set_xticklabels(labels_bar, fontsize=9)
        ax_thick2.set_ylabel("Grosor D_PP (A)")
        ax_thick2.legend(fontsize=8, loc="upper left")

        y_min = min(min(means_bar) - max(stds_bar) * 1.5, 30) if means_bar else 30
        y_max = max(max(means_bar) + max(stds_bar) * 2.5, 55, dpp_total + 2) if means_bar else 55
        ax_thick2.set_ylim(y_min, y_max)
        _despine(ax_thick2)

        _panel_label(ax_inter, "H", x=-0.4)
        interdig = analysis.interdigitation_map(membrane, bins=80)
        im_i = ax_inter.imshow(
            interdig.T,
            origin="lower",
            extent=[0, membrane.Lx / 10, 0, membrane.Ly / 10],
            cmap="YlOrRd",
            aspect="equal",
            interpolation="bilinear",
        )
        cb_i = plt.colorbar(im_i, ax=ax_inter, shrink=0.85, pad=0.02)
        cb_i.set_label("Índice interdig.", fontsize=8.5, labelpad=3)
        cb_i.ax.tick_params(labelsize=6)
        ax_inter.set_xlabel("x (nm)")
        ax_inter.set_ylabel("y (nm)")

        return _save(fig, f"R2_validacion_sim{seed:04d}", dpi, subdir="R2")


def plot_R3_organizacion(membrane: BicapaCryoET, dpi: int = 300) -> str:
    """Genera la figura R3: mapas de fase Lo/Ld, PIPs y parámetro de orden."""
    seed = membrane.seed
    Lx, Ly = membrane.Lx / 10, membrane.Ly / 10
    ext = [0, Lx, 0, Ly]

    def _raft_map_smooth_local(membrane, leaflet, bins=180, sigma=2.2):
        Hr = np.zeros((bins, bins))
        Ht = np.zeros((bins, bins))
        for lip in leaflet:
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
        with np.errstate(invalid="ignore"):
            result = np.clip(sm / np.maximum(cnt, 1e-9), 0, 1)
        return result

    raft_map_ext = _raft_map_smooth_local(membrane, membrane.outer_leaflet, bins=180, sigma=2.2)
    raft_map_int = _raft_map_smooth_local(membrane, membrane.inner_leaflet, bins=180, sigma=2.2)
    order_map = analysis.order_parameter_map(membrane, bins=100)
    pip_map = analysis.pip_density_map(membrane, bins=100)

    cmap_lo_ld = mcolors.LinearSegmentedColormap.from_list(
        "lo_ld",
        [
            (0.0, "#1d4e7a"),
            (0.25, "#4a85b0"),
            (0.50, "#f0ede4"),
            (0.75, "#c9693a"),
            (1.0, "#7a2e00"),
        ],
        N=256,
    )

    PIP_CLR = {
        "PI": "#d4ac0d",
        "PI3P": "#f39c12",
        "PI4P": "#e67e22",
        "PI5P": "#e74c3c",
        "PI34P2": "#a04000",
        "PIP2": "#c0392b",
        "PIP3": "#7b241c",
    }

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"Organización lateral | Simulación = {seed}", fontsize=11, fontweight="bold", y=0.98
        )

        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            left=0.05,
            right=0.97,
            top=0.93,
            bottom=0.07,
            hspace=0.28,
            wspace=0.30,
        )

        ax_a = fig.add_subplot(gs[0, 0])
        ax_b = fig.add_subplot(gs[0, 1])
        ax_c = fig.add_subplot(gs[0, 2])
        ax_d = fig.add_subplot(gs[1, 0])
        ax_e = fig.add_subplot(gs[1, 1])
        ax_f = fig.add_subplot(gs[1, 2])

        _panel_label(ax_a, "A")
        im0 = ax_a.imshow(
            raft_map_ext.T,
            origin="lower",
            extent=ext,
            cmap=cmap_lo_ld,
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="bilinear",
        )

        xg_ext = np.linspace(0, Lx, raft_map_ext.shape[0])
        yg_ext = np.linspace(0, Ly, raft_map_ext.shape[1])
        if raft_map_ext.min() <= 0.65 <= raft_map_ext.max():
            ax_a.contour(
                xg_ext,
                yg_ext,
                raft_map_ext.T,
                levels=[0.65],
                colors=["#1a1a1a"],
                linewidths=1.5,
                linestyles="-",
                alpha=0.90,
            )

        cb0 = plt.colorbar(im0, ax=ax_a, shrink=0.85, pad=0.02)
        cb0.set_label("Fracción Lo", fontsize=8.5, labelpad=3)
        cb0.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb0.set_ticklabels(["0\n(Ld)", "0.25", "0.50", "0.75", "1.0\n(Lo)"], fontsize=8)
        cb0.ax.tick_params(labelsize=6)

        lo_pct_ext = float(raft_map_ext.mean()) * 100
        n_dom_ext = len(getattr(membrane, "rafts_outer", []))
        ax_a.text(
            0.03,
            0.97,
            f"Lo: {lo_pct_ext:.1f}%  |  {n_dom_ext} dominio(s)",
            transform=ax_a.transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(fc="white", ec=C["neutral"], lw=0.5, pad=2.5, alpha=0.9),
        )
        ax_a.set_xlabel("x (nm)")
        ax_a.set_ylabel("y (nm)")
        ax_a.set_title("Mapa de fase Lo/Ld (monocapa externa)", pad=5)

        _panel_label(ax_b, "B")
        ax_b.imshow(
            raft_map_ext.T,
            origin="lower",
            extent=ext,
            cmap=cmap_lo_ld,
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="bilinear",
            alpha=0.55,
        )

        if raft_map_ext.min() <= 0.65 <= raft_map_ext.max():
            ax_b.contour(
                xg_ext,
                yg_ext,
                raft_map_ext.T,
                levels=[0.65],
                colors=["#1a1a1a"],
                linewidths=1.2,
                linestyles="--",
                alpha=0.80,
            )

        pip_handles = []
        n_total_pip = 0
        for sp in sorted(PIP_CLR.keys()):
            pts = [
                (l.head_pos[0] / 10, l.head_pos[1] / 10)
                for l in membrane.inner_leaflet
                if l.lipid_type.name == sp
            ]
            if not pts:
                continue
            xs, ys = zip(*pts)
            col = PIP_CLR[sp]
            ax_b.scatter(
                xs,
                ys,
                s=12,
                c=col,
                marker="o",
                edgecolors="black",
                linewidths=0.3,
                alpha=0.9,
                zorder=5,
            )
            pip_handles.append(
                mpatches.Patch(fc=col, ec="black", lw=0.4, label=f"{sp}  (n={len(pts)})")
            )
            n_total_pip += len(pts)

        if pip_handles:
            ax_b.legend(handles=pip_handles, loc="upper right", fontsize=8, frameon=True)
        ax_b.text(
            0.03,
            0.03,
            f"{n_total_pip} PIPs total",
            transform=ax_b.transAxes,
            fontsize=8.5,
            bbox=dict(fc="white", ec=C["neutral"], lw=0.5, pad=2, alpha=0.9),
        )
        ax_b.set_xlabel("x (nm)")
        ax_b.set_ylabel("y (nm)")
        ax_b.set_title("Distribución de PIPs (monocapa interna)", pad=5)

        _panel_label(ax_c, "C")
        vlo_sch = np.percentile(order_map, 2)
        vhi_sch = np.percentile(order_map, 98)
        im2 = ax_c.imshow(
            order_map.T,
            origin="lower",
            extent=ext,
            cmap="RdYlGn",
            vmin=vlo_sch,
            vmax=vhi_sch,
            aspect="equal",
            interpolation="bilinear",
        )
        cb2 = plt.colorbar(im2, ax=ax_c, shrink=0.85, pad=0.02)
        cb2.set_label("S_CH", fontsize=8.5, labelpad=3)
        cb2.ax.tick_params(labelsize=6)

        ax_c.set_xlabel("x (nm)")
        ax_c.set_ylabel("y (nm)")
        ax_c.set_title("Parámetro de orden S_CH", pad=5)

        _panel_label(ax_d, "D")
        im3 = ax_d.imshow(
            pip_map.T,
            origin="lower",
            extent=ext,
            cmap="hot_r",
            aspect="equal",
            interpolation="bilinear",
        )
        cb3 = plt.colorbar(im3, ax=ax_d, shrink=0.85, pad=0.02)
        cb3.set_label("Densidad PIP (Da·Å⁻²)", fontsize=8.5, labelpad=3)
        cb3.ax.tick_params(labelsize=6)

        for cl in getattr(membrane, "pip_clusters", []):
            if not cl:
                continue
            cx = np.mean([l.head_pos[0] for l in cl]) / 10
            cy = np.mean([l.head_pos[1] for l in cl]) / 10
            r = np.std([np.hypot(l.head_pos[0] / 10 - cx, l.head_pos[1] / 10 - cy) for l in cl])
            r = max(r, 0.5)
            circ = plt.Circle(
                (cx, cy), r, fill=False, edgecolor="cyan", linewidth=1.2, linestyle=":", alpha=0.85
            )
            ax_d.add_patch(circ)
        ax_d.set_xlabel("x (nm)")
        ax_d.set_ylabel("y (nm)")
        ax_d.set_title("Mapa de densidad de PIPs", pad=5)

        _panel_label(ax_e, "E")
        im4 = ax_e.imshow(
            raft_map_int.T,
            origin="lower",
            extent=ext,
            cmap=cmap_lo_ld,
            vmin=0,
            vmax=1,
            aspect="equal",
            interpolation="bilinear",
        )

        xg_int = np.linspace(0, Lx, raft_map_int.shape[0])
        yg_int = np.linspace(0, Ly, raft_map_int.shape[1])
        if raft_map_int.min() <= 0.65 <= raft_map_int.max():
            ax_e.contour(
                xg_int,
                yg_int,
                raft_map_int.T,
                levels=[0.65],
                colors=["#1a1a1a"],
                linewidths=1.5,
                linestyles="-",
                alpha=0.90,
            )

        cb4 = plt.colorbar(im4, ax=ax_e, shrink=0.85, pad=0.02)
        cb4.set_label("Fracción Lo", fontsize=8.5, labelpad=3)
        cb4.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb4.set_ticklabels(["0\n(Ld)", "0.25", "0.50", "0.75", "1.0\n(Lo)"], fontsize=8)
        cb4.ax.tick_params(labelsize=6)

        lo_pct_int = float(raft_map_int.mean()) * 100
        n_dom_int = len(getattr(membrane, "rafts_inner", []))
        ax_e.text(
            0.03,
            0.97,
            f"Lo: {lo_pct_int:.1f}%  |  {n_dom_int} dominio(s)",
            transform=ax_e.transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(fc="white", ec=C["neutral"], lw=0.5, pad=2.5, alpha=0.9),
        )
        ax_e.set_xlabel("x (nm)")
        ax_e.set_ylabel("y (nm)")
        ax_e.set_title("Mapa de fase Lo/Ld (monocapa interna)", pad=5)

        _panel_label(ax_f, "F")

        target_shape = (100, 100)

        if raft_map_ext.shape != target_shape:
            zoom_factor_ext = (
                target_shape[0] / raft_map_ext.shape[0],
                target_shape[1] / raft_map_ext.shape[1],
            )
            raft_ext_resized = zoom(raft_map_ext, zoom_factor_ext, order=1)
        else:
            raft_ext_resized = raft_map_ext

        if raft_map_int.shape != target_shape:
            zoom_factor_int = (
                target_shape[0] / raft_map_int.shape[0],
                target_shape[1] / raft_map_int.shape[1],
            )
            raft_int_resized = zoom(raft_map_int, zoom_factor_int, order=1)
        else:
            raft_int_resized = raft_map_int

        diff_map = raft_ext_resized - raft_int_resized

        cmap_diff = mcolors.LinearSegmentedColormap.from_list(
            "diff", [(0.0, "#2166ac"), (0.5, "#f7f7f7"), (1.0, "#b2182b")], N=256
        )

        vlo_diff = np.percentile(diff_map, 1)
        vhi_diff = np.percentile(diff_map, 99)
        vmax_diff = max(abs(vlo_diff), abs(vhi_diff))

        im5 = ax_f.imshow(
            diff_map.T,
            origin="lower",
            extent=ext,
            cmap=cmap_diff,
            vmin=-vmax_diff,
            vmax=vmax_diff,
            aspect="equal",
            interpolation="bilinear",
        )
        cb5 = plt.colorbar(im5, ax=ax_f, shrink=0.85, pad=0.02)
        cb5.set_label("Delta Lo (ext − int)", fontsize=8.5, labelpad=3)
        cb5.ax.tick_params(labelsize=6)

        ax_f.contour(
            np.linspace(0, Lx, diff_map.shape[0]),
            np.linspace(0, Ly, diff_map.shape[1]),
            diff_map.T,
            levels=[0],
            colors=["black"],
            linewidths=1.0,
            linestyles="--",
            alpha=0.7,
        )

        mean_diff = float(diff_map.mean())
        std_diff = float(diff_map.std())
        ax_f.text(
            0.03,
            0.97,
            f"mean={mean_diff:+.3f}\nstd={std_diff:.3f}",
            transform=ax_f.transAxes,
            fontsize=8.5,
            va="top",
            bbox=dict(fc="white", ec=C["neutral"], lw=0.5, pad=2.5, alpha=0.9),
        )
        ax_f.set_xlabel("x (nm)")
        ax_f.set_ylabel("y (nm)")
        ax_f.set_title("Asimetría Lo (monocapa ext − int)", pad=5)

        return _save(fig, f"R3_organizacion_lateral_sim{seed:04d}", dpi, subdir="R3")


def plot_R4_campos(membrane: BicapaCryoET, dpi: int = 300) -> str:
    """Genera la figura R4: cuadrícula 3×4 con 12 campos biofísicos."""
    seed = membrane.seed
    bins = 64

    campos = {
        "c0  Densidad cryo-ET\n(Da·Å⁻²)": (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins)
            + analysis.density_map(membrane, membrane.inner_leaflet, bins=bins)
        ),
        "c1  Grosor local\n(Å)": analysis.thickness_map(membrane, bins=bins),
        "c2  Rugosidad ext.\n(Å)": analysis.roughness_map(
            membrane, membrane.outer_leaflet, bins=bins
        ),
        "c3  Rugosidad int.\n(Å)": analysis.roughness_map(
            membrane, membrane.inner_leaflet, bins=bins
        ),
        "c4  Raft ext.\n[0,1]": analysis.raft_fraction_map(
            membrane, membrane.outer_leaflet, bins=bins
        ),
        "c5  Raft int.\n[0,1]": analysis.raft_fraction_map(
            membrane, membrane.inner_leaflet, bins=bins
        ),
        "c6  PIPs densidad\n(Da·Å⁻²)": analysis.pip_density_map(membrane, bins=bins),
        "c7  Asimetría\n(Da·Å⁻²)": (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins, sigma=2.0)
            - analysis.density_map(membrane, membrane.inner_leaflet, bins=bins, sigma=2.0)
        ),
        "c8  Sección XZ\n(Da·Å⁻²)": analysis.xz_projection(membrane, bx=bins * 2, bz=bins)[0],
        "c9  Orden S_CH\n[0,1]": analysis.order_parameter_map(membrane, bins=bins),
        "c10 Interdig.\n[0,1]": analysis.interdigitation_map(membrane, bins=bins),
        "c11 ED limpia\n(e·Å⁻³)": electron_density_projection(membrane, bins_xy=bins, sigma=0.8),
    }

    cmaps = {
        "c0": "gray",
        "c1": "viridis",
        "c2": "magma",
        "c3": "magma",
        "c4": "RdYlBu_r",
        "c5": "RdYlBu_r",
        "c6": "hot_r",
        "c7": "bwr",
        "c8": "gray",
        "c9": "RdYlGn",
        "c10": "YlOrRd",
        "c11": "gray",
    }

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        fig.suptitle(
            f"Campos de Entrenamiento | Simulacion = {seed}  " f"({bins}×{bins} px por campo)",
            fontsize=11,
            fontweight="bold",
            y=0.99,
        )
        plt.subplots_adjust(left=0.05, right=0.96, top=0.94, bottom=0.04, hspace=0.40, wspace=0.34)

        ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]
        for i, (ax, (title, arr)) in enumerate(zip(axes.ravel(), campos.items())):
            ch_key = f"c{i}"
            cmap = cmaps.get(ch_key, "viridis")

            vlo = np.percentile(arr, 2)
            vhi = np.percentile(arr, 98)
            if abs(vhi - vlo) < 1e-10:
                vlo, vhi = arr.min(), arr.max()
            im = ax.imshow(
                arr.T,
                origin="lower",
                extent=ext,
                cmap=cmap,
                vmin=vlo,
                vmax=vhi,
                aspect="equal",
                interpolation="bilinear",
            )
            # Colorbar más ancho para acomodar el label de unidades con claridad
            cb = plt.colorbar(im, ax=ax, shrink=0.88, pad=0.03, fraction=0.08)
            # Extraer unidades del título (segunda línea)
            _parts = title.split("\n")
            _unit_str = _parts[1].strip() if len(_parts) > 1 else ""
            if _unit_str:
                ax.text(
                    0.97,
                    0.06,
                    _unit_str,
                    transform=ax.transAxes,
                    fontsize=8,
                    fontweight="bold",
                    ha="right",
                    va="bottom",
                    color=C["line"],
                    bbox=dict(
                        boxstyle="round,pad=0.25",
                        facecolor="white",
                        alpha=0.90,
                        edgecolor=C["neutral"],
                        linewidth=0.6,
                    ),
                    zorder=10,
                )
                
            ax.set_title(title.split("\n")[0], fontsize=8.5, pad=3)
            ax.set_xlabel("x (nm)", fontsize=8.5)
            ax.set_ylabel("y (nm)", fontsize=8.5)
            ax.tick_params(labelsize=6)

        return _save(fig, f"R4_campos_entrenamiento_sim{seed:04d}", dpi, subdir="R4")


def plot_R5_cryoET(membrane: BicapaCryoET, dpi: int = 300) -> str:
    """Genera la figura R5: calidad de simulación cryo-ET (CTF, ruido, PSD)."""
    seed = membrane.seed
    bins = 90
    pixel_A = membrane.Lx / bins

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"Calidad de la simulación cryo-ET | Simulación = {seed}",
            fontsize=11,
            fontweight="bold",
            y=0.98,
        )

        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            left=0.06,
            right=0.97,
            top=0.92,
            bottom=0.08,
            hspace=0.35,
            wspace=0.36,
        )
        ax_clean = fig.add_subplot(gs[0, 0])
        ax_ctf = fig.add_subplot(gs[0, 1])
        ax_noisy = fig.add_subplot(gs[0, 2])
        ax_psd = fig.add_subplot(gs[1, 0])
        ax_curves = fig.add_subplot(gs[1, 1])
        ax_xz = fig.add_subplot(gs[1, 2])

        ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]

        proj_clean = electron_density_projection(membrane, bins_xy=bins, sigma=1.5)
        proj_ctf = apply_ctf_2d(
            proj_clean, pixel_size_angstrom=pixel_A, defocus_um=2.0, b_factor=200.0
        )
        rng = np.random.default_rng(seed)
        proj_noisy = add_noise(proj_ctf, snr=0.10, rng=rng)

        def _show_img(ax, img, title, label):
            _panel_label(ax, label)
            vlo, vhi = np.percentile(img, 1), np.percentile(img, 99)
            ax.imshow(
                img.T,
                origin="lower",
                extent=ext,
                cmap="gray",
                vmin=vlo,
                vmax=vhi,
                aspect="equal",
                interpolation="bilinear",
            )
            ax.set_title(title)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

        _show_img(ax_clean, proj_clean, "Imagen limpia (sin CTF, sin ruido)", "A")
        _show_img(ax_ctf, proj_ctf, "CTF aplicado (Df=2\u03bcm, Cs=2.7mm)", "B")
        _show_img(ax_noisy, proj_noisy, "CTF + ruido (SNR~0.10)", "C")

        _panel_label(ax_psd, "D")

        def _radial_psd(img):
            f = np.fft.fftshift(np.fft.fft2(img))
            psd = np.abs(f) ** 2
            ny, nx = psd.shape
            yc, xc = ny // 2, nx // 2
            r = np.sqrt(
                (np.arange(ny)[:, None] - yc) ** 2 + (np.arange(nx)[None, :] - xc) ** 2
            ).astype(int)
            r_max = min(yc, xc)
            radial = np.array([psd[r == i].mean() for i in range(r_max)])
            L_nm = bins * (pixel_A / 10.0)
            q = np.arange(r_max) / L_nm
            return q[1:], radial[1:]

        for img, lbl, col in [
            (proj_clean, "Limpia", C["lo"]),
            (proj_ctf, "CTF", C["ld"]),
            (proj_noisy, "Noisy", C["pip"]),
        ]:
            q_r, psd_r = _radial_psd(img)
            ax_psd.semilogy(q_r, psd_r, label=lbl, color=col, lw=1.3)

        ax_psd.set_xlabel("q (nm-1)")
        ax_psd.set_ylabel("PSD promediado radialmente")
        ax_psd.set_title("PSD radial promediado", pad=5)
        ax_psd.legend(fontsize=8)
        _despine(ax_psd)

        _panel_label(ax_curves, "E")
        from simulacion.ctf_sim import compute_ctf

        q_max_nm = 0.5 / (pixel_A / 10)
        q_nm = np.linspace(0.001, q_max_nm, 300)
        Fx = (q_nm * 0.1).reshape(-1, 1)
        Fy = np.zeros_like(Fx)
        defoci = [1.0, 2.0, 3.5]
        ctf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for df, col in zip(defoci, ctf_colors):
            ctf_1d = compute_ctf(Fx, Fy, defocus_um=df, b_factor=200.0).flatten()
            ax_curves.plot(q_nm, ctf_1d, lw=1.4, color=col, label=f"Δf = {df} μm")
        ax_curves.axhline(0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
        ax_curves.set_xlabel("Frecuencia espacial (nm-1)")
        ax_curves.set_ylabel("CTF")
        ax_curves.set_ylim(-1.1, 1.1)
        ax_curves.set_title("Curvas CTF", pad=5)
        ax_curves.legend(fontsize=8)
        _despine(ax_curves)

        _panel_label(ax_xz, "F")
        Hxz, xe, ze = analysis.xz_projection(membrane, bx=180, bz=90)
        ax_xz.imshow(
            Hxz.T,
            origin="lower",
            extent=[xe[0], xe[-1], ze[0], ze[-1]],
            cmap="gray",
            aspect="auto",
            interpolation="bilinear",
        )
        ax_xz.axhline(
            membrane.geometry.z_outer / 10,
            color=C["lo"],
            lw=0.9,
            ls="--",
            alpha=0.7,
            label="Cabezas ext.",
        )
        ax_xz.axhline(
            membrane.geometry.z_inner / 10,
            color=C["ld"],
            lw=0.9,
            ls="--",
            alpha=0.7,
            label="Cabezas int.",
        )
        ax_xz.set_xlabel("x (nm)")
        ax_xz.set_ylabel("z (nm)")
        ax_xz.set_title("Sección transversal XZ densidad electrónica", pad=5)
        ax_xz.legend(fontsize=8, loc="upper right")

        return _save(fig, f"R5_calidad_cryoET_sim{seed:04d}", dpi, subdir="R5")


def plot_R6_multisimulacion(stats: Dict, dpi: int = 300) -> str:
    """Genera la figura R6: comparativa multi-simulación con strip-plots."""
    records = stats.get("records", [])
    if len(records) < 2:
        print("  R6 requiere >=2 simulaciones, saltando.")
        return ""

    seeds = [r["seed"] for r in records]
    kc = np.array(stats.get("kc", []))
    thick = np.array(stats.get("thickness", []))
    sch_g = np.array(stats.get("sch_lo", []))
    sch_f = np.array(stats.get("sch_ld", []))
    val_s = np.array(stats.get("val_scores", [np.nan] * len(seeds)))
    N = len(seeds)

    val_s = np.nan_to_num(val_s, nan=0.0)
    if val_s.max() <= 1.0 and val_s.max() > 0:
        val_s = val_s * 100.0

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(18.0, 12.0))
        fig.patch.set_facecolor("white")

        gs = gridspec.GridSpec(
            2,
            3,
            figure=fig,
            hspace=0.20,
            wspace=0.58,
            top=0.92,
            bottom=0.06,
            left=0.07,
            right=0.97,
        )

        ax_A = fig.add_subplot(gs[0, 0])
        _plot_strip_r6(
            ax_A,
            kc,
            1.0,
            color=C["lo"],
            ref_lo=20,
            ref_hi=45,
            ref_label="Ref. [20-45]",
            ylabel="kc (kBT·nm-2)",
            title="",
        )
        ax_A.set_ylim(16, 48)
        _panel_label(ax_A, "A")

        ax_B = fig.add_subplot(gs[0, 1])
        _plot_strip_r6(
            ax_B,
            thick,
            1.0,
            color=C["ld"],
            ref_lo=35,
            ref_hi=50,
            ref_label="Ref. [35-50]",
            ylabel="D_PP (A)",
            title="",
        )
        ax_B.set_ylim(30, 54)
        _panel_label(ax_B, "B")

        ax_C = fig.add_subplot(gs[0, 2])
        for xp, vals, col, lbl, band in [
            (0.82, sch_g, C["lo"], "Lo", (0.85, 0.95)),
            (1.55, sch_f, C["ld"], "Ld", (0.60, 0.75)),
        ]:
            mu = vals.mean()
            sd = vals.std(ddof=1)
            sem = sd / np.sqrt(N)
            ax_C.axhspan(*band, color=C["grid"], alpha=0.4, zorder=0)
            ax_C.fill_between(
                [xp - 0.20, xp + 0.20], mu - sd, mu + sd, color=col, alpha=0.18, zorder=2
            )
            for sign in (+1, -1):
                ax_C.plot(
                    [xp - 0.20, xp + 0.20],
                    [mu + sign * sd] * 2,
                    color=col,
                    lw=1.0,
                    ls="--",
                    alpha=0.6,
                    zorder=3,
                )
            ax_C.plot([xp - 0.20, xp + 0.20], [mu, mu], color=col, lw=2.2, zorder=4)
            ax_C.errorbar(
                xp,
                mu,
                yerr=sem,
                fmt="none",
                color=C["line"],
                capsize=4,
                capthick=1.2,
                elinewidth=1.2,
                zorder=5,
            )
            jitter = np.linspace(-0.10, 0.10, N)
            for ji, vi in zip(jitter, vals):
                ax_C.scatter(
                    xp + ji, vi, s=28, color=col, edgecolors="white", linewidths=0.5, zorder=6
                )
            ax_C.text(xp, band[0] - 0.005, lbl, ha="center", va="top", fontsize=8.5, color=col)

        ax_C.set_xlim(0.42, 2.05)
        ax_C.set_xticks([0.82, 1.55])
        ax_C.set_xticklabels(["Lo", "Ld"], fontsize=9)
        ax_C.set_ylabel("S_CH", labelpad=4)
        ax_C.set_ylim(0.56, 0.98)
        ax_C.set_title("", fontweight="bold", pad=5, loc="left")
        _despine(ax_C)
        _panel_label(ax_C, "C")

        # Panel D: diagrama de barras de calidad biofísica
        ax_D = fig.add_subplot(gs[1, 0])
        cols_q = [C["pass"] if q >= 80 else C["fail"] for q in val_s]
        bars = ax_D.bar(
            range(1, N + 1),
            val_s,
            color=cols_q,
            width=0.55,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax_D.axhline(80, color=C["fail"], lw=1.2, ls="--", label="Umbral 80%", zorder=4)
        ax_D.set_ylim(0, 108)
        if N > 10:
            _step = 5
            _tick_idx = list(range(0, N, _step))
            _tick_pos_d = [1] + [i + 1 for i in range(4, N, 5)]
            if N not in _tick_pos_d:
                _tick_pos_d.append(N)
            _tick_lbl_d = [str(p) for p in _tick_pos_d]
        else:
            _tick_pos_d = list(range(1, N + 1))
            _tick_lbl_d = [str(i) for i in range(1, N + 1)]
        ax_D.set_xticks(_tick_pos_d)
        ax_D.set_xticklabels(_tick_lbl_d, fontsize=8.5, rotation=0, ha="center")
        ax_D.set_xlabel("Simulación", labelpad=4)
        ax_D.set_ylabel("Calidad biofísica (%)", labelpad=4)
        ax_D.set_yticks(range(0, 101, 10))  # major: 0, 10, 20 … 100
        ax_D.yaxis.set_minor_locator(plt.matplotlib.ticker.MultipleLocator(5))  # minor: every 5
        ax_D.tick_params(axis="y", which="minor", length=3, width=0.6, color=C["neutral"])
        ax_D.tick_params(axis="y", which="major", length=5, width=0.8)
        ax_D.set_title("", fontweight="bold", pad=5, loc="left")
        ax_D.legend(
            loc="lower right", frameon=True, framealpha=0.9, edgecolor=C["neutral"], fontsize=8.5
        )
        _light_grid(ax_D)
        _panel_label(ax_D, "D")

        ax_E = fig.add_subplot(gs[1, 1])
        sp_vals = {}
        for r in records:
            for sp, f in r.get("comp_outer", {}).items():
                sp_vals.setdefault(sp, []).append(f * 100)
        sp_sorted = sorted(sp_vals, key=lambda k: -np.mean(sp_vals[k]))
        means = [np.mean(sp_vals[sp]) for sp in sp_sorted]
        stds = [np.std(sp_vals[sp]) for sp in sp_sorted]
        sp_colors_def = {
            "POPC": "#3a86ff",
            "POPE": "#e63946",
            "POPS": "#fb8500",
            "SM": "#2dc653",
            "CHOL": "#adb5bd",
            "GM1": "#d4a017",
            "PI": "#9b5de5",
            "PIP2": "#c0392b",
            "PI4P": "#e67e22",
            "PI3P": "#f39c12",
            "PIP3": "#7b241c",
            "PlsPE": "#c0392b",
            "PI34P2": "#a04000",
            "PI5P": "#e74c3c",
        }
        cols_e = [sp_colors_def.get(sp, "#888888") for sp in sp_sorted]
        yp = np.arange(len(sp_sorted))
        ax_E.barh(
            yp,
            means,
            xerr=stds,
            color=cols_e,
            height=0.55,
            edgecolor="white",
            linewidth=0.7,
            error_kw=dict(ecolor=C["line"], capsize=3.5, capthick=0.9, elinewidth=0.9),
            zorder=3,
        )
        ax_E.set_yticks(yp)
        ax_E.set_yticklabels(sp_sorted, fontsize=8.5)
        ax_E.set_xlabel("Fracción molar (%) · media ± DE", labelpad=4)
        ax_E.set_xlim(0, 47)
        ax_E.set_title("", fontweight="bold", pad=5, loc="left")
        _light_grid(ax_E, axis="x")
        _panel_label(ax_E, "E")

        ax_F = fig.add_subplot(gs[1, 2])
        n_rafts = [r.get("n_rafts_outer", 0) + r.get("n_rafts_inner", 0) for r in records]
        n_pips = [r.get("n_pip_clusters", 0) for r in records]
        w = 0.32
        x = np.arange(1, N + 1)
        ax_F.bar(
            x - w / 2,
            n_rafts,
            width=w,
            label="Balsas (Lo)",
            color=C["lo"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        ax_F.bar(
            x + w / 2,
            n_pips,
            width=w,
            label="Clusters PIP",
            color=C["pip"],
            edgecolor="white",
            linewidth=0.7,
            zorder=3,
        )
        if N > 10:
            _tick_pos_f = [1] + [i + 1 for i in range(4, N, 5)]
            if N not in _tick_pos_f:
                _tick_pos_f.append(N)
            _tick_lbl_f = [str(p) for p in _tick_pos_f]
        else:
            _tick_pos_f = list(x)
            _tick_lbl_f = [str(i) for i in range(1, N + 1)]
        ax_F.set_xticks(_tick_pos_f)
        ax_F.set_xticklabels(_tick_lbl_f, fontsize=8.5, rotation=0, ha="center")
        ax_F.set_xlabel("Simulación", labelpad=4)
        ax_F.set_ylabel("N.º de estructuras", labelpad=4)
        ax_F.set_ylim(0, 4.5)
        ax_F.set_title("", fontweight="bold", pad=5, loc="left")
        ax_F.legend(
            loc="upper right", frameon=True, framealpha=0.9, edgecolor=C["neutral"], fontsize=8.5
        )
        _light_grid(ax_F)
        _panel_label(ax_F, "F")

        fig.suptitle(
            f"Comparativa multi-simulación | N = {N} simulaciones",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        return _save(fig, "R6_comparativa_multisimulacion", dpi, subdir="R6")


def plot_R6b_justificacion_N(stats: Dict, dpi: int = 300) -> str:
    """Genera la figura R6b: convergencia acumulada para justificar N (dinámico según simulaciones introducidas)."""
    records = stats.get("records", [])
    if len(records) < 2:
        return ""

    seeds = [r["seed"] for r in records]
    kc = np.array([r["kc"] for r in records])
    thick = np.array([r["thickness_mean_A"] for r in records])
    sch_g = np.array([r["sch_lo"] for r in records])
    sch_f = np.array([r["sch_ld"] for r in records])
    N = len(seeds)
    X = np.arange(1, N + 1)

    KC_CUM = _cumulative_stats(kc)
    DPP_CUM = _cumulative_stats(thick)
    SCHG_CUM = _cumulative_stats(sch_g)
    SCHF_CUM = _cumulative_stats(sch_f)

    val_scores = np.array([r.get("val_score", 0.0) for r in records])
    best_idx = int(np.argmax(val_scores))
    best_seed_num = seeds[best_idx]
    best_x = best_idx + 1

    X_F = X.astype(float)

    CV_COLOR = C["line"]

    PANELS_CFG = [
        dict(
            cum=KC_CUM,
            col_m=C["lo"],
            ylabel="kc (kBT\u00b7nm\u207b\u00b2)",
            title="",
            lbl="A",
            row=0,
            col=0,
        ),
        dict(cum=DPP_CUM, col_m=C["ld"], ylabel="D_PP (\u00c5)", title="", lbl="B", row=0, col=1),
        dict(cum=SCHG_CUM, col_m=C["lo"], ylabel="S_CH (Lo)", title="", lbl="C", row=1, col=0),
        dict(cum=SCHF_CUM, col_m=C["ld"], ylabel="S_CH (Ld)", title="", lbl="D", row=1, col=1),
    ]

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(16, 12))
        fig.patch.set_facecolor("white")

        gs = gridspec.GridSpec(
            2,
            2,
            figure=fig,
            hspace=0.42,
            wspace=0.64,
            top=0.93,
            bottom=0.07,
            left=0.08,
            right=0.92,
        )

        for p in PANELS_CFG:
            ax1 = fig.add_subplot(gs[p["row"], p["col"]])
            ax2 = ax1.twinx()

            m, sem, cv = p["cum"]

            if N >= 3:
                cs = CubicSpline(X_F, cv)
                xf = np.linspace(X_F[0], X_F[-1], 200)
                cvf = cs(xf)
            else:
                xf, cvf = X_F, cv

            ax2.scatter(X, cv, s=28, color=CV_COLOR, marker="s", zorder=5, clip_on=False)
            ax2.plot(xf, cvf, color=CV_COLOR, lw=1.4, ls="--", zorder=3)
            ax2.axhline(10.0, color=CV_COLOR, lw=0.7, ls=":", alpha=0.50, zorder=1)

            ylim_cv = max(20, float(cv.max()) * 1.30)
            ax2.set_ylim(0, ylim_cv)
            ax2.set_ylabel("CV (%)", color=CV_COLOR, labelpad=3, fontsize=11)
            ax2.tick_params(axis="y", labelcolor=CV_COLOR, labelsize=7, width=0.6, length=2.5)
            ax2.spines["right"].set_color(CV_COLOR)
            ax2.spines["right"].set_linewidth(0.7)
            ax2.spines["top"].set_visible(False)
            ax1.fill_between(
                X,
                m[-1] * 0.95,
                m[-1] * 1.05,
                color=p["col_m"],
                alpha=0.07,
                zorder=1,
                hatch="///",
                linewidth=0,
            )
            ax1.fill_between(X, m - sem, m + sem, color=p["col_m"], alpha=0.22, zorder=2)
            ax1.plot(
                X,
                m,
                color=p["col_m"],
                lw=2.0,
                marker="o",
                markersize=6,
                markerfacecolor="white",
                markeredgewidth=1.5,
                markeredgecolor=p["col_m"],
                zorder=4,
            )
            ax1.axvline(N, color=C["neutral"], lw=0.8, ls="--", zorder=1, clip_on=True)

            ax1.scatter(
                [best_x],
                [m[best_idx]],
                s=120,
                marker="*",
                color="#d4ac0d",
                edgecolors="#a07800",
                linewidths=0.7,
                zorder=7,
                clip_on=False,
                label="_nolegend_",
            )

            ax1.spines["top"].set_visible(False)
            ax1.set_ylabel(p["ylabel"], labelpad=5)
            # X-axis: always N=1, then N=5, 10, 15 … up to N
            _xt_base = np.arange(5, N + 1, 5)
            _xt = np.concatenate([[1], _xt_base]).astype(int)
            # If N itself not already included, append it
            if N not in _xt:
                _xt = np.concatenate([_xt, [N]])
            ax1.set_xticks(_xt)
            ax1.set_xticklabels(
                [f"N={i}" for i in _xt],
                fontsize=8.5,
                rotation=0 if len(_xt) <= 8 else 45,
                ha="center" if len(_xt) <= 8 else "right",
            )
            ax1.set_xlim(0.5, N + 0.5)

            _fm = m[-1]
            _fsem = sem[-1]
            _fcv = cv[-1]
            _fmt_m = f"{_fm:.4g}"
            _fmt_sem = f"{_fsem:.4f}" if _fsem < 0.01 else f"{_fsem:.3f}"
            _stats_txt = f"μ = {_fmt_m}\n" f"±{_fmt_sem} SEM\n" f"CV = {_fcv:.2f} %"
            ax1.text(
                0.97,
                0.97,
                _stats_txt,
                transform=ax1.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                color=C["line"],
                bbox=dict(
                    boxstyle="round,pad=0.35", fc="white", ec=C["neutral"], lw=0.7, alpha=0.88
                ),
                zorder=8,
            )
            ax1.set_xlabel("Simulaciones acumuladas", labelpad=4, fontsize=12)
            ax1.set_title(p["title"], fontweight="bold", pad=6, loc="left", fontsize=14)
            _panel_label(ax1, p["lbl"], x=-0.13, y=1.06)
            _light_grid(ax1)
            if "S_CH (Lo)" in p["ylabel"] or "S_CH (Ld)" in p["ylabel"]:
                rng = float(m.max() - m.min())
                margin = max(0.004, rng * 2.0)
                ax1.set_ylim(float(m.min()) - margin, float(m.max()) + margin)
            elif "S_CH" in p["title"]:
                rng = float(m.max() - m.min())
                margin = max(0.003, rng * 2.0)
                ax1.set_ylim(float(m[-1]) - margin, float(m[-1]) + margin)

            leg_h = [
                Line2D(
                    [0],
                    [0],
                    color=p["col_m"],
                    lw=2.0,
                    marker="o",
                    markersize=5,
                    markerfacecolor="white",
                    markeredgewidth=1.5,
                    markeredgecolor=p["col_m"],
                    label="Media acumulada",
                ),
                mpatches.Patch(color=p["col_m"], alpha=0.22, label="Banda \u00b1SEM"),
                Line2D(
                    [0],
                    [0],
                    color=CV_COLOR,
                    lw=1.4,
                    ls="--",
                    marker="s",
                    markersize=4,
                    label="CV (%) interpolado",
                ),
                Line2D(
                    [0], [0], color=CV_COLOR, lw=0.7, ls=":", alpha=0.6, label="Umbral CV < 10 %"
                ),
                Line2D(
                    [0],
                    [0],
                    linestyle="none",
                    marker="*",
                    markersize=9,
                    color="#d4ac0d",
                    markeredgecolor="#a07800",
                    markeredgewidth=0.7,
                    label=f"Mejor score (Seed {best_seed_num})",
                ),
            ]
            ax1.legend(
                handles=leg_h,
                loc="upper left",
                frameon=True,
                framealpha=0.93,
                edgecolor=C["neutral"],
                fontsize=10,
                handlelength=2.0,
                borderpad=0.6,
            )

        fig.suptitle(
            f"Análisis de convergencia y reproducibilidad  |  Justificación de N = {N}",
            fontsize=14,
            fontweight="bold",
            y=0.99,
        )

        fig.text(
            0.5,
            0.01,
            (
                f"CV = \u03c3/\u03bc \u00d7 100.  Banda sombreada = \u00b1SEM acumulado.  "
                f"\u2605 Mejor score global: Seed {best_seed_num} (N={best_x})."
            ),
            ha="center",
            va="bottom",
            fontsize=9,
            style="italic",
            color=C["neutral"],
        )

        return _save(fig, "R6b_justificacion_N", dpi, subdir="R6")


def generate_anexo_pdf(stats: Dict, output_path: str = "", dpi: int = 150) -> str:
    """
    Genera un PDF de anexo ligero con resumen tabular de las simulaciones.

    Incluye:
      - Tabla de parámetros biofísicos por simulación (kc, D_PP, S_CH Lo/Ld,
        calidad, N rafts, N PIP clusters).
      - Medias y desviaciones estándar del conjunto.
      - Nota de la seed con mejor score global.

    El PDF está diseñado para ser compacto (1 página A4 apaisada) y no contiene
    figuras de análisis, sólo datos tabulados como anexo del TFM.
    """
    records = stats.get("records", [])
    if not records:
        print("  generate_anexo_pdf: no hay registros, saltando.")
        return ""

    if not output_path:
        output_path = os.path.join(_results_dir(), "R6", "pdf", "Anexo_1_resumen_simulaciones.pdf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    seeds = [r["seed"] for r in records]
    kc_vals = [r["kc"] for r in records]
    dpp_vals = [r["thickness_mean_A"] for r in records]
    schlo = [r["sch_lo"] for r in records]
    schld = [r["sch_ld"] for r in records]
    qual = [r.get("val_score", 0.0) for r in records]
    n_rafts = [r.get("n_rafts_outer", 0) + r.get("n_rafts_inner", 0) for r in records]
    n_pips = [r.get("n_pip_clusters", 0) for r in records]

    # Escalar quality 0-1 → 0-100 si necesario
    qual_arr = np.array(qual, dtype=float)
    if qual_arr.max() <= 1.0 and qual_arr.max() > 0:
        qual_arr *= 100.0
    best_idx = int(np.argmax(qual_arr))

    col_headers = [
        "Sim",
        "Seed",
        "kc\n(kBT·nm⁻²)",
        "D_PP\n(Å)",
        "S_CH\n(Lo)",
        "S_CH\n(Ld)",
        "Calidad\n(%)",
        "N rafts",
        "N PIP\nclusters",
    ]
    rows = []
    for i, r in enumerate(records):
        rows.append(
            [
                f"sim{i + 1}",
                str(seeds[i]),
                f"{kc_vals[i]:.2f}",
                f"{dpp_vals[i]:.2f}",
                f"{schlo[i]:.4f}",
                f"{schld[i]:.4f}",
                f"{qual_arr[i]:.1f}",
                str(n_rafts[i]),
                str(n_pips[i]),
            ]
        )

    # Fila de estadísticos resumen
    def _fmt_stat(arr):
        a = np.array(arr, dtype=float)
        return f"{a.mean():.3g} ± {a.std(ddof=1):.3g}"

    summary_row = [
        "Media ± DE",
        "—",
        _fmt_stat(kc_vals),
        _fmt_stat(dpp_vals),
        _fmt_stat(schlo),
        _fmt_stat(schld),
        _fmt_stat(qual_arr),
        _fmt_stat(n_rafts),
        _fmt_stat(n_pips),
    ]

    N = len(records)
    n_cols = len(col_headers)

    with plt.rc_context(PUB_RC):
        # Altura dinámica: ~0.32 cm por fila + cabecera + márgenes
        row_h = 0.30
        fig_h = max(5.0, (N + 3) * row_h + 2.5)
        fig = plt.figure(figsize=(16.0, fig_h))
        fig.patch.set_facecolor("white")

        # Título
        fig.text(
            0.5,
            0.97,
            "Anexo 1 – Resumen de simulaciones",
            ha="center",
            va="top",
            fontsize=13,
            fontweight="bold",
            color=C["line"],
        )
        fig.text(
            0.5,
            0.92,
            (
                f"N = {N} simulaciones independientes · parámetros biofísicos finales · "
                f"Valores extraídos de la corrida completa (seed aleatoria por simulación)"
            ),
            ha="center",
            va="top",
            fontsize=9,
            color=C["neutral"],
            style="italic",
        )

        # Eje invisible para la tabla
        ax = fig.add_axes([0.02, 0.08, 0.96, 0.80])
        ax.axis("off")

        all_rows = rows + [summary_row]
        tbl = ax.table(
            cellText=all_rows,
            colLabels=col_headers,
            loc="upper center",
            cellLoc="center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(8.5)
        tbl.scale(1.0, 1.55)
        tbl.auto_set_column_width(list(range(n_cols)))

        # Estilo cabecera
        HEADER_BG = "#2c2c2c"
        for j in range(n_cols):
            cell = tbl[0, j]
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color="white", fontweight="bold", fontsize=8)

        # Estilo filas alternadas
        for i in range(1, N + 1):
            bg = "#f7f7f7" if i % 2 == 0 else "white"
            # Highlight seed con mejor score
            if i - 1 == best_idx:
                bg = "#fffbe6"
            for j in range(n_cols):
                cell = tbl[i, j]
                cell.set_facecolor(bg)
                if i - 1 == best_idx:
                    cell.set_text_props(fontweight="bold")
                # Calidad: colorear valor
                if j == 6:
                    q = qual_arr[i - 1]
                    cell.set_text_props(
                        color=C["pass"] if q >= 80 else C["fail"], fontweight="bold"
                    )

        # Fila de resumen (última)
        for j in range(n_cols):
            cell = tbl[N + 1, j]
            cell.set_facecolor("#dde8f0")
            cell.set_text_props(fontweight="bold", fontsize=8)

        # Nota al pie
        fig.text(
            0.5,
            0.025,
            (
                f"★  Seed con mejor score global: Seed {seeds[best_idx]}  "
                f"(sim{best_idx + 1}, calidad = {qual_arr[best_idx]:.1f} %).  "
                "kc = módulo de bending; D_PP = grosor cabeza-cabeza; "
                "S_CH = parámetro de orden acil; Lo = fase ordenada; Ld = fase desordenada."
            ),
            ha="center",
            va="bottom",
            fontsize=8,
            color=C["neutral"],
            style="italic",
        )

        fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"  -> {os.path.relpath(output_path)}")
    return output_path


def main():
    """Punto de entrada CLI para generar todas las figuras de resultados."""
    parser = argparse.ArgumentParser(
        description="Figuras de resultados para TFM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python run_results.py --sims 1
  python run_results.py --sims 1 2 3 4 5
  python run_results.py --sims 1 --only R1 R3 R4
  python run_results.py --sims 1 --dpi 300 --size 50 50
        """,
    )
    parser.add_argument(
        "--sims", type=int, nargs="+", required=True, metavar="N", help="Simulaciones a procesar"
    )
    parser.add_argument(
        "--size",
        type=float,
        nargs=2,
        default=[50.0, 50.0],
        metavar=("X", "Y"),
        help="Tamaño en nm (default: 50 50)",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        metavar="Rn",
        help="Sólo generar estas secciones (R1 R2 R3 R4 R5 R6)",
    )
    parser.add_argument(
        "--dpi", type=int, default=300, help="Resolución de salida en DPI (default: 300)"
    )
    args = parser.parse_args()

    only = set(args.only) if args.only else {"R1", "R2", "R3", "R4", "R5", "R6"}
    size_nm = tuple(args.size)

    print(
        f"\nrun_results.py — {len(args.sims)} simulación(es) | "
        f"{size_nm[0]:.0f}×{size_nm[1]:.0f} nm | DPI={args.dpi}"
    )
    print(f"Secciones: {', '.join(sorted(only))}")
    print(f"Salida: {RESULTS_DIR}\n")

    stats_records = []

    for seed in args.sims:
        print(f"->Simulación = {seed}")
        b = BicapaCryoET(size_nm=size_nm, seed=seed)
        b.build()

        results = None
        if "R2" in only or "R6" in only:
            print("  Calculando benchmarks...")
            results = run_all_benchmarks(b)

        if "R1" in only:
            plot_R1_caracterizacion(b, dpi=args.dpi)
        if "R2" in only:
            plot_R2_validacion(b, results=results, dpi=args.dpi)
        if "R3" in only:
            plot_R3_organizacion(b, dpi=args.dpi)
        if "R4" in only:
            plot_R4_campos(b, dpi=args.dpi)
        if "R5" in only:
            plot_R5_cryoET(b, dpi=args.dpi)

        if "R6" in only:
            sup = b.outer_leaflet
            inf = b.inner_leaflet
            sup_xy = np.array([[l.head_pos[0], l.head_pos[1]] for l in sup])
            inf_xy = np.array([[l.head_pos[0], l.head_pos[1]] for l in inf])
            sup_z = np.array([l.head_pos[2] for l in sup])
            inf_z = np.array([l.head_pos[2] for l in inf])
            tree = KDTree(inf_xy)
            _, idxs = tree.query(sup_xy, k=1)
            paired = sup_z - inf_z[idxs]
            thickness_mean = float(paired.mean())

            sch = b.get_sch_by_domain()

            val_score = 0.0
            if results and "summary" in results:
                val_score = results["summary"].get("accuracy_pct", 0.0)

            rec = {
                "seed": seed,
                "kc": b.bending_modulus,
                "sigma": b.surface_tension,
                "thickness_mean_A": thickness_mean,
                "thickness_total_A": float(b.geometry.total_thick),
                "n_rafts_outer": len(b.rafts_outer),
                "n_rafts_inner": len(b.rafts_inner),
                "n_pip_clusters": len(b.pip_clusters),
                "sch_lo": sch["lo"],
                "sch_ld": sch["ld"],
                "sch_chol": sch["chol"],
                "comp_outer": dict(b.comp_outer),
                "val_score": val_score,
            }
            stats_records.append(rec)

    if "R6" in only and len(stats_records) >= 2:
        print("\n->R6 multi-simulación")
        stats = {
            "records": stats_records,
            "kc": [r["kc"] for r in stats_records],
            "thickness": [r["thickness_mean_A"] for r in stats_records],
            "sch_lo": [r["sch_lo"] for r in stats_records],
            "sch_ld": [r["sch_ld"] for r in stats_records],
            "val_scores": [r.get("val_score", 0.0) for r in stats_records],
        }
        plot_R6_multisimulacion(stats, dpi=args.dpi)
        plot_R6b_justificacion_N(stats, dpi=args.dpi)
        generate_anexo_pdf(stats, dpi=args.dpi)
    elif "R6" in only:
        print("  R6 requiere >=2 simulaciones.")
    print(f"\n[OK] Listo. Resultados en: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
