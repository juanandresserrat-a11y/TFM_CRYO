"""
results.py
Generación de figuras de resultados para TFM / artículo científico.

  R1  Caracterización biofísica de la bicapa
      Composición lipídica, parámetros elásticos, perfil de densidad electrónica
      y distribución de grosor.

  R2  Validación cuantitativa contra la literatura
      Los seis benchmarks biofísicos representados como tabla + barras de
      porcentaje de éxito, con los rangos de referencia anotados.

  R3  Organización lateral: dominios Lo/Ld y clusters de PIPs
      Mapa 2D de fase + scatter de PIPs + función de correlación radial -
      muestra la segregación lipídica y la co-localización de señalización.

  R4  Comparativa multi-semilla
      Violinplots de kc, grosor, S_CH y score de validación sobre N semillas -
      demuestra la diversidad y reproducibilidad del dataset.

  R5  Galería de canales de training
      Los 12 canales .npy en una cuadrícula 3×4, normalizados para comparación
      visual.

  R6  Calidad de la simulación cryo-ET
      Comparación imagen limpia / CTF / ruido + espectro de potencia
      radialmente promediado.

Uso rápido:
    python results.py --sims 1             # todas las figuras, semilla 1
    python results.py --sims 1 2 3 4 5     # R4 multi-semilla incluida
    python results.py --sims 1 --only R1 R3 R5   # sólo esas secciones
    python results.py --sims 1 --dpi 300   # resolución de publicación

"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import BicapaCryoET, OUTPUT_DIR
import analysis
from validation import run_all_benchmarks
from electron_density import electron_density_profile, electron_density_projection
from ctf_sim import apply_ctf_2d, add_noise

# Estilo global

RESULTS_DIR = os.path.join(OUTPUT_DIR, "resultados")

C = {
    "lo":      "#b89a68",   # Lo / gel
    "ld":      "#a8c8e0",   # Ld / fluido
    "pip":     "#c0392b",   # PIPs
    "chol":    "#adb5bd",   # Colesterol
    "line":    "#2c2c2c",   # Líneas y bordes
    "grid":    "#e8e8e8",   # Grids suaves
    "pass":    "#2dc653",   # Benchmark pass
    "close":   "#f4a261",   # Benchmark close
    "fail":    "#e63946",   # Benchmark fail
    "neutral": "#6c757d",   # Texto secundario
    "bg":      "#fafafa",   # Fondo de paneles
}

PUB_RC = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman"],
    "mathtext.default": "regular",
    "text.usetex":      False,
    "font.size":        9,
    "axes.titlesize":   9.5,
    "axes.labelsize":   8.5,
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "axes.linewidth":   0.7,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.major.width":0.6,
    "ytick.major.width":0.6,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "legend.fontsize":  7.5,
    "legend.framealpha":0.92,
    "legend.edgecolor": "#cccccc",
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.facecolor":"white",
    "pdf.fonttype":     42,   # fuentes embebidas en PDF
    "ps.fonttype":      42,
}

def _results_dir() -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR

def _save(fig: plt.Figure, name: str, dpi: int = 300, subdir: str = "") -> str:
    if subdir:
        d = os.path.join(_results_dir(), subdir)
        os.makedirs(d, exist_ok=True)
    else:
        d = _results_dir()
    path = os.path.join(d, name + ".pdf")
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  → {os.path.relpath(path)}")
    return path

def _panel_label(ax, letter: str, x: float = -0.12, y: float = 1.06):
    """Etiqueta de panel estilo artículo: A, B, C… en negrita."""
    ax.text(x, y, letter, transform=ax.transAxes,
            fontsize=11, fontweight="bold", va="top", ha="left",
            color=C["line"])

def _despine(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# Funciones Resultados

def plot_R1_caracterizacion(membrane: BicapaCryoET, dpi: int = 300) -> str:
    seed = membrane.seed
    g = membrane.geometry
    todos = membrane.outer_leaflet + membrane.inner_leaflet

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(14, 9))
        fig.suptitle(
            f"R1 — Caracterización biofísica · seed = {seed} · "
            f"{membrane.Lx/10:.0f}×{membrane.Ly/10:.0f} nm",
            fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.08, right=0.96,
                               top=0.92, bottom=0.08,
                               hspace=0.48, wspace=0.40)

        ax_comp  = fig.add_subplot(gs[0, :2])   # A - composición (ancho)
        ax_param = fig.add_subplot(gs[0, 2])    # B - parámetros
        ax_ed    = fig.add_subplot(gs[1, :2])   # C - perfil ED
        ax_thick = fig.add_subplot(gs[1, 2])    # D - grosor

        _panel_label(ax_comp, "A")
        comp_data = {
            "Externa": membrane.comp_outer,
            "Interna": membrane.comp_inner,
        }
        all_species = sorted(
            set(list(membrane.comp_outer.keys()) + list(membrane.comp_inner.keys()))
        )

        sp_colors = {
            "POPC":   "#3a86ff", "POPE":   "#e63946", "PlsPE":  "#c0392b",
            "POPS":   "#fb8500", "SM":     "#2dc653", "CHOL":   "#adb5bd",
            "GM1":    "#d4a017", "PI":     "#9b5de5", "PI3P":   "#f39c12",
            "PI4P":   "#e67e22", "PI5P":   "#e74c3c", "PI34P2": "#a04000",
            "PIP2":   "#c0392b", "PIP3":   "#7b241c",
        }
        leaflets = list(comp_data.keys())
        y_pos = np.arange(len(leaflets))
        left = np.zeros(len(leaflets))
        handles = []
        for sp in all_species:
            vals = np.array([comp_data[lf].get(sp, 0) * 100 for lf in leaflets])
            col = sp_colors.get(sp, "#888888")
            bars = ax_comp.barh(y_pos, vals, left=left, color=col,
                                height=0.52, edgecolor="white", linewidth=0.4)
            if any(v > 0 for v in vals):
                handles.append(mpatches.Patch(color=col, label=sp))
                for i, (v, lf) in enumerate(zip(vals, leaflets)):
                    if v > 2.5:
                        ax_comp.text(left[i] + v / 2, y_pos[i],
                                     f"{v:.0f}%", ha="center", va="center",
                                     fontsize=6.5, color="white", fontweight="bold")
            left += vals

        ax_comp.set_yticks(y_pos)
        ax_comp.set_yticklabels(leaflets, fontsize=8)
        ax_comp.set_xlabel("Fracción molar (%)")
        ax_comp.set_xlim(0, 102)
        ax_comp.set_title("A  Composición lipídica por monocapa")
        ax_comp.legend(handles=handles, loc="lower right", ncol=4,
                       fontsize=6.5, frameon=True)
        ax_comp.axvline(100, color=C["neutral"], lw=0.6, ls="--", alpha=0.4)
        _despine(ax_comp)

        _panel_label(ax_param, "B")
        params = {
            r"kc (k_BT · nm^{2})": {
                "val": membrane.bending_modulus,
                "ref_lo": 20, "ref_hi": 45,
                "unit": "k_BT · nm^{2}", "color": C["lo"]
            },
            r"sigma (k_BT · nm^{-2})": {
                "val": membrane.surface_tension * 1000,
                "ref_lo": 0.1, "ref_hi": 5.0,
                "unit": "x 10^{-3} k_BT * nm^{-2}", "color": C["ld"]
            },
            r"D_PP (Å)": {
                "val": g.total_thick,
                "ref_lo": 35, "ref_hi": 50,
                "unit": "Å", "color": "#fb8500"
            },
        }
        for i, (label, p) in enumerate(params.items()):
            v, lo, hi = p["val"], p["ref_lo"], p["ref_hi"]
            rng = hi - lo
            frac = np.clip((v - lo) / rng, 0, 1)
            ax_param.barh(i, frac, color=p["color"], height=0.55,
                          edgecolor="white", linewidth=0.4, alpha=0.85)
            ax_param.barh(i, 1.0, color=C["grid"], height=0.55,
                          edgecolor=C["neutral"], linewidth=0.4,
                          alpha=0.3, zorder=0)
            status = "[OK]" if lo <= v <= hi else "[!]"
            ax_param.text(1.03, i, f"{status} {v:.1f} {p['unit']}",
                          va="center", fontsize=7, color=C["line"])

        ax_param.set_yticks(range(len(params)))
        ax_param.set_yticklabels(list(params.keys()), fontsize=7.5)
        ax_param.set_xlim(0, 1.0)
        ax_param.set_xticks([0, 0.5, 1.0])
        ax_param.set_xticklabels(["Mín ref.", "50%", "Máx ref."], fontsize=6.5)
        ax_param.set_title("B  Parámetros elásticos")
        _despine(ax_param)

        _panel_label(ax_ed, "C")
        z_cent, ed_prof = electron_density_profile(membrane, bins_z=300)
        z_nm = z_cent / 10.0
        ax_ed.plot(z_nm, ed_prof, color=C["lo"], lw=1.6, label="ED media")
        ax_ed.fill_between(z_nm, 0.334, ed_prof,
                           where=(ed_prof > 0.334),
                           color=C["lo"], alpha=0.22, label="Región densa")
        ax_ed.fill_between(z_nm, ed_prof, 0.334,
                           where=(ed_prof < 0.334),
                           color=C["ld"], alpha=0.22, label="Región diluida")
        ax_ed.axhline(0.334, color=C["neutral"], lw=0.9, ls="--",
                      label="Agua bulk (0.334 e * A^-3)")
        ax_ed.axvline(g.z_outer / 10, color=C["lo"], lw=0.8, ls=":", alpha=0.7)
        ax_ed.axvline(g.z_inner / 10, color=C["lo"], lw=0.8, ls=":", alpha=0.7)
        ax_ed.set_xlabel("Posición axial Z (nm)")
        ax_ed.set_ylabel("Densidad electrónica (e * A^-3)")
        ax_ed.set_title("C  Perfil de densidad electrónica — patrón dark-bright-dark")
        ax_ed.legend(fontsize=7, loc="upper right")
        _despine(ax_ed)

        _panel_label(ax_thick, "D")
        T = analysis.thickness_map(membrane, bins=90)
        T_flat = T[T > 0].ravel()

        R = analysis.raft_fraction_map(membrane,
                                       membrane.outer_leaflet + membrane.inner_leaflet,
                                       bins=90)
        T_lo = T[(R > 0.5) & (T > 0)].ravel()
        T_ld = T[(R < 0.5) & (T > 0)].ravel()

        bins_h = np.linspace(T_flat.min() * 0.97, T_flat.max() * 1.03, 35)
        ax_thick.hist(T_flat, bins=bins_h, density=True,
                      color=C["neutral"], alpha=0.35, label="Global")
        if len(T_lo) > 5:
            ax_thick.hist(T_lo, bins=bins_h, density=True,
                          color=C["lo"], alpha=0.65, label=f"Lo  ({T_lo.mean():.1f} Å)")
        if len(T_ld) > 5:
            ax_thick.hist(T_ld, bins=bins_h, density=True,
                          color=C["ld"], alpha=0.65, label=f"Ld  ({T_ld.mean():.1f} Å)")
        ax_thick.axvline(T_flat.mean(), color=C["line"], lw=1.0, ls="--",
                         label=f"Media {T_flat.mean():.1f} Å")
        ax_thick.set_xlabel("Grosor D_PP (Å)")
        ax_thick.set_ylabel("Densidad de prob. (A^-1)")
        ax_thick.set_title("D  Distribución de grosor Lo / Ld")
        ax_thick.legend(fontsize=7)
        _despine(ax_thick)

        return _save(fig, f"R1_caracterizacion_seed{seed:04d}", dpi, subdir="R1")


def plot_R2_validacion(membrane: BicapaCryoET,
                       results: Optional[Dict] = None,
                       dpi: int = 300) -> str:
    seed = membrane.seed
    if results is None:
        print("  Calculando benchmarks...")
        results = run_all_benchmarks(membrane)

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15, 10))
        summary = results.get("summary", {"score": 0, "passed": 0, "total": 6})
        accuracy_pct = summary.get('accuracy_pct', summary['score']*100)
        fig.suptitle(
            f"R2 — Validación biofísica · seed = {seed}  "
            f"[Accuracy media: {accuracy_pct:.1f}% "
            f"· {summary['passed']}/{summary['total']} benchmarks]",
            fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(3, 3, figure=fig,
                               left=0.07, right=0.96,
                               top=0.88, bottom=0.07,
                               hspace=0.55, wspace=0.42)

        ax_tbl  = fig.add_subplot(gs[0, :2])
        ax_bar  = fig.add_subplot(gs[0, 2])
        ax_helf = fig.add_subplot(gs[1, 0])
        ax_sch  = fig.add_subplot(gs[1, 1])
        ax_ed   = fig.add_subplot(gs[1, 2])
        ax_acf  = fig.add_subplot(gs[2, 0])
        ax_thick2 = fig.add_subplot(gs[2, 1])
        ax_inter  = fig.add_subplot(gs[2, 2])

        _panel_label(ax_tbl, "A")

        rows = [
            ("Helfrich pendiente",  "slope_high_q", "helfrich",
             "−4 ± 0.3", "accuracy"),
            ("Grosor D_PP (Å)",     "mean_nm",       "thickness",
             "35–50", "accuracy_diff"),
            ("ΔD Lo−Ld (Å)",        "diff_A",        "thickness",
             "3–6", "accuracy_diff"),
            ("S_CH gel",           "gel_mean",      "order",
             "0.85–0.95", "accuracy_gel"),
            ("S_CH fluido",        "fluid_mean",    "order",
             "0.60–0.75", "accuracy_fluid"),
            ("Long. corr. Lo (nm)","xi_nm",         "raft_corr",
             "5–25", "accuracy"),
            ("Interdig. Lo > Ld",  "lo_gt_ld",      "interdig",
             "True", "accuracy"),
            ("ED cabeza (e * A^-3)","ed_head_peak",   "electron_ed",
             "0.44–0.50", "accuracy_head"),
            ("ED cola (e * A^-3)", "ed_tail",         "electron_ed",
             "0.28–0.31", "accuracy_tail"),
        ]

        ax_tbl.axis("off")
        col_labels = ["Parámetro", "Valor medido", "Ref. bibliográfica", ""]
        col_widths = [0.36, 0.24, 0.28, 0.09]
        x_positions = [0.02]
        for w in col_widths[:-1]:
            x_positions.append(x_positions[-1] + w)

        for j, (lbl, x) in enumerate(zip(col_labels, x_positions)):
            ax_tbl.text(x, 1.18, lbl, transform=ax_tbl.transAxes,
                        fontsize=8, fontweight="bold", va="top",
                        color=C["line"])

        ax_tbl.plot([0, 1], [1.08, 1.08], color=C["line"], lw=0.8,
                    transform=ax_tbl.transAxes, clip_on=False)

        n = len(rows)
        for i, (label, key, bench, ref, pass_key) in enumerate(rows):
            y = 0.90 - (i + 1) * (0.90 / (n + 1))

            # Obtener resultados del benchmark específico
            bench_results = results.get(bench, {})
            val = bench_results.get(key, "—")

            # Evaluar el estado en tres niveles (PASS / CLOSE / FAIL)
            passed_val = bench_results.get(pass_key, False)
            if isinstance(passed_val, (int, float)):
                acc_norm = passed_val if passed_val <= 1.0 else passed_val / 100.0
            else:
                acc_norm = 1.0 if bool(passed_val) else 0.0

            if acc_norm >= 0.70:
                tier, status_col, bg_col = "PASS",  C["pass"],  "#f6fff8"
            elif acc_norm >= 0.40:
                tier, status_col, bg_col = "CLOSE", C["close"], "#fff8f0"
            else:
                tier, status_col, bg_col = "FAIL",  C["fail"],  "#fff5f5"

            # Formatear valor
            if isinstance(val, float):
                val_str = f"{val:.3f}"
            elif isinstance(val, bool):
                val_str = "Sí" if val else "No"
            else:
                val_str = str(val)

            ax_tbl.axhspan(y - 0.04, y + 0.04, color=bg_col, alpha=0.7,
                           transform=ax_tbl.transAxes)
            ax_tbl.text(x_positions[0] + 0.01, y, label, transform=ax_tbl.transAxes,
                        fontsize=7.5, va="center")
            ax_tbl.text(x_positions[1], y, val_str, transform=ax_tbl.transAxes,
                        fontsize=7.5, va="center", color=C["line"], ha="center")
            ax_tbl.text(x_positions[2], y, ref, transform=ax_tbl.transAxes,
                        fontsize=7, va="center", color=C["neutral"], ha="center")
            ax_tbl.text(x_positions[3], y, tier,
                        transform=ax_tbl.transAxes, fontsize=9,
                        va="center", ha="center", color=status_col, fontweight="bold")

        ax_tbl.set_xlim(0, 1)
        ax_tbl.set_ylim(0, 1.05)

        _panel_label(ax_bar, "B")
        bench_names = ["Helfrich", "Grosor", "Orden\nS_CH",
                       "Corr.\nLo", "Interdig.", "Densidad\nelec."]
        bench_keys  = ["helfrich", "thickness", "order",
                       "raft_corr", "interdig", "electron_ed"]
        scores = []
        for bk in bench_keys:
            b = results.get(bk, {})
            # Usar accuracy combinada directamente cuando existe (evita diluir
            # con sub-scores individuales, igual que en validation.py)
            if "accuracy" in b and isinstance(b["accuracy"], (int, float)):
                scores.append(b["accuracy"] / 100.0)
            else:
                accs = [v for k, v in b.items()
                        if k.startswith("accuracy") and isinstance(v, (int, float))]
                if accs:
                    scores.append(np.mean(accs) / 100.0)
                elif "pass" in b and isinstance(b["pass"], bool):
                    scores.append(1.0 if b["pass"] else 0.0)
                else:
                    scores.append(0.0)

        def _bar_col(s):
            if s >= 0.70: return C["pass"]
            if s >= 0.40: return C["close"]
            return C["fail"]
        bar_cols = [_bar_col(s) for s in scores]
        ax_bar.barh(range(len(scores)), scores, color=bar_cols,
                    height=0.6, edgecolor="white", linewidth=0.5)
        ax_bar.axvline(1.0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
        ax_bar.set_yticks(range(len(bench_names)))
        ax_bar.set_yticklabels(bench_names, fontsize=7)
        ax_bar.set_xlim(0, 1.2)
        ax_bar.set_xlabel("Fraccion superada")
        ax_bar.set_title("B  Score por benchmark")
        for i, s in enumerate(scores):
            ax_bar.text(s + 0.03, i, f"{s*100:.0f}%", va="center", fontsize=7)
        _despine(ax_bar)

        _panel_label(ax_helf, "C")
        h = results.get("helfrich", {})
        q_c = h.get("q_centers", [])
        p_m = h.get("p_mean", [])
        if len(q_c) > 3 and len(p_m) > 3:
            q_arr = np.asarray(q_c)
            p_arr = np.asarray(p_m)
            mask = (q_arr > 0) & (p_arr > 0)
            ax_helf.loglog(q_arr[mask], p_arr[mask],
                           "o", ms=3.5, color=C["lo"], alpha=0.8,
                           label="Simulación")

            idx_hq = q_arr > q_arr.mean()
            if idx_hq.sum() > 2:
                q_fit = q_arr[idx_hq & mask]
                A = np.median(p_arr[idx_hq & mask] * q_fit**4)
                ax_helf.loglog(q_fit, A / q_fit**4, "--",
                               color=C["fail"], lw=1.2, label="q^{-4} Helfrich")
        ax_helf.set_xlabel("q (nm^-1)")
        ax_helf.set_ylabel(r"<|hq|^2> (nm^2)")
        ax_helf.set_title("C  Espectro de fluctuaciones\nHelfrich")
        ax_helf.legend(fontsize=6.5)
        _despine(ax_helf)

        _panel_label(ax_sch, "D")
        op = results.get("order", {})
        s_gel   = op.get("s_gel",   [])
        s_fluid = op.get("s_fluid", [])
        if s_gel:
            ax_sch.hist(s_gel, bins=30, density=True,
                        color=C["lo"], alpha=0.75,
                        label=f"Gel  S_mean={op.get('gel_mean',0):.3f}")
        if s_fluid:
            ax_sch.hist(s_fluid, bins=30, density=True,
                        color=C["ld"], alpha=0.75,
                        label=f"Fluido  S_mean={op.get('fluid_mean',0):.3f}")
        ax_sch.axvspan(0.85, 0.95, color=C["lo"], alpha=0.12, label="Ref. gel")
        ax_sch.axvspan(0.60, 0.75, color=C["ld"], alpha=0.12, label="Ref. fluido")
        ax_sch.set_xlabel("S_CH")
        ax_sch.set_ylabel("PDF")
        ax_sch.set_title("D  Parámetro de orden S_CH")
        ax_sch.legend(fontsize=6.5)
        _despine(ax_sch)

        _panel_label(ax_ed, "E")
        z_c, ed_p = electron_density_profile(membrane, bins_z=200)
        ax_ed.plot(ed_p, z_c / 10, color=C["lo"], lw=1.5, label="Simulación")
        ax_ed.axvspan(0.44, 0.50, color=C["lo"], alpha=0.15,
                      label="Ref. cabeza")
        ax_ed.axvspan(0.28, 0.31, color=C["ld"], alpha=0.15,
                      label="Ref. cola")
        ax_ed.axvline(0.334, color=C["neutral"], lw=0.8, ls="--", alpha=0.6)
        ax_ed.set_xlabel("Densidad electrónica (e * A^-3)")
        ax_ed.set_ylabel("Z (nm)")
        ax_ed.set_title("E  Perfil ED vs. ref.\n[Nagle 2000]")
        ax_ed.legend(fontsize=6.5)
        _despine(ax_ed)

        _panel_label(ax_acf, "F")
        rc = results.get("raft_corr", {})
        r_v = rc.get("r_vals", [])
        acf = rc.get("acf",    [])
        xi  = rc.get("xi_nm", None)
        if len(r_v) > 2 and len(acf) > 2:
            ax_acf.plot(r_v, acf, color=C["lo"], lw=1.5)
            ax_acf.axhline(0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
            ax_acf.fill_between(r_v, 0, acf,
                                 where=np.asarray(acf) > 0,
                                 color=C["lo"], alpha=0.2)
            if xi is not None:
                ax_acf.axvline(xi, color=C["fail"], lw=1.0, ls=":",
                               label=f"xi = {xi:.1f} nm")
                ax_acf.legend(fontsize=6.5)
        ax_acf.set_xlabel("Radio (nm)")
        ax_acf.set_ylabel("ACF")
        ax_acf.set_title("F  Correlación espacial\ndominios Lo")
        _despine(ax_acf)

        _panel_label(ax_thick2, "G")
        th = results.get("thickness", {})
        T = analysis.thickness_map(membrane, bins=90)
        R = analysis.raft_fraction_map(membrane,
                                       membrane.outer_leaflet + membrane.inner_leaflet,
                                       bins=90)
        T_lo = T[(R > 0.5) & (T > 0)].ravel()
        T_ld = T[(R < 0.5) & (T > 0)].ravel()
        bins_t = np.linspace(T[T > 0].min() * 0.97, T.max() * 1.03, 30)
        if len(T_lo) > 3:
            ax_thick2.hist(T_lo, bins=bins_t, density=True, color=C["lo"],
                           alpha=0.7, label=f"Lo {th.get('lo_mean_nm', 0)*10:.1f} Å")
        if len(T_ld) > 3:
            ax_thick2.hist(T_ld, bins=bins_t, density=True, color=C["ld"],
                           alpha=0.7, label=f"Ld {th.get('ld_mean_nm', 0)*10:.1f} Å")
        ax_thick2.set_xlabel("D_PP (Å)")
        ax_thick2.set_ylabel("PDF")
        ax_thick2.set_title("G  Bimodalidad Lo/Ld\n[Pinigin 2022]")
        ax_thick2.legend(fontsize=7)
        _despine(ax_thick2)

        _panel_label(ax_inter, "H")
        interdig = analysis.interdigitation_map(membrane, bins=80)
        im_i = ax_inter.imshow(interdig.T, origin="lower",
                               extent=[0, membrane.Lx/10, 0, membrane.Ly/10],
                               cmap="YlOrRd", aspect="equal",
                               interpolation="bilinear")
        cb_i = plt.colorbar(im_i, ax=ax_inter, shrink=0.85, pad=0.02)
        cb_i.set_label("Índice interdig.", fontsize=7, labelpad=3)
        cb_i.ax.tick_params(labelsize=6)
        ax_inter.set_xlabel("x (nm)")
        ax_inter.set_ylabel("y (nm)")
        ax_inter.set_title("H  Mapa interdigitación\ntrans-leaflet [Chaisson 2025]")

        return _save(fig, f"R2_validacion_seed{seed:04d}", dpi, subdir="R2")


def plot_R3_organizacion(membrane: BicapaCryoET, dpi: int = 300) -> str:
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

    # Colormap personalizado Lo/Ld
    cmap_lo_ld = mcolors.LinearSegmentedColormap.from_list(
        "lo_ld",
        [(0.0, "#1d4e7a"),   # Ld puro
         (0.25, "#4a85b0"),
         (0.50, "#f0ede4"),  # (transición)
         (0.75, "#c9693a"),
         (1.0, "#7a2e00")],  # Lo puro
        N=256)

    # Colores PIPs
    PIP_CLR = {
        "PI": "#d4ac0d", "PI3P": "#f39c12", "PI4P": "#e67e22",
        "PI5P": "#e74c3c", "PI34P2": "#a04000",
        "PIP2": "#c0392b", "PIP3": "#7b241c",
    }

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(
            f"R3 — Organización lateral · seed = {seed}",
            fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.05, right=0.97,
                               top=0.93, bottom=0.07,
                               hspace=0.38, wspace=0.30)

        # Crear los 6 ejes explícitamente para el nuevo layout 2×3
        ax_a = fig.add_subplot(gs[0, 0])  # A: Fracción Lo externa
        ax_b = fig.add_subplot(gs[0, 1])  # B: PIPs sobre mapa externo
        ax_c = fig.add_subplot(gs[0, 2])  # C: Parámetro de orden S_CH
        ax_d = fig.add_subplot(gs[1, 0])  # D: Densidad de PIPs
        ax_e = fig.add_subplot(gs[1, 1])  # E: Fracción Lo interna
        ax_f = fig.add_subplot(gs[1, 2])  # F: Comparativa externa vs interna

        _panel_label(ax_a, "A")
        im0 = ax_a.imshow(raft_map_ext.T, origin="lower", extent=ext,
                          cmap=cmap_lo_ld, vmin=0, vmax=1,
                          aspect="equal", interpolation="bilinear")

        xg_ext = np.linspace(0, Lx, raft_map_ext.shape[0])
        yg_ext = np.linspace(0, Ly, raft_map_ext.shape[1])
        if raft_map_ext.min() <= 0.65 <= raft_map_ext.max():
            ax_a.contour(xg_ext, yg_ext, raft_map_ext.T, levels=[0.65],
                         colors=["#1a1a1a"], linewidths=1.5,
                         linestyles="-", alpha=0.90)

        cb0 = plt.colorbar(im0, ax=ax_a, shrink=0.85, pad=0.02)
        cb0.set_label("Fracción Lo", fontsize=7, labelpad=3)
        cb0.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb0.set_ticklabels(["0\n(Ld)", "0.25", "0.50", "0.75", "1.0\n(Lo)"], fontsize=7)
        cb0.ax.tick_params(labelsize=6)

        lo_pct_ext = float(raft_map_ext.mean()) * 100
        n_dom_ext = len(getattr(membrane, "rafts_outer", []))
        ax_a.text(0.03, 0.97,
                  f"Lo: {lo_pct_ext:.1f}%  |  {n_dom_ext} dominio(s)",
                  transform=ax_a.transAxes, fontsize=7,
                  va="top", bbox=dict(fc="white", ec=C["neutral"],
                                     lw=0.5, pad=2.5, alpha=0.9))
        ax_a.set_title("A  Fracción de fase Lo — externa")
        ax_a.set_xlabel("x (nm)")
        ax_a.set_ylabel("y (nm)")

        _panel_label(ax_b, "B")
        ax_b.imshow(raft_map_ext.T, origin="lower", extent=ext,
                    cmap=cmap_lo_ld, vmin=0, vmax=1,
                    aspect="equal", interpolation="bilinear", alpha=0.55)

        if raft_map_ext.min() <= 0.65 <= raft_map_ext.max():
            ax_b.contour(xg_ext, yg_ext, raft_map_ext.T, levels=[0.65],
                         colors=["#1a1a1a"], linewidths=1.2,
                         linestyles="--", alpha=0.80)

        pip_handles = []
        n_total_pip = 0
        for sp in sorted(PIP_CLR.keys()):
            pts = [(l.head_pos[0]/10, l.head_pos[1]/10)
                   for l in membrane.inner_leaflet
                   if l.lipid_type.name == sp]
            if not pts:
                continue
            xs, ys = zip(*pts)
            col = PIP_CLR[sp]
            ax_b.scatter(xs, ys, s=12, c=col, marker="o",
                         edgecolors="black", linewidths=0.3,
                         alpha=0.9, zorder=5)
            pip_handles.append(mpatches.Patch(fc=col, ec="black", lw=0.4,
                                              label=f"{sp}  (n={len(pts)})"))
            n_total_pip += len(pts)

        if pip_handles:
            ax_b.legend(handles=pip_handles, loc="upper right",
                        fontsize=6.5, frameon=True)
        ax_b.text(0.03, 0.03, f"{n_total_pip} PIPs total",
                  transform=ax_b.transAxes, fontsize=7,
                  bbox=dict(fc="white", ec=C["neutral"], lw=0.5,
                           pad=2, alpha=0.9))
        ax_b.set_title("B  PIPs sobre mapa de fase Lo/Ld")
        ax_b.set_xlabel("x (nm)")
        ax_b.set_ylabel("y (nm)")

        _panel_label(ax_c, "C")
        vlo_sch = np.percentile(order_map, 2)
        vhi_sch = np.percentile(order_map, 98)
        im2 = ax_c.imshow(order_map.T, origin="lower", extent=ext,
                          cmap="RdYlGn", vmin=vlo_sch, vmax=vhi_sch,
                          aspect="equal", interpolation="bilinear")
        cb2 = plt.colorbar(im2, ax=ax_c, shrink=0.85, pad=0.02)
        cb2.set_label("S_CH", fontsize=7, labelpad=3)
        cb2.ax.tick_params(labelsize=6)

        # Añadir referencias de valores esperados
        ax_c.axhspan(0.85, 0.95, xmin=0, xmax=0.15, color=C["lo"], alpha=0.3, transform=ax_c.transAxes)
        ax_c.axhspan(0.60, 0.75, xmin=0, xmax=0.15, color=C["ld"], alpha=0.3, transform=ax_c.transAxes)
        ax_c.text(0.02, 0.90, "Gel\n0.85-0.95", transform=ax_c.transAxes, fontsize=5.5, color=C["lo"], va="top")
        ax_c.text(0.02, 0.70, "Fluido\n0.60-0.75", transform=ax_c.transAxes, fontsize=5.5, color=C["ld"], va="top")

        ax_c.set_title("C  Parámetro de orden S_CH\n[Piggot 2017]")
        ax_c.set_xlabel("x (nm)")
        ax_c.set_ylabel("y (nm)")

        _panel_label(ax_d, "D")
        im3 = ax_d.imshow(pip_map.T, origin="lower", extent=ext,
                          cmap="hot_r", aspect="equal",
                          interpolation="bilinear")
        cb3 = plt.colorbar(im3, ax=ax_d, shrink=0.85, pad=0.02)
        cb3.set_label("Densidad PIP (Da · Å^{-2})", fontsize=7, labelpad=3)
        cb3.ax.tick_params(labelsize=6)

        for cl in getattr(membrane, "pip_clusters", []):
            if not cl:
                continue
            cx = np.mean([l.head_pos[0] for l in cl]) / 10
            cy = np.mean([l.head_pos[1] for l in cl]) / 10
            r = np.std([np.hypot(l.head_pos[0]/10 - cx, l.head_pos[1]/10 - cy) for l in cl])
            r = max(r, 0.5)
            circ = plt.Circle((cx, cy), r, fill=False,
                              edgecolor="cyan", linewidth=1.2,
                              linestyle=":", alpha=0.85)
            ax_d.add_patch(circ)
        ax_d.set_title("D  Densidad de PIPs\n[Di Paolo & De Camilli 2006]")
        ax_d.set_xlabel("x (nm)")
        ax_d.set_ylabel("y (nm)")

        _panel_label(ax_e, "E")
        im4 = ax_e.imshow(raft_map_int.T, origin="lower", extent=ext,
                          cmap=cmap_lo_ld, vmin=0, vmax=1,
                          aspect="equal", interpolation="bilinear")

        xg_int = np.linspace(0, Lx, raft_map_int.shape[0])
        yg_int = np.linspace(0, Ly, raft_map_int.shape[1])
        if raft_map_int.min() <= 0.65 <= raft_map_int.max():
            ax_e.contour(xg_int, yg_int, raft_map_int.T, levels=[0.65],
                         colors=["#1a1a1a"], linewidths=1.5,
                         linestyles="-", alpha=0.90)

        cb4 = plt.colorbar(im4, ax=ax_e, shrink=0.85, pad=0.02)
        cb4.set_label("Fracción Lo", fontsize=7, labelpad=3)
        cb4.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
        cb4.set_ticklabels(["0\n(Ld)", "0.25", "0.50", "0.75", "1.0\n(Lo)"], fontsize=7)
        cb4.ax.tick_params(labelsize=6)

        lo_pct_int = float(raft_map_int.mean()) * 100
        n_dom_int = len(getattr(membrane, "rafts_inner", []))
        ax_e.text(0.03, 0.97,
                  f"Lo: {lo_pct_int:.1f}%  |  {n_dom_int} dominio(s)",
                  transform=ax_e.transAxes, fontsize=7,
                  va="top", bbox=dict(fc="white", ec=C["neutral"],
                                     lw=0.5, pad=2.5, alpha=0.9))
        ax_e.set_title("E  Fracción de fase Lo — interna")
        ax_e.set_xlabel("x (nm)")
        ax_e.set_ylabel("y (nm)")

        _panel_label(ax_f, "F")

        target_shape = (100, 100)

        if raft_map_ext.shape != target_shape:
            zoom_factor_ext = (target_shape[0] / raft_map_ext.shape[0], 
                             target_shape[1] / raft_map_ext.shape[1])
            raft_ext_resized = zoom(raft_map_ext, zoom_factor_ext, order=1)
        else:
            raft_ext_resized = raft_map_ext

        if raft_map_int.shape != target_shape:
            zoom_factor_int = (target_shape[0] / raft_map_int.shape[0],
                             target_shape[1] / raft_map_int.shape[1])
            raft_int_resized = zoom(raft_map_int, zoom_factor_int, order=1)
        else:
            raft_int_resized = raft_map_int

        diff_map = raft_ext_resized - raft_int_resized

        # Colormap divergente para diferencias
        cmap_diff = mcolors.LinearSegmentedColormap.from_list(
            "diff", [(0.0, "#2166ac"), (0.5, "#f7f7f7"), (1.0, "#b2182b")], N=256)

        vlo_diff = np.percentile(diff_map, 1)
        vhi_diff = np.percentile(diff_map, 99)
        vmax_diff = max(abs(vlo_diff), abs(vhi_diff))

        im5 = ax_f.imshow(diff_map.T, origin="lower", extent=ext,
                          cmap=cmap_diff, vmin=-vmax_diff, vmax=vmax_diff,
                          aspect="equal", interpolation="bilinear")
        cb5 = plt.colorbar(im5, ax=ax_f, shrink=0.85, pad=0.02)
        cb5.set_label("Delta Lo (ext − int)", fontsize=7, labelpad=3)
        cb5.ax.tick_params(labelsize=6)

        ax_f.contour(np.linspace(0, Lx, diff_map.shape[0]), 
                     np.linspace(0, Ly, diff_map.shape[1]),
                     diff_map.T, levels=[0],
                     colors=["black"], linewidths=1.0, linestyles="--", alpha=0.7)

        # Estadísticas de diferencia
        mean_diff = float(diff_map.mean())
        std_diff = float(diff_map.std())
        ax_f.text(0.03, 0.97,
                  f"mean={mean_diff:+.3f}\nstd={std_diff:.3f}",
                  transform=ax_f.transAxes, fontsize=7,
                  va="top", bbox=dict(fc="white", ec=C["neutral"],
                                     lw=0.5, pad=2.5, alpha=0.9))
        ax_f.set_title("F  Asimetría Lo externa vs. interna\nDelta = ext − int")
        ax_f.set_xlabel("x (nm)")
        ax_f.set_ylabel("y (nm)")

        return _save(fig, f"R3_organizacion_lateral_seed{seed:04d}", dpi, subdir="R3")


def plot_R4_multisemilla(stats: Dict, dpi: int = 300) -> str:
    records = stats.get("records", [])
    if len(records) < 2:
        print("  R4 requiere ≥2 semillas, saltando.")
        return ""

    seeds = [r["seed"] for r in records]
    kc    = np.array(stats.get("kc", []))
    thick = np.array(stats.get("thickness", []))
    sch_g = np.array(stats.get("sch_gel", []))
    sch_f = np.array(stats.get("sch_fluid", []))
    val_s = np.array(stats.get("val_scores", [np.nan]*len(seeds)))

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"R4 — Comparativa multi-semilla · N = {len(seeds)} simulaciones",
            fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.07, right=0.96,
                               top=0.92, bottom=0.09,
                               hspace=0.50, wspace=0.40)
        axes = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(3)]

        def _violin(ax, data, color, label, ref_lo=None, ref_hi=None):
            data = np.asarray(data)
            data = data[~np.isnan(data)]
            if len(data) < 2:
                ax.text(0.5, 0.5, "Sin datos", transform=ax.transAxes,
                        ha="center", va="center", fontsize=8, color=C["neutral"])
                ax.set_xticks([])
                _despine(ax)
                return
            vp = ax.violinplot([data], positions=[0], widths=0.6,
                               showmedians=True, showextrema=True)
            vp["bodies"][0].set_facecolor(color)
            vp["bodies"][0].set_alpha(0.65)
            vp["cmedians"].set_color(C["line"])
            vp["cmaxes"].set_color(C["neutral"])
            vp["cmins"].set_color(C["neutral"])
            vp["cbars"].set_color(C["neutral"])
            ax.scatter(np.zeros(len(data)) + np.random.uniform(-0.12, 0.12, len(data)),
                       data, s=14, color=C["line"], alpha=0.5, zorder=3)
            if ref_lo is not None:
                ax.axhspan(ref_lo, ref_hi, color=color, alpha=0.12,
                           label=f"Ref. [{ref_lo}–{ref_hi}]")
                ax.legend(fontsize=6.5)
            ax.text(0.97, 0.97,
                    f"μ={np.mean(data):.2f}\nσ={np.std(data):.2f}",
                    transform=ax.transAxes, fontsize=7, va="top", ha="right",
                    bbox=dict(fc="white", ec=C["neutral"], lw=0.5, pad=2))
            ax.set_xticks([])
            _despine(ax)

        _panel_label(axes[0], "A")
        _violin(axes[0], kc, C["lo"], "kc",
                ref_lo=20, ref_hi=45)
        axes[0].set_ylabel("kc (k_BT · nm^{2})")
        axes[0].set_title("A  Módulo de curvatura kc")

        _panel_label(axes[1], "B")
        _violin(axes[1], thick, C["ld"], "D_PP",
                ref_lo=35, ref_hi=50)
        axes[1].set_ylabel("D_PP (Å)")
        axes[1].set_title("B  Grosor de bicapa D_PP")

        _panel_label(axes[2], "C")
        vp_g = axes[2].violinplot([sch_g], positions=[-0.22], widths=0.38,
                                   showmedians=True)
        vp_f = axes[2].violinplot([sch_f], positions=[0.22], widths=0.38,
                                   showmedians=True)
        vp_g["bodies"][0].set_facecolor(C["lo"])
        vp_g["bodies"][0].set_alpha(0.65)
        vp_f["bodies"][0].set_facecolor(C["ld"])
        vp_f["bodies"][0].set_alpha(0.65)
        for vp in [vp_g, vp_f]:
            vp["cmedians"].set_color(C["line"])
            for key in ["cmaxes","cmins","cbars"]:
                vp[key].set_color(C["neutral"])
        axes[2].axhspan(0.85, 0.95, color=C["lo"], alpha=0.10)
        axes[2].axhspan(0.60, 0.75, color=C["ld"], alpha=0.10)
        axes[2].set_xticks([-0.22, 0.22])
        axes[2].set_xticklabels(["Gel", "Fluido"], fontsize=7.5)
        axes[2].set_ylabel("S_CH")
        axes[2].set_title("C  Parámetro de orden S_CH")
        _despine(axes[2])

        _panel_label(axes[3], "D")
        val_pct = val_s * 100
        bar_cols = []
        for v in val_pct:
            if np.isnan(v):
                bar_cols.append(C["neutral"])
            elif v >= 80:
                bar_cols.append(C["pass"])
            elif v >= 60:
                bar_cols.append(C["neutral"])
            else:
                bar_cols.append(C["fail"])
        axes[3].bar(range(len(seeds)), val_pct, color=bar_cols,
                    edgecolor="white", linewidth=0.4)
        axes[3].axhline(80, color=C["pass"], lw=1.0, ls="--", alpha=0.6,
                        label="Umbral 80%")
        axes[3].set_xticks(range(len(seeds)))
        axes[3].set_xticklabels([f"s{s}" for s in seeds],
                                 rotation=45, ha="right", fontsize=6.5)
        axes[3].set_ylabel("Score validación (%)")
        axes[3].set_ylim(0, 110)
        axes[3].set_title("D  Calidad biofísica por semilla")
        axes[3].legend(fontsize=7)
        _despine(axes[3])

        _panel_label(axes[4], "E")
        sp_vals: Dict[str, list] = {}
        for r in records:
            for sp, f in r.get("comp_outer", {}).items():
                sp_vals.setdefault(sp, []).append(f * 100)
        sp_sorted = sorted(sp_vals, key=lambda k: -np.mean(sp_vals[k]))
        means = [np.mean(sp_vals[sp]) for sp in sp_sorted]
        stds  = [np.std(sp_vals[sp])  for sp in sp_sorted]
        sp_colors_def = {
            "POPC":"#3a86ff","POPE":"#e63946","POPS":"#fb8500",
            "SM":"#2dc653","CHOL":"#adb5bd","GM1":"#d4a017",
            "PI":"#9b5de5","PIP2":"#c0392b","PI4P":"#e67e22",
            "PI3P":"#f39c12","PIP3":"#7b241c","PlsPE":"#c0392b",
            "PI34P2":"#a04000","PI5P":"#e74c3c",
        }
        cols_e = [sp_colors_def.get(sp, "#888888") for sp in sp_sorted]
        axes[4].barh(range(len(sp_sorted)), means, xerr=stds,
                     color=cols_e, height=0.6,
                     edgecolor="white", linewidth=0.4,
                     error_kw=dict(ecolor=C["neutral"], capsize=2, lw=0.8))
        axes[4].set_yticks(range(len(sp_sorted)))
        axes[4].set_yticklabels(sp_sorted, fontsize=7)
        axes[4].set_xlabel("Fracción molar (%) · media ± DE")
        axes[4].set_title("E  Composición externa · diversidad Dirichlet")
        _despine(axes[4])

        _panel_label(axes[5], "F")
        n_rafts = [r.get("n_rafts_outer", 0) + r.get("n_rafts_inner", 0)
                   for r in records]
        n_pips  = [r.get("n_pip_clusters", 0) for r in records]
        x = np.arange(len(seeds))
        axes[5].bar(x - 0.2, n_rafts, 0.38, color=C["lo"],
                    edgecolor="white", linewidth=0.4, label="Balsas (Lo)")
        axes[5].bar(x + 0.2, n_pips,  0.38, color=C["pip"],
                    edgecolor="white", linewidth=0.4, label="Clusters PIP")
        axes[5].set_xticks(x)
        axes[5].set_xticklabels([f"s{s}" for s in seeds],
                                 rotation=45, ha="right", fontsize=6.5)
        axes[5].set_ylabel("Número de estructuras")
        axes[5].set_title("F  Heterogeneidad estructural")
        axes[5].legend(fontsize=7)
        _despine(axes[5])

        return _save(fig, "R4_comparativa_multisemilla", dpi, subdir="R4")


def plot_R5_canales(membrane: BicapaCryoET, dpi: int = 300) -> str:
    seed = membrane.seed
    bins = 64

    channels = {
        "c0  Densidad cryo-ET":    (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins)
            + analysis.density_map(membrane, membrane.inner_leaflet, bins=bins)
        ),
        "c1  Grosor local":        analysis.thickness_map(membrane, bins=bins),
        "c2  Rugosidad ext.":      analysis.roughness_map(membrane, membrane.outer_leaflet, bins=bins),
        "c3  Rugosidad int.":      analysis.roughness_map(membrane, membrane.inner_leaflet, bins=bins),
        "c4  Raft ext.":           analysis.raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins),
        "c5  Raft int.":           analysis.raft_fraction_map(membrane, membrane.inner_leaflet, bins=bins),
        "c6  PIPs densidad":       analysis.pip_density_map(membrane, bins=bins),
        "c7  Asimetría comp.":     (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins, sigma=2.0)
            - analysis.density_map(membrane, membrane.inner_leaflet, bins=bins, sigma=2.0)
        ),
        "c8  Sección XZ":          analysis.xz_projection(membrane, bx=bins*2, bz=bins)[0],
        "c9  Orden S_CH":      analysis.order_parameter_map(membrane, bins=bins),
        "c10 Interdigitación":     analysis.interdigitation_map(membrane, bins=bins),
        "c11 ED limpia (prior)":   electron_density_projection(membrane, bins_xy=bins, sigma=0.8),
    }

    cmaps = {
        "c0": "gray",   "c1": "viridis", "c2": "magma",   "c3": "magma",
        "c4": "RdYlBu_r","c5":"RdYlBu_r","c6": "hot_r",   "c7": "bwr",
        "c8": "gray",   "c9":"RdYlGn", "c10":"YlOrRd",  "c11":"gray",
    }

    with plt.rc_context(PUB_RC):
        fig, axes = plt.subplots(3, 4, figsize=(14, 10))
        fig.suptitle(
            f"R5 — Canales de training para ML · seed = {seed}  "
            f"({bins}×{bins} px por canal)",
            fontsize=11, fontweight="bold", y=0.99)
        plt.subplots_adjust(left=0.04, right=0.97, top=0.94, bottom=0.04,
                            hspace=0.38, wspace=0.28)

        ext = [0, membrane.Lx/10, 0, membrane.Ly/10]
        for ax, (title, arr) in zip(axes.ravel(), channels.items()):
            ch_key = title.split()[0]
            cmap = cmaps.get(ch_key, "viridis")

            vlo = np.percentile(arr, 2)
            vhi = np.percentile(arr, 98)
            if abs(vhi - vlo) < 1e-10:
                vlo, vhi = arr.min(), arr.max()
            im = ax.imshow(arr.T, origin="lower", extent=ext,
                           cmap=cmap, vmin=vlo, vmax=vhi,
                           aspect="equal", interpolation="bilinear")
            cb = plt.colorbar(im, ax=ax, shrink=0.9, pad=0.02,
                              fraction=0.046)
            cb.ax.tick_params(labelsize=5.5)
            ax.set_title(title, fontsize=7.5, pad=3)
            ax.set_xlabel("x (nm)", fontsize=6.5)
            ax.set_ylabel("y (nm)", fontsize=6.5)
            ax.tick_params(labelsize=6)

        return _save(fig, f"R5_canales_training_seed{seed:04d}", dpi, subdir="R5")


def plot_R6_cryoET(membrane: BicapaCryoET, dpi: int = 300) -> str:
    seed = membrane.seed
    bins = 90
    pixel_A = membrane.Lx / bins

    with plt.rc_context(PUB_RC):
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle(
            f"R6 — Calidad de la simulación cryo-ET · seed = {seed}",
            fontsize=11, fontweight="bold", y=0.98)

        gs = gridspec.GridSpec(2, 3, figure=fig,
                               left=0.06, right=0.97,
                               top=0.92, bottom=0.08,
                               hspace=0.48, wspace=0.36)
        ax_clean  = fig.add_subplot(gs[0, 0])
        ax_ctf    = fig.add_subplot(gs[0, 1])
        ax_noisy  = fig.add_subplot(gs[0, 2])
        ax_psd    = fig.add_subplot(gs[1, 0])
        ax_curves = fig.add_subplot(gs[1, 1])
        ax_xz     = fig.add_subplot(gs[1, 2])

        ext = [0, membrane.Lx/10, 0, membrane.Ly/10]

        proj_clean = electron_density_projection(membrane, bins_xy=bins, sigma=1.5)
        proj_ctf   = apply_ctf_2d(proj_clean, pixel_size_angstrom=pixel_A,
                                  defocus_um=2.0, b_factor=200.0)
        rng = np.random.default_rng(seed)
        proj_noisy = add_noise(proj_ctf, snr=0.10, rng=rng)

        def _show_img(ax, img, title, label):
            _panel_label(ax, label)
            vlo, vhi = np.percentile(img, 1), np.percentile(img, 99)
            ax.imshow(img.T, origin="lower", extent=ext,
                      cmap="gray", vmin=vlo, vmax=vhi,
                      aspect="equal", interpolation="bilinear")
            ax.set_title(title)
            ax.set_xlabel("x (nm)")
            ax.set_ylabel("y (nm)")

        _show_img(ax_clean, proj_clean, "A  Imagen limpia\n(sin CTF, sin ruido)", "A")
        _show_img(ax_ctf,   proj_ctf,   "B  CTF aplicado\n(Δf = 2 μm, Cs = 2.7 mm)", "B")
        _show_img(ax_noisy, proj_noisy, "C  CTF + ruido\n(SNR ≈ 0.10)", "C")

        _panel_label(ax_psd, "D")
        def _radial_psd(img):
            f = np.fft.fftshift(np.fft.fft2(img))
            psd = np.abs(f)**2
            ny, nx = psd.shape
            yc, xc = ny // 2, nx // 2
            r = np.sqrt((np.arange(ny)[:, None] - yc)**2 +
                        (np.arange(nx)[None, :] - xc)**2).astype(int)
            r_max = min(yc, xc)
            radial = np.array([psd[r == i].mean() for i in range(r_max)])
            q = np.arange(r_max) / (r_max * pixel_A / 10)  # nm^-1
            return q[1:], radial[1:]

        for img, lbl, col in [
            (proj_clean, "Limpia", C["lo"]),
            (proj_ctf,   "CTF",    C["ld"]),
            (proj_noisy, "Noisy",  C["pip"]),
        ]:
            q_r, psd_r = _radial_psd(img)
            ax_psd.semilogy(q_r, psd_r, label=lbl, color=col, lw=1.3)

        ax_psd.set_xlabel("q (nm^-1)")
        ax_psd.set_ylabel("PSD promediado radialmente")
        ax_psd.set_title("D  Espectro de potencia\nradialmente promediado")
        ax_psd.legend(fontsize=7)
        _despine(ax_psd)

        _panel_label(ax_curves, "E")
        from ctf_sim import compute_ctf
        # q en nm^-1, convertir a Å^-1 para compute_ctf (1 nm^-1 = 0.1 Å^-1)
        q_max_nm = 0.5 / (pixel_A / 10)  # nm^-1
        q_nm = np.linspace(0.001, q_max_nm, 300)
        Fx = (q_nm * 0.1).reshape(-1, 1)  # convertir a Å^-1
        Fy = np.zeros_like(Fx)
        defoci = [1.0, 2.0, 3.5]
        ctf_colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        for df, col in zip(defoci, ctf_colors):
            ctf_1d = compute_ctf(Fx, Fy, defocus_um=df, b_factor=200.0).flatten()
            ax_curves.plot(q_nm, ctf_1d, lw=1.4, color=col, label=f"Δf = {df} μm")
        ax_curves.axhline(0, color=C["neutral"], lw=0.7, ls="--", alpha=0.5)
        ax_curves.set_xlabel("Frecuencia espacial (nm^-1)")
        ax_curves.set_ylabel("CTF")
        ax_curves.set_ylim(-1.1, 1.1)
        ax_curves.set_title("E  Curvas CTF\n[Dubochet 1988]")
        ax_curves.legend(fontsize=7)
        _despine(ax_curves)

        _panel_label(ax_xz, "F")
        Hxz, xe, ze = analysis.xz_projection(membrane, bx=180, bz=90)
        ax_xz.imshow(Hxz.T, origin="lower",
                     extent=[xe[0], xe[-1], ze[0]*10, ze[-1]*10],
                     cmap="gray", aspect="auto", interpolation="bilinear")
        ax_xz.axhline(membrane.geometry.z_outer / 10, color=C["lo"],
                      lw=0.9, ls="--", alpha=0.7, label="Cabezas ext.")
        ax_xz.axhline(membrane.geometry.z_inner / 10, color=C["ld"],
                      lw=0.9, ls="--", alpha=0.7, label="Cabezas int.")
        ax_xz.set_xlabel("x (nm)")
        ax_xz.set_ylabel("z (Å)")
        ax_xz.set_title("F  Sección transversal XZ\ndensidad electrónica proyectada")
        ax_xz.legend(fontsize=6.5, loc="upper right")

        return _save(fig, f"R6_calidad_cryoET_seed{seed:04d}", dpi, subdir="R6")


def main():
    parser = argparse.ArgumentParser(
        description="Figuras de resultados para TFM / artículo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python results.py --sims 1      
  python results.py --sims 1 2 3 4 5     # R4 multi-semilla incluida
  python results.py --sims 27 --only R1 R3 R5
  python results.py --sims 1 --dpi 300 --size 50 50
        """)
    parser.add_argument("--sims", type=int, nargs="+", required=True,
                        metavar="N", help="Semillas a procesar")
    parser.add_argument("--size", type=float, nargs=2, default=[50.0, 50.0],
                        metavar=("X", "Y"), help="Tamaño en nm (default: 50 50)")
    parser.add_argument("--only", nargs="+", default=None,
                        metavar="Rn",
                        help="Sólo generar estas secciones (R1 R2 R3 R4 R5 R6)")
    parser.add_argument("--dpi", type=int, default=300,
                        help="Resolución de salida en DPI (default: 300)")
    args = parser.parse_args()

    only = set(args.only) if args.only else {"R1","R2","R3","R4","R5","R6"}
    size_nm = tuple(args.size)

    print(f"\nresults.py — {len(args.sims)} semilla(s) | "
          f"{size_nm[0]:.0f}×{size_nm[1]:.0f} nm | DPI={args.dpi}")
    print(f"Secciones: {', '.join(sorted(only))}")
    print(f"Salida: {RESULTS_DIR}\n")

    # Para R4 acumulamos stats de todas las semillas
    stats_records: List[dict] = []

    for seed in args.sims:
        print(f"─── seed = {seed} ───────────────────────────────────")
        b = BicapaCryoET(size_nm=size_nm, seed=seed)
        b.build()

        results = None
        if "R2" in only or "R4" in only:
            print("  Calculando benchmarks...")
            results = run_all_benchmarks(b)

        if "R1" in only:
            plot_R1_caracterizacion(b, dpi=args.dpi)
        if "R2" in only:
            plot_R2_validacion(b, results=results, dpi=args.dpi)
        if "R3" in only:
            plot_R3_organizacion(b, dpi=args.dpi)
        if "R5" in only:
            plot_R5_canales(b, dpi=args.dpi)
        if "R6" in only:
            plot_R6_cryoET(b, dpi=args.dpi)

        if "R4" in only:
            T = analysis.thickness_map(b)
            todos = b.outer_leaflet + b.inner_leaflet
            s_gel   = [l.order_param for l in todos if l.lipid_type.phase=="gel"]
            s_fluid = [l.order_param for l in todos if l.lipid_type.phase=="fluid"]
            rec = {
                "seed": seed,
                "kc":   b.bending_modulus,
                "sigma":b.surface_tension,
                "thickness_mean_A": float(T.mean()),
                "n_rafts_outer":    len(b.rafts_outer),
                "n_rafts_inner":    len(b.rafts_inner),
                "n_pip_clusters":   len(b.pip_clusters),
                "sch_gel":   float(np.mean(s_gel))   if s_gel   else 0.0,
                "sch_fluid": float(np.mean(s_fluid)) if s_fluid else 0.0,
                "comp_outer": dict(b.comp_outer),
                "val_score":  results["summary"]["score"] if results else 0.0,
            }
            stats_records.append(rec)

    # R4 después de procesar todas las semillas
    if "R4" in only and len(stats_records) >= 2:
        print("\n─── R4 multi-semilla ────────────────────────────────")
        stats = {
            "records":    stats_records,
            "kc":         [r["kc"]   for r in stats_records],
            "thickness":  [r["thickness_mean_A"] for r in stats_records],
            "sch_gel":    [r["sch_gel"]   for r in stats_records],
            "sch_fluid":  [r["sch_fluid"] for r in stats_records],
            "val_scores": [r.get("val_score", 0.0) for r in stats_records],
        }
        plot_R4_multisemilla(stats, dpi=args.dpi)
    elif "R4" in only:
        print("  R4 requiere ≥2 semillas, saltando.")

    print(f"\n[OK] Listo. PDFs en: {RESULTS_DIR}")

if __name__ == "__main__":
    main()