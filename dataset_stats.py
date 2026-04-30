"""
dataset_stats.py
================
Estadisticas del dataset completo y figuras de publicacion.

Genera dos tipos de salidas:
  1. Estadisticas de dataset (N semillas)
     - Variabilidad de composicion entre semillas
     - Distribuciones de parametros fisicos (kc, sigma, grosor)
     - Cobertura del espacio de configuraciones

  2. Panel de comparacion CTF
     - Imagen original (densidad de masa)
     - Imagen con CTF correcta + missing wedge + ruido
     - Diferencia para mostrar el efecto del realismo

  3. Figura resumen de validacion sobre multiples semillas
     - Scatter plot de benchmarks PASS/FAIL por semilla
     - Distribucion de scores de validacion

Uso:
    from dataset_stats import compute_dataset_stats, plot_dataset_summary
    stats = compute_dataset_stats(range(20))
    plot_dataset_summary(stats)
"""

from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import analysis
from builder import BicapaCryoET, OUTPUT_DIR
from lipid_types import LIPID_TYPES

STATS_DIR = os.path.join(OUTPUT_DIR, "stats")
PLT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
}


def _stats_dir():
    os.makedirs(STATS_DIR, exist_ok=True)
    return STATS_DIR


def compute_dataset_stats(
    seeds: List[int],
    size_nm: Tuple[float, float] = (50.0, 50.0),
    run_validation: bool = True,
) -> Dict:
    """
    Calcula estadisticas del dataset para una lista de semillas.

    Construye cada membrana y extrae metricas escalares.
    No guarda figuras — solo acumula datos numericos.

    Parametros
    ----------
    seeds : list of int
        Semillas a procesar.
    run_validation : bool
        Si True, ejecuta los benchmarks cuantitativos por semilla.

    Retorna
    -------
    dict con estadisticas agregadas.
    """
    from validation import run_all_benchmarks

    records = []
    print("Calculando estadisticas para %d semillas..." % len(seeds))

    for seed in seeds:
        b = BicapaCryoET(size_nm=size_nm, seed=seed)
        b.build()

        T = analysis.thickness_map(b)
        todos = b.outer_leaflet + b.inner_leaflet
        s_gel = [l.order_param for l in todos if l.lipid_type.phase == "gel"]
        s_fluid = [l.order_param for l in todos if l.lipid_type.phase == "fluid"]

        rec = {
            "seed": seed,
            "kc": b.bending_modulus,
            "sigma": b.surface_tension,
            "thickness_mean_A": float(T.mean()),
            "thickness_std_A": float(T.std()),
            "n_lipids": len(todos),
            "n_rafts_outer": len(b.rafts_outer),
            "n_rafts_inner": len(b.rafts_inner),
            "n_pip_clusters": len(b.pip_clusters),
            "sch_gel": float(np.mean(s_gel)) if s_gel else 0.0,
            "sch_fluid": float(np.mean(s_fluid)) if s_fluid else 0.0,
            "comp_outer": {k: round(v, 4) for k, v in b.comp_outer.items()},
            "comp_inner": {k: round(v, 4) for k, v in b.comp_inner.items()},
        }

        if run_validation:
            val = run_all_benchmarks(b)
            rec["val_score"] = val["summary"]["score"]
            rec["val_pass"] = val["summary"]["passed"]
            rec["val_total"] = val["summary"]["total"]

        records.append(rec)
        print("  seed=%d | kc=%.1f | T=%.1fA | val=%.0f%%" % (
            seed, b.bending_modulus, T.mean(),
            rec.get("val_score", 0) * 100,
        ))

    arr = np.array
    stats = {
        "n_seeds": len(records),
        "seeds": [r["seed"] for r in records],
        "kc": [r["kc"] for r in records],
        "sigma": [r["sigma"] for r in records],
        "thickness": [r["thickness_mean_A"] for r in records],
        "sch_gel": [r["sch_gel"] for r in records],
        "sch_fluid": [r["sch_fluid"] for r in records],
        "n_rafts": [r["n_rafts_outer"] + r["n_rafts_inner"] for r in records],
        "records": records,
    }
    if run_validation:
        scores = [r.get("val_score", 0) for r in records]
        stats["val_scores"] = scores
        stats["val_mean"] = float(np.mean(scores))
        stats["val_min"] = float(np.min(scores))
        print("\nScore validacion: %.0f%% ± %.0f%% (min %.0f%%)" % (
            np.mean(scores)*100, np.std(scores)*100, np.min(scores)*100
        ))

    path = os.path.join(_stats_dir(), "dataset_stats.json")
    with open(path, "w") as f:
        safe_records = [{k: v for k, v in r.items() if k not in ("comp_outer","comp_inner")} for r in records]
        json.dump({"summary": {k: v for k, v in stats.items() if k != "records"},
                   "records": safe_records}, f, indent=2)
    print("  -> stats/dataset_stats.json")
    return stats


def plot_dataset_summary(
    stats: Dict,
    save_path: Optional[str] = None,
):
    """
    Panel resumen del dataset completo.

    4 paneles:
      1. Distribucion de kc vs composicion de CHOL+SM
      2. Distribucion de grosores (variabilidad inter-semilla)
      3. Scatter S_CH gel vs fluido (separacion de fases)
      4. Scores de validacion por semilla
    """
    records = stats["records"]
    n = len(records)

    kc_vals = np.array(stats["kc"])
    thick_vals = np.array(stats["thickness"])
    sch_gel_vals = np.array(stats["sch_gel"])
    sch_fluid_vals = np.array(stats["sch_fluid"])

    chol_sm = np.array([
        r["comp_outer"].get("CHOL", 0) + r["comp_outer"].get("SM", 0)
        for r in records
    ])

    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(2, 3, figure=fig, wspace=0.38, hspace=0.45)
        fig.suptitle(
            "Estadisticas del dataset sintetico — %d semillas\n"
            "Variabilidad inter-muestra y cobertura del espacio de configuraciones"
            % n,
            fontsize=13, fontweight="bold",
        )

        ax1 = fig.add_subplot(gs[0, 0])
        sc = ax1.scatter(chol_sm * 100, kc_vals, c=thick_vals,
                         cmap="RdYlBu_r", s=60, alpha=0.8, edgecolors="#333333", lw=0.5)
        cb = fig.colorbar(sc, ax=ax1, pad=0.02)
        cb.set_label("Grosor medio (Å)", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        m, b = np.polyfit(chol_sm * 100, kc_vals, 1)
        x_fit = np.linspace(chol_sm.min()*100, chol_sm.max()*100, 50)
        ax1.plot(x_fit, m*x_fit + b, "--", color="#e63946", lw=1.5, alpha=0.7,
                 label="r=%.2f" % np.corrcoef(chol_sm, kc_vals)[0,1])
        ax1.set_xlabel("CHOL + SM en monocapa ext. (%)", fontsize=9)
        ax1.set_ylabel("kc (kBT·nm²)", fontsize=9)
        ax1.set_title(
            "kc vs composicion\n"
            "Mayor CHOL/SM → mayor rigidez [6]",
            fontsize=9, fontweight="bold",
        )
        ax1.legend(fontsize=8)

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.hist(thick_vals, bins=min(n//2 + 1, 20), color="#3a86ff",
                 alpha=0.7, edgecolor="#1a5fbf", lw=0.8)
        ax2.axvline(thick_vals.mean(), color="#e63946", lw=2, ls="--",
                    label="μ=%.1fÅ" % thick_vals.mean())
        ax2.axvspan(thick_vals.mean() - thick_vals.std(),
                    thick_vals.mean() + thick_vals.std(),
                    alpha=0.15, color="#e63946", label="±σ=%.1fÅ" % thick_vals.std())
        ax2.set_xlabel("Grosor total medio (Å)", fontsize=9)
        ax2.set_ylabel("N semillas", fontsize=9)
        ax2.set_title(
            "Distribucion de grosores\nVariabilidad composicional Dirichlet",
            fontsize=9, fontweight="bold",
        )
        ax2.legend(fontsize=8)

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(sch_fluid_vals, sch_gel_vals, c=kc_vals,
                    cmap="Greens", s=60, alpha=0.8, edgecolors="#333333", lw=0.5)
        ax3.axvspan(*[0.55, 0.75], alpha=0.1, color="#3a86ff", label="Ref Ld")
        ax3.axhspan(*[0.80, 0.95], alpha=0.1, color="#2dc653", label="Ref Lo")
        ax3.set_xlabel("S_CH fluido (Ld)", fontsize=9)
        ax3.set_ylabel("S_CH gel (Lo)", fontsize=9)
        ax3.set_title(
            "Separacion de fases S_CH\nLo vs Ld por semilla [7,8]",
            fontsize=9, fontweight="bold",
        )
        ax3.legend(fontsize=8)

        ax4 = fig.add_subplot(gs[1, 0])
        n_rafts = np.array(stats["n_rafts"])
        ax4.bar(range(n), n_rafts, color="#9b5de5", alpha=0.8,
                edgecolor="#5a189a", lw=0.5)
        ax4.set_xlabel("Semilla (indice)", fontsize=9)
        ax4.set_ylabel("N dominios raft (sup + inf)", fontsize=9)
        ax4.set_title(
            "Variabilidad de dominios raft\nNucleacion estocastica controlada",
            fontsize=9, fontweight="bold",
        )

        ax5 = fig.add_subplot(gs[1, 1])
        comp_outer_chol = [r["comp_outer"].get("CHOL", 0) * 100 for r in records]
        comp_outer_sm = [r["comp_outer"].get("SM", 0) * 100 for r in records]
        comp_outer_popc = [r["comp_outer"].get("POPC", 0) * 100 for r in records]
        ax5.errorbar(
            ["CHOL", "SM", "POPC"],
            [np.mean(comp_outer_chol), np.mean(comp_outer_sm), np.mean(comp_outer_popc)],
            yerr=[np.std(comp_outer_chol), np.std(comp_outer_sm), np.std(comp_outer_popc)],
            fmt="o", ms=8, capsize=5, color="#3a86ff", lw=2, elinewidth=2,
        )
        ax5.axhline(30, color="#adb5bd", ls=":", lw=1, label="Base CHOL 30%")
        ax5.axhline(24, color="#2dc653", ls=":", lw=1, label="Base SM 24%")
        ax5.axhline(33, color="#3a86ff", ls=":", lw=1, label="Base POPC 33%")
        ax5.set_ylabel("Fraccion monocapa ext. (%)", fontsize=9)
        ax5.set_title(
            "Composicion: media ± std\nDistribucion Dirichlet (CV~12%%)",
            fontsize=9, fontweight="bold",
        )
        ax5.legend(fontsize=7.5)

        ax6 = fig.add_subplot(gs[1, 2])
        if "val_scores" in stats:
            scores = np.array(stats["val_scores"]) * 100
            seeds_arr = np.array(stats["seeds"])
            colors_v = ["#2dc653" if s >= 80 else "#fb8500" if s >= 60 else "#e63946"
                        for s in scores]
            bars = ax6.bar(range(n), scores, color=colors_v,
                           edgecolor="#333333", lw=0.5, alpha=0.9)
            ax6.axhline(80, color="#2dc653", ls="--", lw=1.5,
                        label="80%% PASS (bueno)")
            ax6.axhline(60, color="#fb8500", ls="--", lw=1.5,
                        label="60%% PASS (aceptable)")
            ax6.set_xlabel("Semilla (indice)", fontsize=9)
            ax6.set_ylabel("Score validacion (%%)", fontsize=9)
            ax6.set_ylim(0, 105)
            ax6.set_title(
                "Score validacion por semilla\nVerde >80%% | Naranja >60%% | Rojo <60%%",
                fontsize=9, fontweight="bold",
            )
            ax6.legend(fontsize=8)
        else:
            ax6.text(0.5, 0.5, "Validacion\nno ejecutada",
                     ha="center", va="center", transform=ax6.transAxes, fontsize=12)

        if save_path is None:
            save_path = os.path.join(_stats_dir(), "dataset_summary.png")
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % save_path)
    return save_path


def plot_ctf_comparison(
    membrane: "BicapaCryoET",
    defocus_values: Optional[List[float]] = None,
    snr: float = 0.1,
    save_path: Optional[str] = None,
):
    """
    Panel comparando imagen sin CTF, con CTF y con CTF+ruido+missing wedge.

    Justifica el realismo del pipeline de simulacion TEM.
    """
    from ctf_sim import simulate_projection, apply_missing_wedge, add_noise
    from electron_density import electron_density_projection

    if defocus_values is None:
        defocus_values = [1.0, 2.0, 3.0]

    bins = 90
    proj_raw = analysis.density_map(membrane, membrane.outer_leaflet, bins=bins) \
             + analysis.density_map(membrane, membrane.inner_leaflet, bins=bins)
    proj_ed = electron_density_projection(membrane, bins_xy=bins)

    n_rows = 2
    n_cols = len(defocus_values) + 1

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 9))
        fig.suptitle(
            "Comparacion de modelos de contraste — seed=%d\n"
            "Fila 1: densidad electronica | Fila 2: CTF + missing wedge + ruido"
            % membrane.seed,
            fontsize=12, fontweight="bold",
        )

        ext = [0, membrane.Lx / 10, 0, membrane.Ly / 10]

        axes[0, 0].imshow(proj_raw.T, origin="lower", cmap="gray",
                          extent=ext, aspect="equal")
        axes[0, 0].set_title("Densidad de masa\n(actual ch0)", fontsize=9, fontweight="bold")
        axes[0, 0].set_xlabel("X (nm)", fontsize=8)
        axes[0, 0].set_ylabel("Y (nm)", fontsize=8)

        axes[1, 0].imshow(proj_ed.T, origin="lower", cmap="gray",
                          extent=ext, aspect="equal")
        axes[1, 0].set_title("Densidad electronica\n(fisicamente correcto)", fontsize=9, fontweight="bold")
        axes[1, 0].set_xlabel("X (nm)", fontsize=8)
        axes[1, 0].set_ylabel("Y (nm)", fontsize=8)

        for col, df in enumerate(defocus_values, start=1):
            proj_ctf_only = simulate_projection(
                membrane, defocus_um=df, snr=100.0,
                use_electron_density=True, bins_xy=bins
            )
            proj_full = simulate_projection(
                membrane, defocus_um=df, snr=snr,
                use_electron_density=True, bins_xy=bins
            )

            axes[0, col].imshow(proj_ctf_only.T, origin="lower", cmap="gray",
                                extent=ext, aspect="equal")
            axes[0, col].set_title(
                "CTF solo\ndf=%.1f μm" % df, fontsize=9, fontweight="bold"
            )
            axes[0, col].set_xlabel("X (nm)", fontsize=8)
            if col > 0:
                axes[0, col].set_ylabel("")

            im = axes[1, col].imshow(proj_full.T, origin="lower", cmap="gray",
                                      extent=ext, aspect="equal")
            axes[1, col].set_title(
                "CTF + MW + ruido\ndf=%.1f μm | SNR=%.2f" % (df, snr),
                fontsize=9, fontweight="bold",
            )
            axes[1, col].set_xlabel("X (nm)", fontsize=8)
            if col > 0:
                axes[1, col].set_ylabel("")

        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(
                _stats_dir(), "ctf_comparison_seed%04d.png" % membrane.seed
            )
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % save_path)
    return save_path


def plot_mrc_comparison(
    membrane: "BicapaCryoET",
    save_path: Optional[str] = None,
):
    """
    Compara el volumen MRC generado (para PolNet) con y sin CTF.
    Muestra los tres slices ortogonales en ambas versiones.
    """
    from ctf_sim import simulate_volume
    from analysis import volumetric_density

    H_orig, edges = volumetric_density(membrane, bins_xy=55, bins_z=40)
    H_sim = simulate_volume(membrane, defocus_um=2.0, snr=0.1, bins_xy=55, bins_z=40)

    cx = 0.5 * (edges[0][:-1] + edges[0][1:])
    cy = 0.5 * (edges[1][:-1] + edges[1][1:])
    cz = 0.5 * (edges[2][:-1] + edges[2][1:])

    with plt.rc_context(PLT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(
            "Volumen MRC para PolNet — seed=%d\n"
            "Arriba: densidad masa (envio a PolNet) | Abajo: simulacion TEM completa"
            % membrane.seed,
            fontsize=12, fontweight="bold",
        )

        def plot_row(row, H, vols, title_prefix):
            mid_y = len(cy) // 2
            mid_x = len(cx) // 2
            g = membrane.geometry

            axes[row, 0].imshow(
                H[:, mid_y, :].T, origin="lower", cmap="gray_r", aspect="auto",
                extent=[cx[0], cx[-1], cz[0], cz[-1]]
            )
            for zl, col in [(g.z_outer/10, "#2dc653"), (g.z_inner/10, "#e63946")]:
                axes[row, 0].axhline(zl, color=col, lw=0.9, ls="--", alpha=0.7)
            axes[row, 0].set_title("%s — Slice XZ" % title_prefix, fontsize=9, fontweight="bold")
            axes[row, 0].set_xlabel("X (nm)", fontsize=8)
            axes[row, 0].set_ylabel("Z rel. (nm)", fontsize=8)

            axes[row, 1].imshow(
                H[mid_x, :, :].T, origin="lower", cmap="gray_r", aspect="auto",
                extent=[cy[0], cy[-1], cz[0], cz[-1]]
            )
            for zl, col in [(g.z_outer/10, "#2dc653"), (g.z_inner/10, "#e63946")]:
                axes[row, 1].axhline(zl, color=col, lw=0.9, ls="--", alpha=0.7)
            axes[row, 1].set_title("%s — Slice YZ" % title_prefix, fontsize=9, fontweight="bold")
            axes[row, 1].set_xlabel("Y (nm)", fontsize=8)

            z_lo_idx = np.argmin(np.abs(cz - g.z_outer/10 * 0.5))
            z_hi_idx = np.argmin(np.abs(cz - g.z_outer/10 * 1.3))
            Hxy = H[:, :, z_lo_idx:z_hi_idx+1].max(axis=2)
            axes[row, 2].imshow(
                Hxy.T, origin="lower", cmap="gray_r", aspect="equal",
                extent=[cx[0], cx[-1], cy[0], cy[-1]]
            )
            axes[row, 2].set_title("%s — MIP XY (cabezas)" % title_prefix, fontsize=9, fontweight="bold")
            axes[row, 2].set_xlabel("X (nm)", fontsize=8)
            axes[row, 2].set_ylabel("Y (nm)", fontsize=8)

        plot_row(0, H_orig, None, "Densidad masa (-> PolNet)")
        plot_row(1, H_sim,  None, "CTF + MW + ruido (objetivo)")

        fig.tight_layout()
        if save_path is None:
            save_path = os.path.join(
                _stats_dir(), "mrc_comparison_seed%04d.png" % membrane.seed
            )
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % save_path)
    return save_path
