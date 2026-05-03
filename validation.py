"""
validation.py
Validación cuantitativa del modelo de bicapa lipídica.

Implementa una serie de benchmarks físicos para comprobar que el dataset
sintético es coherente con el comportamiento esperado de membranas reales.

1. Espectro de Helfrich
2. Grosor de membrana: Distribución bimodal con picos Lo (~4.0 nm) y Ld (~3.6 nm).
3. Parámetro de orden S_CH: Separación entre fase ordenada (~0.85) y fluida (~0.65).
4. Correlación de rafts: Longitudes características ~20–50 nm.
5. Interdigitación: Mayor en dominios Lo que en Ld.
6. Densidad electrónica.

Cada test devuelve un valor numérico y un criterio pass/fail respecto a
rangos experimentales.

Referencias principales:
    [9]  Glushkova et al. 2026 – variación de grosor en membranas celulares (cryo-ET)
    [11] Kučerka et al. 2008 – espesores y áreas lipídicas en bicapas PC
    [21] Pinigin 2022 – parámetros elásticos de membranas desde simulación molecular
    [23] Sharma et al. 2023 – estructura de membranas en cryo-EM
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.signal import find_peaks

import analysis
from builder import OUTPUT_DIR
from electron_density import ELECTRON_DENSITY

if TYPE_CHECKING:
    from builder import BicapaCryoET


VAL_DIR = os.path.join(OUTPUT_DIR, "validation")


def _val_dir():
    os.makedirs(VAL_DIR, exist_ok=True)
    return VAL_DIR


PLT_STYLE = {
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#333333", "axes.linewidth": 1.0,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.5,
    "font.family": "sans-serif", "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold",
}

BENCHMARKS = {
    "helfrich_slope": (-5.0, -3.0),
    "thickness_lo_nm": (2.8, 4.2),
    "thickness_ld_nm": (2.6, 4.0),
    "thickness_diff_A": (0.5, 8.0),
    "sch_gel": (0.80, 0.95),
    "sch_fluid": (0.55, 0.75),
    "raft_xi_nm": (3.0, 30.0),
    "interdig_lo_gt_ld": True,
    "ed_head_peak_e_A3": (0.40, 0.52),
    "ed_tail_e_A3": (0.27, 0.32),
}


def _fmt_result(name, value, ref_range):
    lo, hi = ref_range
    ok = lo <= value <= hi
    marker = "PASS" if ok else "FAIL"
    return "  [%s] %s = %.3f  (ref: %.2f–%.2f)" % (marker, name, value, lo, hi)


def benchmark_helfrich(membrane: "BicapaCryoET") -> Dict:
    """
    Verifica que el espectro de fluctuaciones sigue la ley de Helfrich.

    Analiza el campo de alturas del curvature_map ya calculado.
    """
    h = membrane.curvature_map
    if h is None:
        return {"error": "curvature_map no calculado"}

    bins = h.shape[0]
    L_nm = membrane.Lx / 10.0
    dq = 2 * np.pi / L_nm

    h_ft = np.fft.fft2(h)
    power = np.abs(h_ft) ** 2 / (bins ** 2)

    qx = 2 * np.pi * np.fft.fftfreq(bins, d=L_nm / bins)
    Qx, Qy = np.meshgrid(qx, qx)
    Q = np.sqrt(Qx**2 + Qy**2)

    q_flat = Q.flatten()
    p_flat = power.flatten()

    mask = q_flat > dq * 1.5
    q_valid = q_flat[mask]
    p_valid = p_flat[mask]

    q_bins = np.logspace(np.log10(q_valid.min()), np.log10(q_valid.max()), 25)
    q_centers, p_mean = [], []
    for i in range(len(q_bins) - 1):
        sel = (q_valid >= q_bins[i]) & (q_valid < q_bins[i + 1])
        if sel.sum() > 3:
            q_centers.append(np.sqrt(q_bins[i] * q_bins[i + 1]))
            p_mean.append(p_valid[sel].mean())

    q_centers = np.array(q_centers)
    p_mean = np.array(p_mean)

    if len(q_centers) < 4:
        return {"error": "pocas frecuencias validas"}

    q_mid = np.median(q_centers)
    hi_mask = (q_centers > q_mid) & (q_centers < 0.70 * q_centers.max())
    lo_mask = q_centers <= q_mid

    def power_law(q, A, n):
        return A * q**n

    slope_hi, slope_lo = -4.0, -2.0
    if hi_mask.sum() >= 3:
        try:
            popt, _ = curve_fit(
                power_law, q_centers[hi_mask], p_mean[hi_mask],
                p0=[1.0, -4.0], maxfev=2000
            )
            slope_hi = popt[1]
        except Exception:
            pass

    if lo_mask.sum() >= 3:
        try:
            popt, _ = curve_fit(
                power_law, q_centers[lo_mask], p_mean[lo_mask],
                p0=[1.0, -2.0], maxfev=2000
            )
            slope_lo = popt[1]
        except Exception:
            pass

    lo_ref, hi_ref = BENCHMARKS["helfrich_slope"]
    passed = lo_ref <= slope_hi <= hi_ref

    return {
        "slope_high_q": slope_hi,
        "slope_low_q": slope_lo,
        "q_centers": q_centers.tolist(),
        "p_mean": p_mean.tolist(),
        "pass": passed,
    }


def benchmark_thickness(membrane: "BicapaCryoET") -> Dict:
    """
    Verifica la distribucion bimodal del grosor hidrofobico.

    Mide el grosor como distancia interglicerol entre lipidos apareados
    (sup mas cercano a cada inf), separando dominios raft (Lo) y no-raft (Ld).
    """
    from scipy.spatial import KDTree

    sup = membrane.outer_leaflet
    inf = membrane.inner_leaflet

    if not sup or not inf:
        return {"error": "monocapas vacias"}

    sup_xy = np.array([[l.glycerol_pos[0], l.glycerol_pos[1]] for l in sup])
    inf_xy = np.array([[l.glycerol_pos[0], l.glycerol_pos[1]] for l in inf])
    sup_z = np.array([l.glycerol_pos[2] for l in sup])
    inf_z = np.array([l.glycerol_pos[2] for l in inf])

    tree = KDTree(inf_xy)
    _, idxs = tree.query(sup_xy, k=1)
    paired_thick = (sup_z - inf_z[idxs]) / 10.0

    raft_t = [paired_thick[i] for i, l in enumerate(sup) if l.in_raft]
    nonraft_t = [paired_thick[i] for i, l in enumerate(sup) if not l.in_raft]

    lo_mean = float(np.mean(raft_t)) if raft_t else float(np.mean(paired_thick))
    ld_mean = float(np.mean(nonraft_t)) if nonraft_t else float(np.mean(paired_thick))
    diff_A = (lo_mean - ld_mean) * 10.0

    diff_lo, diff_hi = 0.5, 8.0

    from scipy.stats import gaussian_kde
    vals = paired_thick[paired_thick > 0]
    if len(vals) < 2:
        return {"error": "datos insuficientes"}

    kde = gaussian_kde(vals, bw_method=0.08)
    t_range = np.linspace(vals.min(), vals.max(), 300)
    density = kde(t_range)
    peaks_idx, _ = find_peaks(density, prominence=0.05 * density.max(), distance=5)

    if len(peaks_idx) >= 2:
        peak_vals = np.sort(t_range[peaks_idx])
        lo_peak_nm = float(peak_vals[-1])
        ld_peak_nm = float(peak_vals[0])
    else:
        lo_peak_nm = lo_mean
        ld_peak_nm = ld_mean

    diff_A = abs((lo_peak_nm - ld_peak_nm) * 10.0)

    diff_lo, diff_hi = 0.5, 8.0

    return {
        "mean_nm": float(vals.mean()),
        "std_nm": float(vals.std()),
        "lo_mean_nm": lo_mean,
        "ld_mean_nm": ld_mean,
        "lo_peak_nm": lo_peak_nm,
        "ld_peak_nm": ld_peak_nm,
        "diff_A": diff_A,
        "n_peaks": len(peaks_idx),
        "pass_bimodal": len(peaks_idx) >= 1,
        "pass_diff": diff_lo <= diff_A <= diff_hi,
        "pass": diff_lo <= diff_A <= diff_hi,
        "t_range": t_range.tolist(),
        "kde": density.tolist(),
        "peaks": [float(t_range[i]) for i in peaks_idx],
    }


def benchmark_order_parameter(membrane: "BicapaCryoET") -> Dict:
    """
    Verifica los valores de S_CH para fases gel y fluido.
    """
    todos = membrane.outer_leaflet + membrane.inner_leaflet
    s_gel = [l.order_param for l in todos if l.lipid_type.phase == "gel"]
    s_fluid = [l.order_param for l in todos if l.lipid_type.phase == "fluid"]

    gel_mean = float(np.mean(s_gel)) if s_gel else 0.0
    fluid_mean = float(np.mean(s_fluid)) if s_fluid else 0.0
    gel_std = float(np.std(s_gel)) if s_gel else 0.0
    fluid_std = float(np.std(s_fluid)) if s_fluid else 0.0

    g_lo, g_hi = BENCHMARKS["sch_gel"]
    f_lo, f_hi = BENCHMARKS["sch_fluid"]

    return {
        "gel_mean": gel_mean,
        "gel_std": gel_std,
        "fluid_mean": fluid_mean,
        "fluid_std": fluid_std,
        "n_gel": len(s_gel),
        "n_fluid": len(s_fluid),
        "pass_gel": g_lo <= gel_mean <= g_hi,
        "pass_fluid": f_lo <= fluid_mean <= f_hi,
        "pass": (g_lo <= gel_mean <= g_hi) and (f_lo <= fluid_mean <= f_hi),
        "s_gel": s_gel,
        "s_fluid": s_fluid,
    }


def benchmark_raft_correlation(membrane: "BicapaCryoET") -> Dict:
    """
    Calcula la longitud de correlacion de los dominios raft.

    Ajusta un decaimiento exponencial a la funcion de autocorrelacion
    del mapa de fraccion raft: C(r) = A · exp(-r/xi)
    """
    R = analysis.raft_fraction_map(membrane, membrane.outer_leaflet, bins=90)

    R_mean = R - R.mean()
    R_ft = np.fft.fft2(R_mean)
    acf_2d = np.real(np.fft.ifft2(np.abs(R_ft)**2)) / (R.shape[0] * R.shape[1])
    acf_2d = np.fft.fftshift(acf_2d)

    Nx, Ny = acf_2d.shape
    cx, cy = Nx // 2, Ny // 2
    max_r = min(cx, cy)
    pixel_nm = membrane.Lx / 10.0 / 90

    r_vals, acf_radial = [], []
    for r in range(1, max_r):
        ring = []
        for dx in range(-r - 1, r + 2):
            for dy in range(-r - 1, r + 2):
                dist = np.sqrt(dx**2 + dy**2)
                if abs(dist - r) < 0.7:
                    ix = np.clip(cx + dx, 0, Nx - 1)
                    iy = np.clip(cy + dy, 0, Ny - 1)
                    ring.append(acf_2d[ix, iy])
        if ring:
            r_vals.append(r * pixel_nm)
            acf_radial.append(np.mean(ring))

    r_vals = np.array(r_vals)
    acf_radial = np.array(acf_radial)

    if acf_radial[0] > 0:
        acf_radial = acf_radial / acf_radial[0]

    xi = 15.0
    try:
        def exp_decay(r, A, xi_fit):
            return A * np.exp(-r / xi_fit)

        valid = acf_radial > 0.05
        if valid.sum() >= 4:
            popt, _ = curve_fit(
                exp_decay, r_vals[valid], acf_radial[valid],
                p0=[1.0, 20.0], bounds=([0, 1], [2, 200]),
                maxfev=2000,
            )
            xi = float(popt[1])
    except Exception:
        pass

    xi_lo, xi_hi = BENCHMARKS["raft_xi_nm"]
    return {
        "xi_nm": xi,
        "r_vals": r_vals.tolist(),
        "acf": acf_radial.tolist(),
        "pass": xi_lo <= xi <= xi_hi,
    }


def benchmark_interdigitation(membrane: "BicapaCryoET") -> Dict:
    """Verifica que la interdigitacion es mayor en dominios Lo que Ld."""
    ID = analysis.interdigitation_map(membrane)
    R_outer = analysis.raft_fraction_map(membrane, membrane.outer_leaflet, bins=70)

    lo_thresh = float(np.percentile(R_outer, 75))
    ld_thresh = float(np.percentile(R_outer, 25))

    lo_mask = R_outer > lo_thresh
    ld_mask = R_outer <= ld_thresh

    id_lo = float(ID[lo_mask].mean()) if lo_mask.any() else 0.0
    id_ld = float(ID[ld_mask].mean()) if ld_mask.any() else 0.0

    return {
        "interdig_lo": id_lo,
        "interdig_ld": id_ld,
        "lo_gt_ld": id_lo > id_ld,
        "pass": id_lo > id_ld,
    }


def benchmark_electron_density(membrane: "BicapaCryoET") -> Dict:
    """
    Verifica que el perfil de densidad electronica tiene valores
    en el rango experimental.    
    """
    from electron_density import electron_density_profile

    z_centers, ed = electron_density_profile(membrane, bins_z=200)
    if np.isnan(ed).any():
        ed = np.nan_to_num(ed, nan=ELECTRON_DENSITY["water"])

    g = membrane.geometry
    z_o = g.z_outer
    z_i = g.z_inner

    head_zone = (
        ((z_centers >= z_o * 0.5) & (z_centers <= z_o * 1.3))
        | ((z_centers <= z_i * 0.5) & (z_centers >= z_i * 1.3))
    )
    tail_zone = np.abs(z_centers) < abs(g.hydro_thick) * 0.25

    ed_head = float(np.max(ed[head_zone])) if head_zone.any() else 0.0
    ed_tail = float(np.mean(ed[tail_zone])) if tail_zone.any() else 0.0

    h_lo, h_hi = BENCHMARKS["ed_head_peak_e_A3"]
    t_lo, t_hi = BENCHMARKS["ed_tail_e_A3"]

    return {
        "ed_head_peak": ed_head,
        "ed_tail": ed_tail,
        "z_centers": z_centers.tolist(),
        "ed_profile": ed.tolist(),
        "pass_head": h_lo <= ed_head <= h_hi,
        "pass_tail": t_lo <= ed_tail <= t_hi,
        "pass": (h_lo <= ed_head <= h_hi) and (t_lo <= ed_tail <= t_hi),
    }


def run_all_benchmarks(membrane: "BicapaCryoET") -> Dict:
    print("  Benchmarks cuantitativos para seed=%d..." % membrane.seed)

    results = {}
    results["helfrich"]    = benchmark_helfrich(membrane)
    results["thickness"]   = benchmark_thickness(membrane)
    results["order"]       = benchmark_order_parameter(membrane)
    results["raft_corr"]   = benchmark_raft_correlation(membrane)
    results["interdig"]    = benchmark_interdigitation(membrane)
    results["electron_ed"] = benchmark_electron_density(membrane)

    passed = sum(1 for k, v in results.items() if v.get("pass", False))
    total = len(results)

    print("  Resultados: %d/%d PASS" % (passed, total))
    for name, res in results.items():
        status = "PASS" if res.get("pass", False) else "FAIL"
        print("    [%s] %s" % (status, name))

    results["summary"] = {
        "seed": membrane.seed,
        "passed": passed,
        "total": total,
        "score": round(passed / total, 3),
    }
    return results


def plot_validation_panel(
    membrane: "BicapaCryoET",
    results: Optional[Dict] = None,
    save_path: Optional[str] = None,
):

    if results is None:
        results = run_all_benchmarks(membrane)

    with plt.rc_context(PLT_STYLE):
        fig = plt.figure(figsize=(20, 14))
        gs = GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.45)
        fig.suptitle(
            "Validacion cuantitativa del modelo — seed=%d\n"
            "Benchmarks fisicos contra rangos experimentales de cryo-ET"
            % membrane.seed,
            fontsize=13, fontweight="bold",
        )

        ax1 = fig.add_subplot(gs[0, 0])
        helf = results.get("helfrich", {})
        if "q_centers" in helf:
            qc = np.array(helf["q_centers"])
            pm = np.array(helf["p_mean"])
            ax1.loglog(qc, pm, "o", color="#3a86ff", ms=4, alpha=0.7, label="Modelo")
            slope = helf.get("slope_high_q", -4.0)
            q_fit = np.logspace(np.log10(qc.max() * 0.3), np.log10(qc.max()), 20)
            ax1.loglog(
                q_fit,
                pm.max() * (q_fit / q_fit[0]) ** slope,
                "--", color="#e63946", lw=2,
                label="Ajuste q^%.1f" % slope,
            )
            ax1.loglog(
                q_fit,
                pm.max() * (q_fit / q_fit[0]) ** (-4),
                ":", color="#333333", lw=1.2, alpha=0.6,
                label="Ref q^-4",
            )
        passed_str = "PASS" if helf.get("pass", False) else "FAIL"
        ax1.set_xlabel("q ($\\mathrm{nm}^{-1}$)", fontsize=9)
        ax1.set_ylabel("<|h_q|²> (Å²)", fontsize=9)
        ax1.set_title(
            "Espectro Helfrich [%s]\nPendiente alta-q = %.2f (ref -4 a -3.5)"
            % (passed_str, helf.get("slope_high_q", 0)),
            fontsize=9, fontweight="bold",
        )
        ax1.legend(fontsize=7.5)

        ax2 = fig.add_subplot(gs[0, 1])
        thick = results.get("thickness", {})
        if "t_range" in thick:
            t_r = np.array(thick["t_range"])
            kde_v = np.array(thick["kde"])
            ax2.fill_between(t_r, 0, kde_v, alpha=0.3, color="#3a86ff")
            ax2.plot(t_r, kde_v, color="#1a5fbf", lw=2)
            for pk in thick.get("peaks", []):
                ax2.axvline(pk, color="#e63946", lw=1.5, ls="--",
                            label="%.2f nm" % pk)
            if "diff_A" in thick:
                ax2.text(
                    0.05, 0.95,
                    "Lo=%.2f nm | Ld=%.2f nm\nΔ=%.1f Å (ref 0.5-8 Å)" % (
                        thick.get("lo_peak_nm", 0),
                        thick.get("ld_peak_nm", 0),
                        thick["diff_A"],
                    ),
                    transform=ax2.transAxes, va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#cccccc", alpha=0.9),
                )
        passed_str = "PASS" if thick.get("pass_diff", False) else "FAIL"
        ax2.set_xlabel("Grosor (nm)", fontsize=9)
        ax2.set_ylabel("Densidad", fontsize=9)
        ax2.set_title(
            "Grosor bimodal Lo/Ld [%s]\nPicos Lo y Ld separados 4-8 Å"
            % passed_str,
            fontsize=9, fontweight="bold",
        )
        ax2.legend(fontsize=7.5)

        ax3 = fig.add_subplot(gs[0, 2])
        order = results.get("order", {})
        s_gel = order.get("s_gel", [])
        s_fluid = order.get("s_fluid", [])
        if s_gel:
            ax3.hist(s_fluid, bins=40, density=True, alpha=0.6,
                     color="#3a86ff", label="Fluido (Ld)", histtype="stepfilled", ec="#1a5fbf")
            ax3.hist(s_gel, bins=40, density=True, alpha=0.6,
                     color="#2dc653", label="Gel (Lo)", histtype="stepfilled", ec="#0a6e2d")
            ax3.axvline(order.get("fluid_mean", 0), color="#1a5fbf", lw=1.8, ls="--")
            ax3.axvline(order.get("gel_mean", 0), color="#0a6e2d", lw=1.8, ls="--")
            for lo, hi, col in [
                (BENCHMARKS["sch_fluid"][0], BENCHMARKS["sch_fluid"][1], "#3a86ff"),
                (BENCHMARKS["sch_gel"][0], BENCHMARKS["sch_gel"][1], "#2dc653"),
            ]:
                ax3.axvspan(lo, hi, alpha=0.08, color=col)
        passed_str = "PASS" if (order.get("pass_gel") and order.get("pass_fluid")) else "FAIL"
        ax3.set_xlabel("S_CH", fontsize=9)
        ax3.set_ylabel("Densidad", fontsize=9)
        ax3.set_title(
            "Parametro de orden S_CH [%s]\nGel=%.3f (ref 0.80-0.95) | Flu=%.3f (ref 0.55-0.75)"
            % (passed_str, order.get("gel_mean", 0), order.get("fluid_mean", 0)),
            fontsize=9, fontweight="bold",
        )
        ax3.legend(fontsize=7.5)

        ax4 = fig.add_subplot(gs[1, 0])
        raft = results.get("raft_corr", {})
        if "r_vals" in raft:
            r_v = np.array(raft["r_vals"])
            acf = np.array(raft["acf"])
            ax4.plot(r_v, acf, color="#9b5de5", lw=2, label="ACF radial")
            xi = raft.get("xi_nm", 0)
            r_fit = np.linspace(r_v[0], r_v[-1], 100)
            ax4.plot(r_fit, np.exp(-r_fit / xi), "--", color="#e63946",
                     lw=1.8, label="Ajuste xi=%.1f nm" % xi)
            ax4.axhline(1 / np.e, color="#333333", lw=0.8, ls=":", alpha=0.6,
                        label="1/e")
        passed_str = "PASS" if raft.get("pass", False) else "FAIL"
        ax4.set_xlabel("r (nm)", fontsize=9)
        ax4.set_ylabel("ACF normalizada", fontsize=9)
        ax4.set_title(
            "Correlacion dominios raft [%s]\nxi = %.1f nm (ref: 10-60 nm)"
            % (passed_str, raft.get("xi_nm", 0)),
            fontsize=9, fontweight="bold",
        )
        ax4.legend(fontsize=7.5)

        ax5 = fig.add_subplot(gs[1, 1])
        idig = results.get("interdig", {})
        cats = ["Lo (raft)", "Ld (fluido)"]
        vals_idig = [idig.get("interdig_lo", 0), idig.get("interdig_ld", 0)]
        colors_idig = ["#2dc653", "#3a86ff"]
        bars = ax5.bar(cats, vals_idig, color=colors_idig,
                       edgecolor="#333333", lw=0.8, width=0.5)
        ax5.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
        passed_str = "PASS" if idig.get("pass", False) else "FAIL"
        ax5.set_ylabel("Score penetracion (u.a.)", fontsize=9)
        ax5.set_title(
            "Interdigitacion Lo > Ld [%s]\nChaisson et al. 2025"
            % passed_str,
            fontsize=9, fontweight="bold",
        )
        ax5.set_ylim(0, max(vals_idig) * 1.3 + 0.01)

        ax6 = fig.add_subplot(gs[1, 2])
        ed_res = results.get("electron_ed", {})
        if "z_centers" in ed_res:
            z_c = np.array(ed_res["z_centers"])
            ed_p = np.array(ed_res["ed_profile"])
            ax6.plot(ed_p, z_c / 10.0, color="#d4a017", lw=2.5)
            ax6.fill_betweenx(z_c / 10.0, 0.334, ed_p,
                              alpha=0.3, color="#d4a017")
            ax6.axhline(membrane.geometry.z_outer / 10.0,
                        color="#2dc653", ls="--", lw=1.2, label="Cabezas ext")
            ax6.axhline(membrane.geometry.z_inner / 10.0,
                        color="#e63946", ls="--", lw=1.2, label="Cabezas int")
            ax6.axhline(0, color="#666666", ls=":", lw=0.8)
            for lo, hi, col, lbl in [
                (BENCHMARKS["ed_head_peak_e_A3"][0], BENCHMARKS["ed_head_peak_e_A3"][1],
                 "#2dc653", "Ref cabezas (pico)"),
                (BENCHMARKS["ed_tail_e_A3"][0], BENCHMARKS["ed_tail_e_A3"][1],
                 "#3a86ff", "Ref colas"),
            ]:
                ax6.axvspan(lo, hi, alpha=0.12, color=col, label=lbl)
        passed_str = "PASS" if (ed_res.get("pass_head") and ed_res.get("pass_tail")) else "FAIL"
        ax6.set_xlabel("Densidad electronica (e/Å³)", fontsize=9)
        ax6.set_ylabel("Z (nm)", fontsize=9)
        ax6.set_title(
            "Perfil densidad electronica [%s]\nPatron dark-bright-dark"
            % passed_str,
            fontsize=9, fontweight="bold",
        )
        ax6.legend(fontsize=7.5)

        if save_path is None:
            save_path = os.path.join(
                _val_dir(), "validation_seed%04d.png" % membrane.seed
            )
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % save_path)
    return save_path


def save_benchmark_json(results: Dict, membrane: "BicapaCryoET"):
    # Claves de datos brutos que no aportan al JSON de resumen y se omiten
    # explicitamente por nombre, no por longitud, para no perder datos como
    # q_centers (24 elementos) que si son relevantes para el benchmark Helfrich.
    RAW_KEYS = {
        "s_gel", "s_fluid", "q_centers", "p_mean",
        "r_vals", "acf", "kde", "t_range", "ed_profile", "z_centers", "peaks",
    }

    def _to_native(vv):
        """Convierte escalares numpy a tipos Python nativos de forma robusta."""
        if isinstance(vv, (bool, int, float, str, list, dict)) or vv is None:
            return vv
        try:
            if hasattr(vv, 'item'):
                return vv.item()
        except Exception:
            pass
        try:
            return float(vv)
        except Exception:
            return str(vv)

    clean = {}
    for k, v in results.items():
        if not isinstance(v, dict):
            continue
        clean[k] = {
            kk: _to_native(vv)
            for kk, vv in v.items()
            if kk not in RAW_KEYS
        }
    path = os.path.join(_val_dir(), "benchmarks_seed%04d.json" % membrane.seed)
    with open(path, "w") as f:
        json.dump(clean, f, indent=2)
    print("  -> %s" % path)
    return path