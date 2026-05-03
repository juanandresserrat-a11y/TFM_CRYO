"""
model_3d.py
Modelo 3D de densidad electrónica de la bicapa lipídica.

Genera un volumen 3D en el que cada voxel representa densidad electrónica
derivada de la composición y organización de la membrana.

Incluye:
  1. Densidad por tipo de cabeza y cola (P, N, O vs C, H)
  2. Efecto de insaturaciones en el empaquetamiento de colas
  3. Diferencias Lo vs Ld (rafts vs fase fluida)
  4. Asimetría entre monocapas
  5. Enriquecimiento local en PIPs
  6. Corrección por fluctuaciones de curvatura (Helfrich)

El resultado es un volumen MRC (55×55×80, ~9 Å/voxel) compatible con
ChimeraX, IMOD y PolNet.

Referencias principales:
    [4]  Helfrich 1973 – elasticidad de membranas y fluctuaciones de curvatura en bicapas lipídicas
    [11] Kučerka et al. 2008 – determinación experimental de espesores y áreas por lípido en bicapas
    [16] Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos de cryo-ET
    [18] Nagle & Tristram-Nagle 2000 – estructura de bicapas y perfiles de densidad electrónica
    [20] Piggot et al. 2017 – cálculo de parámetros de orden acil (S_CH) en simulaciones lipídicas
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
import mrcfile

from builder import OUTPUT_DIR
from electron_density import (
    ELECTRON_DENSITY, LIPID_ED_HEADGROUP, LIPID_ED_TAIL
)
from analysis import midplane_map

if TYPE_CHECKING:
    from builder import BicapaCryoET


MODEL3D_DIR = os.path.join(OUTPUT_DIR, "model3d")


def _model3d_dir():
    os.makedirs(MODEL3D_DIR, exist_ok=True)
    return MODEL3D_DIR


# Factor de reduccion de densidad por insaturacion
# Cada doble enlace cis reduce la densidad de empaquetamiento local
UNSATURATION_PENALTY = 0.035


def _tail_density_with_unsaturation(lipid_name: str, z_frac: float) -> float:
    """
    Densidad electronica de una cola lipidica en funcion de z_frac [0,1].

    z_frac = 0 inicio en glicerol
    z_frac = 1 extremo metilo terminal

    Incluye tres efectos principales

    1 insaturaciones
       cada doble enlace reduce la densidad local ~3.5 por kinks cis
       POPC depresion en centro de sn2
       PIPs depresion mas acusada por multiples dobles enlaces
       SM y CHOL sin reduccion apreciable

    2 colesterol
       perfil axial diferenciado
       anillo esteroide mas denso que la cola isooctilo
       anillo ~0.302 e A3
       cola ~0.280 e A3

    3 plasmalogenos
       ausencia de carbonilo en sn1 reduce densidad inicial
       z_frac < 0.1 menor ED que fosfolipidos ester convencionales
    """
    from lipid_types import LIPID_TYPES
    lt = LIPID_TYPES.get(lipid_name)
    if lt is None:
        return ELECTRON_DENSITY["tail_fluid"]

    # Caso raro: gradiente axial del colesterol
    if lipid_name == "CHOL":
        # Anillo esteroide rigido (segmentos 0-4): alta densidad
        # Cadena isooctilo (segmentos 5-9): menor densidad
        if z_frac <= 0.45:
            return 0.302  # anillo esteroide compacto
        else:
            return 0.280  # isooctilo ramificado, menos empaquetado

    base_ed = LIPID_ED_TAIL.get(lipid_name, 0.292)

    ndb = lt.ndb[1]
    dbpos = lt.dbpos[1]
    nc = lt.nc[1]

    if ndb > 0 and dbpos is not None and nc > 0:
        z_db = dbpos / nc
        for _ in range(ndb):
            penalty = UNSATURATION_PENALTY * np.exp(
                -0.5 * ((z_frac - z_db) / 0.15) ** 2
            )
            base_ed -= penalty

    # Plasmalogeno: sin carbonilo ester en inicio de sn1
    if lipid_name == "PlsPE" and z_frac < 0.12:
        base_ed -= 0.004  # sin grupo C=O del enlace ester

    return float(np.clip(base_ed, 0.25, 0.32))


def build_physical_volume(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 80,
    voxel_angstrom: float = 9.0,
    sigma_xy: float = 1.2,
    sigma_z: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Construye el volumen 3D de densidad electronica fisicamente completo.

    El volumen captura el patron dark-bright-dark de la bicapa (cabezas-colas-cabezas),
    el contraste entre dominios Lo y Ld, la señal de PIPs en la monocapa interna,
    el efecto de las insaturaciones en colas acil, la asimetria entre monocapas
    y las fluctuaciones de Helfrich corregidas por plano medio local.
    """
    g = membrane.geometry
    z_mid_grid = midplane_map(membrane, bins=bins_xy)

    z_half_nm = (g.total_thick / 10.0) / 2.0 + 0.8
    z_edges   = np.linspace(-z_half_nm, z_half_nm, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz_nm     = z_edges[1] - z_edges[0]
    dz_A      = dz_nm * 10.0

    vol    = np.full((bins_xy, bins_xy, bins_z), ELECTRON_DENSITY["water"], dtype=np.float32)
    labels = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.uint8)
    weight = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.float32)

    todos = membrane.outer_leaflet + membrane.inner_leaflet

    for l in todos:
        lname    = l.lipid_type.name
        ed_head  = LIPID_ED_HEADGROUP.get(lname, 0.460)
        ed_glyc  = 0.390

        ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
        iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)

        z_ref = z_mid_grid[ix, iy] / 10.0
        z_h   = (l.head_pos[2]     / 10.0) - z_ref
        z_g   = (l.glycerol_pos[2] / 10.0) - z_ref

        hg_half_nm = l.lipid_type.hg_thick / 10.0 / 2.0
        hg_safe    = max(hg_half_nm, 0.1)

        for iz, zc in enumerate(z_centers):
            w_head = np.exp(-0.5 * ((zc - z_h) / hg_safe) ** 2)
            if w_head > 0.03:
                old_w = weight[ix, iy, iz]
                new_w = old_w + w_head
                vol[ix, iy, iz] = (vol[ix, iy, iz]*old_w + ed_head*w_head) / new_w
                weight[ix, iy, iz] = new_w

                label_val = 1 if l.leaflet == "sup" else 4
                if l.is_pip:
                    label_val = 5
                if labels[ix, iy, iz] == 0 or w_head > weight[ix, iy, iz] * 0.5:
                    labels[ix, iy, iz] = label_val

        glyc_thick = max(l.lipid_type.glyc_offset / 10.0, 0.05)
        wg = np.exp(-0.5 * ((z_centers - z_g) / glyc_thick) ** 2)
        mask_g = wg > 0.05
        if mask_g.any():
            old_w = weight[ix, iy, mask_g]
            new_w = old_w + wg[mask_g]
            vol[ix, iy, mask_g] = (vol[ix, iy, mask_g]*old_w + ed_glyc*wg[mask_g]) / new_w
            weight[ix, iy, mask_g] = new_w

        if l.tail1 and len(l.tail1) >= 2:
            nc = l.lipid_type.nc[0]
            for seg_i in range(len(l.tail1) - 1):
                pt_a = l.tail1[seg_i]
                pt_b = l.tail1[seg_i + 1]
                z_mid_seg = ((pt_a[2] + pt_b[2]) / 2.0 / 10.0) - z_ref
                z_frac = seg_i / max(len(l.tail1) - 2, 1)

                ed_tail_local = _tail_density_with_unsaturation(lname, z_frac)

                iz_seg = np.argmin(np.abs(z_centers - z_mid_seg))
                for diz in range(-1, 2):
                    iz2 = np.clip(iz_seg + diz, 0, bins_z - 1)
                    wt = np.exp(-0.5 * (diz * dz_nm / 0.25) ** 2) * 0.4
                    if wt > 0.05:
                        old_w = weight[ix, iy, iz2]
                        new_w = old_w + wt
                        vol[ix, iy, iz2] = (vol[ix, iy, iz2]*old_w + ed_tail_local*wt) / new_w
                        weight[ix, iy, iz2] = new_w

                        if labels[ix, iy, iz2] == 0:
                            labels[ix, iy, iz2] = 2 if l.in_raft else 3

    # Proteinas transmembrana (perturbaciones no diferenciadas)
    # Las proteinas se modelan como cilindros de densidad electronica
    # intermedia sin categorizar y que atraviesan la bicapa de extremo 
    # a extremo. No modelado directamente, solo considerado para realismo.
    ED_PROTEIN = 0.400
    if hasattr(membrane, "perturbations") and membrane.perturbations:
        for pert in membrane.perturbations:
            px, py = pert["pos"][0] / membrane.Lx, pert["pos"][1] / membrane.Ly
            ix_p = int(px * bins_xy) % bins_xy
            iy_p = int(py * bins_xy) % bins_xy
            r_vox = max(1, int(pert["radius"] / voxel_angstrom))
            for dix in range(-r_vox, r_vox + 1):
                for diy in range(-r_vox, r_vox + 1):
                    if dix**2 + diy**2 <= r_vox**2:
                        ixx = (ix_p + dix) % bins_xy
                        iyy = (iy_p + diy) % bins_xy
                        for iz in range(bins_z):
                            vol[ixx, iyy, iz] = (
                                vol[ixx, iyy, iz] * 0.3 + ED_PROTEIN * 0.7
                            )
                            if labels[ixx, iyy, iz] in (0, 2, 3):
                                labels[ixx, iyy, iz] = 6  # label 6 = proteina

    from scipy.ndimage import binary_closing
    for lv in [1, 2, 3, 4, 5, 6]:
        mask = labels == lv
        closed = binary_closing(mask, structure=np.ones((3,3,2), dtype=bool))
        labels[closed & (labels == 0)] = lv

    vol_smooth = gaussian_filter(vol, sigma=[sigma_xy, sigma_xy, sigma_z])

    lo_voxels  = vol_smooth[(labels == 2)]
    ld_voxels  = vol_smooth[(labels == 3)]
    head_voxels = vol_smooth[(labels == 1) | (labels == 4)]
    pip_voxels      = vol_smooth[labels == 5]
    protein_voxels  = vol_smooth[labels == 6]

    stats = {
        "ed_head_mean":  float(head_voxels.mean()) if head_voxels.size else 0.0,
        "ed_tail_Lo":    float(lo_voxels.mean())   if lo_voxels.size  else 0.0,
        "ed_tail_Ld":    float(ld_voxels.mean())   if ld_voxels.size  else 0.0,
        "ed_pip":        float(pip_voxels.mean())   if pip_voxels.size else 0.0,
        "ed_water":      ELECTRON_DENSITY["water"],
        "ed_protein":     float(protein_voxels.mean()) if protein_voxels.size else 0.0,
        "n_protein_objects": len(membrane.perturbations) if hasattr(membrane,"perturbations") else 0,
        "contrast_Lo_Ld": float(lo_voxels.mean() - ld_voxels.mean()) if lo_voxels.size and ld_voxels.size else 0.0,
        "voxel_angstrom": voxel_angstrom,
        "shape":         (bins_xy, bins_xy, bins_z),
        "z_half_nm":     z_half_nm,
    }

    return vol_smooth.astype(np.float32), labels, stats


def export_physical_model_mrc(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 80,
    voxel_angstrom: float = 9.0,
) -> Dict[str, str]:
    """
    Exporta el modelo 3D fisico completo como archivos MRC.

    Genera tres archivos:
      bilayer_physical_seed{N}.mrc       densidad electronica en e/A3
      bilayer_physical_seed{N}_norm.mrc  normalizado [0,255] para PolNet
      bilayer_physical_seed{N}_labels.mrc etiquetas semanticas 6 clases

    Retorna dict con rutas.
    """
    print("  Construyendo modelo 3D fisico para seed=%d..." % membrane.seed)

    vol, labels, stats = build_physical_volume(
        membrane, bins_xy=bins_xy, bins_z=bins_z, voxel_angstrom=voxel_angstrom
    )

    d = _model3d_dir()

    path_ed = os.path.join(d, "bilayer_physical_seed%04d.mrc" % membrane.seed)
    with mrcfile.new(path_ed, overwrite=True) as mrc:
        mrc.set_data(vol.T)
        mrc.voxel_size = voxel_angstrom

    vol_norm = (vol - vol.min()) / (vol.max() - vol.min()) * 255.0
    path_norm = os.path.join(d, "bilayer_physical_seed%04d_norm.mrc" % membrane.seed)
    with mrcfile.new(path_norm, overwrite=True) as mrc:
        mrc.set_data(vol_norm.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    path_lbl = os.path.join(d, "bilayer_physical_seed%04d_labels.mrc" % membrane.seed)
    with mrcfile.new(path_lbl, overwrite=True) as mrc:
        mrc.set_data(labels.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    print("  -> model3d/bilayer_physical_seed%04d.mrc  %dx%dx%d voxels @ %.0f A" % (
        membrane.seed, bins_xy, bins_xy, bins_z, voxel_angstrom))
    print("  ED cabezas: %.3f e/A3 | Lo: %.3f | Ld: %.3f | Contraste: %.4f e/A3" % (
        stats["ed_head_mean"], stats["ed_tail_Lo"], stats["ed_tail_Ld"],
        stats["contrast_Lo_Ld"]))

    return {
        "density_ea3": path_ed,
        "density_norm": path_norm,
        "labels":       path_lbl,
        "stats":        stats,
    }


def plot_physical_model(
    membrane: "BicapaCryoET",
    vol: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    stats: Optional[dict] = None,
    save_dir: str = "CryoET/model3d",
) -> str:
    """
    Genera el panel de visualizacion del modelo 3D fisico.

    5 subplots:
      1. Slice XZ central — perfil dark-bright-dark con fases anotadas
      2. Slice YZ central — segunda vista transversal
      3. Perfil 1D de densidad electronica — curva Z con regiones anotadas
      4. Proyeccion XY cabezas — contraste Lo vs Ld en la cara superior
      5. Mapa de insaturaciones — densidad de cola relativa por celda
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.colors import ListedColormap

    os.makedirs(save_dir, exist_ok=True)

    if vol is None or labels is None:
        vol, labels, stats = build_physical_volume(membrane)

    bins_xy, _, bins_z = vol.shape
    g  = membrane.geometry
    z_half_nm = stats["z_half_nm"]
    z_edges   = np.linspace(-z_half_nm, z_half_nm, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    Lx = membrane.Lx / 10.0
    Ly = membrane.Ly / 10.0
    ext_xy = [0, Lx, 0, Ly]
    ext_xz = [0, Lx, -z_half_nm, z_half_nm]
    ext_yz = [0, Ly, -z_half_nm, z_half_nm]

    PLT = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.4,
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
    }

    label_colors = {
        0: "#f0f4ff",  # agua
        1: "#2dc653",  # cabeza ext
        2: "#e63946",  # nucleo Lo
        3: "#3a86ff",  # nucleo Ld
        4: "#fb8500",  # cabeza int
        5: "#9b5de5",  # PIP cluster
        6: "#ffffff",  # proteina transmembrana
    }

    mid_y = bins_xy // 2
    mid_x = bins_xy // 2

    with plt.rc_context(PLT):
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        fig.suptitle(
            "Modelo 3D fisico de la bicapa lipidica — seed=%d\n"
            "Densidad electronica con insaturaciones, fases Lo/Ld y PIPs"
            % membrane.seed,
            fontsize=12, fontweight="bold",
        )

        ax = axes[0, 0]
        slc_xz = vol[:, mid_y, :]
        im = ax.imshow(slc_xz.T, origin="lower", cmap="gray",
                       extent=ext_xz, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label="$e \\cdot \\AA^{-3}$")
        ax.axhline(g.z_outer/10 - stats["z_half_nm"]*0, color="#2dc653",
                   lw=1.2, ls="--", alpha=0.85, label="Cabezas ext")
        ax.axhline(g.z_inner/10, color="#fb8500",
                   lw=1.2, ls="--", alpha=0.85, label="Cabezas int")
        ax.axhline(0.0, color="#adb5bd", lw=0.8, ls=":", alpha=0.7, label="Plano medio")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Z relativo (nm)")
        ax.set_title("Slice XZ — Patron dark-bright-dark\ncabezas (oscuro) + nucleo (claro)")
        ax.legend(fontsize=7.5, loc="upper right")

        ax = axes[0, 1]
        slc_yz = vol[mid_x, :, :]
        im2 = ax.imshow(slc_yz.T, origin="lower", cmap="gray",
                        extent=ext_yz, aspect="auto")
        plt.colorbar(im2, ax=ax, shrink=0.85, label="$e \\cdot \\AA^{-3}$")
        ax.set_xlabel("Y (nm)"); ax.set_ylabel("Z relativo (nm)")
        ax.set_title("Slice YZ — Segunda vista transversal\nAsimetria cabezas ext/int visible")

        ax = axes[0, 2]
        ed_profile = vol.mean(axis=(0, 1))
        ax.plot(ed_profile, z_centers, color="#d4a017", lw=2.5, label="ED media")
        ax.fill_betweenx(z_centers, ELECTRON_DENSITY["water"], ed_profile,
                         alpha=0.35, color="#d4a017")
        ax.axhline(g.z_outer/10, color="#2dc653", lw=1.2, ls="--",
                   label="Cabezas ext (%.1f nm)" % (g.z_outer/10))
        ax.axhline(g.z_inner/10, color="#fb8500", lw=1.2, ls="--",
                   label="Cabezas int (%.1f nm)" % (g.z_inner/10))
        ax.axhline(0.0, color="#adb5bd", lw=0.8, ls=":", alpha=0.7)
        ax.axvspan(0.27, 0.31, alpha=0.12, color="#3a86ff", label="Ref colas")
        ax.axvspan(0.44, 0.50, alpha=0.12, color="#2dc653", label="Ref cabezas")
        ax.axvline(ELECTRON_DENSITY["water"], color="#888888", lw=0.8, ls=":",
                   alpha=0.7, label="Agua (0.334)")
        ax.set_xlabel("Densidad electronica ($e \\cdot \\AA^{-3}$)")
        ax.set_ylabel("Z (nm)")
        ax.set_title("Perfil 1D — ED a lo largo del eje Z\nPatron dark-bright-dark cuantificado")
        ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1, 0]
        head_proj = vol[:, :, :].max(axis=2)
        from analysis import raft_fraction_map
        raft_map = raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins_xy)
        im3 = ax.imshow(head_proj.T, origin="lower", cmap="gray",
                        extent=ext_xy, aspect="equal")
        plt.colorbar(im3, ax=ax, shrink=0.85, label="ED max ($e \\cdot \\AA^{-3}$)")
        cs = ax.contour(np.linspace(0, Lx, bins_xy),
                        np.linspace(0, Ly, bins_xy),
                        raft_map.T, levels=[0.4], colors=["#e63946"],
                        linewidths=1.2, linestyles="--")
        ax.clabel(cs, fmt="Lo/Ld", fontsize=7)
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Proyeccion maxima XY\nContorno rojo = dominio Lo (raft)")

        ax = axes[1, 1]
        from analysis import order_parameter_map, pip_density_map
        s_ch_map = order_parameter_map(membrane, bins=bins_xy)
        im4 = ax.imshow(s_ch_map.T, origin="lower", cmap="RdYlGn",
                        extent=ext_xy, aspect="equal", vmin=0.55, vmax=0.95)
        plt.colorbar(im4, ax=ax, shrink=0.85, label="S_CH")
        pip_map = pip_density_map(membrane, bins=bins_xy)
        if pip_map.max() > 0:
            pip_norm = pip_map / pip_map.max()
            ax.contour(np.linspace(0, Lx, bins_xy),
                       np.linspace(0, Ly, bins_xy),
                       pip_norm.T, levels=[0.4],
                       colors=["#9b5de5"], linewidths=1.5, linestyles=":")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Mapa S_CH — Orden molecular\nVerde=Lo ordenado | Rojo=Ld fluido\n"
                     "Circulo morado = cluster PIPs")

        ax = axes[1, 2]
        from lipid_types import LIPID_TYPES

        tail_density_map_arr = np.zeros((bins_xy, bins_xy), dtype=np.float32)
        count_map = np.zeros((bins_xy, bins_xy), dtype=np.float32)

        for l in membrane.outer_leaflet + membrane.inner_leaflet:
            ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
            iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)
            lt = l.lipid_type
            ndb_total = lt.ndb[0] + lt.ndb[1]
            ed_base = LIPID_ED_TAIL.get(lt.name, 0.292)
            ed_with_unsat = ed_base - ndb_total * UNSATURATION_PENALTY * 0.5
            tail_density_map_arr[ix, iy] += ed_with_unsat
            count_map[ix, iy] += 1

        with np.errstate(all="ignore"):
            td = np.where(count_map > 0, tail_density_map_arr / count_map, 0.292)

        from scipy.ndimage import gaussian_filter as gf
        td_smooth = gf(td, sigma=2.0)
        im5 = ax.imshow(td_smooth.T, origin="lower", cmap="RdBu",
                        extent=ext_xy, aspect="equal",
                        vmin=td_smooth.min(), vmax=td_smooth.max())
        plt.colorbar(im5, ax=ax, shrink=0.85, label="ED cola ($e \\cdot \\AA^{-3}$)")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Densidad de cola — Efecto insaturaciones\nRojo=saturada (Lo/SM) | Azul=insaturada (Ld/POPC)")

        patches_leyenda = [
            mpatches.Patch(facecolor="#2dc653", label="Cabezas ext (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_head_mean"]),
            mpatches.Patch(facecolor="#e63946", label="Nucleo Lo (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_tail_Lo"]),
            mpatches.Patch(facecolor="#3a86ff", label="Nucleo Ld (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_tail_Ld"]),
            mpatches.Patch(facecolor="#9b5de5", label="PIPs (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_pip"]),
            mpatches.Patch(facecolor="#f0f4ff", edgecolor="#aaa",
                           label="Agua (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_water"]),
            mpatches.Patch(facecolor="#ffffff", edgecolor="#555",
                           label="Proteina TM (%d obj., %.3f $e \\cdot \\AA^{-3}$)" % (
                               stats.get("n_protein_objects", 0),
                               stats.get("ed_protein", 0.400))),
        ]
        
        fig.legend(handles=patches_leyenda, loc="lower center", ncol=3,
                   fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.02))

        plt.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.11, hspace=0.55, wspace=0.35)
        path = os.path.join(save_dir, "model3d_seed%04d.png" % membrane.seed)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % path)
    return path
