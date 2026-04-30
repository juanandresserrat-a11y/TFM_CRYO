"""
model_3d.py
===========
Modelo volumetrico 3D fisicamente completo de la bicapa lipidica.

Construye un volumen 3D donde cada voxel tiene densidad electronica
calculada a partir de las contribuciones atomicas de cada grano CG,
incorporando:

  1. DENSIDAD ELECTRONICA POR ESPECIE — cabezas polares (P, N, O)
     dispersan mas electrones que colas acil (C, H). Valores por especie
     de Nagle & Tristram-Nagle 2000 y Kucerka et al. 2008.

  2. EFECTO DE LAS INSATURACIONES — los dobles enlaces cis introducen
     kinks geometricos que reducen la densidad local de empaquetamiento
     en la zona del doble enlace. POPC (C18:1) tiene menos densidad de
     cola que SM (C24:0) en la misma region Z.

  3. CONTRASTE LO VS LD — los dominios raft (SM, CHOL, GM1) tienen
     colas mas densas por ser saturadas y compactas. Los dominios fluidos
     (POPC, POPE) tienen menor densidad de cola por las insaturaciones.

  4. ASIMETRIA ENTRE MONOCAPAS — la diferencia composicional entre
     monocapa externa (SM/CHOL/GM1) e interna (POPE/POPS/PIPs) se
     refleja en perfiles de densidad electronica distintos para cada cara.

  5. SEÑAL DE PIPs — los fosfoinositidos tienen la cabeza mas densa
     (hasta 0.498 e/A3 para PIP3 vs 0.458 para POPC) por el mayor
     numero de grupos fosfato con electronegatividad alta. Su cluster
     en la monocapa interna produce una senal diferencial detectable.

  6. FLUCTUACIONES HELFRICH — el campo de curvatura termica se aplica
     de forma coherente: la densidad en cada bin Z se calcula relativa
     al plano medio local h(x,y), eliminando el efecto de la ondulacion
     global y centrando la bicapa en cada columna.

El resultado es un volumen MRC de 55x55x80 voxels a 9 A/voxel que
puede importarse directamente en UCSF ChimeraX, IMOD o PolNet.

Referencias:
  Nagle & Tristram-Nagle, BBA 2000 [densidad electronica]
  Kucerka et al., Biophys. J. 2008 [ED por especie]
  Piggot et al., JCTC 2017 [S_CH, insaturaciones]
  Chaisson et al., JCIM 2025 [interdigitacion]
  Martinez-Sanchez et al., IEEE TMI 2024 [PolNet, formato MRC]
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


# ── Factor de reduccion de densidad por insaturacion ─────────────────
# Cada doble enlace cis reduce la densidad de empaquetamiento local
# porque el kink geometrico crea espacio vacio entre cadenas adyacentes.
# Efecto estimado de ~3-5% de reduccion por doble enlace (Piggot 2017).
UNSATURATION_PENALTY = 0.035


def _tail_density_with_unsaturation(lipid_name: str, z_frac: float) -> float:
    """
    Densidad electronica de un segmento de cola en funcion de z_frac [0,1].

    z_frac = 0: inicio de la cola (glicerol)
    z_frac = 1: extremo de la cola (metilo terminal)

    Modela tres efectos fisicos:

    1. INSATURACIONES (Piggot 2017): cada doble enlace reduce la densidad
       local ~3.5% por el kink geometrico cis que crea espacio entre cadenas.
       POPC (1 db en C9 de sn2): depresion central.
       PIPs (4 db en sn2):      depresion marcada en cola aracidonica.
       SM/CHOL (0 db):         sin reduccion, densidad uniforme.

    2. GRADIENTE AXIAL DEL COLESTEROL (Nagle & Tristram-Nagle 2000):
       El anillo esteroide rigido (z_frac 0-0.45) tiene mayor densidad
       que la cadena isooctilo flexible (z_frac 0.45-1.0).
       Anillo:    ~0.302 e/A3 (3 ciclos fusionados, compactos)
       Isooctilo: ~0.280 e/A3 (ramificado, menos empaquetado)
       Esta diferencia axial es detectable en perfiles de ED experimentales.

    3. PLASMALOGENOS (PlsPE): enlace vinil-eter en sn1 sin carbonilo.
       El segmento inicial (z_frac < 0.1) tiene menor ED que POPE
       porque falta el grupo O del enlace ester.
    """
    from lipid_types import LIPID_TYPES
    lt = LIPID_TYPES.get(lipid_name)
    if lt is None:
        return ELECTRON_DENSITY["tail_fluid"]

    # CASO ESPECIAL: gradiente axial del colesterol
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

    El volumen captura:
    - Patron dark-bright-dark de la bicapa (cabezas-colas-cabezas)
    - Contraste entre dominios Lo y Ld por diferencias de empaquetamiento
    - Señal de PIPs en monocapa interna (mayor densidad de cabeza)
    - Efecto de insaturaciones en colas acil
    - Asimetria entre monocapas
    - Fluctuaciones Helfrich corregidas por plano medio local

    Parametros
    ----------
    bins_xy : int
        Resolucion lateral (default 55 → 9 A/voxel para parche de 50 nm)
    bins_z : int
        Resolucion axial (default 80 → mayor resolucion en Z que en XY
        para capturar bien el perfil de la bicapa)
    voxel_angstrom : float
        Tamano de voxel declarado en el MRC. A 9 A es compatible con
        PolNet (10 A) y tiene mayor fidelidad en Z.
    sigma_xy, sigma_z : float
        Suavizado gaussiano en voxels. Simula la PSF del microscopio.

    Retorna
    -------
    (vol, labels, stats)
        vol    : array (bins_xy, bins_xy, bins_z), float32, en e/A3
        labels : array (bins_xy, bins_xy, bins_z), uint8
                 0=agua, 1=cabeza ext, 2=nucleo Lo, 3=nucleo Ld,
                 4=cabeza int, 5=PIP cluster
        stats  : dict con metricas fisicas del volumen
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

    # ── Proteinas transmembrana (perturbaciones no diferenciadas) ──────────
    # Las proteinas se modelan como cilindros de densidad electronica
    # intermedia (~0.40 e/A3) que atraviesan la bicapa de extremo a extremo.
    # Esta densidad es mayor que las colas (0.29-0.31) pero menor que las
    # cabezas (0.44-0.50), reproduciendo el contraste de proteinas en cryo-ET.
    # Referencia: Singer & Nicolson 1972 (modelo de mosaico fluido).
    ED_PROTEIN = 0.400  # densidad tipica de proteina transmembrana en e/A3
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
    }

    mid_y = bins_xy // 2
    mid_x = bins_xy // 2

    with plt.rc_context(PLT):
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
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
        plt.colorbar(im, ax=ax, shrink=0.85, label="e/Å³")
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
        plt.colorbar(im2, ax=ax, shrink=0.85, label="e/Å³")
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
        ax.set_xlabel("Densidad electronica (e/Å³)")
        ax.set_ylabel("Z (nm)")
        ax.set_title("Perfil 1D — ED a lo largo del eje Z\nPatron dark-bright-dark cuantificado")
        ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1, 0]
        head_proj = vol[:, :, :].max(axis=2)
        from analysis import raft_fraction_map
        raft_map = raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins_xy)
        im3 = ax.imshow(head_proj.T, origin="lower", cmap="gray",
                        extent=ext_xy, aspect="equal")
        plt.colorbar(im3, ax=ax, shrink=0.85, label="ED max (e/Å³)")
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
        plt.colorbar(im5, ax=ax, shrink=0.85, label="ED cola (e/Å³)")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Densidad de cola — Efecto insaturaciones\nRojo=saturada (Lo/SM) | Azul=insaturada (Ld/POPC)")

        patches_leyenda = [
            mpatches.Patch(facecolor="#2dc653", label="Cabezas ext (%.3f e/Å³)" % stats["ed_head_mean"]),
            mpatches.Patch(facecolor="#e63946", label="Nucleo Lo (%.3f e/Å³)" % stats["ed_tail_Lo"]),
            mpatches.Patch(facecolor="#3a86ff", label="Nucleo Ld (%.3f e/Å³)" % stats["ed_tail_Ld"]),
            mpatches.Patch(facecolor="#9b5de5", label="PIPs (%.3f e/Å³)" % stats["ed_pip"]),
            mpatches.Patch(facecolor="#f0f4ff", edgecolor="#aaa",
                           label="Agua (%.3f e/Å³)" % stats["ed_water"]),
        ]
        
        fig.legend(handles=patches_leyenda, loc="lower center", ncol=5,
                   fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.0))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        path = os.path.join(save_dir, "model3d_seed%04d.png" % membrane.seed)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % path)
    return path
