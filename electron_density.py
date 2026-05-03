"""
electron_density.py
Densidad electronica correcta para simulacion de contraste cryo-ET.

La cryo-ET mide la dispersion de electrones, no la masa molecular.
Las cabezas polares (P, N, O) dispersan mas electrones que las colas
acil (C, H), generando el contraste caracteristico de doble banda
oscura-clara-oscura.

Este modulo calcula:
  1. Perfiles de densidad electronica por region de la bicapa
  2. Volumen 3D de densidad electronica
  3. Proyecciones XY con contraste fisico

Valores de referencia (e/Å³):
  Agua bulk: 0.334
  Cabezas polares: 0.44–0.47
  Glicerol: 0.38–0.40
  Colas acil: 0.28–0.31
  Nucleo CH2: 0.29

Referencias principales:
    [11] Kučerka et al. 2008 – determinación experimental de espesores y áreas de densidad electrónica en bicapas lipídicas
    [18] Nagle & Tristram-Nagle 2000 – modelo estructural de bicapas y perfiles de densidad electrónica en sistemas lipídicos
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from builder import BicapaCryoET


ELECTRON_DENSITY = {
    "head_outer": 0.460,
    "head_inner": 0.455,
    "glycerol":   0.390,
    "tail_gel":   0.290,
    "tail_fluid": 0.295,
    "water":      0.334,
}

LIPID_ED_HEADGROUP: Dict[str, float] = {
    "POPC":   0.458,
    "POPE":   0.452,
    "POPS":   0.461,
    "PI":     0.468,
    "PI3P":   0.475,
    "PI4P":   0.476,
    "PI5P":   0.474,
    "PI34P2": 0.483,
    "PIP2":   0.490,
    "PIP3":   0.498,
    "SM":     0.462,
    "CHOL":   0.385,
    "GM1":    0.472,
    "PlsPE":  0.448,   # sin carbonilo ester en sn1 → menor que POPE (0.452)
}

LIPID_ED_TAIL: Dict[str, float] = {
    "POPC":   0.294,
    "POPE":   0.293,
    "POPS":   0.293,
    "PI":     0.291,
    "PI3P":   0.291,
    "PI4P":   0.291,
    "PI5P":   0.291,
    "PI34P2": 0.291,
    "PIP2":   0.291,
    "PIP3":   0.291,
    "SM":     0.287,
    "CHOL":   0.302,
    "GM1":    0.289,
    "PlsPE":  0.291,   # sn2 identico a POPE, sn1 vinilo-eter similar
}


def electron_density_profile(
    membrane: "BicapaCryoET",
    bins_z: int = 200,
    sigma: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perfil 1D de densidad electronica a lo largo del eje Z.

    Distingue las cabezas polares (alta densidad electronica)
    del nucleo hidrofobico (baja densidad), reproduciendo el
    patron dark-bright-dark observado en cryo-ET.

    Retorna
      (z_centers, ed_profile)
      z_centers en Å, ed_profile en electrones/Å³
    """
    todos = membrane.outer_leaflet + membrane.inner_leaflet
    zs = np.array([l.head_pos[2] for l in todos])
    z_min = zs.min() - 5.0
    z_max = zs.max() + 5.0
    z_edges = np.linspace(z_min, z_max, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = z_edges[1] - z_edges[0]

    ed = np.full(bins_z, ELECTRON_DENSITY["water"])

    g = membrane.geometry

    for l in todos:
        lname = l.lipid_type.name
        ed_head = LIPID_ED_HEADGROUP.get(lname, 0.460)
        ed_tail = LIPID_ED_TAIL.get(lname, 0.292)
        ed_glyc = 0.390

        z_h = l.head_pos[2]
        z_g = l.glycerol_pos[2]
        hg_thick = l.lipid_type.hg_thick
        glyc_thick = l.lipid_type.glyc_offset

        sign = -1 if l.leaflet == "sup" else 1

        iz_head = np.clip(
            np.searchsorted(z_edges, z_h) - 1, 0, bins_z - 1
        )
        hg_safe = max(hg_thick, 1.0)
        i_range_head = max(1, int(hg_safe / dz))
        for diz in range(-i_range_head, i_range_head + 1):
            iz = np.clip(iz_head + diz, 0, bins_z - 1)
            w = np.exp(-0.5 * (diz * dz / (hg_safe * 0.4)) ** 2)
            ed[iz] = ed[iz] * (1 - w) + ed_head * w

        iz_glyc = np.clip(
            np.searchsorted(z_edges, z_g) - 1, 0, bins_z - 1
        )
        glyc_safe = max(glyc_thick, 0.5)
        i_range_glyc = max(1, int(glyc_safe / dz))
        for diz in range(-i_range_glyc, i_range_glyc + 1):
            iz = np.clip(iz_glyc + diz, 0, bins_z - 1)
            w = 0.5 * np.exp(-0.5 * (diz * dz / (glyc_safe * 0.5)) ** 2)
            ed[iz] = ed[iz] * (1 - w) + ed_glyc * w

        if l.tail1 and len(l.tail1) > 2:
            z_tail_start = z_g
            z_tail_end = l.tail1[-1][2]
            iz_s = np.clip(int((z_tail_start - z_min) / dz), 0, bins_z - 1)
            iz_e = np.clip(int((z_tail_end - z_min) / dz), 0, bins_z - 1)
            iz_lo, iz_hi = min(iz_s, iz_e), max(iz_s, iz_e)
            for iz in range(iz_lo, iz_hi + 1):
                w = 0.3
                ed[iz] = ed[iz] * (1 - w) + ed_tail * w

    ed_smooth = gaussian_filter(ed, sigma=sigma / dz)
    return z_centers, ed_smooth


def electron_density_volume(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 40,
    sigma_xy: float = 1.5,
    sigma_z: float = 0.8,
) -> Tuple[np.ndarray, tuple]:
    """
    Volumen 3D de densidad electronica en electrones/Å³.

    Cada lipido aporta su contribucion segun region:
      - Cabezas polares: alta densidad electronica
      - Colas acil: baja densidad electronica
      - Agua: fondo basal

    El resultado es compatible con volumetric_density() de analysis.py
    y puede usarse en export_mrc.py para mayor realismo fisico.
    """
    g = membrane.geometry
    z_half = (g.total_thick / 10.0) / 2.0 + 0.5

    from analysis import midplane_map
    z_mid_grid = midplane_map(membrane, bins=bins_xy)

    vol = np.full(
        (bins_xy, bins_xy, bins_z),
        ELECTRON_DENSITY["water"],
        dtype=np.float32,
    )

    x_edges = np.linspace(0, membrane.Lx / 10, bins_xy + 1)
    y_edges = np.linspace(0, membrane.Ly / 10, bins_xy + 1)
    z_edges = np.linspace(-z_half, z_half, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz = z_edges[1] - z_edges[0]

    todos = membrane.outer_leaflet + membrane.inner_leaflet

    for l in todos:
        lname = l.lipid_type.name
        ed_head = LIPID_ED_HEADGROUP.get(lname, 0.460)
        ed_tail = LIPID_ED_TAIL.get(lname, 0.292)

        ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
        iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)
        z_ref = z_mid_grid[ix, iy]

        z_h_rel = (l.head_pos[2] - z_ref) / 10.0
        z_g_rel = (l.glycerol_pos[2] - z_ref) / 10.0
        hg_half = l.lipid_type.hg_thick / 10.0 / 2.0

        for iz, zc in enumerate(z_centers):
            w_head = np.exp(-0.5 * ((zc - z_h_rel) / hg_half) ** 2)
            if w_head > 0.05:
                vol[ix, iy, iz] = (
                    vol[ix, iy, iz] * (1 - w_head) + ed_head * w_head
                )

            if l.tail1 and len(l.tail1) > 2:
                z_tail_end_rel = (l.tail1[-1][2] - z_ref) / 10.0
                z_tail_lo = min(z_g_rel, z_tail_end_rel)
                z_tail_hi = max(z_g_rel, z_tail_end_rel)
                if z_tail_lo <= zc <= z_tail_hi:
                    w_tail = 0.4
                    vol[ix, iy, iz] = (
                        vol[ix, iy, iz] * (1 - w_tail) + ed_tail * w_tail
                    )

    vol_smooth = gaussian_filter(vol, sigma=[sigma_xy, sigma_xy, sigma_z])

    edges = (x_edges, y_edges, z_edges)
    return vol_smooth, edges


def electron_density_projection(
    membrane: "BicapaCryoET",
    bins_xy: int = 90,
    sigma: float = 1.8,
) -> np.ndarray:
    """
    Proyeccion XY de la densidad electronica integrada en Z.
    Simula la imagen de cryo-ET con contraste fisicamente correcto:
    cabezas polares = zonas densas, nucleo = zona clara.
    """
    vol, edges = electron_density_volume(membrane, bins_xy=bins_xy, bins_z=40)
    proj = vol.sum(axis=2)
    return gaussian_filter(proj, sigma=sigma)
