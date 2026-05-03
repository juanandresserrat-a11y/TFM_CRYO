"""
analysis.py
Funciones de análisis cuantitativo de la bicapa.

Todas las funciones reciben un objeto BicapaCryoET ya construido
y devuelven arrays numpy. No generan figuras ni escriben archivos.

Esto permite:
  - Usar los mapas en training sin necesidad de matplotlib
  - Probar los análisis de forma independiente de la visualización
  - Cachear resultados costosos externamente

Referencias principales:

    [3]  Chakraborty et al. 2020 – efecto del colesterol en la rigidez de membranas insaturadas
    [4]  Chaisson 2025 – cuantificación de la interdigitación en bicapas simuladas
    [5]  Kučerka 2011 – espesores de bicapa y áreas lipídicas en PC comunes
    [7]  Singer & Nicolson 1972 – modelo de mosaico fluido de membranas
    [8]  Smith et al. 2018 – buenas prácticas en simulación de membranas lipídicas
    [9]  Martinez-Sanchez 2024 – generación de datasets sintéticos para cryo-ET
    [10] Helfrich 1973 – elasticidad de membranas y modelo de fluctuaciones
    [14] Liu et al. 2021 – simulaciones de membranas a doble resolución
    [15] Lučič et al. 2013 – cryo-electron tomography in situ
    [17] Moebel et al. 2021 – deep learning en cryo-ET
    [20] Piggot 2017 – cálculo de parámetros de orden S_CH
    [21] Pinigin 2022 – parámetros elásticos en membranas lipídicas
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

if TYPE_CHECKING:
    from builder import BicapaCryoET
    from geometry import LipidInstance


def density_map(
    membrane: "BicapaCryoET",
    lipids: List["LipidInstance"],
    bins: int = 90,
    sigma: float = 1.8,
) -> np.ndarray:
    """Mapa de densidad de masa (Da/Å²) proyectado en XY."""
    H = np.zeros((bins, bins))
    ab = (membrane.Lx / bins) * (membrane.Ly / bins)
    for lip in lipids:
        xi = int(lip.head_pos[0] / membrane.Lx * bins) % bins
        yi = int(lip.head_pos[1] / membrane.Ly * bins) % bins
        H[xi, yi] += lip.lipid_type.mass
    return gaussian_filter(H / ab, sigma=sigma)


def roughness_map(
    membrane: "BicapaCryoET",
    lipids: List["LipidInstance"],
    bins: int = 70,
    sigma: float = 1.2,
) -> np.ndarray:
    """
    Rugosidad local σ_z(x,y): desviación estándar de z_cabeza por celda.
    Diferencia Lo (~0.6 Å) de Ld (~1.8 Å).
    """
    S, S2, C = (np.zeros((bins, bins)) for _ in range(3))
    for lip in lipids:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        z = lip.head_pos[2]
        S[ix, iy] += z
        S2[ix, iy] += z * z
        C[ix, iy] += 1
    with np.errstate(all="ignore"):
        mu = np.where(C > 0, S / C, 0)
        var = np.where(C > 1, S2 / C - mu**2, 0)
    return gaussian_filter(np.sqrt(np.clip(var, 0, None)), sigma=sigma)


def thickness_map(
    membrane: "BicapaCryoET",
    bins: int = 70,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Grosor local (Å): diferencia z_cabeza_sup − z_cabeza_inf por celda.
    Lo (SM/CHOL) genera ~4 Å más de grosor que Ld (POPC).
    """
    def avg_z(lipids):
        S, C = np.zeros((bins, bins)), np.zeros((bins, bins))
        for lip in lipids:
            ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
            iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
            S[ix, iy] += lip.head_pos[2]
            C[ix, iy] += 1
        with np.errstate(all="ignore"):
            return np.where(C > 0, S / C, np.nan)

    d = avg_z(membrane.outer_leaflet) - avg_z(membrane.inner_leaflet)
    return gaussian_filter(np.nan_to_num(d, nan=float(np.nanmean(d))), sigma)


def raft_fraction_map(
    membrane: "BicapaCryoET",
    lipids: List["LipidInstance"],
    bins: int = 90,
    sigma: float = 2.5,
) -> np.ndarray:
    """Fracción local de lípidos en fase Lo (raft) por celda."""
    Hr, Ht = np.zeros((bins, bins)), np.zeros((bins, bins))
    for lip in lipids:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        Ht[ix, iy] += 1
        if lip.in_raft:
            Hr[ix, iy] += 1
    with np.errstate(all="ignore"):
        return gaussian_filter(np.where(Ht > 0, Hr / Ht, 0), sigma=sigma)


def pip_density_map(
    membrane: "BicapaCryoET",
    bins: int = 90,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Densidad de fosfoinosítidos (Da/Å²) en la monocapa interna.
    Los PIPs de mayor orden reciben peso adicional proporcional.
    """
    H = np.zeros((bins, bins))
    ab = (membrane.Lx / bins) * (membrane.Ly / bins)
    for lip in membrane.inner_leaflet:
        if not lip.is_pip:
            continue
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        H[ix, iy] += (
            (1 + lip.lipid_type.pip_order * 0.5) * lip.lipid_type.mass
        )
    return gaussian_filter(H / ab, sigma=sigma)


def order_parameter_map(
    membrane: "BicapaCryoET",
    bins: int = 70,
    sigma: float = 1.5,
) -> np.ndarray:
    """
    Mapa 2D de S_CH medio por celda (ambas monocapas).

    Lo (gel): S_CH ~0.85–0.95 | Ld (fluido): S_CH ~0.60–0.75.
    """
    S_sum = np.zeros((bins, bins))
    C = np.zeros((bins, bins))
    for lip in membrane.outer_leaflet + membrane.inner_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        S_sum[ix, iy] += lip.order_param
        C[ix, iy] += 1
    with np.errstate(all="ignore"):
        return gaussian_filter(np.where(C > 0, S_sum / C, 0), sigma=sigma)


def midplane_map(
    membrane: "BicapaCryoET",
    bins: int = 70,
) -> np.ndarray:
    """
    Plano medio local z_mid(x,y) = (z_sup + z_inf) / 2.

    Suavizado gaussiano para reducir ruido en bins con pocos lípidos.
    Necesario para corregir la ondulación Helfrich antes de calcular
    la interdigitación.
    """
    S_s, C_s = np.zeros((bins, bins)), np.zeros((bins, bins))
    S_i, C_i = np.zeros((bins, bins)), np.zeros((bins, bins))
    for lip in membrane.outer_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        S_s[ix, iy] += lip.head_pos[2]
        C_s[ix, iy] += 1
    for lip in membrane.inner_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        S_i[ix, iy] += lip.head_pos[2]
        C_i[ix, iy] += 1
    S_s_sm = gaussian_filter(S_s, sigma=2.5)
    C_s_sm = gaussian_filter(C_s, sigma=2.5)
    S_i_sm = gaussian_filter(S_i, sigma=2.5)
    C_i_sm = gaussian_filter(C_i, sigma=2.5)
    with np.errstate(all="ignore"):
        z_s = np.where(C_s_sm > 0.1, S_s_sm / C_s_sm, membrane.geometry.z_outer)
        z_i = np.where(C_i_sm > 0.1, S_i_sm / C_i_sm, membrane.geometry.z_inner)
    return (z_s + z_i) / 2.0


def interdigitation_map(
    membrane: "BicapaCryoET",
    bins: int = 70,
    sigma: float = 2.0,
) -> np.ndarray:
    """
    Penetración normalizada de colas acil trans-leaflet con cooperatividad
    de dominio Lo (CHOL–SM).
    """
    z_mid = midplane_map(membrane, bins=bins)
    score_sum = np.zeros((bins, bins))
    count = np.zeros((bins, bins))

    for lip in membrane.outer_leaflet + membrane.inner_leaflet:
        ix = min(int(lip.head_pos[0] / membrane.Lx * bins), bins - 1)
        iy = min(int(lip.head_pos[1] / membrane.Ly * bins), bins - 1)
        z_ref = z_mid[ix, iy]

        half_len = lip.lipid_type.tail_length / 2.0
        if half_len < 1.0:
            continue
        phase_factor = 1.0
        if lip.lipid_type.name == "CHOL":
            phase_factor = 2.2 if lip.in_raft else 0.4
        elif lip.lipid_type.name == "SM" and lip.in_raft:
            phase_factor = 1.15

        for tail in [lip.tail1, lip.tail2]:
            if not tail or len(tail) < 2:
                continue
            z_end = tail[-1][2]
            pen = (z_ref - z_end) if lip.leaflet == "sup" else (z_end - z_ref)
            score_sum[ix, iy] += max(0.0, pen / half_len) * phase_factor
            count[ix, iy] += 1

    with np.errstate(all="ignore"):
        raw = np.where(count > 0, score_sum / count, 0)
    R = raft_fraction_map(
        membrane, membrane.outer_leaflet + membrane.inner_leaflet, bins=bins
    )
    cooperative_boost = 1.0 + 0.20 * R
    raw = raw * cooperative_boost

    return gaussian_filter(raw, sigma=sigma)


def z_profile(
    membrane: "BicapaCryoET",
    bins: int = 160,
    sigma: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """ Perfil de densidad de masa a lo largo del eje Z."""
    todos = membrane.outer_leaflet + membrane.inner_leaflet
    zs = np.array([l.head_pos[2] for l in todos])
    ms = np.array([l.lipid_type.mass for l in todos])
    edges = np.linspace(zs.min() - 3, zs.max() + 3, bins + 1)
    H = np.zeros(bins)
    for z, m in zip(zs, ms):
        ix = np.clip(np.searchsorted(edges, z) - 1, 0, bins - 1)
        H[ix] += m
    return 0.5 * (edges[:-1] + edges[1:]), gaussian_filter(H, sigma=sigma)


def xz_projection(
    membrane: "BicapaCryoET",
    bx: int = 220,
    bz: int = 110,
    sigma: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Proyección de densidad en el plano XZ (sección transversal).

    Las cabezas polares reciben peso 1.5× frente a las colas (0.5×),
    mejorando el contraste de doble banda.
    """
    todos = membrane.outer_leaflet + membrane.inner_leaflet
    xs = np.array([l.head_pos[0] for l in todos]) / 10.0
    zs = np.array([l.head_pos[2] for l in todos]) / 10.0
    g = membrane.geometry
    ms = np.array([
        l.lipid_type.mass * (
            1.5 if abs(l.head_pos[2] - (
                g.z_outer if l.leaflet == "sup" else g.z_inner
            )) < 5 else 0.5
        )
        for l in todos
    ])
    zr = [zs.min() - 0.3, zs.max() + 0.3]
    H, xe, ze = np.histogram2d(
        xs, zs, bins=[bx, bz],
        range=[[0, membrane.Lx / 10], zr], weights=ms,
    )
    return gaussian_filter(H, sigma=sigma), xe, ze


def volumetric_density(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 40,
) -> Tuple[np.ndarray, Tuple]:
    """
    Densidad de masa 3D con coordenadas Z relativas al plano medio local.

    Elimina el efecto de la ondulación Helfrich en los slices,
    centrando la bicapa en z=0 local.
    """
    todos = membrane.outer_leaflet + membrane.inner_leaflet
    xs = np.array([l.head_pos[0] for l in todos]) / 10.0
    ys = np.array([l.head_pos[1] for l in todos]) / 10.0
    ms = np.array([l.lipid_type.mass for l in todos])

    z_mid_grid = midplane_map(membrane, bins=bins_xy)
    zs_rel = np.zeros(len(todos))
    for i, l in enumerate(todos):
        ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
        iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)
        zs_rel[i] = (l.head_pos[2] - z_mid_grid[ix, iy]) / 10.0

    g = membrane.geometry
    z_half = (g.total_thick / 10.0) / 2.0 + 0.5
    H, edges = np.histogramdd(
        np.column_stack([xs, ys, zs_rel]),
        bins=[bins_xy, bins_xy, bins_z],
        range=[[0, membrane.Lx / 10], [0, membrane.Ly / 10], [-z_half, z_half]],
        weights=ms,
    )
    return gaussian_filter(H, sigma=[1.2, 1.2, 0.8]), edges