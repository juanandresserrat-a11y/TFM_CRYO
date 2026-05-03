"""
physics.py
Física de la bicapa: curvatura Helfrich y generación de cadenas acil.

Funciones puras (sin estado): reciben parámetros y devuelven resultados.
No dependen de la clase BicapaCryoET, por lo que pueden probarse
de forma independiente.

Referencias principales:
    [3]  Chakraborty et al. 2020 – dependencia del módulo de bending (kc) con la composición lipídica de la membrana
    [10] Helfrich 1973 – energía elástica de membranas y descripción de la curvatura en bicapas
    [20] Piggot et al. 2017 – cálculo del parámetro de orden acil S_CH desde simulaciones moleculares
    [21] Pinigin 2022 – espectro de fluctuaciones de membrana y parámetros elásticos efectivos
    [26] Smith et al. 2019 – modelado de kinks en cadenas acil y geometría de dobles enlaces en lípidos
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import Generator

from lipid_types import LIPID_TYPES


_KC_WEIGHTS: Dict[str, float] = {
    "POPC":   1.00, "POPE":   1.05, "POPS":  1.10,
    "PI":     1.00, "PI3P":   1.00, "PI4P":  1.00,
    "PI5P":   1.00, "PI34P2": 1.05,
    "PIP2":   1.20, "PIP3":   1.30,
    "SM":     1.60, "CHOL":   2.00, "GM1":   1.80,
}


def bending_modulus_from_composition(
    comp_outer: Dict[str, float],
    comp_inner: Dict[str, float],
) -> float:
    """
    Calcula el módulo de bending kc (kBT·nm²) a partir de la composición
    lipídica de ambas monocapas.

    La presencia de colesterol y esfingomielina incrementa la rigidez de
    la membrana al condensar el empaquetamiento de las cadenas acilo.

    comp_outer : dict
        Composición de la monocapa externa {lipido: fracción}
    comp_inner : dict
        Composición de la monocapa interna {lipido: fracción}
    """
    def leaflet_kc(comp):
        return sum(_KC_WEIGHTS.get(k, 1.0) * f for k, f in comp.items())

    kc_norm = 0.5 * (leaflet_kc(comp_outer) + leaflet_kc(comp_inner))
    return float(np.clip(20.0 + 20.0 * (kc_norm - 1.0), 18.0, 45.0))


def generate_helfrich_map(
    Lx_angstrom: float,
    kc: float,
    sigma: float,
    rng: Generator,
    bins: int = 64,
) -> np.ndarray:
    """
    Campo de alturas h(x,y) con espectro de Helfrich.

    kc controla bending, σ la tensión superficial.
    """
    L_nm = Lx_angstrom / 10.0
    qx = 2.0 * np.pi * np.fft.fftfreq(bins, d=L_nm / bins)
    Qx, Qy = np.meshgrid(qx, qx)
    Q2 = Qx**2 + Qy**2
    Q4 = Q2**2

    denom = kc * Q4 + sigma * Q2
    denom[0, 0] = 1.0

    sigma_q = np.sqrt(1.0 / denom) * (L_nm / bins)
    sigma_q[0, 0] = 0.0

    noise = (
        rng.standard_normal((bins, bins))
        + 1j * rng.standard_normal((bins, bins))
    ) * sigma_q

    h_nm = np.real(np.fft.ifft2(noise)) * bins
    return h_nm * 10.0


def generate_tail(
    start: np.ndarray,
    nc: int,
    ndb: int,
    dbpos: Optional[int],
    direction: int,
    tilt: float,
    phi: float,
    phase: str,
    rng: Generator,
    nseg: int = 9,
) -> Tuple[List[np.ndarray], float]:
    """
    Genera una cadena acil como lista de puntos 3D.

    Implementa el kink en dobles enlaces y calcula
    el parámetro de orden S_CH por segmento.
    """
    L = nc * 1.26
    dz_base = direction * L / nseg * np.cos(tilt)
    dr = L / nseg * np.sin(tilt)
    disorder = 0.25 if phase == "gel" else 0.55

    points = [start.copy()]
    current = start.copy()
    order_vals = []

    for s in range(nseg):
        frac = (s + 0.5) / nseg
        kink = 0.0
        if ndb > 0 and dbpos is not None:
            kink = 2.2 * np.exp(-((frac - dbpos / nc) ** 2) / 0.018)

        jx = rng.normal(0, disorder) + kink * np.cos(phi + np.pi / 2)
        jy = rng.normal(0, disorder) + kink * np.sin(phi + np.pi / 2)
        dx = dr * np.cos(phi) + jx
        dy = dr * np.sin(phi) + jy
        dz = dz_base

        seg_len = np.sqrt(dx**2 + dy**2 + dz**2)
        if seg_len > 0:
            cos_theta = abs(dz) / seg_len

            order_vals.append((3.0 * cos_theta**2 - 1.0) / 2.0)

        current = current + np.array([dx, dy, dz])
        points.append(current.copy())

    S_CH = float(np.mean(order_vals)) if order_vals else 0.5
    return points, S_CH