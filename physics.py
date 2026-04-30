"""
physics.py
==========
Física de la bicapa: curvatura Helfrich y generación de cadenas acil.

Funciones puras (sin estado): reciben parámetros y devuelven resultados.
No dependen de la clase BicapaCryoET, por lo que pueden probarse
de forma independiente.

Referencias:
  [4]  Helfrich, Z. Naturforsch. C 1973 – energía elástica de membrana
  [5]  Pinigin, Membranes 2022 – espectro de fluctuaciones
  [6]  Chakraborty et al. PNAS 2020 – kc dependiente de composición
  [7]  Piggot et al. J. Chem. Theory Comput. 2017 – parámetro S_CH
  [2]  Smith et al. LiveCoMS 2019 – kinks en dobles enlaces
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
    Calcula el módulo de bending kc (kBT·nm²) a partir de la
    composición de ambas monocapas.

    CHOL y SM condensan las cadenas acil, aumentando kc [6].
    Rango resultante: 18–45 kBT·nm², coherente con Pinigin 2022 [5].

    Parametros
    ----------
    comp_outer, comp_inner : dict {nombre_lipido: fraccion}

    Retorna
    -------
    float
        kc en kBT·nm².
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
    Genera un campo de alturas h(x,y) con espectro de Helfrich completo:

        <|h_q|²> = kBT / (kc·q⁴ + σ·q²)

    donde kc controla los modos de alta q (bending) y σ suprime
    los modos de larga longitud de onda (tensión superficial) [4, 5].

    Parametros
    ----------
    Lx_angstrom : float
        Tamaño lateral del sistema en Å.
    kc : float
        Módulo de bending en kBT·nm².
    sigma : float
        Tensión superficial en kBT/nm².
    rng : numpy Generator
        Generador aleatorio con semilla fijada.
    bins : int
        Resolución del campo (bins × bins).

    Retorna
    -------
    np.ndarray, shape (bins, bins)
        Campo de alturas en Å.
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
    Genera una cadena acil como lista de nseg+1 puntos 3D.

    Implementa el modelo de kink en el doble enlace según Smith et al.
    LiveCoMS 2019 [2] y calcula S_CH por segmento C–C según Piggot et
    al. J. Chem. Theory Comput. 2017 [7].

    Parametros
    ----------
    start : np.ndarray, shape (3,)
        Punto de inicio de la cadena (glicerol desplazado lateralmente).
    nc : int
        Número de carbonos de la cadena.
    ndb : int
        Número de dobles enlaces.
    dbpos : int o None
        Posición del doble enlace (carbono) a lo largo de la cadena.
    direction : int
        +1 para monocapa interna (creciente en z), -1 para externa.
    tilt : float
        Ángulo de inclinación respecto a la normal (radianes).
    phi : float
        Ángulo azimutal (radianes).
    phase : str
        "gel" (desorden gaussiano 0.25 Å) | "fluid" (0.55 Å).
        CORRECCION: reducido de 0.80 a 0.55 para evitar rebotes excesivos
        que _sanitize_tail tenia que corregir agresivamente, dejando colas
        demasiado rectas y cortas. A 0.55 hay suficiente desorden para
        distinguir fluido de gel pero sin violar la progresion monotona.
    rng : Generator
        Generador aleatorio con semilla.
    nseg : int
        Número de segmentos (puntos = nseg + 1).

    Retorna
    -------
    (points, S_CH_mean)
        points : list[np.ndarray]  — puntos 3D de la cadena
        S_CH_mean : float          — S_CH medio de todos los segmentos
    """
    L = nc * 1.26
    dz_base = direction * L / nseg * np.cos(tilt)
    dr = L / nseg * np.sin(tilt)
    # CORRECCION: disorder reducido de 0.80 a 0.55 para fase fluida
    # El valor anterior (0.80) generaba demasiado ruido transversal que
    # hacia que los segmentos "rebotaran" en Z, forzando a _sanitize_tail
    # a truncar la cola agresivamente. Con 0.55 las colas fluidas mantienen
    # su curvatura natural sin violar la progresion monotona en Z.
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