"""
geometry.py
===========
Dataclasses de geometría y de instancias lipídicas individuales.

Separado de lipid_types.py para que las estructuras de datos que
contienen arrays numpy (mutable) no contaminen el módulo de solo
lectura de tipos.

Dependencias: lipid_types.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from lipid_types import LipidType


@dataclass
class MembraneGeometry:
    """
    Parámetros geométricos medios de la bicapa, calculados a partir
    de las fracciones de composición lipídica.

    Todas las magnitudes en Å.
    """
    hydro_thick: float
    total_thick: float
    z_outer: float
    z_inner: float
    z_mid: float

    def __str__(self):
        return (
            "MembraneGeometry(total=%.1f A, hidro=%.1f A, "
            "z_outer=%.1f A, z_inner=%.1f A)"
            % (self.total_thick, self.hydro_thick, self.z_outer, self.z_inner)
        )


@dataclass
class LipidInstance:
    """
    Instancia individual de un lípido en la bicapa.

    Cada campo de posición es un array numpy 3D en Å.
    Las colas son listas de puntos desde el glicerol hasta el carbono
    terminal, con nseg+1 puntos (nseg=9 por defecto).
    """
    lipid_type: LipidType
    leaflet: str
    head_pos: np.ndarray
    glycerol_pos: np.ndarray
    tail1: List[np.ndarray]
    tail2: Optional[List[np.ndarray]]
    order_param: float
    in_raft: bool
    is_pip: bool

    @property
    def x(self):
        return self.head_pos[0]

    @property
    def y(self):
        return self.head_pos[1]

    @property
    def z(self):
        return self.head_pos[2]

    @property
    def name(self):
        return self.lipid_type.name
