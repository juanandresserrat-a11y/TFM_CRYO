"""
lipid_types.py
==============
Definiciones de tipos lipídicos y composiciones de membrana.

Este módulo es de solo lectura: no depende de ningún otro módulo
del paquete. Todos los demás módulos lo importan.

Referencias:
  [17] Kucerka et al. Biochim. Biophys. Acta 2011 – áreas y grosores
  [10] Daleke, J. Lipid Res. 2003 – asimetría composicional
  [11] Di Paolo & De Camilli, Nature 2006 – fosfoinosítidos
"""

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class LipidType:
    """
    Propiedades biofísicas de una especie lipídica.

    Todos los valores de longitud están en Å; la masa en Da.
    El campo pip_order indica el número de grupos fosfato adicionales
    en el anillo de inositol (0 = no-PIP, 1 = monofosfato, etc.).
    """
    name: str
    area: float
    tail_length: float
    hg_thick: float
    hg_radius: float
    glyc_offset: float
    nc: Tuple[int, int]
    ndb: Tuple[int, int]
    dbpos: Tuple[Optional[int], Optional[int]]
    mass: float
    phase: str
    is_raft: bool
    charge: int
    color: str
    color_tail1: str
    color_tail2: str
    pip_order: int = 0


LIPID_TYPES = {

    "POPC": LipidType(
        "POPC", 64.3, 14.5, 9.0, 4.5, 3.5,
        (16, 18), (0, 1), (None, 9),
        760.1, "fluid", False, 0,
        "#3a86ff", "#90c8ff", "#1a5fbf",
    ),
    "POPE": LipidType(
        "POPE", 59.0, 14.3, 8.5, 3.8, 3.5,
        (16, 18), (0, 1), (None, 9),
        717.0, "fluid", False, 0,
        "#e63946", "#f4a5aa", "#9b1e28",
    ),
    # Plasmalogeno PE: enlace vinil-eter en sn1.
    # ~18-20% de los PE de membrana plasmatica de mamifero.
    # head_ED=0.448 (sin carbonilo ester en sn1 vs POPE 0.452).
    # Masa: 699 Da (15 Da menos que POPE, sin O del enlace ester).
    # Referencia: Braverman & Moser, BBA 2012.
    "PlsPE": LipidType(
        "PlsPE", 59.5, 14.3, 8.2, 3.8, 3.5,
        (16, 18), (0, 1), (None, 9),
        699.0, "fluid", False, 0,
        "#c0392b", "#e57373", "#7b0000",
    ),
    "POPS": LipidType(
        "POPS", 59.7, 14.2, 9.5, 4.0, 3.5,
        (16, 18), (0, 1), (None, 9),
        761.0, "fluid", False, -1,
        "#fb8500", "#ffd166", "#c84b00",
    ),

    "PI": LipidType(
        "PI", 67.0, 14.5, 10.0, 5.5, 3.5,
        (18, 20), (0, 4), (None, 5),
        857.0, "fluid", False, -1,
        "#9b5de5", "#c897f5", "#5a189a",
    ),
    "PI3P": LipidType(
        "PI3P", 68.5, 14.5, 11.0, 6.0, 3.5,
        (18, 20), (0, 4), (None, 5),
        937.0, "fluid", False, -2,
        "#c77dff", "#e0aaff", "#7b2d8b", 1,
    ),
    "PI4P": LipidType(
        "PI4P", 69.0, 14.5, 11.2, 6.0, 3.5,
        (18, 20), (0, 4), (None, 5),
        937.0, "fluid", False, -2,
        "#7209b7", "#b5179e", "#480ca8", 1,
    ),
    "PI5P": LipidType(
        "PI5P", 68.0, 14.5, 10.8, 5.8, 3.5,
        (18, 20), (0, 4), (None, 5),
        937.0, "fluid", False, -2,
        "#560bad", "#7b2fbe", "#3a0ca3", 1,
    ),
    "PI34P2": LipidType(
        "PI(3,4)P2", 72.0, 14.8, 12.5, 6.8, 3.5,
        (18, 20), (0, 4), (None, 5),
        1017.0, "fluid", False, -3,
        "#f72585", "#ff85c2", "#b5179e", 2,
    ),
    "PIP2": LipidType(
        "PIP2", 75.0, 14.8, 13.0, 7.2, 3.5,
        (18, 20), (0, 4), (None, 5),
        1017.0, "fluid", False, -4,
        "#f1c40f", "#f8e08a", "#c09000", 2,
    ),
    "PIP3": LipidType(
        "PIP3", 80.0, 14.8, 14.5, 8.0, 3.5,
        (18, 20), (0, 4), (None, 5),
        1097.0, "fluid", False, -5,
        "#ff6b35", "#ffb494", "#c23b00", 3,
    ),


    "SM": LipidType(
        "SM", 47.0, 16.5, 10.0, 4.5, 4.0,
        (18, 24), (1, 0), (4, None),
        731.0, "gel", True, 0,
        "#2dc653", "#95e8b4", "#0a6e2d",
    ),
    "CHOL": LipidType(
        "CHOL", 38.5, 17.0, 4.0, 2.2, 0.0,
        (27, 0), (1, 0), (5, None),
        386.7, "gel", True, 0,
        "#adb5bd", "#dee2e6", "#6c757d",
    ),
    "GM1": LipidType(
        "GM1", 85.0, 17.0, 15.0, 9.0, 4.0,
        (18, 20), (1, 0), (4, None),
        1545.0, "gel", True, -1,
        "#d4a017", "#f0d080", "#8a6200",
    ),
}


COMP_OUTER_BASE = {
    "POPC": 0.33,
    "SM":   0.24,
    "CHOL": 0.30,
    "GM1":  0.05,
    "PI":   0.04,
    "POPE": 0.04,
}

COMP_INNER_BASE = {
    "POPC":   0.18,
    "POPE":   0.19,
    "PlsPE":  0.05,   # ~20% de los PE son plasmalogenos (Braverman & Moser 2012)
    "POPS":   0.14,
    "CHOL":   0.28,
    "PI":     0.05,
    "PIP2":   0.04,
    "PI4P":   0.03,
    "PI3P":   0.02,
    "PIP3":   0.01,
    "PI34P2": 0.01,
}


DIRICHLET_CONCENTRATION = 50
