"""
builder.py
Construcción y organización de la bicapa.

Referencias principales:
    [3]  Chakraborty et al. 2020 – dependencia del módulo de bending con composición lipídica
    [4]  Helfrich 1973 – elasticidad de membranas y curvatura
    [5]  Pinigin 2022 – espectro de fluctuaciones y parámetros elásticos
    [6]  Di Paolo & De Camilli 2006 – regulación de fosfoinosítidos (PIPs)
    [11] Kučerka et al. 2008 – espesores y áreas lipídicas en bicapas
    [12] Simons & Ikonen 1997 – organización en lipid rafts
    [13] Lingwood & Simons 2010 – rafts como principio organizador de membrana
    [14] Liu et al. 2021 – simulaciones de membranas a doble resolución
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.random import Generator, default_rng
from scipy.spatial import KDTree

from geometry import LipidInstance, MembraneGeometry
from lipid_types import (
    COMP_INNER_BASE, COMP_OUTER_BASE,
    DIRICHLET_CONCENTRATION, LIPID_TYPES, LipidType,
)
from physics import (
    bending_modulus_from_composition,
    generate_helfrich_map,
    generate_tail,
)

OUTPUT_DIR = "CryoET"


class BicapaCryoET:
    """Instantánea estática de membrana plasmática de mamífero."""

    def __init__(self, size_nm: Tuple[float, float] = (50, 50), seed: int = 42):
        self.seed = seed
        self.rng: Generator = default_rng(seed)
        self.size_nm = size_nm
        self.Lx = size_nm[0] * 10
        self.Ly = size_nm[1] * 10

        self.outer_leaflet: List[LipidInstance] = []
        self.inner_leaflet: List[LipidInstance] = []
        self.perturbations: List[Dict] = []
        self.rafts_outer: List[List[LipidInstance]] = []
        self.rafts_inner: List[List[LipidInstance]] = []
        self.pip_clusters: List[List[LipidInstance]] = []

        self.geometry: Optional[MembraneGeometry] = None
        self.curvature_map: Optional[np.ndarray] = None
        self.bending_modulus: float = 25.0
        self.surface_tension: float = 0.01
        self.perturbation_density: float = 0.012

        self.comp_outer: Dict[str, float] = {}
        self.comp_inner: Dict[str, float] = {}

    # Directorios

    def seed_dir(self) -> str:
        d = os.path.join(OUTPUT_DIR, "seed%04d" % self.seed)
        os.makedirs(d, exist_ok=True)
        return d

    def training_dir(self) -> str:
        d = os.path.join(OUTPUT_DIR, "training", "seed%04d" % self.seed)
        os.makedirs(d, exist_ok=True)
        return d

    # Composición

    def _random_composition(
        self,
        base: Dict[str, float],
        concentration: float = DIRICHLET_CONCENTRATION,
    ) -> Dict[str, float]:
        """
        Composición aleatoria mediante distribución Dirichlet.
        CV ~12% reproduce la variabilidad biológica intercelular.
        """
        alpha = np.array([base.get(k, 0.0) for k in LIPID_TYPES]) * concentration
        values = self.rng.dirichlet(alpha + 1e-8)
        return {k: v for k, v in zip(LIPID_TYPES, values) if v > 0.01}

    # Geometría

    def _calculate_geometry(self) -> MembraneGeometry:
        """
        Grosor y posición de cabezas calculados desde las fracciones
        lipídicas. Datos de Kucerka et al. 2011.
        """
        def mean_prop(comp, prop):
            return sum(
                getattr(LIPID_TYPES[k], prop) * f
                for k, f in comp.items() if k in LIPID_TYPES
            )

        tail_o = mean_prop(self.comp_outer, "tail_length")
        tail_i = mean_prop(self.comp_inner, "tail_length")
        hg_o   = mean_prop(self.comp_outer, "hg_thick")
        hg_i   = mean_prop(self.comp_inner, "hg_thick")
        glyc_o = mean_prop(self.comp_outer, "glyc_offset")
        glyc_i = mean_prop(self.comp_inner, "glyc_offset")

        hydro = tail_o + tail_i
        total = hydro + hg_o + hg_i + glyc_o + glyc_i

        return MembraneGeometry(
            hydro_thick=hydro,
            total_thick=total,
            z_outer=tail_o + glyc_o,
            z_inner=-(tail_i + glyc_i),
            z_mid=(tail_o + glyc_o - tail_i - glyc_i) / 2.0,
        )

    def get_local_z(self, x: float, y: float, bins: int = 64) -> float:
        """Interpola la curvatura Helfrich en la posición (x, y)."""
        if self.curvature_map is None:
            return 0.0
        ix = int(x / self.Lx * bins) % bins
        iy = int(y / self.Ly * bins) % bins
        return float(self.curvature_map[ix, iy])

    def _generate_chol_body(self, glycerol_pos, sign, phi, rng):
        """
        Cuerpo del colesterol: anillos esteroide A-D + isooctilo.

        Espiral compacta de 12 segmentos con progresión Z monótona.
        """
        segments = []
        n_seg = 12
        height = 17.0
        seg_dz = height / n_seg  # ~1.42 Å por plano Z

        cx = glycerol_pos[0] + np.sin(phi) * 0.3
        cy = glycerol_pos[1] + np.cos(phi) * 0.3

        for i in range(n_seg):
            t = i / max(n_seg - 1, 1)
            z = glycerol_pos[2] + sign * (i * seg_dz + rng.normal(0, 0.15))

            # Anillos A-D (~55 % inicial): radio ~2.1 Å
            # Isooctilo (~45 % final): radio decrece hasta ~0.8 Å
            if t < 0.55:
                r_base = 2.1
            else:
                r_base = 1.4 * (1.0 - 0.4 * (t - 0.55) / 0.45)

            r = max(r_base + rng.normal(0, 0.15), 0.5)
            angle = phi + i * (2 * np.pi / 3.2) + rng.normal(0, 0.1)

            segments.append(np.array([
                cx + r * np.cos(angle),
                cy + r * np.sin(angle),
                z,
            ]))

        return segments, 0.90  # S_CH elevado: cuerpo rígido de fase gel

    def _sanitize_tail(self, tail, glycerol_pos, sign, lipid_name):
        """
        Corrige la geometría de colas acil para evitar intersección
        con la región de cabezas polares o glicerol.
        
        El paso por segmento se calcula desde tail_length de cada
        especie, esto evita que colas cortas (POPC 14.5 Å) viajen tan lejos
        como colas largas (SM 16.5 Å), preservando el contraste de
        interdigitación Lo > Ld.
        """
        if not tail or len(tail) == 0 or lipid_name == "CHOL":
            return tail

        # Paso máximo por segmento dependiente de la especie
        ltype_data = LIPID_TYPES.get(lipid_name)
        if ltype_data and ltype_data.tail_length > 0:
            seg_step = (ltype_data.tail_length / 9) * 1.30
        else:
            seg_step = 2.0  # fallback conservador

        sanitized = []
        z_glyc = glycerol_pos[2]
        z_prev = z_glyc
        glyc_zone = 2.0  # margen mínimo desde el glicerol (Å)

        for i, seg in enumerate(tail):
            seg = np.array(seg, dtype=float)
            z = seg[2]

            if sign < 0:  # Monocapa externa: Z decrece hacia el interior
                if z > z_prev + 0.8:
                    z = z_prev - 0.5
                # Límite máximo proporcional a la longitud de cadena de la especie
                z_limit = z_glyc - glyc_zone - i * seg_step
                if z > z_limit:
                    z = z_limit
            else:          # Monocapa interna: Z crece hacia el interior
                if z < z_prev - 0.8:
                    z = z_prev + 0.5
                z_limit = z_glyc + glyc_zone + i * seg_step
                if z < z_limit:
                    z = z_limit

            seg[2] = z
            sanitized.append(seg)
            z_prev = z

        return sanitized

    def _create_lipid(
        self,
        lipid_name: str,
        x: float,
        y: float,
        z_base: float,
        leaflet: str,
    ) -> LipidInstance:
        ltype = LIPID_TYPES[lipid_name]
        sign = -1 if leaflet == "sup" else 1

        z_head = z_base + self.get_local_z(x, y) + self.rng.normal(0, 0.5)
        head_pos = np.array([x, y, z_head])

        tilt = self.rng.uniform(3, 12 if ltype.phase == "gel" else 27) * np.pi / 180
        phi = self.rng.uniform(0, 2.0 * np.pi)

        glyc_z = z_head + sign * ltype.glyc_offset
        glyc_xy = np.sin(tilt) * ltype.glyc_offset * 0.3
        glycerol_pos = np.array([
            x + glyc_xy * np.cos(phi),
            y + glyc_xy * np.sin(phi),
            glyc_z,
        ])

        dphi = np.pi / 5.0

        if lipid_name == "CHOL":
            # CHOL: cuerpo anular, no cadena. glyc_offset=0 → glycerol=head.
            tail1, s1 = self._generate_chol_body(glycerol_pos, sign, phi, self.rng)
            tail2, s2 = None, s1
        else:
            nc1, nc2 = ltype.nc
            ndb1, ndb2 = ltype.ndb
            dbp1, dbp2 = ltype.dbpos

            start1 = glycerol_pos + np.array([
                1.1 * np.cos(phi - dphi),
                1.1 * np.sin(phi - dphi), 0.0,
            ])
            start2 = glycerol_pos + np.array([
                1.1 * np.cos(phi + dphi),
                1.1 * np.sin(phi + dphi), 0.0,
            ])

            tail1, s1 = generate_tail(
                start1, nc1, ndb1, dbp1, sign, tilt, phi - dphi,
                ltype.phase, self.rng,
            )
            tail2, s2 = generate_tail(
                start2, nc2, ndb2, dbp2, sign, tilt, phi + dphi,
                ltype.phase, self.rng,
            )

            tail1 = self._sanitize_tail(tail1, glycerol_pos, sign, lipid_name)
            if tail2 is not None:
                tail2 = self._sanitize_tail(tail2, glycerol_pos, sign, lipid_name)

        return LipidInstance(
            lipid_type=ltype,
            leaflet=leaflet,
            head_pos=head_pos,
            glycerol_pos=glycerol_pos,
            tail1=tail1,
            tail2=tail2,
            order_param=(s1 + s2) / 2.0 if tail2 else s1,
            in_raft=False,
            is_pip=ltype.pip_order > 0,
        )

    def _populate_leaflet(
        self,
        composition: Dict[str, float],
        z_base: float,
        leaflet: str,
    ) -> List[LipidInstance]:
        """Grilla hexagonal con jitter gaussiano del 20%."""
        area_t = self.Lx * self.Ly
        area_m = sum(
            LIPID_TYPES[k].area * f
            for k, f in composition.items() if k in LIPID_TYPES
        )
        n_total = int(area_t / area_m * 1.4)

        counts: Dict[str, int] = {}
        remaining = n_total
        for i, (ltype, frac) in enumerate(
            sorted(composition.items(), key=lambda x: -x[1])
        ):
            if ltype not in LIPID_TYPES:
                continue
            counts[ltype] = (
                int(n_total * frac) if i < len(composition) - 1 else remaining
            )
            remaining -= counts[ltype]

        nx = max(1, int(np.sqrt(n_total * self.Lx / self.Ly)))
        ny = max(1, n_total // nx + 1)
        dx, dy = self.Lx / nx, self.Ly / ny
        positions = []
        for ix in range(nx + 2):
            for iy in range(ny + 2):
                xp = (ix * dx + (0.5 * dx if iy % 2 else 0)
                      + self.rng.normal(0, dx * 0.20))
                yp = iy * dy + self.rng.normal(0, dy * 0.20)
                positions.append((xp % self.Lx, yp % self.Ly))
        self.rng.shuffle(positions)
        positions = positions[:n_total]

        fr_raft = sum(
            f for k, f in composition.items()
            if LIPID_TYPES.get(k, _null_lt()).is_raft
        )
        n_nuc = max(2, int(np.sqrt((self.Lx / 10) * (self.Ly / 10) * fr_raft / 50)))
        raft_centers = [
            (self.rng.uniform(0.1 * self.Lx, 0.9 * self.Lx),
             self.rng.uniform(0.1 * self.Ly, 0.9 * self.Ly))
            for _ in range(n_nuc)
        ]
        raft_radius = (
            np.sqrt(self.Lx * self.Ly * fr_raft / (np.pi * n_nuc)) * 1.2
        )

        fr_pip = sum(
            f for k, f in composition.items()
            if LIPID_TYPES.get(k, _null_lt()).pip_order > 0
        )
        pip_centers: List[Tuple[float, float]] = []
        pip_radius = 0.0
        if fr_pip > 0.01 and leaflet == "inf":
            n_pip = max(3, round(fr_pip * 8))
            pip_centers = [
                (self.rng.uniform(0.1 * self.Lx, 0.9 * self.Lx),
                 self.rng.uniform(0.1 * self.Ly, 0.9 * self.Ly))
                for _ in range(n_pip)
            ]
            pip_radius = np.sqrt(
                self.Lx * self.Ly * fr_pip / (np.pi * n_pip)
            )

        lipids: List[LipidInstance] = []
        pos_idx = 0
        for ltype, count in counts.items():
            for _ in range(count):
                if pos_idx >= len(positions):
                    break
                x, y = positions[pos_idx]
                pos_idx += 1

                if (LIPID_TYPES[ltype].pip_order > 0 and pip_centers
                        and not any(
                            np.hypot(x - cx, y - cy) < pip_radius
                            for cx, cy in pip_centers
                        )):
                    cx, cy = pip_centers[self.rng.integers(len(pip_centers))]
                    r = self.rng.uniform(0, pip_radius)
                    theta = self.rng.uniform(0, 2.0 * np.pi)
                    x = (cx + r * np.cos(theta)) % self.Lx
                    y = (cy + r * np.sin(theta)) % self.Ly

                lipid = self._create_lipid(ltype, x, y, z_base, leaflet)
                lipid.in_raft = (
                    LIPID_TYPES[ltype].is_raft
                    and any(
                        np.hypot(x - cx, y - cy) < raft_radius
                        for cx, cy in raft_centers
                    )
                )
                lipids.append(lipid)

        return lipids

    # Perturbaciones

    def _insert_perturbations(self):
        """Objetos transmembrana con repulsión suave sobre lípidos vecinos."""
        area = (self.Lx / 10) * (self.Ly / 10)
        n_obj = int(area * self.perturbation_density)

        for _ in range(n_obj):
            x = self.rng.uniform(0.1 * self.Lx, 0.9 * self.Lx)
            y = self.rng.uniform(0.1 * self.Ly, 0.9 * self.Ly)
            z = (
                (self.geometry.z_outer + self.geometry.z_inner) / 2.0
                + self.get_local_z(x, y)
            )
            radius = self.rng.uniform(8, 20)
            self.perturbations.append({"pos": np.array([x, y, z]), "radius": radius})
            for capa in [self.outer_leaflet, self.inner_leaflet]:
                for lip in capa:
                    dx = lip.head_pos[0] - x
                    dy = lip.head_pos[1] - y
                    dist = np.hypot(dx, dy)
                    if 0 < dist < radius * 1.5:
                        force = (radius * 1.5 - dist) / (radius * 1.5) * 6.0
                        lip.head_pos[0] = (lip.head_pos[0] + (dx / dist) * force) % self.Lx
                        lip.head_pos[1] = (lip.head_pos[1] + (dy / dist) * force) % self.Ly

    # Clusters

    def _detect_clusters(self):
        """BFS sobre KDTree para etiquetar rafts y clusters de PIPs."""
        def find_clusters(lipids, condition, min_size=4):
            subset = [l for l in lipids if condition(l)]
            if len(subset) < min_size:
                return []
            coords = np.array([[l.head_pos[0], l.head_pos[1]] for l in subset])
            tree = KDTree(coords)
            r_link = np.sqrt(self.Lx * self.Ly / len(subset)) * 1.6
            visited = set()
            clusters = []
            for i in range(len(subset)):
                if i in visited:
                    continue
                queue, cl = [i], {i}
                visited.add(i)
                while queue:
                    cur = queue.pop(0)
                    for nb in tree.query_ball_point(coords[cur], r=r_link):
                        if nb not in visited:
                            visited.add(nb)
                            cl.add(nb)
                            queue.append(nb)
                if len(cl) >= min_size:
                    clusters.append([subset[k] for k in cl])
            return clusters

        self.rafts_outer = find_clusters(self.outer_leaflet, lambda l: l.in_raft)
        self.rafts_inner = find_clusters(self.inner_leaflet, lambda l: l.in_raft)
        self.pip_clusters = find_clusters(self.inner_leaflet, lambda l: l.is_pip, min_size=3)

    # Build

    def build(self) -> "BicapaCryoET":
        """
        Construye la bicapa completa con semilla determinista.
        
        Los dominios Lo se generan son instantáneas estáticas, 
        no equilibrio real.
        """
        self.rng = default_rng(self.seed)
        self.perturbation_density = 0.008 + self.rng.uniform(0, 0.008)
        self.surface_tension = self.rng.uniform(0.001, 0.020)

        self.comp_outer = self._random_composition(COMP_OUTER_BASE)
        self.comp_inner = self._random_composition(COMP_INNER_BASE)
        self.geometry = self._calculate_geometry()
        self.bending_modulus = bending_modulus_from_composition(
            self.comp_outer, self.comp_inner
        )
        self.curvature_map = generate_helfrich_map(
            self.Lx, self.bending_modulus, self.surface_tension, self.rng
        )
        self.outer_leaflet = self._populate_leaflet(
            self.comp_outer, self.geometry.z_outer, "sup"
        )
        self.inner_leaflet = self._populate_leaflet(
            self.comp_inner, self.geometry.z_inner, "inf"
        )
        self._insert_perturbations()
        self._detect_clusters()

        print(
            "  seed=%d | %.0fx%.0f nm | lipidos=%d | "
            "rafts=%d/%d | pips=%d | kc=%.0f | sigma=%.3f"
            % (
                self.seed, self.Lx / 10, self.Ly / 10,
                len(self.outer_leaflet) + len(self.inner_leaflet),
                len(self.rafts_outer), len(self.rafts_inner),
                len(self.pip_clusters),
                self.bending_modulus, self.surface_tension,
            )
        )
        return self


def _null_lt() -> LipidType:
    """LipidType vacío para comparaciones seguras con dict.get()."""
    return LipidType(
        "", 0, 0, 0, 0, 0,
        (0, 0), (0, 0), (None, None),
        0, "", False, 0, "", "", "",
    )