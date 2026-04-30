"""
export_positions.py
===================
Exportacion de posiciones 3D de todos los granos CG de la bicapa.

Este modulo genera los archivos de posiciones que PolNet necesita
para insertar membranas con composicion lipidica conocida dentro
de un tomograma sintetico. Produce dos formatos:

  1. PDB  — formato estandar de biologia estructural.
            Cada grano CG (cabeza, glicerol, segmento de cola) es
            un ATOM con coordenadas X,Y,Z en Angstrom.
            Compatible con PyMOL, UCSF ChimeraX y VMD.

  2. CSV  — tabla con una fila por grano, columnas:
            lipid_id, lipid_type, leaflet, bead_type, bead_idx,
            x, y, z, order_param, in_raft, is_pip
            Formato directo para analisis en Python/R/Julia.

  3. PolNet particle list — formato propio de PolNet para
            macromoleculas como puntos con orientacion:
            type, label, x, y, z, q1, q2, q3, q4
            Permite a PolNet colocar proteinas de membrana
            alineadas con los lipidos ya posicionados.

Resolucion espacial de las coordenadas:
  - Las posiciones de cabeza/glicerol son continuas (sub-Angstrom)
  - Cada segmento de cola mide ~2.24 A (nc*1.26 / nseg)
  - Precision de float64: ~1e-15 A (irrelevante; el jitter
    estocástico es de 0.25-1.75 A dependiendo de la fase)

Comparacion con formatos de PolNet:
  - PolNet coloca macromoleculas como densidades MRC (10 A/voxel)
  - Este archivo permite colocarlas como posiciones exactas
  - Util para proteinas de membrana ancladas a lipidos especificos

Curvatura de los parches simulados:
  - Todas las simulaciones son parches PLANOS con ondulaciones
    termicas de Helfrich (RMS ~3-5 A, equivalente a R_curv > 60 nm)
  - Los parches representan fragmentos de membrana plasmatica en
    condiciones de vitrificacion instantanea (cryo-ET)
  - La curvatura de organulos (ER ~25 nm, vesiculas ~50 nm) queda
    fuera del alcance de esta version del modelo y constituye una
    linea de trabajo futuro

Referencia PolNet:
  Martinez-Sanchez et al. IEEE Trans. Med. Imaging 2024
  doi:10.1109/TMI.2024.3398401
"""

from __future__ import annotations

import csv
import os
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np

from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET
    from geometry import LipidInstance


POS_DIR = os.path.join(OUTPUT_DIR, "positions")


def _pos_dir():
    os.makedirs(POS_DIR, exist_ok=True)
    return POS_DIR


def _bead_name(bead_type: str, lipid_name: str) -> str:
    codes = {
        "HEAD":  "HD",
        "GLYC":  "GL",
        "TAIL1": "T1",
        "TAIL2": "T2",
    }
    return "%s%s" % (codes.get(bead_type, "XX"), lipid_name[:2])


def _lipid_residue_name(lipid_name: str) -> str:
    mapping = {
        "POPC": "PPC", "POPE": "PPE", "POPS": "PPS",
        "PI":   "PPI", "PI3P": "P3P", "PI4P": "P4P",
        "PI5P": "P5P", "PI34P2": "P34", "PIP2": "PP2",
        "PIP3": "PP3", "SM":   "SPM", "CHOL": "CHL",
        "GM1":  "GM1",
    }
    return mapping.get(lipid_name, lipid_name[:3].upper())


def export_pdb(
    membrane: "BicapaCryoET",
    path: Optional[str] = None,
    wrap_periodic: bool = True,
) -> str:
    """
    Exporta todas las posiciones de granos CG en formato PDB.

    Estructura del archivo:
      - Cada lipido es un RESIDUO con su tipo como resName
      - Cada grano (cabeza, glicerol, segmentos de cola) es un ATOM
      - La cadena A es la monocapa externa (sup), B la interna (inf)
      - El campo B-factor almacena el parametro de orden S_CH
      - El campo occupancy es 1.0 para raft, 0.5 para no-raft

    Limitacion PDB: coordenadas en Angstrom, max 99999 atomos por
    cadena, max 9999 residuos (se usan indices ciclicos si se supera).

    Parametros
    ----------
    wrap_periodic : bool
        Si True, las coordenadas X,Y se envuelven al rango [0, Lx].
        Recomendado para compatibilidad con visualizadores.
    """
    if path is None:
        path = os.path.join(_pos_dir(), "bilayer_seed%04d.pdb" % membrane.seed)

    lines = []
    lines.append(
        "REMARK  BicapaCryoET v15 — seed %d — %.0fx%.0f nm"
        % (membrane.seed, membrane.Lx / 10, membrane.Ly / 10)
    )
    lines.append(
        "REMARK  Resolucion espacial: posiciones continuas en Angstrom"
    )
    lines.append(
        "REMARK  Grano de cola: %.2f A/segmento (9 segmentos por cadena)"
        % (14.5 * 1.26 / 9)
    )
    lines.append(
        "REMARK  Ondulacion Helfrich RMS: %.1f A (parche plano)"
        % (membrane.curvature_map.std() if membrane.curvature_map is not None else 0)
    )
    lines.append(
        "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1"
        % (membrane.Lx, membrane.Ly, 80.0)
    )

    atom_idx = 1
    res_idx = 1

    def write_atom(atom_name, res_name, chain, res_num, x, y, z,
                   occupancy=1.0, bfactor=0.0):
        nonlocal atom_idx
        if wrap_periodic:
            x = x % membrane.Lx
            y = y % membrane.Ly
        line = (
            "ATOM  %5d %-4s %3s %1s%4d    "
            "%8.3f%8.3f%8.3f%6.2f%6.2f          %2s  "
            % (
                atom_idx % 99999, atom_name, res_name, chain,
                res_num % 9999,
                x, y, z,
                occupancy, bfactor,
                atom_name[:1],
            )
        )
        lines.append(line)
        atom_idx += 1

    for leaflet_lips, chain in [
        (membrane.outer_leaflet, "A"),
        (membrane.inner_leaflet, "B"),
    ]:
        for lip in leaflet_lips:
            lname = lip.lipid_type.name
            res_name = _lipid_residue_name(lname)
            occ = 1.0 if lip.in_raft else 0.50
            bf = round(lip.order_param * 100.0, 2)

            write_atom("HD  ", res_name, chain, res_idx,
                       *lip.head_pos, occ, bf)

            if lip.lipid_type.glyc_offset > 0:
                write_atom("GL  ", res_name, chain, res_idx,
                           *lip.glycerol_pos, occ, bf)

            if lip.tail1:
                for seg_i, pt in enumerate(lip.tail1):
                    aname = "T1%02d" % seg_i if seg_i < 100 else "T1XX"
                    write_atom(aname, res_name, chain, res_idx,
                               *pt, occ, bf)

            if lip.tail2:
                for seg_i, pt in enumerate(lip.tail2):
                    aname = "T2%02d" % seg_i if seg_i < 100 else "T2XX"
                    write_atom(aname, res_name, chain, res_idx,
                               *pt, occ, bf)

            res_idx += 1

        lines.append("TER")

    lines.append("END")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")

    n_atoms = atom_idx - 1
    print(
        "  -> %s  (%d atomos CG, %d residuos)"
        % (os.path.basename(path), n_atoms, res_idx - 1)
    )
    return path


def export_csv_positions(
    membrane: "BicapaCryoET",
    path: Optional[str] = None,
    include_tails: bool = True,
) -> str:
    """
    Exporta todas las posiciones en formato CSV tabulado.

    Columnas:
      lipid_id     — indice global del lipido (0-based)
      lipid_type   — nombre de la especie (POPC, SM, etc.)
      leaflet      — sup | inf
      bead_type    — HEAD | GLYC | TAIL1_N | TAIL2_N
      x, y, z      — coordenadas en Angstrom
      order_param  — S_CH del lipido
      in_raft      — 1 si pertenece a dominio Lo, 0 si no
      is_pip       — 1 si es fosfoinositido, 0 si no
      phase        — gel | fluid

    Este formato permite analisis directo en pandas, R o Julia.
    """
    if path is None:
        path = os.path.join(
            _pos_dir(), "positions_seed%04d.csv" % membrane.seed
        )

    fieldnames = [
        "lipid_id", "lipid_type", "leaflet", "bead_type",
        "x", "y", "z",
        "order_param", "in_raft", "is_pip", "phase",
    ]

    todos = [
        (lip, i) for i, lip in enumerate(membrane.outer_leaflet)
    ] + [
        (lip, i + len(membrane.outer_leaflet))
        for i, lip in enumerate(membrane.inner_leaflet)
    ]

    n_rows = 0
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for lip, lid in todos:
            base = {
                "lipid_id":   lid,
                "lipid_type": lip.lipid_type.name,
                "leaflet":    lip.leaflet,
                "order_param": round(float(lip.order_param), 4),
                "in_raft":    int(lip.in_raft),
                "is_pip":     int(lip.is_pip),
                "phase":      lip.lipid_type.phase,
            }

            def row(btype, pt):
                d = dict(base)
                d["bead_type"] = btype
                d["x"] = round(float(pt[0]), 3)
                d["y"] = round(float(pt[1]), 3)
                d["z"] = round(float(pt[2]), 3)
                return d

            writer.writerow(row("HEAD", lip.head_pos))
            if lip.lipid_type.glyc_offset > 0:
                writer.writerow(row("GLYC", lip.glycerol_pos))
            n_rows += 1

            if include_tails:
                if lip.tail1:
                    for si, pt in enumerate(lip.tail1):
                        writer.writerow(row("TAIL1_%d" % si, pt))
                if lip.tail2:
                    for si, pt in enumerate(lip.tail2):
                        writer.writerow(row("TAIL2_%d" % si, pt))

    print(
        "  -> %s  (%d lipidos, include_tails=%s)"
        % (os.path.basename(path), len(todos), include_tails)
    )
    return path


def export_polnet_particle_list(
    membrane: "BicapaCryoET",
    path: Optional[str] = None,
) -> str:
    """
    Exporta las posiciones en el formato de lista de particulas de PolNet.

    PolNet usa este formato para macromoleculas colocadas en el tomograma.
    Cada cabeza lipidica es una particula con posicion (x,y,z) y orientacion
    expresada como cuaternion (q1,q2,q3,q4).

    La orientacion se calcula a partir del vector cabeza->glicerol,
    que apunta hacia el interior de la bicapa (normal local del lipido).

    Formato de columnas (compatible con PolNet CSV output):
      type      — tipo de objeto (lipid_HEAD)
      label     — nombre de especie lipidica
      x, y, z   — coordenadas en Angstrom
      q1,q2,q3,q4 — cuaternion de orientacion (unidad)

    Este archivo permite a PolNet:
      1. Colocar proteinas de membrana alineadas con los lipidos
      2. Generar subtomos alineados para subtomogram averaging
      3. Verificar la densidad local de lipidos en el volumen

    Referencia:
      PolNet particle list spec: github.com/anmartinezs/polnet
    """
    if path is None:
        path = os.path.join(
            _pos_dir(), "polnet_particles_seed%04d.csv" % membrane.seed
        )

    def vec_to_quaternion(v: np.ndarray) -> Tuple[float, float, float, float]:
        """
        Cuaternion que rota el eje Z (0,0,1) hasta el vector v.
        Usado para orientar las cabezas lipidicas hacia el interior.
        """
        v = v / (np.linalg.norm(v) + 1e-12)
        z_axis = np.array([0.0, 0.0, 1.0])
        cross = np.cross(z_axis, v)
        dot = np.dot(z_axis, v)
        cross_norm = np.linalg.norm(cross)

        if cross_norm < 1e-8:
            if dot > 0:
                return (1.0, 0.0, 0.0, 0.0)
            else:
                return (0.0, 1.0, 0.0, 0.0)

        angle = np.arctan2(cross_norm, dot)
        axis = cross / cross_norm
        s = np.sin(angle / 2.0)
        return (
            float(np.cos(angle / 2.0)),
            float(axis[0] * s),
            float(axis[1] * s),
            float(axis[2] * s),
        )

    rows = []
    todos = membrane.outer_leaflet + membrane.inner_leaflet

    for lip in todos:
        orient_vec = lip.glycerol_pos - lip.head_pos
        q = vec_to_quaternion(orient_vec)

        rows.append({
            "type":   "lipid_HEAD",
            "label":  lip.lipid_type.name,
            "leaflet": lip.leaflet,
            "in_raft": int(lip.in_raft),
            "is_pip":  int(lip.is_pip),
            "phase":   lip.lipid_type.phase,
            "x": round(float(lip.head_pos[0]), 3),
            "y": round(float(lip.head_pos[1]), 3),
            "z": round(float(lip.head_pos[2]), 3),
            "q1": round(q[0], 6),
            "q2": round(q[1], 6),
            "q3": round(q[2], 6),
            "q4": round(q[3], 6),
        })

    fieldnames = [
        "type", "label", "leaflet", "in_raft", "is_pip", "phase",
        "x", "y", "z", "q1", "q2", "q3", "q4",
    ]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        "  -> %s  (%d particulas, formato PolNet)"
        % (os.path.basename(path), len(rows))
    )
    return path


def export_all_positions(
    membrane: "BicapaCryoET",
    include_tails: bool = True,
) -> dict:
    """
    Exporta los tres formatos de posiciones en una sola llamada.

    Retorna
    -------
    dict con rutas a 'pdb', 'csv', 'polnet'
    """
    print("  Exportando posiciones 3D para seed=%d..." % membrane.seed)
    return {
        "pdb":    export_pdb(membrane),
        "csv":    export_csv_positions(membrane, include_tails=include_tails),
        "polnet": export_polnet_particle_list(membrane),
    }
