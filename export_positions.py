"""
export_positions.py
Exportacion de posiciones 3D de todos los granos CG de la bicapa.

Genera tres formatos de salida:

  1. PDB
     Cada grano coarse-grained (cabeza, glicerol, colas) se guarda
     como un ATOM con coordenadas X,Y,Z en Å.
     Compatible con PyMOL, ChimeraX y VMD.

  2. CSV
     Tabla por grano con:
       lipid_id, lipid_type, leaflet, bead_type, bead_idx,
       x, y, z, order_param, in_raft, is_pip

  3. PolNet particle list
     Formato de puntos con orientación:
       type, label, x, y, z, q1, q2, q3, q4
     Permite insertar proteínas alineadas con la membrana.

Escala espacial:
  - Cabezas y glicerol: posiciones continuas sub-Å
  - Colas: segmentos ~2.2 Å
  - Float64 solo garantiza estabilidad numérica (no resolución física)

Relación con PolNet:
  - PolNet usa densidades MRC (~10 Å/voxel)
  - Este formato permite colocación exacta de partículas
  - Útil para proteínas ancladas a lípidos específicos

Geometría del sistema:
  - Parches planos con fluctuaciones térmicas tipo Helfrich
  - RMS ~3–5 Å (radio de curvatura efectivo > 60 nm)
  - Representa fragmentos de membrana plasmática vitrificada
  - Curvaturas fuertes (orgánulos) no están modeladas

Referencia principal:
    [16] Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos de cryo-ET
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

    El archivo representa la bicapa como una estructura tipo proteína
    para poder visualizarla en herramientas estándar como PyMOL o ChimeraX.

    Estructura del archivo:
      - Cada lípido se guarda como un residuo (resName indica el tipo de lípido)
      - Cada grano (cabeza, glicerol y segmentos de cola) se escribe como un ATOM
      - Cadena A = monocapa externa, cadena B = monocapa interna
      - El B-factor almacena el parámetro de orden S_CH
      - El campo occupancy se usa como indicador sencillo de estado:
          1.0 → dominio raft
          0.5 → resto de la membrana

    Limitaciones del formato PDB:
      - Coordenadas en Å
      - Número máximo de átomos por cadena
      - Número máximo de residuos por estructura (si se excede, se reutilizan índices)

    Parametros
      wrap_periodic : bool
        Si es True, las coordenadas X e Y se envuelven al rango [0, Lx]
        para que la membrana pueda visualizarse sin discontinuidades en bordes
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

    Cada fila representa un grano CG de la bicapa, lo que permite
    analizar la estructura sin necesidad de cargar formatos moleculares
    más complejos.

    Columnas:
      lipid_id     — índice global del lípido (0-based)
      lipid_type   — especie lipídica (POPC, SM, etc.)
      leaflet      — monocapa superior (sup) o inferior (inf)
      bead_type    — HEAD | GLYC | TAIL1_N | TAIL2_N
      x, y, z      — coordenadas en Å
      order_param  — parámetro de orden S_CH
      in_raft      — 1 si pertenece a dominio Lo, 0 si no
      is_pip       — 1 si es fosfoinositido, 0 en caso contrario
      phase        — gel | fluid

    Este formato está pensado para análisis directo en pandas, R.
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
    Exporta las posiciones en el formato de lista de partículas de PolNet.

    PolNet utiliza este formato para representar macromoléculas dentro del tomograma.
    Aquí, cada cabeza lipídica se trata como una partícula con posición (x, y, z)
    y orientación en forma de cuaternión (q1, q2, q3, q4).

    La orientación se calcula a partir del vector cabeza→glicerol, que define la
    dirección local del lípido y apunta aproximadamente hacia el interior de la bicapa.

    Formato de columnas (compatible con el CSV de PolNet):
      type        — tipo de objeto (lipid_HEAD)
      label       — especie lipídica
      x, y, z     — coordenadas en Å
      q1,q2,q3,q4 — cuaternión unitario de orientación

    Este archivo permite a PolNet:
      1. Colocar proteínas de membrana alineadas con la geometría lipídica
      2. Generar subtomos consistentes para subtomogram averaging
      3. Evaluar la distribución local de lípidos en el volumen simulado
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
    
    print("  Exportando posiciones 3D para seed=%d..." % membrane.seed)
    return {
        "pdb":    export_pdb(membrane),
        "csv":    export_csv_positions(membrane, include_tails=include_tails),
        "polnet": export_polnet_particle_list(membrane),
    }
