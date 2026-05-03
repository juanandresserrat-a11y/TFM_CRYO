"""
export_paraview.py
Exportacion para ParaView por simulacion.

Estructura de salida:
  CryoET/paraview/simulacion{N}/
    bilayer_sim{N}.vtp          todos los granos + bonds
    bilayer_sim{N}_heads.vtp    cabezas polares
    bilayer_sim{N}_tails.vtp    colas acil (sn1, sn2, CHOL)
    bilayer_sim{N}_rafts.vtp    dominios Lo
    bilayer_sim{N}_pips.vtp     regiones PIPs
    bilayer_sim{N}_chol.vtp     colesterol (region 4)
    bilayer_sim{N}_ext.vtp      monocapa externa
    bilayer_sim{N}_int.vtp      monocapa interna
    bilayer_heads_sim{N}.pdb    cabezas en formato PDB

Propiedades:
  is_pip   = 1 si lipido PIP
  pip_head = 1 en el grano cabeza de PIP
  region   = 0 cabeza, 1 glicerol, 2 sn1, 3 sn2, 4 CHOL

El colesterol se asigna completamente a region 4.

Uso en ParaView:
  Open -> bilayer_sim{N}.vtp
  Apply -> Tube filter (radius 1.2)
  Color -> electron_density

Referencias principales:
    [4] Chaisson et al. 2025 – cuantificación de interdigitación en bicapas simuladas mediante interacciones trans-bicapa
    [11] Kučerka et al. 2008 – determinación experimental de espesores y áreas lipídicas en bicapas fosfolipídicas
    [18] Nagle & Tristram-Nagle 2000 – estructura de bicapas lipídicas y modelos de densidad electrónica
    [20] Piggot et al. 2017 – cálculo del parámetro de orden S_CH a partir de simulaciones de lípidos
"""

from __future__ import annotations
import os
from typing import TYPE_CHECKING, Optional
import numpy as np
from builder import OUTPUT_DIR
from electron_density import LIPID_ED_HEADGROUP, LIPID_ED_TAIL

if TYPE_CHECKING:
    from builder import BicapaCryoET

LIPID_ID = {
    "POPC":0,"POPE":1,"POPS":2,"PI":3,"PI3P":4,"PI4P":5,"PI5P":6,
    "PI34P2":7,"PIP2":8,"PIP3":9,"SM":10,"CHOL":11,"GM1":12,"PlsPE":13,
}

PIP_SPECIES = {"PI","PI3P","PI4P","PI5P","PI34P2","PIP2","PIP3"}
UNSATURATION_PENALTY = 0.035


def _sim_pv_dir(seed):
    d = os.path.join(OUTPUT_DIR, "paraview", "simulacion%04d" % seed)
    os.makedirs(d, exist_ok=True)
    return d


def _tail_ed(lname, seg_i, seg_total, ndb):
    """ED de un segmento de cola con efecto de insaturacion."""
    base = LIPID_ED_TAIL.get(lname, 0.292)
    if lname == "CHOL":
        return 0.302 if seg_i / max(seg_total-1,1) <= 0.45 else 0.280
    if ndb > 0 and seg_total > 1:
        frac = seg_i / (seg_total - 1)
        base -= ndb * UNSATURATION_PENALTY * np.exp(-0.5*((frac - 0.5)/0.2)**2)
    if lname == "PlsPE" and seg_i == 0:
        base -= 0.004
    return float(np.clip(base, 0.25, 0.32))


def _build_atoms_and_bonds(membrane):
    """Construye arrays de posiciones, propiedades y conectividad."""
    xs, ys, zs = [], [], []
    lipid_id_arr, leaflet_arr, bead_type_arr, seg_idx_arr = [], [], [], []
    order_p, in_raft_arr, is_pip_arr, ed_arr = [], [], [], []
    ndb_arr, phase_arr, nc_arr = [], [], []
    region_arr, is_head_arr, is_glycerol_arr, is_tail_arr = [], [], [], []
    pip_head_arr = []
    bonds = []

    for monocapa, chain_id in [
        (membrane.outer_leaflet, 0),
        (membrane.inner_leaflet, 1),
    ]:
        for lip in monocapa:
            lt    = lip.lipid_type
            lname = lt.name
            lid   = LIPID_ID.get(lname, 99)
            op    = float(lip.order_param)
            ir    = int(lip.in_raft)
            # CORRECCION: PIP solo si es especie PIP Y monocapa interna
            ip    = int(lname in PIP_SPECIES and chain_id == 1)
            ph    = 0 if lt.phase == "gel" else 1
            nc    = lt.nc[0]
            ndb   = lt.ndb[0] + lt.ndb[1]
            ed_h  = LIPID_ED_HEADGROUP.get(lname, 0.460)
            is_chol = (lname == "CHOL")

            def add(pt, bt, si, ed_val, region_code, pip_head_val=0):
                idx = len(xs)
                xs.append(float(pt[0])); ys.append(float(pt[1])); zs.append(float(pt[2]))
                lipid_id_arr.append(lid); leaflet_arr.append(chain_id)
                bead_type_arr.append(bt); seg_idx_arr.append(si)
                order_p.append(op); in_raft_arr.append(ir)
                is_pip_arr.append(ip); ed_arr.append(float(ed_val))
                ndb_arr.append(ndb); phase_arr.append(ph); nc_arr.append(nc)
                region_arr.append(region_code)
                is_head_arr.append(1 if region_code == 0 else 0)
                is_glycerol_arr.append(1 if region_code == 1 else 0)
                is_tail_arr.append(1 if region_code in (2, 3, 4) else 0)
                pip_head_arr.append(pip_head_val)
                return idx

            # CABEZA POLAR
            # pip_head=1 SOLO si es PIP en monocapa interna
            i_head = add(lip.head_pos, 0, 0, ed_h, 0,
                         pip_head_val=ip)  # ip ya es 0 para externos

            # GLICEROL (no CHOL)
            if lt.glyc_offset > 0:
                i_glyc = add(lip.glycerol_pos, 1, 0, 0.390, 1)
                bonds.append((i_head, i_glyc))
                prev = i_glyc
            else:
                prev = i_head

            # TAIL sn1 — region 4 para CHOL, 2 para el resto
            if lip.tail1 and len(lip.tail1) > 0:
                n_seg = len(lip.tail1)
                i_prev = prev
                reg = 4 if is_chol else 2
                i_first_tail = None
                for si, pt in enumerate(lip.tail1):
                    i_t = add(pt, 2, si, _tail_ed(lname, si, n_seg, ndb), reg)
                    bonds.append((i_prev, i_t))
                    # CHOL: bonds cruzados para cuerpo anular
                    if is_chol and si >= 2 and si % 2 == 0 and i_first_tail is not None:
                        bonds.append((i_first_tail, i_t))
                    if si == 0:
                        i_first_tail = i_t
                    i_prev = i_t

            # TAIL sn2 — region 3 (solo fosfolipidos, no CHOL)
            if lip.tail2 and len(lip.tail2) > 0:
                n_seg = len(lip.tail2)
                i_prev = prev  # prev es i_glyc para fosfolipidos
                for si, pt in enumerate(lip.tail2):
                    i_t = add(pt, 3, si, _tail_ed(lname, si, n_seg, ndb), 3)
                    bonds.append((i_prev, i_t))
                    i_prev = i_t

    # Proteinas transmembrana: cada perturbacion se representa como una
    # columna de N_SLICES puntos que atraviesa la bicapa de z_inner a z_outer.
    # region=5, is_protein=1 permiten filtrarlas en ParaView independientemente
    # del resto de la molecula. electron_density=0.400 e/A3 es el valor
    # tipico de una proteina integrada en membrana (entre colas y cabezas).
    is_protein_arr = [0] * len(xs)
    ED_PROTEIN = 0.400
    g = getattr(membrane, "geometry", None)
    z_bottom = float(g.z_inner - 5.0) if g else -60.0
    z_top    = float(g.z_outer + 5.0) if g else  60.0
    N_SLICES = 12

    for pert in getattr(membrane, "perturbations", []):
        px, py = float(pert["pos"][0]), float(pert["pos"][1])
        z_pts  = np.linspace(z_bottom, z_top, N_SLICES)
        prev_idx = None
        for si, z_pt in enumerate(z_pts):
            idx = len(xs)
            xs.append(px); ys.append(py); zs.append(float(z_pt))
            lipid_id_arr.append(99); leaflet_arr.append(2)
            bead_type_arr.append(9); seg_idx_arr.append(si)
            order_p.append(0.0); in_raft_arr.append(0)
            is_pip_arr.append(0); pip_head_arr.append(0)
            ed_arr.append(ED_PROTEIN); ndb_arr.append(0)
            phase_arr.append(2); nc_arr.append(0)
            region_arr.append(5); is_head_arr.append(0)
            is_glycerol_arr.append(0); is_tail_arr.append(0)
            is_protein_arr.append(1)
            if prev_idx is not None:
                bonds.append((prev_idx, idx))
            prev_idx = idx

    atoms = {
        "x": np.array(xs, np.float32), "y": np.array(ys, np.float32),
        "z": np.array(zs, np.float32),
        "lipid_id":         np.array(lipid_id_arr, np.int32),
        "leaflet":          np.array(leaflet_arr,  np.int32),
        "bead_type":        np.array(bead_type_arr,np.int32),
        "seg_idx":          np.array(seg_idx_arr,  np.int32),
        "order_param":      np.array(order_p,      np.float32),
        "in_raft":          np.array(in_raft_arr,  np.int32),
        "is_pip":           np.array(is_pip_arr,   np.int32),
        "pip_head":         np.array(pip_head_arr, np.int32),
        "electron_density": np.array(ed_arr,       np.float32),
        "n_doublebonds":    np.array(ndb_arr,      np.int32),
        "phase":            np.array(phase_arr,    np.int32),
        "chain_length":     np.array(nc_arr,       np.int32),
        "region":           np.array(region_arr,   np.int32),
        "is_head":          np.array(is_head_arr,  np.int32),
        "is_glycerol":      np.array(is_glycerol_arr, np.int32),
        "is_tail":          np.array(is_tail_arr,  np.int32),
        "is_protein":       np.array(is_protein_arr, np.int32),
    }
    return atoms, bonds


def _write_vtp(path, atoms, bonds, prop_keys=None):
    """Escribe un VTP con los arrays y bonds dados."""
    if prop_keys is None:
        prop_keys = [k for k in atoms if k not in ("x","y","z")]

    n  = len(atoms["x"])
    nb = len(bonds)
    FLOATS = {"order_param","electron_density"}

    def farr(name, a):
        return '<DataArray type="Float32" Name="%s" format="ascii">\n%s\n</DataArray>\n' % (
            name, " ".join("%.4f" % float(x) for x in a))

    def iarr(name, a):
        return '<DataArray type="Int32" Name="%s" format="ascii">\n%s\n</DataArray>\n' % (
            name, " ".join(str(int(x)) for x in a))

    pts  = " ".join("%.3f %.3f %.3f" % (float(x),float(y),float(z))
                    for x,y,z in zip(atoms["x"],atoms["y"],atoms["z"]))
    conn = " ".join("%d %d" % (int(a),int(b)) for a,b in bonds)
    offs = " ".join(str(2*(i+1)) for i in range(nb))

    vtp = ('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="0.1" '
           'byte_order="LittleEndian">\n<PolyData>\n'
           '<Piece NumberOfPoints="%d" NumberOfVerts="0" NumberOfLines="%d" '
           'NumberOfStrips="0" NumberOfPolys="0">\n' % (n, nb))
    vtp += '<Points>\n<DataArray type="Float32" NumberOfComponents="3" format="ascii">\n'
    vtp += pts + '\n</DataArray>\n</Points>\n<PointData>\n'
    for key in prop_keys:
        if key in atoms:
            vtp += farr(key, atoms[key]) if key in FLOATS else iarr(key, atoms[key])
    vtp += '</PointData>\n'
    if nb > 0:
        vtp += ('<Lines>\n'
                '<DataArray type="Int32" Name="connectivity" format="ascii">\n%s\n</DataArray>\n'
                '<DataArray type="Int32" Name="offsets" format="ascii">\n%s\n</DataArray>\n'
                '</Lines>\n') % (conn, offs)
    vtp += '</Piece>\n</PolyData>\n</VTKFile>\n'

    with open(path, "w") as f:
        f.write(vtp)
    return path


def export_vtp(membrane, d=None):
    """VTP principal con todos los granos, bonds y propiedades."""
    if d is None:
        d = _sim_pv_dir(membrane.seed)
    path = os.path.join(d, "bilayer_sim%04d.vtp" % membrane.seed)

    atoms, bonds = _build_atoms_and_bonds(membrane)
    prop_order = [
        "region","is_head","is_glycerol","is_tail","is_protein",
        "order_param","in_raft","is_pip","pip_head","electron_density",
        "lipid_id","leaflet","bead_type","seg_idx",
        "n_doublebonds","phase","chain_length",
    ]
    _write_vtp(path, atoms, bonds, prop_order)
    size_mb = os.path.getsize(path) / 1e6
    print("  -> %s  (%d granos, %d enlaces, %.1f MB)" % (
        os.path.basename(path), len(atoms["x"]), len(bonds), size_mb))
    return path, atoms, bonds


def export_vtp_by_region(membrane, atoms, bonds, d=None):
    """
    VTPs separados por region anatomica.

    Correcciones:
    - _pips.vtp: solo las CABEZAS de PIPs (pip_head=1), no toda la molecula
    - _chol.vtp: granos con region=4 (cuerpo del colesterol)
    """
    if d is None:
        d = _sim_pv_dir(membrane.seed)
    sid = membrane.seed

    PROPS = ["region","order_param","in_raft","is_pip","pip_head",
             "electron_density","lipid_id","leaflet","phase","n_doublebonds"]

    def subvtp(tag, mask):
        idx_old = np.where(mask)[0]
        if len(idx_old) == 0:
            return None
        remap  = {int(o): int(ni) for ni, o in enumerate(idx_old)}
        sub    = {k: v[mask] for k,v in atoms.items()}
        sbonds = [(remap[a], remap[b]) for a,b in bonds
                  if a in remap and b in remap]
        p = os.path.join(d, "bilayer_sim%04d_%s.vtp" % (sid, tag))
        _write_vtp(p, sub, sbonds, PROPS)
        print("  -> %s  (%d pts, %d bonds)" % (
            os.path.basename(p), len(idx_old), len(sbonds)))
        return p

    masks = {
        "heads":    atoms["region"] == 0,
        "tails":    atoms["is_tail"] == 1,
        "rafts":    atoms["in_raft"] == 1,
        "pips":     atoms["pip_head"] == 1,
        "chol":     atoms["region"] == 4,
        "proteins": atoms["is_protein"] == 1,
        "ext":      (atoms["leaflet"] == 0) & (atoms["is_protein"] == 0),
        "int":      (atoms["leaflet"] == 1) & (atoms["is_protein"] == 0),
    }
    paths = {}
    for tag, mask in masks.items():
        r = subvtp(tag, mask)
        if r: paths[tag] = r
    return paths


def export_pdb_heads(membrane, d=None):
    """PDB solo cabezas polares. Compatible PyMOL/ChimeraX/VMD."""
    if d is None:
        d = _sim_pv_dir(membrane.seed)
    path = os.path.join(d, "bilayer_heads_sim%04d.pdb" % membrane.seed)

    RES = {"POPC":"PPC","POPE":"PPE","POPS":"PPS","PI":"PPI",
           "PI3P":"P3P","PI4P":"P4P","PI5P":"P5P","PI34P2":"P34",
           "PIP2":"PP2","PIP3":"PP3","SM":"SPM","CHOL":"CHL",
           "GM1":"GM1","PlsPE":"PLS"}
    lines = [
        "REMARK BicapaCryoET | simulacion=%d | %.0fx%.0f nm"
        % (membrane.seed, membrane.Lx/10, membrane.Ly/10),
        "REMARK B-factor=S_CH*100 | Occ: 1.0=raft 0.5=fluido | Chain A=ext B=int",
        "CRYST1%9.3f%9.3f%9.3f  90.00  90.00  90.00 P 1           1"
        % (membrane.Lx, membrane.Ly, 100.0),
    ]
    ai = ri = 1
    for mono, chain in [(membrane.outer_leaflet,"A"),(membrane.inner_leaflet,"B")]:
        for lip in mono:
            rn  = RES.get(lip.lipid_type.name, lip.lipid_type.name[:3])
            occ = 1.00 if lip.in_raft else 0.50
            bf  = round(float(lip.order_param)*100, 2)
            x,y,z = lip.head_pos
            lines.append("ATOM  %5d  HD  %3s %1s%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s  "
                % (ai%99999, rn, chain, ri%9999, x, y, z, occ, bf, "O"))
            ai+=1; ri+=1
        lines.append("TER")
    lines.append("END")
    with open(path,"w") as f: f.write("\n".join(lines)+"\n")
    print("  -> %s  (%d cabezas, PDB)" % (os.path.basename(path), ai-1))
    return path


def write_readme(membrane, d):
    path = os.path.join(d, "README.txt")
    with open(path,"w") as f:
        f.write("""INSTRUCCIONES PARAVIEW — Simulacion %d
======================================

ARCHIVO PRINCIPAL:
  bilayer_sim{N}.vtp

PASOS BASICOS:
  1. File -> Open -> bilayer_sim{N}.vtp -> Apply
  2. Filters -> Tube -> Radius=1.2 -> Apply  (cadenas como cilindros)
  3. Colorear por:
     electron_density  → insaturaciones (azul=POPC, rojo=SM)
     in_raft           → Lo/Ld (1=raft, 0=fluido)
     region            → parte anatomica (ver tabla)
     pip_head          → solo cabezas de PIPs resaltadas

ARCHIVOS SEPARADOS (carga directa):
  *_heads.vtp   cabezas polares
  *_tails.vtp   colas acil (sn1 + sn2 + CHOL body)
  *_rafts.vtp   dominio Lo (raft)
  *_pips.vtp    SOLO cabezas de fosfoinositidos
  *_chol.vtp    SOLO cuerpo del colesterol (region=4)
  *_ext.vtp     monocapa externa
  *_int.vtp     monocapa interna

PROPIEDAD region:
  0 = cabeza polar
  1 = glicerol
  2 = cola sn1
  3 = cola sn2
  4 = cuerpo esteroide CHOL

PARA VER INSATURACIONES EN COLAS:
  Abrir *_tails.vtp -> Tube -> Colorear por electron_density
  SM C24:0 (saturado):        rojo uniforme
  POPC C18:1 (1 doble enlace): depresion azul en segmento central
  PIPs C20:4 (4 dobles):      zona azul marcada en sn2

MODELO VOLUMETRICO 3D:
  CryoET/model3d/bilayer_physical_sim{N}_norm.mrc
  ParaView -> Filters -> Contour -> value=0.45
""" % membrane.seed)
    return path


def export_all_paraview(membrane):
    """
    Exporta todos los formatos en CryoET/paraview/simulacion{N}/.
    """
    d = _sim_pv_dir(membrane.seed)
    print("  Exportando ParaView -> simulacion%04d/ (seed=%d)..." % (
        membrane.seed, membrane.seed))

    vtp_path, atoms, bonds = export_vtp(membrane, d)

    paths = {
        "vtp":       vtp_path,
        "pdb_heads": export_pdb_heads(membrane, d),
        "regions":   export_vtp_by_region(membrane, atoms, bonds, d),
    }
    write_readme(membrane, d)

    # Stats summary
    n_chol_grains    = int((atoms["region"]==4).sum())
    n_pip_heads      = int((atoms["pip_head"]==1).sum())
    n_raft_grains    = int((atoms["in_raft"]==1).sum())
    n_protein_grains = int((atoms["is_protein"]==1).sum())
    n_proteins       = len(getattr(membrane, "perturbations", []))
    print("  CHOL granos (region=4): %d | PIP cabezas: %d | Raft granos: %d | Proteinas: %d (%d granos)" % (
        n_chol_grains, n_pip_heads, n_raft_grains, n_proteins, n_protein_grains))
    return paths