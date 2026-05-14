"""
model_3d.py
Modelo 3D de densidad electrónica de la bicapa lipídica.

Genera un volumen 3D en el que cada voxel representa densidad electrónica
derivada de la composición y organización de la membrana.

Incluye:
  1. Densidad por tipo de cabeza y cola (P, N, O vs C, H)
  2. Efecto de insaturaciones en el empaquetamiento de colas
  3. Diferencias Lo vs Ld (rafts vs fase fluida)
  4. Asimetría entre monocapas
  5. Enriquecimiento local en PIPs
  6. Corrección por fluctuaciones de curvatura (Helfrich)

El resultado es un volumen MRC (55×55×80, ~9 Å/voxel) compatible con
ChimeraX, IMOD y PolNet.

Ademas, genera automaticamente un script de estado pvpython autocontenido
en CryoET/paraview/simulacion{N}/visualize_sim{N}.py para visualizacion
interactiva en ParaView. Este script requiere UNICAMENTE el archivo
bilayer_sim{N}.vtp (general); extrae proteinas, CHOL y colas internamente
mediante Thresholds sobre los arrays de punto.

Referencias principales:
    [4]  Helfrich 1973 – elasticidad de membranas y fluctuaciones de curvatura en bicapas lipídicas
    [11] Kučerka et al. 2008 – determinación experimental de espesores y áreas por lípido en bicapas
    [16] Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos de cryo-ET
    [18] Nagle & Tristram-Nagle 2000 – estructura de bicapas y perfiles de densidad electrónica
    [20] Piggot et al. 2017 – cálculo de parámetros de orden acil (S_CH) en simulaciones lipídicas
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.ndimage import binary_closing
import mrcfile

from builder import OUTPUT_DIR
from electron_density import (
    ELECTRON_DENSITY, LIPID_ED_HEADGROUP, LIPID_ED_TAIL
)
from analysis import midplane_map

if TYPE_CHECKING:
    from builder import BicapaCryoET


MODEL3D_DIR = os.path.join(OUTPUT_DIR, "model3d")


def _model3d_dir():
    os.makedirs(MODEL3D_DIR, exist_ok=True)
    return MODEL3D_DIR


UNSATURATION_PENALTY = 0.035


def _tail_density_with_unsaturation(lipid_name: str, z_frac: float) -> float:
    """
    Densidad electronica de una cola lipidica en funcion de z_frac [0,1].

    z_frac = 0 inicio en glicerol
    z_frac = 1 extremo metilo terminal

    Incluye tres efectos principales

    1 insaturaciones
       cada doble enlace reduce la densidad local ~3.5 por kinks cis
       POPC depresion en centro de sn2
       PIPs depresion mas acusada por multiples dobles enlaces
       SM y CHOL sin reduccion apreciable

    2 colesterol
       perfil axial diferenciado
       anillo esteroide mas denso que la cola isooctilo
       anillo ~0.302 e A3
       cola ~0.280 e A3

    3 plasmalogenos
       ausencia de carbonilo en sn1 reduce densidad inicial
       z_frac < 0.1 menor ED que fosfolipidos ester convencionales
    """
    from lipid_types import LIPID_TYPES
    lt = LIPID_TYPES.get(lipid_name)
    if lt is None:
        return ELECTRON_DENSITY["tail_fluid"]

    if lipid_name == "CHOL":
        return 0.302 if z_frac <= 0.45 else 0.280

    base_ed = LIPID_ED_TAIL.get(lipid_name, 0.292)

    ndb = lt.ndb[1]
    dbpos = lt.dbpos[1]
    nc = lt.nc[1]

    if ndb > 0 and dbpos is not None and nc > 0:
        z_db = dbpos / nc
        for _ in range(ndb):
            penalty = UNSATURATION_PENALTY * np.exp(
                -0.5 * ((z_frac - z_db) / 0.15) ** 2
            )
            base_ed -= penalty

    if lipid_name == "PlsPE" and z_frac < 0.12:
        base_ed -= 0.004

    return float(np.clip(base_ed, 0.25, 0.32))


def build_physical_volume(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 80,
    voxel_angstrom: float = 9.0,
    sigma_xy: float = 1.2,
    sigma_z: float = 0.6,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Construye el volumen 3D de densidad electronica fisicamente completo.

    El volumen captura el patron dark-bright-dark de la bicapa (cabezas-colas-cabezas),
    el contraste entre dominios Lo y Ld, la señal de PIPs en la monocapa interna,
    el efecto de las insaturaciones en colas acil, la asimetria entre monocapas
    y las fluctuaciones de Helfrich corregidas por plano medio local.

    Esquema de labels semanticos:
      0  agua
      1  cabeza ext (Ld)
      2  cola Lo (raft)
      3  cola Ld (fluido)
      4  cabeza int (Ld)
      5  PIP
      6  proteina transmembrana
      7  colesterol (cuerpo esteroide)
      8  cabeza ext (Lo/raft)
      9  cabeza int (Lo/raft)
    """
    g = membrane.geometry
    z_mid_grid = midplane_map(membrane, bins=bins_xy)

    z_half_nm = (g.total_thick / 10.0) / 2.0 + 0.8
    z_edges   = np.linspace(-z_half_nm, z_half_nm, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    dz_nm     = z_edges[1] - z_edges[0]

    vol    = np.full((bins_xy, bins_xy, bins_z), ELECTRON_DENSITY["water"], dtype=np.float32)
    labels = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.uint8)
    weight = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.float32)

    todos = membrane.outer_leaflet + membrane.inner_leaflet

    for l in todos:
        lname    = l.lipid_type.name
        ed_head  = LIPID_ED_HEADGROUP.get(lname, 0.460)
        ed_glyc  = 0.390
        is_chol  = (lname == "CHOL")

        ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
        iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)

        z_ref = z_mid_grid[ix, iy] / 10.0
        z_h   = (l.head_pos[2]     / 10.0) - z_ref
        z_g   = (l.glycerol_pos[2] / 10.0) - z_ref

        hg_half_nm = l.lipid_type.hg_thick / 10.0 / 2.0
        hg_safe    = max(hg_half_nm, 0.1)

        for iz, zc in enumerate(z_centers):
            w_head = np.exp(-0.5 * ((zc - z_h) / hg_safe) ** 2)
            if w_head > 0.03:
                old_w = weight[ix, iy, iz]
                new_w = old_w + w_head
                vol[ix, iy, iz] = (vol[ix, iy, iz]*old_w + ed_head*w_head) / new_w
                weight[ix, iy, iz] = new_w

                if l.is_pip:
                    label_val = 5
                elif l.in_raft:
                    label_val = 8 if l.leaflet == "sup" else 9
                else:
                    label_val = 1 if l.leaflet == "sup" else 4

                if labels[ix, iy, iz] == 0 or w_head > old_w * 0.5:
                    labels[ix, iy, iz] = label_val

        if not is_chol:
            glyc_thick = max(l.lipid_type.glyc_offset / 10.0, 0.05)
            wg = np.exp(-0.5 * ((z_centers - z_g) / glyc_thick) ** 2)
            mask_g = wg > 0.05
            if mask_g.any():
                old_w = weight[ix, iy, mask_g]
                new_w = old_w + wg[mask_g]
                vol[ix, iy, mask_g] = (vol[ix, iy, mask_g]*old_w + ed_glyc*wg[mask_g]) / new_w
                weight[ix, iy, mask_g] = new_w

        if l.tail1 and len(l.tail1) >= 2:
            for seg_i in range(len(l.tail1) - 1):
                pt_a = l.tail1[seg_i]
                pt_b = l.tail1[seg_i + 1]
                z_mid_seg = ((pt_a[2] + pt_b[2]) / 2.0 / 10.0) - z_ref
                z_frac = seg_i / max(len(l.tail1) - 2, 1)

                ed_tail_local = _tail_density_with_unsaturation(lname, z_frac)

                iz_seg = np.argmin(np.abs(z_centers - z_mid_seg))
                for diz in range(-1, 2):
                    iz2 = np.clip(iz_seg + diz, 0, bins_z - 1)
                    wt = np.exp(-0.5 * (diz * dz_nm / 0.25) ** 2) * 0.4
                    if wt > 0.05:
                        old_w = weight[ix, iy, iz2]
                        new_w = old_w + wt
                        vol[ix, iy, iz2] = (vol[ix, iy, iz2]*old_w + ed_tail_local*wt) / new_w
                        weight[ix, iy, iz2] = new_w

                        if labels[ix, iy, iz2] == 0:
                            labels[ix, iy, iz2] = 7 if is_chol else (2 if l.in_raft else 3)

    ED_PROTEIN = 0.400
    if hasattr(membrane, "perturbations") and membrane.perturbations:
        for pert in membrane.perturbations:
            px, py = pert["pos"][0] / membrane.Lx, pert["pos"][1] / membrane.Ly
            ix_p = int(px * bins_xy) % bins_xy
            iy_p = int(py * bins_xy) % bins_xy
            r_vox = max(1, int(pert["radius"] / voxel_angstrom))
            for dix in range(-r_vox, r_vox + 1):
                for diy in range(-r_vox, r_vox + 1):
                    if dix**2 + diy**2 <= r_vox**2:
                        ixx = (ix_p + dix) % bins_xy
                        iyy = (iy_p + diy) % bins_xy
                        for iz in range(bins_z):
                            vol[ixx, iyy, iz] = (
                                vol[ixx, iyy, iz] * 0.3 + ED_PROTEIN * 0.7
                            )
                            if labels[ixx, iyy, iz] != 6:
                                labels[ixx, iyy, iz] = 6

    for lv in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        mask = labels == lv
        closed = binary_closing(mask, structure=np.ones((3, 3, 2), dtype=bool))
        labels[closed & (labels == 0)] = lv

    vol_smooth = gaussian_filter(vol, sigma=[sigma_xy, sigma_xy, sigma_z])

    lo_voxels        = vol_smooth[labels == 2]
    ld_voxels        = vol_smooth[labels == 3]
    head_voxels      = vol_smooth[np.isin(labels, (1, 4, 8, 9))]
    pip_voxels       = vol_smooth[labels == 5]
    protein_voxels   = vol_smooth[labels == 6]
    chol_voxels      = vol_smooth[labels == 7]
    raft_head_voxels = vol_smooth[np.isin(labels, (8, 9))]

    stats = {
        "ed_head_mean":      float(head_voxels.mean())       if head_voxels.size      else 0.0,
        "ed_head_raft":      float(raft_head_voxels.mean())  if raft_head_voxels.size else 0.0,
        "ed_tail_Lo":        float(lo_voxels.mean())         if lo_voxels.size        else 0.0,
        "ed_tail_Ld":        float(ld_voxels.mean())         if ld_voxels.size        else 0.0,
        "ed_pip":            float(pip_voxels.mean())        if pip_voxels.size       else 0.0,
        "ed_chol":           float(chol_voxels.mean())       if chol_voxels.size      else 0.0,
        "ed_water":          ELECTRON_DENSITY["water"],
        "ed_protein":        float(protein_voxels.mean())    if protein_voxels.size   else 0.0,
        "n_protein_objects": len(membrane.perturbations)     if hasattr(membrane, "perturbations") else 0,
        "contrast_Lo_Ld":    float(lo_voxels.mean() - ld_voxels.mean())
                             if lo_voxels.size and ld_voxels.size else 0.0,
        "voxel_angstrom":    voxel_angstrom,
        "shape":             (bins_xy, bins_xy, bins_z),
        "z_half_nm":         z_half_nm,
    }

    return vol_smooth.astype(np.float32), labels, stats


def _write_paraview_state_script(membrane: "BicapaCryoET", pv_dir: str) -> str:
    seed = membrane.seed
    seed_str = f"{seed:04d}"

    # Resuelve la ruta al VTP en tiempo de ejecución del script generado,
    # relativa a la ubicación del propio script (independiente del cwd de pvpython).
    _VTP_RESOLVER = f"""\
import os as _os
_vtp_path = _os.path.join(
    _os.path.dirname(_os.path.abspath(__file__)),
    'bilayer_sim{seed_str}.vtp'
)

"""

    _TEMPLATE = """\
# state file generated using paraview version 6.1.0
import paraview
paraview.compatibility.major = 6
paraview.compatibility.minor = 1

from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

renderView1 = CreateView('RenderView')
renderView1.Set(
    ViewSize=[1561, 537],
    CenterOfRotation=[249.61142349243164, 250.3909034729004, -0.2740001678466797],
    CameraPosition=[247.04269848799944, 901.252210160883, 162.94797928816465],
    CameraFocalPoint=[249.6114234924316, 250.39090347290036, -0.2740001678466909],
    CameraViewUp=[0.0016807366477598058, -0.2432396143889024, 0.9699647751935423],
)

SetActiveView(None)

layout1 = CreateLayout(name='Layout #1')
layout1.AssignView(0, renderView1)
layout1.SetSize(1561, 537)

SetActiveView(renderView1)

selection_sources23004 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23004', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 0)',
    Assembly='',
    Selectors=['/'])

selection_sources26087 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.26087', groupname='selection_sources', ElementType='Point Data',
    QueryString='(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter23037 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23037', groupname='selection_sources', Input=selection_sources23004,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources23049 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.23049', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 0)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_sources24680 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24680', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)&(leaflet == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24713 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24713', groupname='selection_sources', Input=selection_sources24680,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter23082 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.23082', groupname='selection_sources', Input=selection_sources23049,
    Expression='s0',
    SelectionNames=['s0'])

selection_filter26120 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.26120', groupname='selection_sources', Input=selection_sources26087,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24479 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24479', groupname='selection_sources', ElementType='Point Data',
    QueryString='(in_raft == 1)&(leaflet == 1)&(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24512 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24512', groupname='selection_sources', Input=selection_sources24479,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24591 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24591', groupname='selection_sources', ElementType='Point Data',
    QueryString='(region == 1)&(leaflet == 0)',
    Assembly='',
    Selectors=['/'])

selection_filter24624 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24624', groupname='selection_sources', Input=selection_sources24591,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources24524 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.24524', groupname='selection_sources', ElementType='Point Data',
    QueryString='(in_raft == 1)&(leaflet == 0)&(is_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter24557 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.24557', groupname='selection_sources', Input=selection_sources24524,
    Expression='s0',
    SelectionNames=['s0'])

selection_sources25799 = CreateSelection(proxyname='SelectionQuerySource', registrationname='selection_sources.25799', groupname='selection_sources', ElementType='Point Data',
    QueryString='(pip_head == 1)',
    Assembly='',
    Selectors=['/'])

selection_filter25832 = CreateSelection(proxyname='AppendSelections', registrationname='selection_filter.25832', groupname='selection_sources', Input=selection_sources25799,
    Expression='s0',
    SelectionNames=['s0'])

selectionSource0 = CreateSelection(proxyname='SelectionQuerySource', registrationname='SelectionSource0', groupname='selections', ElementType='Point Data',
    QueryString='(is_head == 1)',
    Assembly='',
    Selectors=['/'])

appendSelections = CreateSelection(proxyname='AppendSelections', registrationname='AppendSelections', groupname='selections', Input=selectionSource0,
    Expression='s0',
    SelectionNames=['s0'])

tails_Source = XMLPolyDataReader(registrationName='Tails_Source', FileName=[_vtp_path])
tails_Source.Set(
    PointArrayStatus=['region', 'in_raft', 'electron_density', 'phase'],
    TimeArray='None',
)

general = XMLPolyDataReader(registrationName='General', FileName=[_vtp_path])
general.Set(
    PointArrayStatus=['region', 'is_head', 'is_glycerol', 'is_tail', 'is_protein', 'order_param', 'in_raft', 'is_pip', 'pip_head', 'electron_density', 'lipid_id', 'leaflet', 'bead_type', 'seg_idx', 'n_doublebonds', 'phase', 'chain_length'],
    TimeArray='None',
)

head_out = ExtractSelection(registrationName='Head_out', Input=general,
    Selection=selection_filter23037)

glyc_out = ExtractSelection(registrationName='Glyc_out', Input=general,
    Selection=selection_filter24713)

proteins = Threshold(registrationName='Proteins', Input=general)
proteins.Set(
    Scalars=['POINTS', 'is_protein'],
    LowerThreshold=1.0,
    UpperThreshold=1.0,
)

raft_in = ExtractSelection(registrationName='Raft_in', Input=general,
    Selection=selection_filter24512)

cHOL = Threshold(registrationName='CHOL', Input=general)
cHOL.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=4.0,
    UpperThreshold=4.0,
)

head_in = ExtractSelection(registrationName='Head_in', Input=general,
    Selection=selection_filter23082)

glyc_in = ExtractSelection(registrationName='Glyc_in', Input=general,
    Selection=selection_filter24624)

raft_out = ExtractSelection(registrationName='Raft_out', Input=general,
    Selection=selection_filter24557)

tails = Threshold(registrationName='Tails', Input=tails_Source)
tails.Set(
    Scalars=['POINTS', 'region'],
    LowerThreshold=2.0,
    UpperThreshold=3.0,
)

e_density = ExtractSelection(registrationName='E_density', Input=general,
    Selection=selection_filter26120)

pIPs = ExtractSelection(registrationName='PIPs', Input=head_in,
    Selection=selection_filter25832)

tails_Poly = ExtractSurface(registrationName='Tails_Poly', Input=tails)

tails_Tube = Tube(registrationName='Tails_Tube', Input=tails_Poly)
tails_Tube.Set(
    Scalars=['POINTS', ''],
    Vectors=['POINTS', ''],
    NumberofSides=12,
    Radius=1.2,
)

appendSelections.SetSelectionId(general.GetGlobalID())
appendSelections.SetSelectionPort(0)

proteinsDisplay = Show(proteins, renderView1, 'UnstructuredGridRepresentation')
proteinsDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.753, 0.753, 0.753],
    ColorArrayName=['FIELD', ''],
    DiffuseColor=[0.753, 0.753, 0.753],
    MapScalars=0,
    Opacity=0.5,
    GaussianRadius=2.5,
)
proteinsDisplay.ScaleTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]
proteinsDisplay.OpacityTransferFunction.Points = [9.0, 0.0, 0.5, 0.0, 9.001953125, 1.0, 0.5, 0.0]

cHOLDisplay = Show(cHOL, renderView1, 'UnstructuredGridRepresentation')
cHOLDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.831, 0.627, 0.09],
    ColorArrayName=['FIELD', ''],
    DiffuseColor=[0.831, 0.627, 0.09],
    MapScalars=0,
    Opacity=0.85,
    GaussianRadius=2.0,
)
cHOLDisplay.ScaleTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]
cHOLDisplay.OpacityTransferFunction.Points = [2.0, 0.0, 0.5, 0.0, 2.00048828125, 1.0, 0.5, 0.0]

tails_TubeDisplay = Show(tails_Tube, renderView1, 'GeometryRepresentation')

electron_densityLUT = GetColorTransferFunction('electron_density')
electron_densityLUT.Set(
    RGBPoints=[
        0.25, 0.0, 0.2, 1.0,
        0.3987999975681304, 0.5, 0.7, 1.0,
        0.46257142509732924, 1.0, 0.9, 0.0,
        0.49799999594688416, 1.0, 0.0, 0.0,
    ],
    ColorSpace='RGB',
    ScalarRangeInitialized=1.0,
)

tails_TubeDisplay.Set(
    Representation='Surface',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    SelectNormalArray='TubeNormals',
)
tails_TubeDisplay.ScaleTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]
tails_TubeDisplay.OpacityTransferFunction.Points = [-1.0, 0.0, 0.5, 0.0, 1.0, 1.0, 0.5, 0.0]

head_outDisplay = Show(head_out, renderView1, 'UnstructuredGridRepresentation')
head_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.6666666865348816, 0.49803921580314636],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.6666666865348816, 0.49803921580314636],
    GaussianRadius=3.8,
)
head_outDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
head_outDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

head_inDisplay = Show(head_in, renderView1, 'UnstructuredGridRepresentation')
head_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[1.0, 0.6666666865348816, 0.49803921580314636],
    ColorArrayName=[None, ''],
    DiffuseColor=[1.0, 0.6666666865348816, 0.49803921580314636],
    GaussianRadius=3.8,
)
head_inDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
head_inDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

raft_inDisplay = Show(raft_in, renderView1, 'UnstructuredGridRepresentation')
raft_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.7803921699523926, 0.5176470875740051, 0.38823530077934265],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.7803921699523926, 0.5176470875740051, 0.38823530077934265],
    GaussianRadius=3.8,
)
raft_inDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
raft_inDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

glyc_outDisplay = Show(glyc_out, renderView1, 'UnstructuredGridRepresentation')
glyc_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.9372549057006836, 0.6235294342041016, 0.46666666865348816],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.9372549057006836, 0.6235294342041016, 0.46666666865348816],
    GaussianRadius=2.5,
)
glyc_outDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
glyc_outDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

raft_outDisplay = Show(raft_out, renderView1, 'UnstructuredGridRepresentation')
raft_outDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.4588235318660736, 0.33725491166114807],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.4588235318660736, 0.33725491166114807],
    GaussianRadius=3.8,
)
raft_outDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
raft_outDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

glyc_inDisplay = Show(glyc_in, renderView1, 'UnstructuredGridRepresentation')
glyc_inDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.0, 0.7450980544090271, 0.545098066329956],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.0, 0.7450980544090271, 0.545098066329956],
    GaussianRadius=2.5,
)
glyc_inDisplay.ScaleTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]
glyc_inDisplay.OpacityTransferFunction.Points = [1.0, 0.0, 0.5, 0.0, 1.000244140625, 1.0, 0.5, 0.0]

pIPsDisplay = Show(pIPs, renderView1, 'UnstructuredGridRepresentation')
pIPsDisplay.Set(
    Representation='Point Gaussian',
    AmbientColor=[0.6666666865348816, 0.0, 1.0],
    ColorArrayName=[None, ''],
    DiffuseColor=[0.6666666865348816, 0.0, 1.0],
    GaussianRadius=3.8,
)
pIPsDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
pIPsDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

e_densityDisplay = Show(e_density, renderView1, 'UnstructuredGridRepresentation')
e_densityDisplay.Set(
    Representation='Point Gaussian',
    ColorArrayName=['POINTS', 'electron_density'],
    LookupTable=electron_densityLUT,
    Opacity=0.35,
    GaussianRadius=2.5,
)
e_densityDisplay.ScaleTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]
e_densityDisplay.OpacityTransferFunction.Points = [0.0, 0.0, 0.5, 0.0, 1.1757813367477812e-38, 1.0, 0.5, 0.0]

electron_densityLUTColorBar = GetScalarBar(electron_densityLUT, renderView1)
electron_densityLUTColorBar.Set(
    Title='electron_density',
    ComponentTitle='',
)
electron_densityLUTColorBar.Visibility = 1

tails_TubeDisplay.SetScalarBarVisibility(renderView1, True)
e_densityDisplay.SetScalarBarVisibility(renderView1, True)

electron_densityPWF = GetOpacityTransferFunction('electron_density')
electron_densityPWF.Set(
    Points=[0.25, 0.0, 0.5, 0.0, 0.49799999594688416, 1.0, 0.5, 0.0],
    ScalarRangeInitialized=1,
)

timeKeeper1 = GetTimeKeeper()
timeAnimationCue1 = GetTimeTrack()
animationScene1 = GetAnimationScene()
animationScene1.Set(
    ViewModules=renderView1,
    Cues=timeAnimationCue1,
    AnimationTime=0.0,
)

SetActiveSource(e_density)

# RenderAllViews()
# Interact()
# SaveScreenshot("path/to/screenshot.png")
"""

    script = _VTP_RESOLVER + _TEMPLATE

    path = os.path.join(pv_dir, f"visualize_sim{seed_str}.py")
    os.makedirs(pv_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(script)
    print("  -> %s" % path)
    return path


def export_physical_model_mrc(
    membrane: "BicapaCryoET",
    bins_xy: int = 55,
    bins_z: int = 80,
    voxel_angstrom: float = 9.0,
) -> Dict[str, str]:
    """
    Exporta el modelo 3D fisico completo como archivos MRC.

    Genera tres archivos:
      bilayer_physical_seed{N}.mrc       densidad electronica en e/A3
      bilayer_physical_seed{N}_norm.mrc  normalizado [0,255] para PolNet
      bilayer_physical_seed{N}_labels.mrc etiquetas semanticas 10 clases

    Ademas, genera automaticamente el script de visualizacion ParaView
    en CryoET/paraview/simulacion{N}/visualize_sim{N}.py si la carpeta
    de paraview existe (es decir, si se ejecuto previamente --paraview).
    Este script requiere UNICAMENTE el VTP general bilayer_sim{N}.vtp.

    Retorna dict con rutas.
    """
    print("  Construyendo modelo 3D fisico para seed=%d..." % membrane.seed)

    vol, labels, stats = build_physical_volume(
        membrane, bins_xy=bins_xy, bins_z=bins_z, voxel_angstrom=voxel_angstrom
    )

    d = _model3d_dir()

    path_ed = os.path.join(d, "bilayer_physical_seed%04d.mrc" % membrane.seed)
    with mrcfile.new(path_ed, overwrite=True) as mrc:
        mrc.set_data(vol.T)
        mrc.voxel_size = voxel_angstrom

    denom = vol.max() - vol.min()
    vol_norm = (vol - vol.min()) / denom * 255.0 if denom > 0 else vol.copy()

    path_norm = os.path.join(d, "bilayer_physical_seed%04d_norm.mrc" % membrane.seed)
    with mrcfile.new(path_norm, overwrite=True) as mrc:
        mrc.set_data(vol_norm.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    path_lbl = os.path.join(d, "bilayer_physical_seed%04d_labels.mrc" % membrane.seed)
    with mrcfile.new(path_lbl, overwrite=True) as mrc:
        mrc.set_data(labels.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    print("  -> model3d/bilayer_physical_seed%04d.mrc  %dx%dx%d voxels @ %.0f A" % (
        membrane.seed, bins_xy, bins_xy, bins_z, voxel_angstrom))
    print("  ED cabezas: %.3f e/A3 | Raft heads: %.3f | Lo: %.3f | Ld: %.3f | CHOL: %.3f | Contraste: %.4f e/A3" % (
        stats["ed_head_mean"], stats["ed_head_raft"], stats["ed_tail_Lo"],
        stats["ed_tail_Ld"], stats["ed_chol"], stats["contrast_Lo_Ld"]))

    pv_dir = os.path.join(OUTPUT_DIR, "paraview", "simulacion%04d" % membrane.seed)
    if os.path.isdir(pv_dir):
        try:
            _write_paraview_state_script(membrane, pv_dir)
        except Exception as exc:
            print("  [AVISO] No se pudo generar script ParaView: %s" % exc)

    return {
        "density_ea3":  path_ed,
        "density_norm": path_norm,
        "labels":       path_lbl,
        "stats":        stats,
    }


def plot_physical_model(
    membrane: "BicapaCryoET",
    vol: Optional[np.ndarray] = None,
    labels: Optional[np.ndarray] = None,
    stats: Optional[dict] = None,
    save_dir: str = "CryoET/model3d",
) -> str:
    """
    Genera el panel de visualizacion del modelo 3D fisico.

    5 subplots:
      1. Slice XZ central — perfil dark-bright-dark con fases anotadas
      2. Slice YZ central — segunda vista transversal
      3. Perfil 1D de densidad electronica — curva Z con regiones anotadas
      4. Proyeccion XY cabezas — contraste Lo vs Ld en la cara superior
      5. Mapa de insaturaciones — densidad de cola relativa por celda
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    os.makedirs(save_dir, exist_ok=True)

    if vol is None or labels is None:
        vol, labels, stats = build_physical_volume(membrane)

    bins_xy, _, bins_z = vol.shape
    g = membrane.geometry
    z_half_nm = stats["z_half_nm"]
    z_edges   = np.linspace(-z_half_nm, z_half_nm, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    Lx = membrane.Lx / 10.0
    Ly = membrane.Ly / 10.0
    ext_xy = [0, Lx, 0, Ly]
    ext_xz = [0, Lx, -z_half_nm, z_half_nm]
    ext_yz = [0, Ly, -z_half_nm, z_half_nm]

    PLT = {
        "figure.facecolor": "white", "axes.facecolor": "white",
        "axes.edgecolor": "#333333", "axes.linewidth": 0.8,
        "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.4,
        "font.family": "sans-serif", "font.size": 9,
        "axes.titlesize": 10, "axes.titleweight": "bold",
    }

    mid_y = bins_xy // 2
    mid_x = bins_xy // 2

    # CORRECCIÓN: posición de las CABEZAS, no del glicerol
    z_head_ext_nm = g.total_thick / 20.0
    z_head_int_nm = -g.total_thick / 20.0

    with plt.rc_context(PLT):
        fig, axes = plt.subplots(2, 3, figsize=(20, 15))
        fig.suptitle(
            "Modelo 3D fisico de la bicapa lipidica — seed=%d\n"
            "Densidad electronica con insaturaciones, fases Lo/Ld y PIPs"
            % membrane.seed,
            fontsize=12, fontweight="bold",
        )

        ax = axes[0, 0]
        slc_xz = vol[:, mid_y, :]
        im = ax.imshow(slc_xz.T, origin="lower", cmap="gray",
                       extent=ext_xz, aspect="auto")
        plt.colorbar(im, ax=ax, shrink=0.85, label="$e \\cdot \\AA^{-3}$")
        ax.axhline(z_head_ext_nm, color="#2dc653", lw=1.2, ls="--", alpha=0.85, label="Cabezas ext")
        ax.axhline(z_head_int_nm, color="#fb8500", lw=1.2, ls="--", alpha=0.85, label="Cabezas int")
        ax.axhline(0.0, color="#adb5bd", lw=0.8, ls=":", alpha=0.7, label="Plano medio")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Z relativo (nm)")
        ax.set_title("Slice XZ — Patron dark-bright-dark\ncabezas (oscuro) + nucleo (claro)")
        ax.legend(fontsize=7.5, loc="upper right")

        ax = axes[0, 1]
        slc_yz = vol[mid_x, :, :]
        im2 = ax.imshow(slc_yz.T, origin="lower", cmap="gray",
                        extent=ext_yz, aspect="auto")
        plt.colorbar(im2, ax=ax, shrink=0.85, label="$e \\cdot \\AA^{-3}$")
        ax.set_xlabel("Y (nm)"); ax.set_ylabel("Z relativo (nm)")
        ax.set_title("Slice YZ — Segunda vista transversal\nAsimetria cabezas ext/int visible")

        ax = axes[0, 2]
        ed_profile = vol.mean(axis=(0, 1))
        ax.plot(ed_profile, z_centers, color="#d4a017", lw=2.5, label="ED media")
        ax.fill_betweenx(z_centers, ELECTRON_DENSITY["water"], ed_profile,
                         alpha=0.35, color="#d4a017")
        ax.axhline(z_head_ext_nm, color="#2dc653", lw=1.2, ls="--",
                   label="Cabezas ext (%.1f nm)" % z_head_ext_nm)
        ax.axhline(z_head_int_nm, color="#fb8500", lw=1.2, ls="--",
                   label="Cabezas int (%.1f nm)" % z_head_int_nm)
        ax.axhline(0.0, color="#adb5bd", lw=0.8, ls=":", alpha=0.7)
        ax.axvspan(0.27, 0.31, alpha=0.12, color="#3a86ff", label="Ref colas")
        ax.axvspan(0.44, 0.50, alpha=0.12, color="#2dc653", label="Ref cabezas")
        ax.axvline(ELECTRON_DENSITY["water"], color="#888888", lw=0.8, ls=":",
                   alpha=0.7, label="Agua (0.334)")
        ax.set_xlabel("Densidad electronica ($e \\cdot \\AA^{-3}$)")
        ax.set_ylabel("Z (nm)")
        ax.set_title("Perfil 1D — ED a lo largo del eje Z\nPatron dark-bright-dark cuantificado")
        ax.legend(fontsize=7.5, loc="upper left")

        ax = axes[1, 0]
        head_proj = vol[:, :, :].max(axis=2)
        from analysis import raft_fraction_map
        raft_map = raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins_xy)
        im3 = ax.imshow(head_proj.T, origin="lower", cmap="gray",
                        extent=ext_xy, aspect="equal")
        plt.colorbar(im3, ax=ax, shrink=0.85, label="ED max ($e \\cdot \\AA^{-3}$)")
        cs = ax.contour(np.linspace(0, Lx, bins_xy),
                        np.linspace(0, Ly, bins_xy),
                        raft_map.T, levels=[0.4], colors=["#e63946"],
                        linewidths=1.2, linestyles="--")
        ax.clabel(cs, fmt="Lo/Ld", fontsize=7)
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Proyeccion maxima XY\nContorno rojo = dominio Lo (raft)")

        ax = axes[1, 1]
        from analysis import order_parameter_map, pip_density_map
        s_ch_map = order_parameter_map(membrane, bins=bins_xy)
        im4 = ax.imshow(s_ch_map.T, origin="lower", cmap="RdYlGn",
                        extent=ext_xy, aspect="equal", vmin=0.55, vmax=0.95)
        plt.colorbar(im4, ax=ax, shrink=0.85, label="S_CH")
        pip_map = pip_density_map(membrane, bins=bins_xy)
        if pip_map.max() > 0:
            pip_norm = pip_map / pip_map.max()
            ax.contour(np.linspace(0, Lx, bins_xy),
                       np.linspace(0, Ly, bins_xy),
                       pip_norm.T, levels=[0.4],
                       colors=["#9b5de5"], linewidths=1.5, linestyles=":")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Mapa S_CH — Orden molecular\nVerde=Lo ordenado | Rojo=Ld fluido\n"
                     "Circulo morado = cluster PIPs")

        ax = axes[1, 2]
        from lipid_types import LIPID_TYPES

        tail_density_map_arr = np.zeros((bins_xy, bins_xy), dtype=np.float32)
        count_map = np.zeros((bins_xy, bins_xy), dtype=np.float32)

        for l in membrane.outer_leaflet + membrane.inner_leaflet:
            ix = min(int(l.head_pos[0] / membrane.Lx * bins_xy), bins_xy - 1)
            iy = min(int(l.head_pos[1] / membrane.Ly * bins_xy), bins_xy - 1)
            lt = l.lipid_type
            ndb_total = lt.ndb[0] + lt.ndb[1]
            ed_base = LIPID_ED_TAIL.get(lt.name, 0.292)
            ed_with_unsat = ed_base - ndb_total * UNSATURATION_PENALTY * 0.5
            tail_density_map_arr[ix, iy] += ed_with_unsat
            count_map[ix, iy] += 1

        with np.errstate(all="ignore"):
            td = np.where(count_map > 0, tail_density_map_arr / count_map, 0.292)

        from scipy.ndimage import gaussian_filter as gf
        td_smooth = gf(td, sigma=2.0)
        im5 = ax.imshow(td_smooth.T, origin="lower", cmap="RdBu",
                        extent=ext_xy, aspect="equal",
                        vmin=td_smooth.min(), vmax=td_smooth.max())
        plt.colorbar(im5, ax=ax, shrink=0.85, label="ED cola ($e \\cdot \\AA^{-3}$)")
        ax.set_xlabel("X (nm)"); ax.set_ylabel("Y (nm)")
        ax.set_title("Densidad de cola — Efecto insaturaciones\nRojo=saturada (Lo/SM) | Azul=insaturada (Ld/POPC)")

        patches_leyenda = [
            mpatches.Patch(facecolor="#2dc653",
                           label="Cabeza ext (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_head_mean"]),
            mpatches.Patch(facecolor="#1a7a3e",
                           label="Cabeza ext raft (%.3f $e \\cdot \\AA^{-3}$)" % stats.get("ed_head_raft", 0)),
            mpatches.Patch(facecolor="#e63946",
                           label="Nucleo Lo (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_tail_Lo"]),
            mpatches.Patch(facecolor="#3a86ff",
                           label="Nucleo Ld (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_tail_Ld"]),
            mpatches.Patch(facecolor="#9b5de5",
                           label="PIPs (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_pip"]),
            mpatches.Patch(facecolor="#d4a017",
                           label="CHOL (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_chol"]),
            mpatches.Patch(facecolor="#f0f4ff", edgecolor="#aaa",
                           label="Agua (%.3f $e \\cdot \\AA^{-3}$)" % stats["ed_water"]),
            mpatches.Patch(facecolor="#ffffff", edgecolor="#555",
                           label="Proteina TM (%d obj., %.3f $e \\cdot \\AA^{-3}$)" % (
                               stats.get("n_protein_objects", 0),
                               stats.get("ed_protein", 0.400))),
        ]

        fig.legend(handles=patches_leyenda, loc="lower center", ncol=4,
                   fontsize=8.5, frameon=True, bbox_to_anchor=(0.5, 0.02))

        plt.subplots_adjust(left=0.06, right=0.96, top=0.93, bottom=0.11,
                            hspace=0.55, wspace=0.35)
        path = os.path.join(save_dir, "model3d_seed%04d.png" % membrane.seed)
        fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % path)
    return path