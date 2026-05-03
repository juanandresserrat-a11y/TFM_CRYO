"""
export_mrc.py
Exportacion de la bicapa en formato MRC para integracion con PolNet.

PolNet genera tomogramas sinteticos de cryo-ET a partir de volumenes de 
densidad y etiquetas semanticas en formato MRC.

Este modulo genera dos archivos por semilla:
  bilayer_seed{N}.mrc
    volumen 3D de densidad normalizada (entrada PolNet)

  bilayer_seed{N}_labels.mrc
    volumen 3D de etiquetas semanticas

Etiquetas:
  0 fondo (agua)
  1 monocapa externa
  2 nucleo hidrofobico
  3 monocapa interna

Flujo:
  BicapaCryoET.build()
    -> export_mrc()
    -> PolNet config
    -> polnet run
    -> tomograma MRC

Referencia principal:
    [16] Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos de cryo-ET
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import mrcfile
import numpy as np

import analysis
from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET


MRC_DIR = os.path.join(OUTPUT_DIR, "mrc")


def _mrc_dir():
    os.makedirs(MRC_DIR, exist_ok=True)
    return MRC_DIR


def export_density_mrc(
    membrane: "BicapaCryoET",
    voxel_angstrom: float = 10.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> str:
    """
    Exporta el volumen 3D de densidad de masa a formato MRC.

    El volumen se normaliza a [0, 255] en float32 para compatibilidad
    con el rango de grises usado por PolNet. Las coordenadas Z se
    definen respecto al plano medio local con correccion de curvatura
    (Helfrich).

    El formato MRC utiliza orden ZYX internamente, por lo que el
    volumen se transpone antes de la escritura.
    """
    H, _ = analysis.volumetric_density(membrane, bins_xy=bins_xy, bins_z=bins_z)

    H_norm = (H / H.max() * 255.0).astype(np.float32)

    fname = "bilayer_seed%04d.mrc" % membrane.seed
    path = os.path.join(_mrc_dir(), fname)

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(H_norm.T)
        mrc.voxel_size = voxel_angstrom

    print("  -> mrc/%s  (%.0f x %.0f x %.0f voxels, %.0f A/voxel)" % (
        fname,
        H_norm.shape[0], H_norm.shape[1], H_norm.shape[2],
        voxel_angstrom,
    ))
    return path


def export_label_mrc(
    membrane: "BicapaCryoET",
    voxel_angstrom: float = 10.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> str:
    """
    Exporta el volumen 3D de etiquetas semanticas a formato MRC.

    Las etiquetas identifican cada voxel segun su region estructural
    en la bicapa, siguiendo el esquema compatible con PolNet:

      0 fondo (solucion acuosa exterior)
      1 monocapa externa
      2 nucleo hidrofobico
      3 monocapa interna

    Los limites entre regiones se derivan de la geometria media de la
    bicapa (MembraneGeometry) y se aplican en coordenadas Z relativas
    al plano medio local.
    """
    _, edges = analysis.volumetric_density(membrane, bins_xy=bins_xy, bins_z=bins_z)
    cz = 0.5 * (edges[2][:-1] + edges[2][1:])

    g = membrane.geometry

    # hydro_thick es la suma de las colas de AMBAS monocapas, no el grosor
    # de una sola. Usar hydro_thick/2 aqui comprimia el nucleo hasta ~0.65 nm
    # y asignaba label=2 a casi toda la bicapa. Se usa hg_thick promedio
    # de la composicion externa, que refleja correctamente el grosor del
    # grupo cabeza (~9 A para POPC) y da limites correctos (~1.65 nm).
    from lipid_types import LIPID_TYPES
    hg_mean = (
        sum(LIPID_TYPES[k].hg_thick * f for k, f in membrane.comp_outer.items()
            if k in LIPID_TYPES)
        if membrane.comp_outer else 9.0
    )
    hg_half_o = (g.z_outer - hg_mean / 2.0) / 10.0
    hg_half_i = (g.z_inner + hg_mean / 2.0) / 10.0

    labels = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.uint8)

    for iz, z in enumerate(cz):
        if z > hg_half_o:
            labels[:, :, iz] = 1
        elif z < hg_half_i:
            labels[:, :, iz] = 3
        else:
            labels[:, :, iz] = 2

    fname = "bilayer_seed%04d_labels.mrc" % membrane.seed
    path = os.path.join(_mrc_dir(), fname)

    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(labels.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    print("  -> mrc/%s  (labels: 0=fondo 1=ext 2=hidro 3=int)" % fname)
    return path


def export_mrc(
    membrane: "BicapaCryoET",
    voxel_angstrom: float = 10.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> dict:
    """Exporta densidad y etiquetas MRC en una sola llamada."""
    print("  Exportando MRC para seed=%d..." % membrane.seed)
    path_density = export_density_mrc(membrane, voxel_angstrom, bins_xy, bins_z)
    path_labels  = export_label_mrc(membrane, voxel_angstrom, bins_xy, bins_z)
    return {"density": path_density, "labels": path_labels}


def generate_polnet_yaml(
    membrane: "BicapaCryoET",
    tomo_shape: tuple = (400, 400, 200),
    voxel_angstrom: float = 10.0,
    snr: float = 0.1,
    tilt_range: tuple = (-60, 60, 3),
) -> str:
    """
    Genera una plantilla YAML de configuracion para PolNet.

    El YAML referencia directamente los archivos MRC exportados por este
    modulo y puede ejecutarse con:

      polnet --config config_seed{N}.yaml
    """
    density_path = os.path.abspath(os.path.join(
        _mrc_dir(), "bilayer_seed%04d.mrc" % membrane.seed
    ))

    yaml_content = (
        "metadata:\n"
        "  description: BicapaCryoET seed%(seed)d → PolNet pipeline\n"
        "  author: BicapaCryoET v15\n"
        "\n"
        "folders:\n"
        "  root: ./\n"
        "  input: ./polnet_input/\n"
        "  output: ./polnet_output/seed%(seed)04d/\n"
        "\n"
        "global:\n"
        "  ntomos: 1\n"
        "  seed: %(seed)d\n"
        "\n"
        "sample:\n"
        "  voi_shape: [%(sx)d, %(sy)d, %(sz)d]\n"
        "  voxel_size: %(vox)g\n"
        "  membranes:\n"
        "    - mb_type: file\n"
        "      mb_file: %(density)s\n"
        "      mb_thick: %(thick)g\n"
        "\n"
        "tem:\n"
        "  snr: %(snr)g\n"
        "  tilt_range: [%(tmin)d, %(tmax)d, %(tstep)d]\n"
        "\n"
        "# Instrucciones:\n"
        "#   1. Instala PolNet: pip install polnet\n"
        "#   2. Ejecuta: polnet --config %(yaml_name)s\n"
        "#   3. El tomograma simulado estara en polnet_output/seed%(seed)04d/\n"
        "#\n"
        "# Referencia:\n"
        "#   Martinez-Sanchez et al. IEEE Trans. Med. Imaging 2024\n"
        "#   https://doi.org/10.1109/TMI.2024.3398401\n"
    ) % {
        "seed":     membrane.seed,
        "sx":       tomo_shape[0],
        "sy":       tomo_shape[1],
        "sz":       tomo_shape[2],
        "vox":      voxel_angstrom,
        "density":  density_path,
        "thick":    membrane.geometry.total_thick / 10.0,
        "snr":      snr,
        "tmin":     tilt_range[0],
        "tmax":     tilt_range[1],
        "tstep":    tilt_range[2],
        "yaml_name": "config_seed%04d.yaml" % membrane.seed,
    }

    yaml_name = "config_seed%04d.yaml" % membrane.seed
    yaml_path = os.path.join(_mrc_dir(), yaml_name)
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print("  -> mrc/%s  (plantilla PolNet lista)" % yaml_name)
    return yaml_path


def export_double_gaussian_mrc(
    membrane: "BicapaCryoET",
    voxel_angstrom: float = 10.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> str:
    """
    Exporta la membrana como perfil de doble gaussiana para PolNet.

    PolNet representa las membranas como perfiles continuos de densidad
    tipo doble capa gaussiana, sin resolución molecular explícita.

    Esta función construye ese perfil directamente a partir de la geometría
    del modelo, mejorando la compatibilidad con el motor de simulación.

    Para cada voxel:
      rho(z) = A_head * [G(z - z_outer, sigma_hg) + G(z - z_inner, sigma_hg)]
             + A_tail * G(z - z_mid, sigma_tail)

    sigma_hg y sigma_tail se derivan de la geometría media de la bicapa.
    """
    from analysis import midplane_map

    g = membrane.geometry
    z_half = (g.total_thick / 10.0) / 2.0 + 0.5

    x_edges = np.linspace(0, membrane.Lx / 10, bins_xy + 1)
    y_edges = np.linspace(0, membrane.Ly / 10, bins_xy + 1)
    z_edges = np.linspace(-z_half, z_half, bins_z + 1)
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])

    z_mid_grid = midplane_map(membrane, bins=bins_xy)

    sigma_hg   = g.total_thick / 10.0 * 0.12
    sigma_tail = g.hydro_thick  / 10.0 * 0.35
    A_head = 0.47
    A_tail = 0.29

    vol = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.float32)

    for ix in range(bins_xy):
        for iy in range(bins_xy):
            z_ref   = z_mid_grid[ix, iy] / 10.0
            z_outer = (g.z_outer / 10.0) - z_ref
            z_inner = (g.z_inner / 10.0) - z_ref

            for iz, z in enumerate(z_centers):
                rho  = A_head * np.exp(-0.5 * ((z - z_outer) / sigma_hg) ** 2)
                rho += A_head * np.exp(-0.5 * ((z - z_inner) / sigma_hg) ** 2)
                rho += A_tail * np.exp(-0.5 * (z / sigma_tail) ** 2)
                vol[ix, iy, iz] = rho

    from scipy.ndimage import gaussian_filter
    vol_smooth = gaussian_filter(vol, sigma=[1.0, 1.0, 0.6])
    vol_norm   = (vol_smooth / vol_smooth.max() * 255.0).astype(np.float32)

    fname = "bilayer_gaussian_seed%04d.mrc" % membrane.seed
    path  = os.path.join(_mrc_dir(), fname)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(vol_norm.T)
        mrc.voxel_size = voxel_angstrom

    print("  -> mrc/%s  (perfil doble gaussiana, compatible PolNet)" % fname)
    return path


def export_label_mrc_with_closing(
    membrane: "BicapaCryoET",
    voxel_angstrom: float = 10.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> str:
    """Exporta etiquetas semanticas con cierre morfologico 3D."""
    from scipy.ndimage import binary_closing

    _, edges = analysis.volumetric_density(membrane, bins_xy=bins_xy, bins_z=bins_z)
    cz = 0.5 * (edges[2][:-1] + edges[2][1:])
    g  = membrane.geometry

    # Misma corrección que export_label_mrc: usar hg_thick promedio, no hydro_thick
    from lipid_types import LIPID_TYPES
    hg_mean = (
        sum(LIPID_TYPES[k].hg_thick * f for k, f in membrane.comp_outer.items()
            if k in LIPID_TYPES)
        if membrane.comp_outer else 9.0
    )
    hg_half_o = (g.z_outer - hg_mean / 2.0) / 10.0
    hg_half_i = (g.z_inner + hg_mean / 2.0) / 10.0

    labels = np.zeros((bins_xy, bins_xy, bins_z), dtype=np.uint8)
    for iz, z in enumerate(cz):
        if z > hg_half_o:    labels[:, :, iz] = 1
        elif z < hg_half_i:  labels[:, :, iz] = 3
        else:                labels[:, :, iz] = 2

    struct = np.ones((3, 3, 2), dtype=bool)
    for label_val in [1, 2, 3]:
        mask   = (labels == label_val)
        closed = binary_closing(mask, structure=struct, iterations=1)
        labels[closed & (labels == 0)] = label_val

    fname = "bilayer_seed%04d_labels_closed.mrc" % membrane.seed
    path  = os.path.join(_mrc_dir(), fname)
    with mrcfile.new(path, overwrite=True) as mrc:
        mrc.set_data(labels.T.astype(np.float32))
        mrc.voxel_size = voxel_angstrom

    print("  -> mrc/%s  (labels con closing morfologico)" % fname)
    return path
