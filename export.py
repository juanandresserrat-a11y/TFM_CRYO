"""
export.py
=========
Exportación de datos de training: arrays numpy y metadatos JSON.

Genera 11 canales por semilla y actualiza el labels.json global.
No depende de matplotlib, por lo que puede ejecutarse sin entorno gráfico.

Estructura de salida:
  CryoET/
    training/
      labels.json           metadatos acumulativos de todas las semillas
      seed{N}/
        ch0_cryoET.npy      imagen de densidad (entrada para CNN)
        ch1_thickness.npy
        ch2_rough_outer.npy
        ch3_rough_inner.npy
        ch4_raft_outer.npy
        ch5_raft_inner.npy
        ch6_pip_density.npy
        ch8_asymmetry.npy
        ch9_xz_slice.npy
        ch10_order.npy
        ch11_interdig.npy
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, List

import numpy as np

import analysis
from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET


CHANNEL_DESCRIPTIONS = {
    "ch0":  "cryoET_imagen_densidad",
    "ch1":  "grosor_local_bicapa",
    "ch2":  "rugosidad_monocapa_externa",
    "ch3":  "rugosidad_monocapa_interna",
    "ch4":  "fraccion_raft_externa",
    "ch5":  "fraccion_raft_interna",
    "ch6":  "densidad_pip_total",
    "ch8":  "asimetria_composicional",
    "ch9":  "slice_xz_cryoET",
    "ch10": "parametro_orden_S_CH [7,8]",
    "ch11": "interdigitacion_trans_leaflet [9]",
    "ch12": "prior_limpio_densidad_electronica_sin_CTF_ni_ruido",
}

REFERENCES = [
    "Smith et al. LiveCoMS 2019 [2]",
    "Helfrich 1973 [4]",
    "Pinigin Membranes 2022 [5]",
    "Chakraborty et al. PNAS 2020 [6]",
    "Piggot et al. JCTC 2017 [7]",
    "Chaisson et al. JCIM 2025 [9]",
]


def export_training(membrane: "BicapaCryoET", bins: int = 64) -> str:
    """
    Exporta los 11 canales de training y actualiza labels.json.

    Parametros
    ----------
    membrane : BicapaCryoET
        Bicapa ya construida (build() ya ejecutado).
    bins : int
        Resolución de cada canal (bins × bins). Default 64.

    Retorna
    -------
    str
        Ruta al directorio de salida de esta semilla.
    """
    d = membrane.training_dir()


    channels = {
        "ch0_cryoET": (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins)
            + analysis.density_map(membrane, membrane.inner_leaflet, bins=bins)
        ),
        "ch1_thickness":   analysis.thickness_map(membrane, bins=bins),
        "ch2_rough_outer": analysis.roughness_map(membrane, membrane.outer_leaflet, bins=bins),
        "ch3_rough_inner": analysis.roughness_map(membrane, membrane.inner_leaflet, bins=bins),
        "ch4_raft_outer":  analysis.raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins),
        "ch5_raft_inner":  analysis.raft_fraction_map(membrane, membrane.inner_leaflet, bins=bins),
        "ch6_pip_density": analysis.pip_density_map(membrane, bins=bins),
        "ch8_asymmetry":   (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins, sigma=2.0)
            - analysis.density_map(membrane, membrane.inner_leaflet, bins=bins, sigma=2.0)
        ),
        "ch10_order":      analysis.order_parameter_map(membrane, bins=bins),
        "ch11_interdig":   analysis.interdigitation_map(membrane, bins=bins),
    }
    Hxz, _, _ = analysis.xz_projection(membrane, bx=bins * 2, bz=bins)
    channels["ch9_xz_slice"] = Hxz

    try:
        from electron_density import electron_density_projection
        channels["ch12_prior_clean"] = electron_density_projection(
            membrane, bins_xy=bins, sigma=0.8
        )
    except Exception:
        pass


    for name, arr in channels.items():
        np.save(os.path.join(d, "%s.npy" % name), arr)


    g = membrane.geometry
    todos = membrane.outer_leaflet + membrane.inner_leaflet

    gel_order = [l.order_param for l in todos if l.lipid_type.phase == "gel"]
    fluid_order = [l.order_param for l in todos if l.lipid_type.phase == "fluid"]

    meta = {
        "seed": membrane.seed,
        "size_nm": [membrane.Lx / 10, membrane.Ly / 10],
        "kc_kBT_nm2": round(membrane.bending_modulus, 2),
        "sigma_kBT_nm2": round(membrane.surface_tension, 4),
        "grosor_total_A": round(g.total_thick, 2),
        "grosor_hidro_A": round(g.hydro_thick, 2),
        "n_sup": len(membrane.outer_leaflet),
        "n_inf": len(membrane.inner_leaflet),
        "n_balsas_s": len(membrane.rafts_outer),
        "n_balsas_i": len(membrane.rafts_inner),
        "n_pip_clusters": len(membrane.pip_clusters),
        "n_perturbadores": len(membrane.perturbations),
        "densidad_perturbadores": round(membrane.perturbation_density, 4),
        "S_CH_medio_gel": round(float(np.mean(gel_order)) if gel_order else 0, 3),
        "S_CH_medio_fluido": round(float(np.mean(fluid_order)) if fluid_order else 0, 3),
        "comp_outer": {t: round(f, 4) for t, f in membrane.comp_outer.items()},
        "comp_inner": {t: round(f, 4) for t, f in membrane.comp_inner.items()},
        "canales": CHANNEL_DESCRIPTIONS,
        "referencias": REFERENCES,
    }


    labels_file = os.path.join(OUTPUT_DIR, "training", "labels.json")
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    all_meta: List[dict] = []
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            all_meta = json.load(f)
    all_meta = [m for m in all_meta if m.get("seed") != membrane.seed]
    all_meta.append(meta)
    with open(labels_file, "w") as f:
        json.dump(all_meta, f, indent=2)

    print("  -> training/seed%04d/ (%d canales)" % (membrane.seed, len(channels)))
    return d
