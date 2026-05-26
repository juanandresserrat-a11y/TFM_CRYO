"""
export.py
Exportación de datos de training: arrays numpy y metadatos JSON.

Genera 12 canales por semilla y actualiza el labels.json global.
No depende de matplotlib, por lo que puede ejecutarse sin entorno gráfico.

Referencias principales:
    [11]  Martinez-Sanchez et al. 2024 – simulación de contexto celular en datasets sintéticos de cryo-ET
    [13]  Seghiri et al. 2026 – segmentación aumentada de membranas en cryo-ET mediante simulación del contexto celular
    [14]  Moebel et al. 2021 – deep learning para identificación de macromoléculas en tomogramas celulares de cryo-ET
"""

from __future__ import annotations

import json
import os
from typing import TYPE_CHECKING, List

import numpy as np

import analysis
import ctf_sim
from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET


CHANNEL_DESCRIPTIONS = {
    "c0":  "cryoET_imagen_densidad",
    "c1":  "grosor_local_bicapa",
    "c2":  "rugosidad_monocapa_externa",
    "c3":  "rugosidad_monocapa_interna",
    "c4":  "fraccion_raft_externa",
    "c5":  "fraccion_raft_interna",
    "c6":  "densidad_pip_total",
    "c7":  "asimetria_composicional",
    "c8":  "slice_xz_cryoET",
    "c9": "parametro_orden_S_CH [7,8]",
    "c10": "interdigitacion_trans_leaflet [9]",
    "c11": "prior_limpio_densidad_electronica_sin_CTF_ni_ruido",
}

REFERENCES = [
        "Smith et al. LiveCoMS 2019 [24]",
        "Helfrich 1973 [15]",
        "Pinigin Membranes 2022 [16]",
        "Chakraborty et al. PNAS 2020 [17]",
        "Piggot et al. JCTC 2017 [18]",
        "Chaisson et al. JCIM 2025 [22]",
]


def export_training(membrane: "BicapaCryoET", bins: int = 64) -> str:
    """Exporta los 11 campos de training y actualiza labels.json."""
    d = membrane.training_dir()


    channels = {
        "c0_cryoET": ctf_sim.simulate_projection(
            membrane, defocus_um=2.0, snr=0.1,
            use_electron_density=True, bins_xy=bins,
        ),
        "c1_thickness":   analysis.thickness_map(membrane, bins=bins),
        "c2_rough_outer": analysis.roughness_map(membrane, membrane.outer_leaflet, bins=bins),
        "c3_rough_inner": analysis.roughness_map(membrane, membrane.inner_leaflet, bins=bins),
        "c4_raft_outer":  analysis.raft_fraction_map(membrane, membrane.outer_leaflet, bins=bins),
        "c5_raft_inner":  analysis.raft_fraction_map(membrane, membrane.inner_leaflet, bins=bins),
        "c6_pip_density": analysis.pip_density_map(membrane, bins=bins),
        "c7_asymmetry":   (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins, sigma=2.0)
            - analysis.density_map(membrane, membrane.inner_leaflet, bins=bins, sigma=2.0)
        ),
        "c9_order":      analysis.order_parameter_map(membrane, bins=bins),
        "c10_interdig":   analysis.interdigitation_map(membrane, bins=bins),
    }
    Hxz, _, _ = analysis.xz_projection(membrane, bx=bins * 2, bz=bins)
    channels["c8_xz_slice"] = Hxz

    try:
        from electron_density import electron_density_projection
        channels["c11_prior_clean"] = electron_density_projection(
            membrane, bins_xy=bins, sigma=0.8
        )
    except Exception as e:
        print(f"  [WARN] c11_prior_clean no generado: {type(e).__name__}: {e}")
        channels["c11_prior_clean"] = np.zeros_like(channels["c0_cryoET"])


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
        "campos": CHANNEL_DESCRIPTIONS,
        "referencias": REFERENCES,
    }


    labels_file = os.path.join(OUTPUT_DIR, "entrenamiento", "labels.json")
    os.makedirs(os.path.dirname(labels_file), exist_ok=True)
    all_meta: List[dict] = []
    if os.path.exists(labels_file):
        with open(labels_file) as f:
            all_meta = json.load(f)
    all_meta = [m for m in all_meta if m.get("seed") != membrane.seed]
    all_meta.append(meta)
    with open(labels_file, "w") as f:
        json.dump(all_meta, f, indent=2)

    print("  -> entrenamiento/simulacion%04d/  %d campos guardados" % (membrane.seed, len(channels)))
    return d
