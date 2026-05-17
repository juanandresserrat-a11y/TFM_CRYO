# TFM
Generador de datasets sintéticos de membranas lipídicas para Cryo-ET y machine learning.

## Descripción

Este proyecto genera datasets sintéticos de bicapas lipídicas simuladas para aplicaciones
en Cryo-Electron Tomography (Cryo-ET) y machine learning. Incluye herramientas para:

- Construir bicapas lipídicas asimétricas con dominios Lo/Ld, clusters de PIPs y proteínas transmembrana
- Simular contraste cryo-ET realista (CTF, missing wedge, ruido Poisson + Gaussiano)
- Exportar datos en múltiples formatos: canales `.npy`, MRC (PolNet), PDB/CSV, VTP (ParaView)
- Validar propiedades biofísicas contra referencias de la literatura

## Estructura del Repositorio

```
TFM/
├── main.py                  # Punto de entrada principal (CLI)
├── builder.py               # Construcción de la bicapa y lógica principal
├── lipid_types.py           # Definición de tipos lipídicos y parámetros biofísicos
├── geometry.py              # Dataclasses: MembraneGeometry y LipidInstance
├── physics.py               # Física: curvatura Helfrich, cadenas acil, S_CH
├── analysis.py              # Mapas 2D/3D: densidad, rugosidad, grosor, orden
├── electron_density.py      # Densidad electrónica real por tipo lipídico (e·Å⁻³)
├── ctf_sim.py               # Simulación TEM: CTF, missing wedge, ruido
├── figures.py               # Figuras de publicación (9 paneles, 300 DPI)
├── model_3d.py              # Modelo 3D físico: volumen ED, labels, panel visual
├── export.py                # Exportación de canales .npy y labels.json
├── export_mrc.py            # Exportación MRC para PolNet (densidad + etiquetas)
├── export_paraview.py       # Exportación VTP para ParaView (granos CG)
├── export_positions.py      # Posiciones 3D: PDB, CSV, PolNet particle list
├── validation.py            # Benchmarks biofísicos y panel de validación
├── dataset_stats.py         # Estadísticas de dataset multi-semilla
├── bicapa.ipynb             # Notebook interactivo
├── README.md                # Este archivo
├── requirements.txt         # Dependencias
└── CryoET/                  # Outputs generados (creado al ejecutar)
    ├── figuras/
    │   └── simulacion{N}/   # Figuras PNG por semilla
    ├── training/
    │   ├── seed{N}/         # Canales .npy por semilla
    │   └── labels.json      # Metadatos agregados de todas las semillas
    ├── mrc/                 # Volúmenes MRC para PolNet
    ├── paraview/            # Archivos VTP para ParaView
    ├── positions/           # PDB, CSV y particle list
    ├── model3d/             # Volúmenes 3D físicos
    └── validation/          # JSONs de benchmarks y paneles de validación
```

## Uso

```bash
# Instalación de dependencias
pip install -r requirements.txt

# Simulación básica (genera canales .npy + figuras)
python main.py --sims 1 2 3

# Con validación biofísica
python main.py --sims 27 --validate

# Todos los outputs (ParaView + MRC + posiciones + modelo 3D)
python main.py --sims 27 --all

# Dataset completo 50 semillas (solo training, sin figuras)
python main.py --sims $(seq 0 49) --stats

# Desde el notebook interactivo
jupyter notebook bicapa.ipynb
```

### Flags disponibles

| Flag | Descripción |
|---|---|
| `--sims N [N ...]` | Números de semilla a simular (default: 27 42) |
| `--size X Y` | Tamaño de la membrana en nm (default: 50 50) |
| `--paraview` | Exporta VTP + README para ParaView |
| `--model3d` | Volumen 3D de densidad electrónica real (MRC) |
| `--mrc` | MRC simplificado + doble gaussiana para PolNet |
| `--positions` | PDB completo + CSV + PolNet particle list |
| `--validate` | Panel de validación biofísica (9 benchmarks) |
| `--figures` | Figuras de publicación (9 paneles, 300 DPI) |
| `--stats` | Estadísticas del dataset (requiere >1 semilla) |
| `--dpi N` | Resolución de figuras (default: 200) |
| `--all` | Activa: paraview + model3d + validate + mrc + positions |

## Canales de Training Generados

Por cada semilla se generan los siguientes arrays en `CryoET/training/seed{N}/`:

| Archivo | Descripción | Unidades |
|---|---|---|
| `ch0_cryoET.npy` | Densidad proyectada (entrada CNN) | Da·Å⁻² |
| `ch1_thickness.npy` | Grosor local de la bicapa | Å |
| `ch2_rough_outer.npy` | Rugosidad monocapa externa σ_z | Å |
| `ch3_rough_inner.npy` | Rugosidad monocapa interna σ_z | Å |
| `ch4_raft_outer.npy` | Fracción local Lo monocapa externa | [0, 1] |
| `ch5_raft_inner.npy` | Fracción local Lo monocapa interna | [0, 1] |
| `ch6_pip_density.npy` | Densidad de fosfoinosítidos | Da·Å⁻² |
| `ch8_asymmetry.npy` | Asimetría composicional ext − int | Da·Å⁻² |
| `ch9_xz_slice.npy` | Sección transversal XZ | Da·Å⁻² |
| `ch10_order.npy` | Parámetro de orden S_CH medio | [0, 1] |
| `ch11_interdig.npy` | Interdigitación trans-leaflet | [0, 1] |
| `ch12_prior_clean.npy` | Densidad electrónica sin CTF ni ruido | e·Å⁻³ |
| `labels.json` | Metadatos: composición, parámetros físicos, referencias | — |

> **Nota:** El canal ch7 está reservado para uso futuro.

## Formatos de Exportación

### MRC (PolNet)
Compatibles con [PolNet](https://doi.org/10.1109/TMI.2024.3398401) para generar tomogramas sintéticos completos:
- `bilayer_seed{N}.mrc` — densidad 3D normalizada
- `bilayer_seed{N}_labels.mrc` — etiquetas semánticas (0=agua, 1=cabeza ext, 2=núcleo, 3=cabeza int)
- `bilayer_gaussian_seed{N}.mrc` — perfil doble gaussiana para PolNet nativo
- `config_seed{N}.yaml` — plantilla de configuración PolNet lista para usar

### ParaView (VTP)
Cada lípido y proteína se exporta como grafo de granos CG con atributos por punto:
- `region` (0=cabeza, 1=glicerol, 2=cola sn1, 3=cola sn2, 4=CHOL, 5=proteína)
- `is_protein`, `in_raft`, `is_pip`, `order_param`, `electron_density`
- Archivos separados por región: `*_heads.vtp`, `*_tails.vtp`, `*_proteins.vtp`, etc.

### Posiciones 3D
- `bilayer_seed{N}.pdb` — todas las posiciones en formato PDB estándar
- `positions_seed{N}.csv` — tabla completa (lipid_id, tipo, leaflet, xyz, fase...)
- `polnet_particles_seed{N}.csv` — lista de partículas con orientación en cuaternión

## Figuras Generadas

| Figura | Contenido |
|---|---|
| `fig1_perfil_ED` | Perfil de densidad electrónica 1D (patrón dark-bright-dark) |
| `fig2_composicion` | Composición lipídica por monocapa |
| `fig3_helfrich` | Espectro de fluctuaciones de Helfrich y kc |
| `fig4_grosor` | Mapa 2D de grosor local |
| `fig5_mapa_raft` | Mapa 2D de dominios Lo/Ld |
| `fig6_mapa_order` | Mapa 2D de parámetro de orden S_CH |
| `fig7_pip_radial` | Perfil radial de densidad PIP desde centroide |
| `fig8_parametros` | Panel de parámetros biofísicos globales |
| `fig9_mapa_pips_balsas` | Distribución espacial de PIPs sobre dominios Lo/Ld |

## Composición Lipídica por Defecto

| Especie | Monocapa externa | Monocapa interna |
|---|---|---|
| POPC | 33% | 18% |
| SM | 24% | — |
| CHOL | 30% | 28% |
| GM1 | 5% | — |
| POPE | 4% | 19% |
| PlsPE | — | 5% |
| POPS | — | 14% |
| PI/PIPs | 4% | 16% |

La composición varía ligeramente entre semillas mediante muestreo Dirichlet
(concentración = 50) para generar diversidad en el dataset.

## Referencias Principales

- Helfrich, W. (1973). Z. Naturforsch. C — elasticidad de membrana
- Piggot et al. (2017). JCTC — parámetro de orden S_CH
- Chaisson et al. (2025). JCIM — interdigitación trans-leaflet
- Martinez-Sanchez et al. (2024). IEEE Trans. Med. Imaging — PolNet
- Kučerka et al. (2011). BBA Biomembranes — áreas y grosores lipídicos
- Nagle & Tristram-Nagle (2000). BBA Reviews — densidad electrónica
