# TFM
Generador de datasets sintéticos de membranas lipídicas para Cryo-ET y machine learning.

## Descripción

Este proyecto genera datasets sintéticos de bicapas lipídicas simuladas para aplicaciones
en Cryo-Electron Tomography (Cryo-ET) y machine learning. Incluye herramientas para:

- Construir bicapas lipídicas asimétricas con dominios Lo/Ld, clusters de PIPs y proteínas transmembrana
- Simular contraste cryo-ET realista (CTF, missing wedge, ruido Poisson + Gaussiano)
- Exportar datos en múltiples formatos: canales `.npy`, MRC (PolNet), PDB/CSV, VTP (ParaView)
- Validar propiedades biofísicas contra referencias de la literatura
- Generar figuras de publicación para TFM / artículo científico

---

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
├── results.py               # Figuras de resultados para TFM (R1–R8)
├── bicapa.ipynb             # Notebook interactivo
├── README.md                # Este archivo
├── requirements.txt         # Dependencias
└── CryoET/                  # Outputs generados (creado automáticamente)
    ├── figuras/
    │   └── simulacion{N}/   # Figuras PNG de publicación por semilla
    ├── training/
    │   ├── seed{N}/         # Canales .npy por semilla
    │   └── labels.json      # Metadatos agregados de todas las semillas
    ├── mrc/                 # Volúmenes MRC para PolNet
    ├── paraview/            # Archivos VTP para ParaView
    ├── positions/           # PDB, CSV y particle list
    ├── model3d/             # Volúmenes 3D físicos
    ├── validation/          # JSONs de benchmarks y paneles de validación
    └── resultados/          # PDFs de resultados para TFM (R1–R8)
```

---

## 1. Instalación

```bash
cd TFM/
pip install -r requirements.txt
```

---

## 2. Generar semillas — `main.py`

Cada **semilla** genera una membrana diferente con composición lipídica ligeramente
distinta (muestreo Dirichlet). Los datos generados son la base para todo lo demás.

### Mínimo — canales de training

```bash
# Una semilla
python main.py --sims 1

# Varias semillas a la vez
python main.py --sims 1 2 3 4 5

# Con tamaño de membrana personalizado (en nm)
python main.py --sims 27 --size 100 100
```

Genera `CryoET/training/seed{N}/` con los 12 canales `.npy` y actualiza `labels.json`.

### Con validación biofísica

```bash
python main.py --sims 27 --validate
python main.py --sims 1 2 3 4 5 --validate --stats
```

### Todos los formatos de exportación

```bash
# Todo de una vez
python main.py --sims 27 --all

# O seleccionar individualmente
python main.py --sims 27 --paraview   # VTP para ParaView
python main.py --sims 27 --mrc        # MRC para PolNet
python main.py --sims 27 --positions  # PDB + CSV + particle list
python main.py --sims 27 --model3d    # Volumen 3D físico
python main.py --sims 27 --figures    # Figuras de publicación PNG
```

### Dataset completo para ML

```bash
# 50 semillas, solo training (rápido, sin figuras)
python main.py --sims $(seq 0 49) --stats

# 100 semillas con validación
python main.py --sims $(seq 0 99) --validate --stats
```

### Flags de main.py

| Flag | Descripción |
|---|---|
| `--sims N [N ...]` | Semillas a simular (default: 27 42) |
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

---

## 3. Generar figuras de resultados — `results.py`

`results.py` genera figuras de publicación en PDF organizadas por secciones
de resultados de TFM. Lee los datos ya generados por `main.py`.

> **Requisito:** ejecutar `main.py --sims N` antes de `results.py --sims N`.

### Uso básico

```bash
# Todas las figuras (R1–R8) para una semilla
python results.py --sims 1

# Varias semillas — activa R4 y R8 multi-semilla automáticamente
python results.py --sims 1 2 3 4 5

# Solo secciones específicas
python results.py --sims 27 --only R1 R3 R5 R7

# Resolución máxima para publicación
python results.py --sims 27 --dpi 300
```

### Flujo completo recomendado

```bash
# Paso 1: generar datos
python main.py --sims 1 2 3 4 5 --validate --figures --all

# Paso 2: generar todas las figuras de resultados
python results.py --sims 1 2 3 4 5

# Paso 3: solo las figuras utilizadas para el TFM
python results.py --sims 1 2 3 4 5 --only R1 R2 R3 R4 R7
```

Los PDFs se guardan en `CryoET/resultados/`.

### Secciones disponibles (R1–R8)

| Sección | PDF generado | Contenido | Semillas |
|---|---|---|---|
| `R1` | `R1_caracterizacion_seed{N}.pdf` | Composición, parámetros elásticos, perfil ED, grosor Lo/Ld | 1 |
| `R2` | `R2_validacion_seed{N}.pdf` | Tabla de benchmarks, Helfrich, S_CH, interdigitación | 1 |
| `R3` | `R3_organizacion_lateral_seed{N}.pdf` | Mapa Lo/Ld, PIPs sobre fase, S_CH 2D, densidad PIPs | 1 |
| `R4` | `R4_comparativa_multisemilla.pdf` | Violinplots kc/grosor/S_CH, score por semilla, composición ± DE | ≥2 |
| `R5` | `R5_canales_training_seed{N}.pdf` | Galería de los 12 canales .npy en cuadrícula 3×4 | 1 |
| `R6` | `R6_calidad_cryoET_seed{N}.pdf` | Imagen limpia / CTF / ruido, espectros PSD, curvas CTF | 1 |
| `R7` | `R7_sintesis_literatura_seed{N}.pdf` | Tabla visual: simulación vs. rangos bibliográficos | 1 |
| `R8` | `R8_dataset_ML.pdf` | Variabilidad kc/grosor/S_CH, heatmaps de composición | ≥2 |

> R4 y R8 requieren al menos 2 semillas y se saltan automáticamente si solo hay una.

### Flags de results.py

| Flag | Descripción |
|---|---|
| `--sims N [N ...]` | Semillas a procesar (ya generadas con main.py) |
| `--size X Y` | Tamaño en nm — debe coincidir con main.py (default: 50 50) |
| `--only R1 R3 ...` | Generar solo esas secciones |
| `--dpi N` | Resolución de salida en DPI (default: 300) |

---

## 4. Notebook interactivo — `bicapa.ipynb`

```bash
jupyter notebook bicapa.ipynb
```

Permite construir y explorar una semilla de forma interactiva, visualizar
cada figura individualmente y exportar datos sin usar la línea de comandos.

---

## 5. Canales de training generados

Por cada semilla en `CryoET/training/seed{N}/`:

| Archivo | Descripción | Unidades |
|---|---|---|
| `c0_cryoET.npy` | Densidad proyectada (entrada CNN) | Da·Å⁻² |
| `c1_thickness.npy` | Grosor local de la bicapa | Å |
| `c2_rough_outer.npy` | Rugosidad monocapa externa σ_z | Å |
| `c3_rough_inner.npy` | Rugosidad monocapa interna σ_z | Å |
| `c4_raft_outer.npy` | Fracción local Lo monocapa externa | [0, 1] |
| `c5_raft_inner.npy` | Fracción local Lo monocapa interna | [0, 1] |
| `c6_pip_density.npy` | Densidad de fosfoinosítidos | Da·Å⁻² |
| `c7_asymmetry.npy` | Asimetría composicional ext − int | Da·Å⁻² |
| `c8_xz_slice.npy` | Sección transversal XZ | Da·Å⁻² |
| `c9_order.npy` | Parámetro de orden S_CH medio | [0, 1] |
| `c10_interdig.npy` | Interdigitación trans-leaflet | [0, 1] |
| `c11_prior_clean.npy` | Densidad electrónica sin CTF ni ruido | e·Å⁻³ |
| `labels.json` | Metadatos: composición, parámetros físicos, referencias | — |

Los canales están numerados consecutivamente c0–c11 (12 canales en total, 64×64 px, float32).

### Cargar los canales en Python

```python
import numpy as np, json, os

seed = 1
base = f"CryoET/training/seed{seed:04d}/"

# Canal individual
c0 = np.load(base + "c0_cryoET.npy")          # shape (64, 64)

# Tensor completo con todos los canales
files = sorted(f for f in os.listdir(base) if f.endswith(".npy"))
tensor = np.stack([np.load(base + f) for f in files])  # shape (12, 64, 64)

# Metadatos de la semilla
with open("CryoET/training/labels.json") as f:
    labels = json.load(f)
meta = next(m for m in labels if m["seed"] == seed)
print(meta["kc_kBT_nm2"], meta["grosor_total_A"], meta["comp_outer"])
```

---

## 6. Formatos de exportación adicionales

### MRC para PolNet

```bash
python main.py --sims 27 --mrc
# Salida: CryoET/mrc/
#   bilayer_seed{N}.mrc            densidad 3D normalizada
#   bilayer_seed{N}_labels.mrc     etiquetas: 0=agua 1=cabeza_ext 2=nucleo 3=cabeza_int
#   bilayer_gaussian_seed{N}.mrc   perfil doble gaussiana compatible PolNet
#   config_seed{N}.yaml            polnet --config config_seed{N}.yaml
```

### ParaView (VTP)

```bash
python main.py --sims 27 --paraview
# Salida: CryoET/paraview/simulacion{N}/
# Abrir en ParaView → Filters → Tube → colorear por electron_density / in_raft / is_protein
```

### Posiciones 3D

```bash
python main.py --sims 27 --positions
# Salida: CryoET/positions/
#   bilayer_seed{N}.pdb            compatible PyMOL, ChimeraX, VMD
#   positions_seed{N}.csv          tabla xyz + fase + orden + raft + PIP
#   polnet_particles_seed{N}.csv   orientación en cuaternión para PolNet
```

---

## 7. Composición lipídica por defecto

| Especie | Monocapa externa | Monocapa interna |
|---|---|---|
| POPC | 33% | 18% |
| SM | 24% | — |
| CHOL | 30% | 28% |
| GM1 | 5% | — |
| POPE | 4% | 19% |
| PlsPE | — | 5% |
| POPS | — | 14% |
| PI / PIPs | 4% | 16% |

La composición varía entre semillas mediante muestreo Dirichlet (k=50).

---

## 8. Referencias principales

- Helfrich, W. (1973). Z. Naturforsch. C - elasticidad de membrana
- Piggot et al. (2017). JCTC - parámetro de orden S_CH
- Chaisson et al. (2025). JCIM - interdigitación trans-leaflet
- Martinez-Sanchez et al. (2024). IEEE Trans. Med. Imaging - PolNet
- Kučerka et al. (2011). BBA Biomembranes - áreas y grosores lipídicos
- Nagle & Tristram-Nagle (2000). BBA Reviews - densidad electrónica
- Pinigin (2022). Membranes - parámetros elásticos de membrana
- Di Paolo & De Camilli (2006). Nature - fosfoinosítidos
