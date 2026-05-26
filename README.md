# TFM — Generador sintético de bicapas lipídicas para Cryo-ET

Genera datasets sintéticos de membranas lipídicas simuladas para Cryo-ET y machine learning, con exportación multiformato y figuras de publicación.

---

## Instalación

```bash
python -m pip install -r requirements.txt --user
```

---

## Uso rápido

```bash
# Paso 1 — generar simulaciones (datos base)
python main.py --sims 1 2 3 4 5 --validate --all

# Paso 2 — generar figuras de resultados
python results.py --sims 1 2 3 4 5
```

Los resultados se guardan en `CryoET/`.

---

## main.py — Generar simulaciones

```bash
python main.py --sims 23            # una simulación
python main.py --sims 1 2 3 4 5     # varias
python main.py --sims 23 --all      # todos los formatos de exportación
```

| Flag | Descripción |
|---|---|
| `--sims N [N ...]` | Semillas a simular (default: 27 42) |
| `--validate` | Benchmarks biofísicos |
| `--paraview` | VTP + script de visualización para ParaView |
| `--model3d` | Volumen 3D de densidad electrónica (MRC) |
| `--mrc` | MRC simplificado para PolNet |
| `--positions` | PDB + CSV + particle list |
| `--figures` | Figuras de publicación PNG |
| `--stats` | Estadísticas del dataset (requiere ≥2 simulaciones) |
| `--all` | Activa: paraview + model3d + validate + mrc + positions |

---

## results.py — Figuras de resultados

```bash
python results.py --sims 1 2 3 4 5         # R1–R6 completo
python results.py --sims 23 --only R1 R3   # secciones específicas
```

| Sección | Contenido |
|---|---|
| `R1` | Composición, parámetros elásticos, perfil ED |
| `R2` | Tabla de benchmarks y scores de validación |
| `R3` | Organización lateral: dominios Lo/Ld y PIPs |
| `R4` | Galería de los 12 campos de entrenamiento |
| `R5` | Calidad cryo-ET: CTF, ruido, espectros PSD |
| `R6` | Comparativa multi-simulación (requiere ≥2) |
| `R6b` | Convergencia acumulada para justificar N=5 (requiere ≥2) |

Los PDFs se guardan en `CryoET/resultados/`.

---

## Visualización en ParaView

### Archivos necesarios

Cada simulación genera en `CryoET/paraview/simulacion{N}/` dos archivos:

| Archivo | Descripción |
|---|---|
| `bicapa_sim{N}.vtp` | **Requerido** — geometría completa de la bicapa |
| `visualize_sim{N}.py` | **Requerido** — script de estado de ParaView |
| `bicapa_*_heads.vtp` | Opcional — solo cabezas polares |
| `bicapa_*_tails.vtp` | Opcional — solo colas acilo |
| `bicapa_*_lo_domain.vtp` | Opcional — dominio Lo aislado |
| `bicapa_*_pips.vtp` | Opcional — clústeres de PIPs |

Los archivos opcionales permiten explorar componentes por separado, pero para la visualización estándar solo se necesitan los dos primeros.

### Cómo abrir

**Opción A — script de estado (recomendado):**
```
ParaView → File → Load State → visualize_sim{N}.py
```
Carga automáticamente el VTP, aplica todos los filtros y configura la cámara y la escala de colores Blue–Green–Orange por densidad electrónica.

**Opción B — abrir el VTP directamente:**
```
ParaView → File → Open → bicapa_sim{N}.vtp
```
Requiere configurar los filtros manualmente (Threshold, Tube, colormaps).

### Lo que verás

La escena cargada por el script muestra dos layouts:
- **Vista principal** — perspectiva lateral con colas como tubos y cabezas/glicerol/CHOL coloreados por densidad electrónica
- **Vista secundaria** — vista cenital de la bicapa completa

---

## Campos de entrenamiento

Por simulación en `CryoET/entrenamiento/simulacion{N}/` (64×64 px, float32):

| Campo | Descripción |
|---|---|
| `c0_cryoET.npy` | Imagen cryo-ET simulada (CTF + ruido) |
| `c1_thickness.npy` | Grosor local (Å) |
| `c2/c3_rough_*.npy` | Rugosidad monocapa externa/interna (Å) |
| `c4/c5_raft_*.npy` | Fracción Lo externa/interna [0,1] |
| `c6_pip_density.npy` | Densidad de fosfoinosítidos |
| `c7_asymmetry.npy` | Asimetría composicional ext − int |
| `c8_xz_slice.npy` | Sección transversal XZ |
| `c9_order.npy` | Parámetro de orden S_CH [0,1] |
| `c10_interdig.npy` | Interdigitación trans-leaflet [0,1] |
| `c11_prior_clean.npy` | Densidad electrónica sin degradación |

```python
import numpy as np
import os

base = "CryoET/entrenamiento/simulacion0001/"
files = sorted(f for f in os.listdir(base) if f.endswith(".npy"))
tensor = np.stack([np.load(base + f) for f in files])  # (12, 64, 64)
```

---

## Estructura de salida

```
CryoET/
├── entrenamiento/simulacion{N}/   # 12 campos .npy + labels.json
├── resultados/                    # PDFs de figuras R1–R6
├── paraview/simulacion{N}/        # VTP + script visualize_sim{N}.py
├── mrc/                           # Volúmenes MRC para PolNet
├── posiciones/                    # PDB + CSV + particle list
├── modelo3d/                      # Volúmenes 3D físicos
└── validacion/                    # JSONs de benchmarks
```

---

## Referencias clave

1. Singer & Nicolson. Science 1972 — modelo de mosaico fluido
2. Simons & Ikonen. Nature 1997 — balsas lipídicas
3. Helfrich W. Z Naturforsch C 1973 — elasticidad de membranas
4. Martinez-Sanchez et al. IEEE Trans Med Imaging 2024 — datos sintéticos cryo-ET
5. Sharma et al. Emerg Top Life Sci 2023 — visualización de membranas con cryo-EM

Bibliografía completa en el TFM.
