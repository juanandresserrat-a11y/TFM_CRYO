# TFM

Generador de datasets sintéticos de membranas lipídicas para Cryo-ET y machine learning.

## Descripción

Este proyecto genera datasets sintéticos de bicapas lipídicas simuladas para aplicaciones en Cryo-Electron Tomography (Cryo-ET) y machine learning. Incluye herramientas para crear visualizaciones y exportar datos en formatos compatibles con ML.

## Estructura del Repositorio

```
TFM/
├── main.py                 # Punto de entrada principal
├── builder.py              # Construcción de la bicapa y lógica principal
├── lipid_types.py          # Definición de tipos lipídicos y parámetros
├── geometry.py             # Cálculos geométricos y generación de colas
├── physics.py              # Física: curvatura Helfrich, elasticidad
├── export.py               # Exportación de canales .npy y JSON
├── visualization.py        # Generación de figuras PNG
├── analysis.py             # Análisis y métricas estadísticas
├── z-testing.ipynb         # Notebook de pruebas
├── README.md               # Este archivo
├── requirements.txt        # Dependencias
├── CryoET/                 # Datos generados (output)
│   ├── seed0001/           # Figuras por semilla
│   ├── seed0002/
│   └── training/           # Canales .npy para ML
│       ├── seed0001/
│       ├── seed0002/
│       └── labels.json     # Metadatos agregados
└── pycache/                # Caché de Python
```
## Archivos Generados

Por cada semilla, se crean los siguientes archivos en el directorio `CRYOET/`:

```
| Archivo               | Descripción                                 |
|-----------------------|---------------------------------------------|
| `ch0_cryoET.npy`      | Imagen de densidad (simula Cryo-ET)         |
| `ch1_thickness.npy`   | Mapa de grosor local                        |
| `ch2/ch3_rough_*.npy` | Rugosidad de cada monocapa                  |
| `ch4/ch5_raft_*.npy`  | Fracción de dominios tipo raft              |
| `ch6_pip_density.npy` | Densidad de fosfoinosítidos                 |
| `ch8_asymmetry.npy`   | Diferencia de densidad entre capas          |
| `ch9_xz_slice.npy`    | Corte transversal lateral                   |
| `ch10_order.npy`      | Parámetro de orden de cadenas               |
| `ch11_interdig.npy`   | Interdigitación trans-leaflet               |
| `labels.json`         | Metadatos (composición, parámetros físicos) |
| `fig*.png`            | Visualizaciones de validación               |
```
