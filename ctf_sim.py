"""
ctf_sim.py
==========
Simulacion completa de adquisicion TEM para cryo-ET.

Implementa los tres efectos fisicos que degradan la imagen experimental:
  1. CTF (Contrast Transfer Function): invierte el contraste a ciertas
     frecuencias espaciales, produciendo halos caracteristicos.
  2. Missing wedge: la geometria de inclinacion limitada (tipicamente
     ±60-70°) deja sectores del espacio de Fourier sin muestrar,
     causando elongacion de estructuras en la direccion del eje de
     inclinacion y perdida de membranas oblicuas.
  3. Ruido: combinacion de ruido de conteo (Poisson) y ruido de lectura
     del detector (Gaussiano).

La aproximacion de PSF gaussiana usada en el modulo de analisis captura
la perdida de resolucion pero NO reproduce la inversion de contraste ni
la anisotropia del missing wedge. Este modulo los implementa correctamente.

Parametros tipicos de cryo-ET:
  Voltaje:    300 kV  (lambda = 0.0197 Å)
  Cs:         2.7 mm  (aberracion esferica)
  Defoco:     1-3 μm  (negativo = subdenfoco convencional)
  SNR salida: 0.05-0.2 (muy ruidoso)
  Tilt:       ±60° a 3°/paso

Fuentes:
  [14] Dubochet et al. Q. Rev. Biophys. 1988
  [16] Lucic, Rigort & Baumeister, J. Cell Biol. 2013
  Frank, Electron Tomography, Springer 2006
  Wade, Ultramicroscopy 1992 (envelopes CTF)
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter

import analysis
from builder import OUTPUT_DIR

if TYPE_CHECKING:
    from builder import BicapaCryoET


LAMBDA_300KV = 0.0197


def wavelength_angstrom(voltage_kv: float = 300.0) -> float:
    """
    Longitud de onda de de Broglie relativista para electrones (Å).

    lambda = h / sqrt(2 m0 e V (1 + eV/2m0c²))
    """
    m0c2_kev = 511.0
    return 12.264 / np.sqrt(voltage_kv * 1e3 * (1.0 + voltage_kv / (2.0 * m0c2_kev)))


def compute_ctf(
    qx: np.ndarray,
    qy: np.ndarray,
    defocus_um: float = 2.0,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    amplitude_contrast: float = 0.07,
    b_factor: float = 200.0,
) -> np.ndarray:
    """
    Funcion de transferencia de contraste CTF(q).

    CTF(q) = -sqrt(1 - Ac²) · sin(chi(q)) + Ac · cos(chi(q))

    donde chi(q) = π·Δf·λ·q² - π/2·Cs·λ³·q⁴  (aberracion de fase)
    y Ac es el contraste de amplitud (~0.07 para cryo-ET).

    El factor de B simula la perdida de coherencia y la envolvente
    del detector: exp(-B·q²/4).

    Parametros
    ----------
    qx, qy : np.ndarray
        Frecuencias espaciales en Å⁻¹ (meshgrid).
    defocus_um : float
        Defoco en micras. Positivo = sobreenfoque, negativo = subenfoque.
        Convención: defoco positivo da contraste correcto para cabezas.
    cs_mm : float
        Aberracion esferica en mm.
    voltage_kv : float
        Voltaje acelerador en kV.
    amplitude_contrast : float
        Fraccion de contraste de amplitud (tipico 0.07-0.10).
    b_factor : float
        Factor B de envolvente (Å²). Valores tipicos: 100-500 Å².

    Retorna
    -------
    np.ndarray
        CTF, shape = qx.shape, valores en [-1, 1].
    """
    lam = wavelength_angstrom(voltage_kv)
    df = defocus_um * 1e4
    cs = cs_mm * 1e7

    q2 = qx**2 + qy**2
    q2[0, 0] = 1e-12

    chi = np.pi * df * lam * q2 - 0.5 * np.pi * cs * lam**3 * q2**2

    ac = amplitude_contrast
    ctf = -np.sqrt(1.0 - ac**2) * np.sin(chi) + ac * np.cos(chi)

    envelope = np.exp(-b_factor * q2 / 4.0)
    ctf = ctf * envelope
    ctf[0, 0] = 0.0
    return ctf


def apply_ctf_2d(
    image: np.ndarray,
    pixel_size_angstrom: float = 9.1,
    defocus_um: float = 2.0,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    amplitude_contrast: float = 0.07,
    b_factor: float = 200.0,
) -> np.ndarray:
    """
    Aplica la CTF a una imagen 2D proyectada.

    Convolucion en el espacio de Fourier: I_ctf = F^-1[ F[I] · CTF ]

    Parametros
    ----------
    image : np.ndarray, shape (Nx, Ny)
        Imagen de densidad proyectada.
    pixel_size_angstrom : float
        Tamano de pixel en Å.

    Retorna
    -------
    np.ndarray
        Imagen con CTF aplicada, misma forma que la entrada.
    """
    Nx, Ny = image.shape
    fx = np.fft.fftfreq(Nx, d=pixel_size_angstrom)
    fy = np.fft.fftfreq(Ny, d=pixel_size_angstrom)
    Fx, Fy = np.meshgrid(fx, fy, indexing="ij")

    ctf = compute_ctf(
        Fx, Fy,
        defocus_um=defocus_um,
        cs_mm=cs_mm,
        voltage_kv=voltage_kv,
        amplitude_contrast=amplitude_contrast,
        b_factor=b_factor,
    )

    I_ft = np.fft.fft2(image)
    I_ctf = np.real(np.fft.ifft2(I_ft * ctf))
    return I_ctf


def apply_missing_wedge(
    volume: np.ndarray,
    tilt_max_deg: float = 60.0,
    tilt_axis: int = 1,
    tilt_min_deg: float = None,
    randomize: bool = False,
    rng=None,
) -> np.ndarray:
    """
    Aplica el artefacto de cuña perdida (missing wedge) a un volumen 3D.

    Soporta rangos de inclinacion asimetricos y ejes variables para
    mejorar la generalizacion de redes neuronales entrenadas con el dataset.

    Parametros
    ----------
    tilt_max_deg : float
        Semi-angulo positivo maximo (grados).
    tilt_min_deg : float, opcional
        Semi-angulo negativo (si None, usa -tilt_max_deg → simetrico).
    randomize : bool
        Si True, randomiza el eje de inclinacion y el rango.
        Util para data augmentation.
    rng : np.random.Generator, opcional
        Generador para reproducibilidad al usar randomize=True.

    Referencia:
      MemBrain-seg augmentation strategy — github.com/teamtomo/membrain-seg
    """
    if randomize:
        if rng is None:
            rng = np.random.default_rng()
        tilt_axis    = int(rng.integers(0, 3))
        tilt_max_deg = float(rng.uniform(55.0, 70.0))
        tilt_min_deg = float(rng.uniform(-70.0, -55.0))
    else:
        if tilt_min_deg is None:
            tilt_min_deg = -tilt_max_deg

    Nx, Ny, Nz = volume.shape
    V_ft = np.fft.fftn(volume)

    fx = np.fft.fftfreq(Nx)
    fy = np.fft.fftfreq(Ny)
    fz = np.fft.fftfreq(Nz)
    Fx, Fy, Fz = np.meshgrid(fx, fy, fz, indexing="ij")

    if tilt_axis == 1:
        q_perp = np.sqrt(Fx**2 + Fz**2)
        angle  = np.degrees(np.arctan2(Fy, q_perp + 1e-12))
    elif tilt_axis == 0:
        q_perp = np.sqrt(Fy**2 + Fz**2)
        angle  = np.degrees(np.arctan2(Fx, q_perp + 1e-12))
    else:
        q_perp = np.sqrt(Fx**2 + Fy**2)
        angle  = np.degrees(np.arctan2(Fz, q_perp + 1e-12))

    wedge_mask = (angle >= tilt_min_deg) & (angle <= tilt_max_deg)
    volume_mw  = np.real(np.fft.ifftn(V_ft * wedge_mask))
    return volume_mw


def add_noise(
    image: np.ndarray,
    snr: float = 0.1,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Aniade ruido realista de cryo-ET.

    Modelo de dos componentes:
      1. Ruido de Poisson: proporcional a la senal (shot noise del haz)
      2. Ruido gaussiano de lectura del detector

    La proporcion relativa de cada componente depende del SNR total.

    Parametros
    ----------
    image : np.ndarray
        Imagen o volumen de entrada.
    snr : float
        Relacion senal-ruido objetivo. 0.1 = muy ruidoso (tipico),
        0.5 = moderado, 1.0 = casi limpio.
    rng : np.random.Generator, opcional
        Generador de numeros aleatorios (para reproducibilidad).

    Retorna
    -------
    np.ndarray
        Imagen con ruido anadido.
    """
    if rng is None:
        rng = np.random.default_rng()

    img_norm = image - image.min()
    if img_norm.max() > 0:
        img_norm = img_norm / img_norm.max()

    signal_std = img_norm.std()
    if signal_std == 0:
        signal_std = 1.0

    noise_std = signal_std / snr

    noise_poisson = rng.poisson(np.abs(img_norm) * 100) / 100.0 - np.abs(img_norm)
    noise_gaussian = rng.normal(0, noise_std * 0.5, image.shape)

    noisy = image + noise_poisson * noise_std + noise_gaussian
    return noisy.astype(np.float32)


def simulate_projection(
    membrane: "BicapaCryoET",
    defocus_um: float = 2.0,
    snr: float = 0.1,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    b_factor: float = 200.0,
    use_electron_density: bool = True,
    bins_xy: int = 90,
) -> np.ndarray:
    """
    Simula una imagen de cryo-ET realista de la bicapa.

    Pipeline:
      1. Calcula la densidad electronica proyectada en XY
      2. Aplica la CTF en el espacio de Fourier
      3. Anade ruido Poisson + Gaussiano

    Parametros
    ----------
    membrane : BicapaCryoET
        Bicapa ya construida.
    defocus_um : float
        Defoco en micras.
    snr : float
        Relacion senal-ruido (0.05-0.2 es realista para cryo-ET).
    use_electron_density : bool
        Si True usa densidad electronica real (recomendado).
        Si False usa densidad de masa (rapido, menos realista).

    Retorna
    -------
    np.ndarray, shape (bins_xy, bins_xy)
        Imagen simulada de cryo-ET.
    """
    pixel_A = membrane.Lx / bins_xy

    if use_electron_density:
        from electron_density import electron_density_projection
        proj = electron_density_projection(membrane, bins_xy=bins_xy)
    else:
        proj = (
            analysis.density_map(membrane, membrane.outer_leaflet, bins=bins_xy)
            + analysis.density_map(membrane, membrane.inner_leaflet, bins=bins_xy)
        )

    proj_ctf = apply_ctf_2d(
        proj,
        pixel_size_angstrom=pixel_A,
        defocus_um=defocus_um,
        cs_mm=cs_mm,
        voltage_kv=voltage_kv,
        amplitude_contrast=0.07,
        b_factor=b_factor,
    )

    rng = np.random.default_rng(membrane.seed)
    proj_noisy = add_noise(proj_ctf, snr=snr, rng=rng)
    return proj_noisy.astype(np.float32)


def simulate_volume(
    membrane: "BicapaCryoET",
    defocus_um: float = 2.0,
    snr: float = 0.1,
    tilt_max_deg: float = 60.0,
    cs_mm: float = 2.7,
    voltage_kv: float = 300.0,
    b_factor: float = 200.0,
    bins_xy: int = 55,
    bins_z: int = 40,
) -> np.ndarray:
    """
    Simula un tomograma 3D completo con CTF, missing wedge y ruido.

    Pipeline:
      1. Densidad electronica 3D
      2. Missing wedge en espacio de Fourier
      3. CTF aplicada slice a slice (aprox. de campo lejano)
      4. Ruido

    Retorna
    -------
    np.ndarray, shape (bins_xy, bins_xy, bins_z)
        Tomograma simulado.
    """
    from electron_density import electron_density_volume

    vol, edges = electron_density_volume(membrane, bins_xy=bins_xy, bins_z=bins_z)
    pixel_A = membrane.Lx / bins_xy

    vol_mw = apply_missing_wedge(vol, tilt_max_deg=tilt_max_deg, tilt_axis=1)

    vol_ctf = np.zeros_like(vol_mw)
    for iz in range(bins_z):
        slc = vol_mw[:, :, iz]
        vol_ctf[:, :, iz] = apply_ctf_2d(
            slc,
            pixel_size_angstrom=pixel_A,
            defocus_um=defocus_um,
            cs_mm=cs_mm,
            voltage_kv=voltage_kv,
            b_factor=b_factor,
        )

    rng = np.random.default_rng(membrane.seed)
    vol_noisy = add_noise(vol_ctf, snr=snr, rng=rng)
    return vol_noisy.astype(np.float32)


def plot_ctf_curves(
    defocus_values_um: list = None,
    pixel_size_angstrom: float = 9.1,
    bins: int = 200,
    voltage_kv: float = 300.0,
    cs_mm: float = 2.7,
    save_path: str = None,
):
    """
    Figura de las curvas CTF para distintos valores de defoco.
    Util para justificar la eleccion del defoco en el TFM.
    """
    import matplotlib.pyplot as plt

    if defocus_values_um is None:
        defocus_values_um = [0.5, 1.0, 2.0, 3.0, 4.0]

    q_max = 0.5 / pixel_size_angstrom
    q = np.linspace(0, q_max, bins)
    Fx = q
    Fy = np.zeros_like(q)

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    for i, df in enumerate(defocus_values_um):
        ctf_1d = compute_ctf(
            Fx.reshape(bins, 1),
            Fy.reshape(bins, 1),
            defocus_um=df,
            cs_mm=cs_mm,
            voltage_kv=voltage_kv,
            b_factor=200.0,
        ).flatten()
        ax.plot(q * 10, ctf_1d, color=colors[i % len(colors)],
                lw=1.8, label="%.1f μm" % df)

    ax.axhline(0, color="#333333", lw=0.8, ls="--", alpha=0.5)
    ax.set_xlabel("Frecuencia espacial (nm⁻¹)", fontsize=11)
    ax.set_ylabel("CTF", fontsize=11)
    ax.set_title(
        "Curvas CTF — Voltaje %.0f kV | Cs=%.1f mm | B=200 Å²\n"
        "La primera inversion de contraste marca la resolucion efectiva"
        % (voltage_kv, cs_mm),
        fontsize=10, fontweight="bold",
    )
    ax.legend(title="Defoco", fontsize=9)
    ax.set_xlim(0, q_max * 10)
    ax.set_ylim(-1.1, 1.1)
    ax.grid(True, alpha=0.3)

    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("  -> %s" % save_path)
    else:
        plt.tight_layout()
        plt.show()
    return fig
