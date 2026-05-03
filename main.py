"""
main.py 
Generador de bicapas lipidicas para cryo-ET

Modelo 3D completo para estudio en ParaView.
La validacion produce figuras con composicion,
perfil ED, PIPs y dominios Lo/Ld.

Uso:
    python main.py --sims 1 2 3            # genera todo por defecto
    python main.py --sims 27 --validate    # con panel de validacion 3D
    python main.py --sims 27 --all         # todos los outputs

Flags:
    --sims N [N ...]   numeros de simulacion (default: 27 42)
    --size X Y         tamano en nm (default: 50 50)
    --paraview         VTP + PDB por simulacion en carpeta propia
    --model3d          volumen 3D de densidad electronica (MRC)
    --mrc              MRC para PolNet (doble gaussiana + labels)
    --positions        PDB completo + CSV + PolNet particle list
    --validate         panel de validacion 3D (composicion, ED, fases)
    --figures          figura de publicacion (8 paneles, 300 DPI)
    --stats            estadisticas del dataset (>1 simulacion)
    --dpi N            resolucion figuras (default: 200)
    --all              paraview + model3d + validate + mrc + positions
"""

import argparse
import os
import sys

sys.path.insert(0, "/home/alumno25/.local/lib/python3.6/site-packages")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from builder import BicapaCryoET
from export import export_training


def parse_args():
    p = argparse.ArgumentParser(
        description="Generador sintetico de bicapas lipidicas para cryo-ET",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--sims", type=int, nargs="+", default=[27, 42],
                   metavar="N", help="Numeros de simulacion (default: 27 42)")
    p.add_argument("--size", type=float, nargs=2, default=[50.0, 50.0],
                   metavar=("X", "Y"))
    p.add_argument("--paraview",  action="store_true",
                   help="VTP + PDB en CryoET/paraview/simulacion{N}/")
    p.add_argument("--model3d",   action="store_true",
                   help="Volumen 3D fisico (MRC con ED real, labels, figura)")
    p.add_argument("--mrc",       action="store_true",
                   help="MRC simplificado + doble gaussiana para PolNet")
    p.add_argument("--positions", action="store_true",
                   help="PDB completo + CSV + PolNet particle list")
    p.add_argument("--validate",  action="store_true",
                   help="Panel de validacion 3D: composicion, ED, fases, PIPs")
    p.add_argument("--figures",   action="store_true",
                   help="Figura de publicacion 8 paneles 300 DPI en CryoET/figuras/simulacion{N}/")
    p.add_argument("--stats",     action="store_true",
                   help="Estadisticas del dataset (requiere >1 simulacion)")
    p.add_argument("--dpi",       type=int, default=200)
    p.add_argument("--all",       action="store_true",
                   help="Activa: paraview + model3d + validate + mrc + positions")
    return p.parse_args()


def run_sim(seed, size_nm, args):
    """Ejecuta una simulacion completa sin figuras 2D."""
    b = BicapaCryoET(size_nm=tuple(size_nm), seed=seed)
    b.build()

    # Arrays de training (canales numpy + labels.json)
    export_training(b)

    if args.paraview or args.all:
        from export_paraview import export_all_paraview
        export_all_paraview(b)

    if args.model3d or args.all:
        from model_3d import export_physical_model_mrc, plot_physical_model
        export_physical_model_mrc(b)
        plot_physical_model(b)

    if args.mrc or args.all:
        from export_mrc import (export_mrc, generate_polnet_yaml,
                                 export_double_gaussian_mrc,
                                 export_label_mrc_with_closing)
        export_mrc(b)
        generate_polnet_yaml(b)
        export_double_gaussian_mrc(b)
        export_label_mrc_with_closing(b)

    if args.positions or args.all:
        from export_positions import export_all_positions
        export_all_positions(b)

    if args.figures or args.all:
        from model_3d import build_physical_volume
        from figures import plot_all_figures
        v, lb, st = build_physical_volume(b)
        plot_all_figures(b, v, lb, st, dpi=args.dpi)

    if args.validate or args.all:
        from validation import run_all_benchmarks, plot_validation_panel, save_benchmark_json
        results = run_all_benchmarks(b)
        plot_validation_panel(b, results)
        save_benchmark_json(results, b)


def main():
    args = parse_args()
    sims    = list(args.sims)
    size_nm = list(args.size)

    print("BicapaCryoET — %d simulacion(es) | %.0fx%.0f nm" % (
        len(sims), size_nm[0], size_nm[1]))

    for seed in sims:
        run_sim(seed, size_nm, args)

    if args.stats and len(sims) > 1:
        from dataset_stats import compute_dataset_stats, plot_dataset_summary
        stats = compute_dataset_stats(sims, size_nm=tuple(size_nm),
                                      run_validation=args.validate)
        plot_dataset_summary(stats)

    print("\nListo. Outputs en: CryoET/")


if __name__ == "__main__":
    main()
