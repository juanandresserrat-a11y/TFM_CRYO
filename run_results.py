"""
run_results.py — Punto de entrada para figuras de resultados.

Delega la ejecución en visualizacion/results.py conservando
la interfaz documentada en el README:

    python run_results.py --sims 1 2 3 4 5
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualizacion.results import main

if __name__ == "__main__":
    main()
