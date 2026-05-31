"""
run_results.py — Punto de entrada para figuras de resultados.

Delega la ejecución en visualizacion/results.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from visualizacion.results import main

if __name__ == "__main__":
    main()
