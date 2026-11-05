"""
Script wrapper/alias untuk ingest_datahistorical_binance.py
"""
import runpy
import os

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    target_script = os.path.join(current_dir, "ingest_datahistorical_binance.py")
    runpy.run_path(target_script, run_name="__main__")

