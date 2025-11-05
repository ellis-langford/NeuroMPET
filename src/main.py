# Entry point
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from src.pipeline import NeuroMPET

if __name__ == "__main__":
    plugin = NeuroMPET()
    plugin.run_pipeline()