"""Run model1 in an isolated process with the shared inference pipeline."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference import run_inference
from model import build_model


if __name__ == '__main__':
    run_inference('model1', build_model)
