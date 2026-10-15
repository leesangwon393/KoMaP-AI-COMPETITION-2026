"""Run the three frozen models sequentially, then ensemble their probabilities."""
from pathlib import Path
import subprocess
import sys


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    for name in ('model1','model2','model3'):
        subprocess.run([sys.executable,str(root/name/'infer.py')],check=True)
    subprocess.run([sys.executable,str(root/'ensemble.py')],check=True)
