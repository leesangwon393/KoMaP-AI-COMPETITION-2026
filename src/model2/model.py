import json
from pathlib import Path
import sys

folder = Path(__file__).resolve().parent
sys.path.insert(0, str(folder / 'code'))
sys.path.insert(0, str(folder.parents[1] / 'src'))
from model_factory import make_model


def build_model():
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    return make_model(branch={'model1': 'R052', 'model2': 'R073'}[folder.name], model_options=config['model_options'])
