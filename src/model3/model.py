import json
from pathlib import Path
import sys

folder = Path(__file__).resolve().parent
sys.path.insert(0, str(folder / 'code'))
from komap_phase.models import SegmentationModel


def build_model():
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    return SegmentationModel(config, pretrained=False)
