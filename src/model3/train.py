"""Original U2_ECA trainer, fresh seed42 initialization, fixed 150 epochs."""
import json
from pathlib import Path
import sys
import torch

folder = Path(__file__).resolve().parent
package = folder.parents[1]
sys.path.insert(0, str(folder / 'code'))
from komap_phase.training import train


if __name__ == '__main__':
    settings = json.loads((package / 'resources/settings.json').read_text())
    config = json.loads((folder.parents[1] / 'resources' / folder.name / 'config.json').read_text())
    torch.set_num_threads(4)
    output = package / 'run/retraining/model3'
    assert not output.exists(), 'Choose a new retraining output; released results are protected.'
    train(config, (package / settings['data_root']).resolve(), output, torch.device(settings['device']), resume=False)
