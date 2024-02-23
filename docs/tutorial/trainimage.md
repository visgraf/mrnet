# Example: Fitting an Image

Importing dependencies:

```python
import os
from pathlib import Path
import torch

from mrnet.training.trainer import MRTrainer
from mrnet.datasets.signals import ImageSignal
from mrnet.networks.mrnet import MRFactory
from mrnet.datasets.pyramids import create_MR_structure
from mrnet.training.listener import TrainingListener

from mrnet.training.utils import load_hyperparameters, get_optim_handler
```

We set a random seed for reproducibility, and read the hyperparameters from a YAML file to the `hyper` dictionary.

```python
torch.manual_seed(777)
#-- hyperparameters in configs --#
hyper = load_hyperparameters('docs/configs/image.yml')
project_name = hyper.get('project_name', 'framework-tests')
```

We need an object from a subclass of `Signal` to hold the training data. In this example, `base_signal` is initialized as an `ImageSignal` from a file located at `data_path`. We recommend setting all arguments from the hyperparameters dictionary to make it easier to track your experiments.

```python
base_signal = ImageSignal.init_fromfile(
                    hyper['data_path'],
                    domain=hyper['domain'],
                    channels=hyper['channels'],
                    sampling_scheme=hyper['sampling_scheme'],
                    width=hyper['width'], height=hyper['height'],
                    batch_size=hyper['batch_size'],
                    color_space=hyper['color_space'])
```

The training set for MR-Net is specified as a MR-Structure, which is basically any sequential data-structure that holds a multiresolution representation of the signal. In this case, a Gaussian (`filter`) Pyramid (`decimation`) with `max_stages` levels is built as the `train_dataset`. The `pmode` parameter specifies how to handle the borders on filtering. The `test_dataset` is also a MR-Structure, but in this case, we don't subsample the data after filtering, building a "Gaussian Tower". This way, the model will be trained in the subsampled dataset, but compared against the full resolution grid at all levels.

```python
train_dataset = create_MR_structure(base_signal,
                                    hyper['max_stages'],
                                    hyper['filter'],
                                    hyper['decimation'],
                                    hyper['pmode'])
test_dataset = create_MR_structure(base_signal,
                                    hyper['max_stages'],
                                    hyper['filter'],
                                    False,
                                    hyper['pmode'])
```

```python
if hyper['width'] == 0:
    hyper['width'] = base_signal.shape[-1]
if hyper['height'] == 0:
    hyper['height'] = base_signal.shape[-1]

# you can substitute this line by your custom handler class
optim_handler = get_optim_handler(hyper.get('optim_handler', 'regular'))
```


Loading the model.

```python
mrmodel = MRFactory.from_dict(hyper)
print("Model: ", type(mrmodel))
```

Training the model.

```python
name = os.path.basename(hyper['data_path'])
logger = TrainingListener(project_name,
                            f"{hyper['model']}{hyper['filter'][0].upper()}{name[0:7]}{hyper['color_space'][0]}",
                            hyper,
                            Path(hyper.get("log_path", "runs")))
mrtrainer = MRTrainer.init_from_dict(mrmodel,
                                    train_dataset,
                                    test_dataset,
                                    logger,
                                    hyper,
                                    optim_handler=optim_handler)
mrtrainer.train(hyper['device'])
```