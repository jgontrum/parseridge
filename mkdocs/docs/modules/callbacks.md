# Callbacks

The [`Trainer`](/modules/training/#dynamic-trainer) is built around a callback system that allows seamless logging, evaluation,
and data manipulation without any need to modify the trainer itself.

We have two distinct groups of callbacks, one for training and one for evaluation.

## Training Callbacks
```python
from parseridge.parser.training.callbacks.base_callback import Callback
```

During training, a [`Callback`](#training-callbacks) can be executed at the following stages:

|Method|Description|
|---------|-----------|
|`on_train_begin`|Called once when the training process is started.|
|`on_epoch_begin`|Called at the beginning of every epoch.|
|`on_batch_begin`|Called at each batch before the forward pass.|
|`on_loss_begin`|Called _after_ the forward pass, but _before_ the loss computation.|
|`on_backward_begin`|Called _after_ the loss computation, but _before_ the backpropagation.|
|`on_backward_end`|Called _after_ the back propagation, but _before_ the parameter update through the optimizer.|
|`on_step_end`|Called _after_ the optimizer, but _before_ all the gradients have been set to zero.|
|`on_batch_end`|Called at the end of a batch.|
|`on_epoch_end`|Called at the end of an epochh.|
|`on_train_end`|Called at the end of the training process.|

As the callback objects are stateful, they can remember information between their executions.
The order of execution is defined by the `_order` class variable.

### Example: Freeze Embeddings

Here is an example for a callback, that allows us to only freeze some embeddings. It is usefull,
if we want to keep the word embeddings static, but learn embeddings for meta tokens like OOV.

```python
@dataclass
class PartialFreezeEmbeddingsCallback(Callback):
    _order = 10

    freeze_indices: torch.Tensor
    embedding_layer: nn.Embedding

    def on_backward_end(self, **kwargs: Any) -> None:
        self.embedding_layer.weight.grad[self.freeze_indices] = 0.0

```

### Example: Save the Model to File

Another example for a callback that works with parameters passed to the methods:
Initially, we create the folder for the model to save in and at the end of the model,
we use `torch.save` to write the model to file.

```python
class SaveModelCallback(Callback):
    _order = 5

    def __init__(self, folder_path: Optional[str] = None):
        self.folder = folder_path
        if self.folder:
            os.makedirs(folder_path, exist_ok=True)

    def on_epoch_end(self, epoch: int, model: Module, **kwargs: Any) -> None:
        if self.folder:
            file_name = f"{self.folder}/epoch_{epoch}.torch"
            torch.save(model.state_dict(), file_name)
```

### List of Training Callbacks

#### Gradient Clipping
```python
from parseridge.parser.training.callbacks import GradientClippingCallback
```

Clips the gradients at a given threshold.

#### Learning Rate Finder
```python
from parseridge.parser.training.callbacks import LearningRateFinderCallback
```

Still experimental callback to dynamically adapt the learning rate to find the optimal values.

#### Learning Rate Scheduler
```python
from parseridge.parser.training.callbacks import LRSchedulerCallback
```

Given a `LRScheduler`, execute it at a given time during training.

#### Training Mode
```python
from parseridge.parser.training.callbacks import ModelTrainingCallback
```

A useful helper that ensures that the model is set to training mode during training
and to evaluation mode during evaluation.

#### Partial Freeze Embeddings
```python
from parseridge.parser.training.callbacks import PartialFreezeEmbeddingsCallback
```

A callback, that allows us to only freeze some embeddings. It is usefull,
if we want to keep the word embeddings static, but learn embeddings for meta tokens like OOV.

#### Progress Bar
```python
from parseridge.parser.training.callbacks import ProgressBarCallback
```
If the training is performed in a Notebook or in commandline, this callback
can display a progress bar with ETA time.

#### Save Model
```python
from parseridge.parser.training.callbacks import SaveModelCallback
```
Saves the model directly after training.

#### Simple Logger
```python
from parseridge.parser.training.callbacks import TrainSimpleLoggerCallback
```
Logs the progress of the training using the standard logger. Useful, if the training is run
on a cluster.


## Evaluation Callbacks
```python
from parseridge.parser.evaluation.callbacks.base_eval_callback import EvalCallback
```

During evaluation, a [`Callback`](#evaluation-callbacks) can be executed at the following stages:

|Method|Description|
|---------|-----------|
|`on_initialization`|Called when the *training* process is initially started.|
|`on_eval_begin`|Called when the evaluation process is started.|
|`on_epoch_begin`|Called at the beginning of every evaliuation epoch.|
|`on_batch_begin`|Called at each batch before the forward pass.|
|`on_batch_end`|Called at the end of a batch.|
|`on_epoch_end`|Called at the end of an epoch.|
|`on_eval_end`|Called at the end of the evaluation.|
|`on_shutdown`|Called at the end of the *training* process.|


### List of Evaluation Callbacks

#### Attention Reporter
```python
from parseridge.parser.evaluation.callbacks import EvalAttentionReporter
```
Saves the attention weights during evaluaton.

#### CSV Reporter
```python
from parseridge.parser.evaluation.callbacks import EvalCSVReporter
```
Logs information about the performance into a CSV file.

#### Google Sheets Reporter
```python
from parseridge.parser.evaluation.callbacks import EvalGoogleSheetsReporter
```
Logs information about the performance into a Google Sheet.

#### Progress Bar
```python
from parseridge.parser.evaluation.callbacks import EvalProgressBarCallback
```
If the training is performed in a Notebook or in commandline, this callback
can display a progress bar with ETA time.

#### Save Sentences
```python
from parseridge.parser.evaluation.callbacks import EvalSaveParsedSentencesCallback
```
Saves the parsed sentences in CONLL format into a file for later evaluation.

#### Simple Logger
```python
from parseridge.parser.evaluation.callbacks import EvalSimpleLogger
```
Logs the progress of the evaluation using the standard logger. Useful, if the training is run
on a cluster.

#### YAML Logger
```python
from parseridge.parser.evaluation.callbacks import EvalYAMLReporter
```
Saves information about the configuration of the experiment and the scores into a YAML file.
