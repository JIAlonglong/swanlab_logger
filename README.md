# swanlab_logger

A unified logging adapter that supports both TensorBoard and SwanLab logging methods simultaneously.

## Features

- Support for both TensorBoard and SwanLab logging
- Unified API interface for simplified logging process
- Support for scalar recording, dictionary recording, and hyperparameter recording
- Automatic error handling to ensure program stability
- Compatibility with TensorBoard's `add_scalar` method calls

## Installation

### Basic Installation

```bash
pip install -e /path/to/swanlab_logger
```

### Installing SwanLab Support

```bash
pip install -e "/path/to/swanlab_logger[swanlab]"
# Or install separately
pip install swanlab
```

## Usage Examples

### Basic Usage

```python
from swanlab_logger import create_logger

# Create logger
global_logger = create_logger(
    log_dir="./logs",
    experiment_name="My_Experiment",
    config={"learning_rate": 0.001, "batch_size": 32},
    use_tensorboard=True,
    use_swanlab=True
)

# Log scalar values
global_logger.log_scalar("loss", 0.5, step=1)
global_logger.log_scalar("accuracy", 0.85, step=1)

# Or use TensorBoard-compatible method
global_logger.add_scalar("reward", 10.5, step=10)

# Log multiple scalar values
global_logger.log_dict({
    "train_loss": 0.4,
    "val_loss": 0.6,
    "lr": 0.0005
}, step=2)

# Log hyperparameters
global_logger.log_hparams(
    hparams={"optimizer": "Adam", "epochs": 100},
    metrics={"best_accuracy": 0.92}
)

# Close logger (automatically called at program end, but better to call manually to ensure data saving)
global_logger.close()
```

### Using in Training Loop

```python
from swanlab_logger import create_logger

epoch = 100

global_logger = create_logger(log_dir="./logs", experiment_name="Training_Run")

for i in range(epoch):
    # Training code...
    loss = compute_loss()
    accuracy = compute_accuracy()
    
    # Log metrics for each epoch
global_logger.log_scalar("train/loss", loss, step=i)
global_logger.log_scalar("train/accuracy", accuracy, step=i)

# End of training
global_logger.close()
```

## API Reference

### create_logger

Factory function to create logger

```python
def create_logger(
    log_dir: Optional[str] = None, 
    experiment_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    use_tensorboard: bool = True,
    use_swanlab: bool = True
) -> SwanLabLogger:
```

**Parameters:**
- `log_dir`: Log saving directory
- `experiment_name`: Experiment name
- `config`: Experiment configuration parameters
- `use_tensorboard`: Whether to use TensorBoard
- `use_swanlab`: Whether to use SwanLab

**Returns:**
- `SwanLabLogger` instance

### SwanLabLogger Class

#### log_scalar

Record single scalar value

```python
def log_scalar(self, tag: str, value: Union[float, int], step: int = None)
```

**Parameters:**
- `tag`: Metric name
- `value`: Metric value
- `step`: Step number

#### add_scalar

TensorBoard-compatible add_scalar method, internally calls log_scalar

```python
def add_scalar(self, tag: str, value: Union[float, int], step: int = None)
```

**Parameters:**
- `tag`: Metric name
- `value`: Metric value
- `step`: Step number

#### log_dict

Record multiple scalar values

```python
def log_dict(self, metrics: Dict[str, Union[float, int]], step: int = None)
```

**Parameters:**
- `metrics`: Metric dictionary {metric name: metric value}
- `step`: Step number

#### log_hparams

Record hyperparameters

```python
def log_hparams(self, hparams: Dict[str, Any], metrics: Dict[str, float] = None)
```

**Parameters:**
- `hparams`: Hyperparameter dictionary
- `metrics`: Corresponding metrics

#### close

Close logger

```python
def close(self)
```

## Notes

1. If SwanLab is not installed, it will automatically downgrade to using only TensorBoard
2. All logging operations have error handling to ensure single logging failures do not affect the entire program
3. It is recommended to manually call the `close()` method at the end of the program to ensure all log data is properly saved
4. When using in a multi-process environment, please note the synchronization of logging

## License

BSD 3-Clause License

## Contributions
<a href="https://github.com/JIAlonglong/swanlab_logger/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=JIAlonglong/swanlab_logger" alt="contrib.rocks image" />
</a>

Issues and improvements are welcome.