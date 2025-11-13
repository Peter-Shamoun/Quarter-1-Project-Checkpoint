# WandB Hyperparameter Sweep Instructions

This guide explains how to run hyperparameter sweeps for the BabyLM RoBERTa training pipeline using WandB Sweeps.

## Overview

The sweep is configured to optimize:
- **Learning rate** (1e-4 to 2e-3)
- **Batch size** (16, 32, 64)
- **Warmup steps** (50k, 100k, 150k)

Using **Bayesian optimization** to minimize **validation perplexity** (`eval_perplexity_mean`).

## Prerequisites

### 1. Environment Setup

Ensure you have all dependencies installed:

```bash
pip install -r requirements.txt
```

### 2. Required Environment Variables

Set the following environment variables:

```bash
export WANDB_API_KEY=your_wandb_api_key
export HF_READ_TOKEN=your_hf_read_token
export HF_WRITE_TOKEN=your_hf_write_token
```

You can get your WandB API key from: https://wandb.ai/settings

### 3. Data Preparation

Ensure your training data is available at:
- `local_data/train_10M/*.train`
- `local_data/dev/*.dev`

## Quick Start

### Step 1: Initialize the Sweep

Run the launch script to create a new sweep:

```bash
python launch_sweep.py --project babylm-sweep --count 10
```

This will:
- Create a sweep on WandB
- Print the sweep ID and agent commands
- Save sweep information to `sweep_info.txt`

**Arguments:**
- `--config`: Path to sweep config (default: `sweep_config.yaml`)
- `--project`: WandB project name (default: `babylm-sweep`)
- `--entity`: WandB entity/team name (default: `baby-lm`)
- `--count`: Maximum number of runs (default: 10)

### Step 2: Run Sweep Agents

After initialization, you'll receive a command like:

```bash
wandb agent baby-lm/babylm-sweep/SWEEP_ID
```

Run this command to start an agent. Each agent will:
1. Fetch hyperparameters from the sweep controller
2. Run a training experiment with those parameters
3. Report results back to WandB
4. Repeat until the sweep is complete

## Running Agents

### Option 1: Single GPU (Local)

The simplest way to run experiments:

```bash
wandb agent baby-lm/babylm-sweep/SWEEP_ID
```

### Option 2: Multiple Agents (Parallel)

To speed up the sweep, run multiple agents in parallel. Open multiple terminals and run the agent command in each:

**Terminal 1:**
```bash
wandb agent baby-lm/babylm-sweep/SWEEP_ID
```

**Terminal 2:**
```bash
wandb agent baby-lm/babylm-sweep/SWEEP_ID
```

**Terminal 3:**
```bash
wandb agent baby-lm/babylm-sweep/SWEEP_ID
```

Each agent will work on a different hyperparameter configuration.

### Option 3: Cluster/SLURM

For running on a computing cluster, adapt your SLURM script:

```bash
#!/bin/bash
#SBATCH --job-name=wandb-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --time=24:00:00

# Set environment variables
export WANDB_API_KEY=your_key
export HF_READ_TOKEN=your_token
export HF_WRITE_TOKEN=your_token

# Set sweep information
export WANDB_SWEEP_ID=your_sweep_id
export WANDB_PROJECT=babylm-sweep
export WANDB_ENTITY=baby-lm

# Run sweep agent
wandb agent ${WANDB_ENTITY}/${WANDB_PROJECT}/${WANDB_SWEEP_ID}
```

Submit multiple jobs to run agents in parallel:

```bash
sbatch sweep_job.sh
sbatch sweep_job.sh
sbatch sweep_job.sh
```

## Monitoring the Sweep

### WandB Dashboard

View real-time results at:
```
https://wandb.ai/baby-lm/babylm-sweep/sweeps/SWEEP_ID
```

The dashboard shows:
- **Parallel Coordinates Plot**: Visualize parameter relationships
- **Parameter Importance**: Which parameters affect performance most
- **Best Runs**: Top performing configurations
- **Progress**: Number of completed runs

### Key Metrics to Monitor

1. **Primary Metric**: `eval_perplexity_mean` (lower is better)
2. **Secondary Metrics**:
   - `eval_blimp_avg`: Average BLIMP accuracy
   - Various BLIMP subtasks
   - `train/loss`: Training loss progression
   - `train/learning_rate`: Learning rate schedule

## Configuration Details

### Sweep Configuration (`sweep_config.yaml`)

The sweep uses:
- **Method**: Bayesian optimization (efficient for limited budget)
- **Early Termination**: Hyperband (stops poor runs early)
- **Training Steps**: Reduced to 150k (from 400k) for faster iteration
- **Evaluation**: BLiMP and GLUE enabled

### Hyperparameter Ranges

| Parameter | Type | Range/Values | Note |
|-----------|------|--------------|------|
| `trainer.lr` | log-uniform | 1e-4 to 2e-3 | Most critical parameter |
| `trainer.batch_size` | categorical | [16, 32, 64] | Affects memory & dynamics |
| `trainer.num_warmup_steps` | categorical | [50k, 100k, 150k] | Affects training stability |

### Fixed Parameters

- `trainer.max_training_steps`: 150,000 (reduced for sweep)
- `experiment.seed`: 42
- `trainer.eval_blimp`: True
- `trainer.eval_glue`: True
- All other settings from `conf/config.yaml`

## Retrieving Best Hyperparameters

### Via WandB Dashboard

1. Go to your sweep page
2. Sort runs by `eval_perplexity_mean`
3. Click on the best run
4. View "Config" tab for hyperparameters

### Via Python API

```python
import wandb

api = wandb.Api()
sweep = api.sweep("baby-lm/babylm-sweep/SWEEP_ID")

# Get best run
best_run = sweep.best_run()
print(f"Best run: {best_run.name}")
print(f"Best perplexity: {best_run.summary['eval_perplexity_mean']}")
print(f"Best config: {best_run.config}")
```

### Via CLI

```bash
wandb sweep baby-lm/babylm-sweep/SWEEP_ID --stop
```

## Training with Best Hyperparameters

Once you've identified the best hyperparameters:

### Option 1: Update Config File

Edit `conf/config.yaml`:

```yaml
trainer: 
  batch_size: 32  # Best value from sweep
  lr: 5e-4        # Best value from sweep
  num_warmup_steps: 100_000  # Best value from sweep
  max_training_steps: 400_000  # Use full training for final model
```

Then run normal training:

```bash
python train.py experiment.name=final-model experiment.group=babylm-final
```

### Option 2: Override via Command Line

```bash
python train.py \
  experiment.name=final-model \
  experiment.group=babylm-final \
  trainer.lr=5e-4 \
  trainer.batch_size=32 \
  trainer.num_warmup_steps=100000 \
  trainer.max_training_steps=400000
```

## Troubleshooting

### Issue: "WANDB_API_KEY not set"

**Solution:** Set your WandB API key:
```bash
export WANDB_API_KEY=your_key
```

### Issue: "HF_READ_TOKEN and HF_WRITE_TOKEN need to be set"

**Solution:** Set your HuggingFace tokens:
```bash
export HF_READ_TOKEN=your_token
export HF_WRITE_TOKEN=your_token
```

### Issue: Agent crashes with CUDA out of memory

**Solution:** 
- The sweep tries batch_size=64 which may be too large
- Edit `sweep_config.yaml` and remove 64 from batch_size values
- Reinitialize the sweep

### Issue: Training is too slow

**Solution:**
- Further reduce `trainer.max_training_steps` in `sweep_config.yaml`
- Run more agents in parallel
- Use early termination (already enabled)

### Issue: Sweep completed but want more runs

**Solution:** You can't add runs to a completed sweep. Instead:
1. Create a new sweep with a narrower parameter range around the best values
2. Or manually run experiments with specific configurations

## Advanced Usage

### Modifying Sweep Configuration

To change hyperparameters or ranges:

1. Edit `sweep_config.yaml`
2. Create a new sweep with `python launch_sweep.py`
3. Run agents with the new sweep ID

**Note:** You cannot modify an existing sweep. You must create a new one.

### Custom Sweep Strategies

The current sweep uses Bayesian optimization. Other options:

- **Grid Search**: Exhaustive search (edit `method: grid` in config)
- **Random Search**: Random sampling (edit `method: random` in config)

### Extending to More Hyperparameters

To sweep additional parameters, add them to `sweep_config.yaml`:

```yaml
parameters:
  # Existing parameters...
  
  # Add new ones:
  data_preprocessing.max_input_length:
    values: [128, 256, 512]
  
  model.model_kwargs.hidden_size:
    values: [256, 512, 768]
```

**Caution:** More parameters = more runs needed for good coverage.

## Best Practices

1. **Start Small**: Run 5-10 experiments first to validate the setup
2. **Monitor Early**: Check first few runs to ensure they're working
3. **Use Early Termination**: Enabled by default, saves compute
4. **Parallel Agents**: Run 2-3 agents simultaneously
5. **Iterate**: Use results to narrow search space for follow-up sweeps
6. **Full Training**: After finding best hyperparameters, retrain with full steps

## Resources

- [WandB Sweeps Documentation](https://docs.wandb.ai/guides/sweeps)
- [Hydra Documentation](https://hydra.cc/)
- [Transformers Trainer](https://huggingface.co/docs/transformers/main_classes/trainer)

## Support

For issues with:
- **WandB Sweeps**: Check [WandB docs](https://docs.wandb.ai/guides/sweeps) or [community forum](https://community.wandb.ai/)
- **Training Code**: Review `train.py` and `src/trainer.py`
- **Configuration**: Check Hydra configs in `conf/` directory

