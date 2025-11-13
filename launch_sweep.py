"""
Launch WandB Sweep for hyperparameter optimization.

This script initializes a WandB sweep and provides instructions for running agents.
"""

import argparse
import os
import sys

import wandb
import yaml


def load_sweep_config(config_path: str) -> dict:
    """Load sweep configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def initialize_sweep(config_path: str, project: str, entity: str = "baby-lm") -> str:
    """
    Initialize a WandB sweep.
    
    Args:
        config_path: Path to the sweep configuration YAML file
        project: WandB project name
        entity: WandB entity/team name
        
    Returns:
        sweep_id: The ID of the created sweep
    """
    sweep_config = load_sweep_config(config_path)
    
    print("=" * 80)
    print("Initializing WandB Sweep with the following configuration:")
    print("=" * 80)
    print(yaml.dump(sweep_config, default_flow_style=False))
    print("=" * 80)
    
    # Initialize the sweep
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project=project,
        entity=entity
    )
    
    return sweep_id


def print_agent_instructions(sweep_id: str, project: str, entity: str, count: int):
    """Print instructions for running sweep agents."""
    print("\n" + "=" * 80)
    print("✓ Sweep initialized successfully!")
    print("=" * 80)
    print(f"\nSweep ID: {sweep_id}")
    print(f"Project: {entity}/{project}")
    print(f"\nView sweep dashboard at:")
    print(f"https://wandb.ai/{entity}/{project}/sweeps/{sweep_id}")
    print("\n" + "=" * 80)
    print("To run sweep agents:")
    print("=" * 80)
    
    # Single GPU agent
    print("\n1. Single GPU Agent:")
    print(f"   wandb agent {entity}/{project}/{sweep_id}")
    
    # Multi-GPU agent
    print("\n2. Multi-GPU Agent (using torchrun):")
    print(f"   # In separate terminals, run {count} agents:")
    for i in range(min(count, 3)):  # Show max 3 examples
        print(f"   wandb agent {entity}/{project}/{sweep_id}")
    if count > 3:
        print(f"   # ... (repeat for {count - 3} more terminals)")
    
    # Cluster/SLURM example
    print("\n3. Cluster/SLURM (see scripts/launch_slurm.wilkes3 for reference):")
    print(f"   export WANDB_SWEEP_ID={sweep_id}")
    print(f"   export WANDB_PROJECT={project}")
    print(f"   export WANDB_ENTITY={entity}")
    print("   sbatch scripts/launch_slurm.wilkes3")
    
    print("\n" + "=" * 80)
    print("Notes:")
    print("=" * 80)
    print("- Each agent will run one trial at a time")
    print("- Run multiple agents in parallel to speed up the sweep")
    print(f"- The sweep will stop after {count} runs (as configured)")
    print("- Ensure HF_READ_TOKEN and HF_WRITE_TOKEN are set as environment variables")
    print("- Monitor progress in the WandB dashboard")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Initialize a WandB sweep for hyperparameter optimization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="sweep_config.yaml",
        help="Path to sweep configuration YAML file (default: sweep_config.yaml)"
    )
    parser.add_argument(
        "--project",
        type=str,
        default="babylm-sweep",
        help="WandB project name (default: babylm-sweep)"
    )
    parser.add_argument(
        "--entity",
        type=str,
        default="baby-lm",
        help="WandB entity/team name (default: baby-lm)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=10,
        help="Maximum number of runs in the sweep (default: 10)"
    )
    
    args = parser.parse_args()
    
    # Check for required environment variables
    if "WANDB_API_KEY" not in os.environ:
        print("ERROR: WANDB_API_KEY environment variable not set!")
        print("Please set it with: export WANDB_API_KEY=your_api_key")
        sys.exit(1)
    
    if "HF_READ_TOKEN" not in os.environ or "HF_WRITE_TOKEN" not in os.environ:
        print("WARNING: HF_READ_TOKEN and HF_WRITE_TOKEN should be set as environment variables")
        print("These are required by train.py")
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"ERROR: Configuration file '{args.config}' not found!")
        sys.exit(1)
    
    try:
        # Initialize the sweep
        sweep_id = initialize_sweep(args.config, args.project, args.entity)
        
        # Print instructions
        print_agent_instructions(sweep_id, args.project, args.entity, args.count)
        
        # Save sweep ID to file for easy reference
        sweep_info_file = "sweep_info.txt"
        with open(sweep_info_file, 'w') as f:
            f.write(f"Sweep ID: {sweep_id}\n")
            f.write(f"Project: {args.entity}/{args.project}\n")
            f.write(f"Dashboard: https://wandb.ai/{args.entity}/{args.project}/sweeps/{sweep_id}\n")
            f.write(f"\nAgent command:\n")
            f.write(f"wandb agent {args.entity}/{args.project}/{sweep_id}\n")
        
        print(f"Sweep information saved to {sweep_info_file}")
        
    except Exception as e:
        print(f"\nERROR: Failed to initialize sweep: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

