import argparse
import os
from typing import Any, Dict, List, Optional

import wandb
import yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate and run hyperparameter sweeps for MPT training.',
    )

    parser.add_argument('--project', type=str, default='llm-foundry-scripts_train')
    parser.add_argument('--entity', type=str, default='local-research-group')
    parser.add_argument('--model_yaml', type=str, required=True,
                      help='Path to the base model YAML configuration')
    parser.add_argument('--sweep_yaml', type=str, required=True,
                      help='Path to save the sweep configuration')
    parser.add_argument('--parameters', type=str, nargs='+', required=True,
                      help='Parameters to sweep over in format "param_name=value1,value2,..."')
    parser.add_argument('--metric', type=str, default='eval/loss',
                      help='Metric to optimize')
    parser.add_argument('--goal', type=str, default='minimize',
                      choices=['minimize', 'maximize'],
                      help='Whether to minimize or maximize the metric')
    parser.add_argument('--method', type=str, default='grid',
                      choices=['grid', 'random', 'bayes'],
                      help='Sweep method')
    return parser.parse_args()

def parse_parameter(param_str: str) -> Dict[str, List[Any]]:
    """Parse a parameter string in format 'param_name=value1,value2,...'"""
    name, values = param_str.split('=')
    # Convert values to appropriate types
    value_list = []
    for v in values.split(','):
        try:
            # Try to convert to float first
            value_list.append(float(v))
        except ValueError:
            # If not a float, keep as string
            value_list.append(v)
    return {name: value_list}

def create_sweep_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Create the sweep configuration dictionary"""
    # Parse all parameters
    parameters = {}
    for param in args.parameters:
        parameters.update(parse_parameter(param))

    # Create the sweep configuration
    sweep_config = {
        'program': 'train.py',
        'method': args.method,
        'metric': {
            'name': args.metric,
            'goal': args.goal
        },
        'parameters': parameters,
        'command': [
            '${env}',
            'cd',
            'llm-foundry/scripts',
            '&&',
            'composer',
            'train/train.py',
            args.model_yaml
        ]
    }

    return sweep_config

def main():
    args = parse_args()

    # Create sweep configuration
    sweep_config = create_sweep_config(args)

    # Save sweep configuration to YAML
    with open(args.sweep_yaml, 'w') as f:
        yaml.dump(sweep_config, f)

    # Initialize wandb
    wandb.init(project=args.project, entity=args.entity)

    # Create sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)

    print(f"Created sweep with ID: {sweep_id}")
    print(f"Sweep configuration saved to: {args.sweep_yaml}")
    print("\nTo start the sweep, run:")
    print(f"wandb agent {args.entity}/{args.project}/{sweep_id}")

if __name__ == '__main__':
    main()
