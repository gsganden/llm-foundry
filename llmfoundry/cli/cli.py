# Copyright 2024 MosaicML LLM Foundry authors
# SPDX-License-Identifier: Apache-2.0

from typing import List, Optional

import typer
from omegaconf import DictConfig
from omegaconf import OmegaConf as om

from llmfoundry.cli import registry_cli
from llmfoundry.train.train import train_from_yaml as trainer

app = typer.Typer(pretty_exceptions_show_locals=False)
app.add_typer(registry_cli.app, name='registry')


@app.command(name='train')
def train(
    yaml_path: str = typer.Argument(
        ...,
        help='Path to the YAML configuration file',
    ),  # type: ignore
    args_list: Optional[List[str]] = typer.
    Argument(None, help='Additional command line arguments'),  # type: ignore
):
    """Run the training with optional overrides from CLI."""
    trainer(yaml_path, args_list)


if __name__ == '__main__':
    app()
