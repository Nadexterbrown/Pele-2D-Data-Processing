"""Main CLI interface"""

import click
from pathlib import Path

from ..workflows.batch_processor import BatchProcessor
from ..config.processing_config import ProcessingConfiguration
from ..infrastructure.logging.mpi_logger import setup_logging


@click.group()
@click.version_option()
def cli():
    """PELE Analysis Toolkit - Professional CFD data processing"""
    pass


@cli.command()
@click.argument('data_dir', type=click.Path(exists=True))
@click.option('--output', '-o', default='./results', help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Verbose logging')
def process(data_dir, output, config, verbose):
    """Process PELE data directory"""

    setup_logging(verbose=verbose)

    # Load configuration
    if config:
        config_obj = ProcessingConfiguration.from_file(config)
    else:
        config_obj = ProcessingConfiguration()

    config_obj.output_directory = output
    config_obj.validate()

    # Run processing
    processor = BatchProcessor(config_obj)
    results = processor.process_directory(data_dir)

    click.echo(f"Processing completed. Results saved to {output}")