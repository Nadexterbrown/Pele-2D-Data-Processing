"""Main processing script - simplified from 2D-Pele-Processing.py"""

# !/usr/bin/env python3

from pele_analysis import BatchProcessor, ProcessingConfiguration, ThermodynamicConditions
from pele_analysis.infrastructure.logging.mpi_logger import setup_logging


def main():
    """Simplified main processing function"""

    # Setup logging
    setup_logging('processing.log', overwrite=True)

    # Configuration
    config = ProcessingConfiguration(
        extraction_location=0.0494854,  # 5cm extraction location
        output_directory='./results',
        flame_temp_threshold=2500.0
    )

    # Enable desired processing
    config.flame.position.enabled = True
    config.flame.velocity.enabled = True
    config.flame.thickness.enabled = True
    config.animation.temperature.enabled = True

    config.validate()

    # Setup thermodynamic conditions
    thermo = ThermodynamicConditions(
        temperature=503.15,
        pressure=10.0 * 1e5,
        equivalence_ratio=1.0,
        fuel_type='H2',
        mechanism_file='../mechanism_files/LiDryer.yaml'
    )

    # Process data
    processor = BatchProcessor(config)
    results = processor.process_directory('../../2D-Test-Data')

    print("Processing completed successfully")


if __name__ == "__main__":
    main()