"""
Units Module for PELE Analysis

Unit conversion system that handles PELE's mixed unit conventions
and provides clean conversion to SI units.
"""

from .converter import UnitConverter, ConversionError
from .definitions import (
    PELE_UNIT_DEFINITIONS,
    SI_BASE_UNITS,
    UnitDefinition,
    UnitType,
    register_custom_unit,
    get_unit_info
)

# Create default converter instance
default_converter = UnitConverter()

# Convenience functions using default converter
def convert_to_si(value, from_unit):
    """Convert value from PELE units to SI using default converter"""
    return default_converter.convert_to_si(value, from_unit)

def convert_value(value, from_unit, to_unit):
    """Convert value between arbitrary units using default converter"""
    return default_converter.convert(value, from_unit, to_unit)

def get_conversion_factor(from_unit, to_unit):
    """Get conversion factor between units using default converter"""
    return default_converter.get_conversion_factor(from_unit, to_unit)

__all__ = [
    'UnitConverter',
    'ConversionError',
    'PELE_UNIT_DEFINITIONS',
    'SI_BASE_UNITS',
    'UnitDefinition',
    'UnitType',
    'register_custom_unit',
    'get_unit_info',
    'default_converter',
    'convert_to_si',
    'convert_value',
    'get_conversion_factor'
]