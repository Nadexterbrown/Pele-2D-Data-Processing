"""
Unit Definitions for PELE Analysis

Comprehensive unit definitions covering PELE's mixed unit conventions,
SI standards, and scientific computing needs.
"""

from dataclasses import dataclass
from typing import Dict, Optional, List
from enum import Enum
import re


class UnitType(Enum):
    """Categories of physical units"""
    LENGTH = "length"
    TIME = "time"
    MASS = "mass"
    TEMPERATURE = "temperature"
    PRESSURE = "pressure"
    ENERGY = "energy"
    POWER = "power"
    VELOCITY = "velocity"
    ACCELERATION = "acceleration"
    DENSITY = "density"
    VISCOSITY = "viscosity"
    THERMAL_CONDUCTIVITY = "thermal_conductivity"
    SPECIFIC_HEAT = "specific_heat"
    DIMENSIONLESS = "dimensionless"
    COMPOUND = "compound"


@dataclass(frozen=True)
class UnitDefinition:
    """
    Definition of a physical unit

    Immutable definition that includes conversion factors and metadata
    for robust unit handling in scientific computing.
    """
    symbol: str  # Unit symbol (e.g., "m", "kg/m³")
    name: str  # Full name (e.g., "meter", "kilogram per cubic meter")
    unit_type: UnitType  # Category of unit
    si_equivalent: str  # Equivalent SI unit symbol
    to_si_factor: float  # Multiplication factor to convert to SI
    description: str = ""  # Human-readable description
    aliases: List[str] = None  # Alternative symbols/names

    def __post_init__(self):
        """Validate unit definition after creation"""
        if self.to_si_factor <= 0:
            raise ValueError(f"Conversion factor must be positive, got {self.to_si_factor}")

        if self.aliases is None:
            object.__setattr__(self, 'aliases', [])


# ===================================================================
# BASE SI UNITS
# ===================================================================

SI_BASE_UNITS: Dict[str, UnitDefinition] = {
    # Length
    'm': UnitDefinition('m', 'meter', UnitType.LENGTH, 'm', 1.0, 'SI base unit of length'),
    'cm': UnitDefinition('cm', 'centimeter', UnitType.LENGTH, 'm', 0.01, 'Common PELE length unit'),
    'mm': UnitDefinition('mm', 'millimeter', UnitType.LENGTH, 'm', 0.001),
    'km': UnitDefinition('km', 'kilometer', UnitType.LENGTH, 'm', 1000.0),

    # Time
    's': UnitDefinition('s', 'second', UnitType.TIME, 's', 1.0, 'SI base unit of time'),
    'ms': UnitDefinition('ms', 'millisecond', UnitType.TIME, 's', 0.001),
    'μs': UnitDefinition('μs', 'microsecond', UnitType.TIME, 's', 1e-6, aliases=['us']),
    'ns': UnitDefinition('ns', 'nanosecond', UnitType.TIME, 's', 1e-9),

    # Mass
    'kg': UnitDefinition('kg', 'kilogram', UnitType.MASS, 'kg', 1.0, 'SI base unit of mass'),
    'g': UnitDefinition('g', 'gram', UnitType.MASS, 'kg', 0.001, 'Common PELE mass unit'),
    'mg': UnitDefinition('mg', 'milligram', UnitType.MASS, 'kg', 1e-6),

    # Temperature
    'K': UnitDefinition('K', 'kelvin', UnitType.TEMPERATURE, 'K', 1.0, 'SI base unit of temperature'),
    '°C': UnitDefinition('°C', 'degree celsius', UnitType.TEMPERATURE, 'K', 1.0, 'Offset by 273.15K', aliases=['C']),

    # Pressure
    'Pa': UnitDefinition('Pa', 'pascal', UnitType.PRESSURE, 'Pa', 1.0, 'SI unit of pressure'),
    'kPa': UnitDefinition('kPa', 'kilopascal', UnitType.PRESSURE, 'Pa', 1000.0),
    'MPa': UnitDefinition('MPa', 'megapascal', UnitType.PRESSURE, 'Pa', 1e6),
    'bar': UnitDefinition('bar', 'bar', UnitType.PRESSURE, 'Pa', 1e5, 'Common pressure unit'),
    'atm': UnitDefinition('atm', 'atmosphere', UnitType.PRESSURE, 'Pa', 101325.0, 'Standard atmosphere'),

    # Energy
    'J': UnitDefinition('J', 'joule', UnitType.ENERGY, 'J', 1.0, 'SI unit of energy'),
    'kJ': UnitDefinition('kJ', 'kilojoule', UnitType.ENERGY, 'J', 1000.0),
    'MJ': UnitDefinition('MJ', 'megajoule', UnitType.ENERGY, 'J', 1e6),
    'cal': UnitDefinition('cal', 'calorie', UnitType.ENERGY, 'J', 4.184),
    'kcal': UnitDefinition('kcal', 'kilocalorie', UnitType.ENERGY, 'J', 4184.0),

    # Dimensionless
    '': UnitDefinition('', 'dimensionless', UnitType.DIMENSIONLESS, '', 1.0, 'No units'),
    '1': UnitDefinition('1', 'unity', UnitType.DIMENSIONLESS, '', 1.0, 'Explicit dimensionless'),
}

# ===================================================================
# PELE-SPECIFIC UNIT DEFINITIONS
# ===================================================================

PELE_UNIT_DEFINITIONS: Dict[str, UnitDefinition] = {

    # PELE uses CGS-based units extensively
    **SI_BASE_UNITS,  # Include all SI base units

    # Compound units common in PELE
    'g/cm³': UnitDefinition('g/cm³', 'grams per cubic centimeter', UnitType.DENSITY, 'kg/m³', 1000.0,
                            'PELE density unit'),
    'kg/m³': UnitDefinition('kg/m³', 'kilograms per cubic meter', UnitType.DENSITY, 'kg/m³', 1.0,
                            'SI density unit'),

    # Pressure units (PELE often uses g/cm/s²)
    'g/cm/s²': UnitDefinition('g/cm/s²', 'grams per centimeter per second squared', UnitType.PRESSURE, 'Pa', 0.1,
                              'PELE pressure unit (CGS)'),

    # Velocity
    'cm/s': UnitDefinition('cm/s', 'centimeters per second', UnitType.VELOCITY, 'm/s', 0.01,
                           'PELE velocity unit'),
    'm/s': UnitDefinition('m/s', 'meters per second', UnitType.VELOCITY, 'm/s', 1.0,
                          'SI velocity unit'),

    # Viscosity
    'g/cm/s': UnitDefinition('g/cm/s', 'grams per centimeter per second', UnitType.VISCOSITY, 'Pa·s', 0.1,
                             'PELE dynamic viscosity unit (poise)'),
    'Pa·s': UnitDefinition('Pa·s', 'pascal second', UnitType.VISCOSITY, 'Pa·s', 1.0,
                           'SI dynamic viscosity unit', aliases=['Pa*s']),

    # Thermal conductivity
    'g·cm/s³/K': UnitDefinition('g·cm/s³/K', 'grams centimeter per second cubed per kelvin',
                                UnitType.THERMAL_CONDUCTIVITY, 'W/m/K', 0.01,
                                'PELE thermal conductivity unit'),
    'g cm²/s³/cm/K': UnitDefinition('g cm²/s³/cm/K',
                                    'grams square centimeter per second cubed per centimeter per kelvin',
                                    UnitType.THERMAL_CONDUCTIVITY, 'W/m/K', 0.01,
                                    'Alternative PELE thermal conductivity notation'),
    'W/m/K': UnitDefinition('W/m/K', 'watts per meter per kelvin', UnitType.THERMAL_CONDUCTIVITY, 'W/m/K', 1.0,
                            'SI thermal conductivity unit'),

    # Specific heat
    'g·cm²/s²/g/K': UnitDefinition('g·cm²/s²/g/K', 'gram square centimeter per second squared per gram per kelvin',
                                   UnitType.SPECIFIC_HEAT, 'J/kg/K', 10000.0,
                                   'PELE specific heat unit'),
    'g cm²/s²/g/K': UnitDefinition('g cm²/s²/g/K', 'gram square centimeter per second squared per gram per kelvin',
                                   UnitType.SPECIFIC_HEAT, 'J/kg/K', 10000.0,
                                   'Alternative PELE specific heat notation'),
    'J/kg/K': UnitDefinition('J/kg/K', 'joules per kilogram per kelvin', UnitType.SPECIFIC_HEAT, 'J/kg/K', 1.0,
                             'SI specific heat unit'),

    # Heat release rate (volumetric power density)
    'g·cm²/s³/cm³': UnitDefinition('g·cm²/s³/cm³', 'gram square centimeter per second cubed per cubic centimeter',
                                   UnitType.POWER, 'W/m³', 100000.0,
                                   'PELE volumetric heat release rate'),
    'g cm²/s³/cm³': UnitDefinition('g cm²/s³/cm³', 'gram square centimeter per second cubed per cubic centimeter',
                                   UnitType.POWER, 'W/m³', 100000.0,
                                   'Alternative PELE heat release notation'),
    'W/m³': UnitDefinition('W/m³', 'watts per cubic meter', UnitType.POWER, 'W/m³', 1.0,
                           'SI volumetric power density'),

    # Production rates (mass per volume per time)
    'g/cm³/s': UnitDefinition('g/cm³/s', 'grams per cubic centimeter per second', UnitType.COMPOUND, 'kg/m³/s', 1000.0,
                              'PELE species production rate'),
    'kg/m³/s': UnitDefinition('kg/m³/s', 'kilograms per cubic meter per second', UnitType.COMPOUND, 'kg/m³/s', 1.0,
                              'SI mass production rate'),

    # Diffusion coefficients
    'cm²/s': UnitDefinition('cm²/s', 'square centimeters per second', UnitType.COMPOUND, 'm²/s', 1e-4,
                            'PELE diffusion coefficient'),
    'm²/s': UnitDefinition('m²/s', 'square meters per second', UnitType.COMPOUND, 'm²/s', 1.0,
                           'SI diffusion coefficient'),
}

# ===================================================================
# UNIT REGISTRY FUNCTIONS
# ===================================================================

_custom_units: Dict[str, UnitDefinition] = {}


def register_custom_unit(unit_def: UnitDefinition) -> None:
    """
    Register a custom unit definition

    Args:
        unit_def: Custom unit definition to register

    Raises:
        ValueError: If unit symbol already exists
    """
    if unit_def.symbol in PELE_UNIT_DEFINITIONS or unit_def.symbol in _custom_units:
        raise ValueError(f"Unit '{unit_def.symbol}' already registered")

    _custom_units[unit_def.symbol] = unit_def


def get_unit_info(unit_symbol: str) -> Optional[UnitDefinition]:
    """
    Get unit definition by symbol

    Args:
        unit_symbol: Symbol to look up (e.g., 'g/cm³')

    Returns:
        UnitDefinition if found, None otherwise
    """
    # Check custom units first
    if unit_symbol in _custom_units:
        return _custom_units[unit_symbol]

    # Check PELE units
    if unit_symbol in PELE_UNIT_DEFINITIONS:
        return PELE_UNIT_DEFINITIONS[unit_symbol]

    # Check aliases
    for unit_def in {**PELE_UNIT_DEFINITIONS, **_custom_units}.values():
        if unit_symbol in unit_def.aliases:
            return unit_def

    return None


def list_units_by_type(unit_type: UnitType) -> List[UnitDefinition]:
    """
    List all units of a specific type

    Args:
        unit_type: Type of units to list

    Returns:
        List of unit definitions of the specified type
    """
    all_units = {**PELE_UNIT_DEFINITIONS, **_custom_units}
    return [unit_def for unit_def in all_units.values() if unit_def.unit_type == unit_type]


def get_all_unit_symbols() -> List[str]:
    """Get list of all registered unit symbols"""
    all_units = {**PELE_UNIT_DEFINITIONS, **_custom_units}
    symbols = list(all_units.keys())

    # Add aliases
    for unit_def in all_units.values():
        symbols.extend(unit_def.aliases)

    return sorted(set(symbols))