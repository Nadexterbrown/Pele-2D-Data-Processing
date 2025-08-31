"""PELE field definitions and mappings"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum


class FieldType(Enum):
    """Field type enumeration"""
    COORDINATE = "coordinate"
    SCALAR = "scalar"
    VECTOR = "vector"
    SPECIES = "species"
    DERIVED = "derived"


@dataclass(frozen=True)
class FieldDefinition:
    """Immutable field definition"""
    name: str
    pele_name: str
    units: str
    description: str
    field_type: FieldType
    cantera_computable: bool = False
    required: bool = True


# Clean, organized field definitions
BASE_FIELD_DEFINITIONS: Dict[str, FieldDefinition] = {
    # Coordinates
    'X': FieldDefinition('X', 'x', 'cm', 'X coordinate', FieldType.COORDINATE),
    'Y': FieldDefinition('Y', 'y', 'cm', 'Y coordinate', FieldType.COORDINATE),

    # Thermodynamic properties
    'Temperature': FieldDefinition('Temperature', 'Temp', 'K', 'Temperature', FieldType.SCALAR),
    'Pressure': FieldDefinition('Pressure', 'pressure', 'g / cm / s^2', 'Pressure', FieldType.SCALAR),
    'Density': FieldDefinition('Density', 'density', 'g / cm^3', 'Density', FieldType.SCALAR, cantera_computable=True),

    # Transport properties
    'Viscosity': FieldDefinition('Viscosity', 'viscosity', 'g / cm / s', 'Dynamic viscosity', FieldType.SCALAR,
                                 cantera_computable=True),
    'Conductivity': FieldDefinition('Conductivity', 'conductivity', 'g cm^2 / s^3 / cm / K', 'Thermal conductivity',
                                    FieldType.SCALAR, cantera_computable=True),
    'Sound_Speed': FieldDefinition('Sound_Speed', 'soundspeed', 'cm / s', 'Speed of sound', FieldType.SCALAR,
                                   cantera_computable=True),

    # Flow properties
    'X_Velocity': FieldDefinition('X_Velocity', 'x_velocity', 'cm / s', 'X-direction velocity', FieldType.SCALAR),
    'Y_Velocity': FieldDefinition('Y_Velocity', 'y_velocity', 'cm / s', 'Y-direction velocity', FieldType.SCALAR),
    'Mach_Number': FieldDefinition('Mach_Number', 'MachNumber', '', 'Mach number', FieldType.DERIVED,
                                   cantera_computable=True),

    # Mixture species properties
    'Cp': FieldDefinition('Cp', 'cp', 'g cm^2 / s^2 / g / K', 'Specific heat at constant pressure', FieldType.SCALAR,
                          cantera_computable=True),
    'Cv': FieldDefinition('Cv', 'cv', 'g cm^2 / s^2 / g / K', 'Specific heat at constant volume', FieldType.SCALAR,
                          cantera_computable=True),

    # Combustion properties
    'Heat_Release_Rate': FieldDefinition('Heat_Release_Rate', 'heatRelease', 'g cm^2 / s^3 / cm^3', 'Heat release rate',
                                         FieldType.SCALAR, cantera_computable=True),
}


class FieldRegistry:
    """Registry for managing field definitions"""

    def __init__(self):
        self._fields = BASE_FIELD_DEFINITIONS.copy()

    def add_species_fields(self, species_list: List[str]) -> None:
        """Add species-specific fields"""
        for species in species_list:
            # Mass fraction
            self._fields[f'Y_{species}'] = FieldDefinition(
                f'Y_{species}', f'Y({species})', '', f'{species} mass fraction', FieldType.SPECIES
            )
            # Diffusion coefficient
            self._fields[f'D_{species}'] = FieldDefinition(
                f'D_{species}', f'D({species})', 'cm^2 / s', f'{species} diffusion coefficient', FieldType.SPECIES,
                cantera_computable=True
            )
            # Production rate
            self._fields[f'W_{species}'] = FieldDefinition(
                f'W_{species}', f'rho_omega_{species}', 'g / cm^3 / s', f'{species} production rate', FieldType.SPECIES
            )

    def get_field(self, field_name: str) -> Optional[FieldDefinition]:
        """Get field definition"""
        return self._fields.get(field_name)

    def get_fields_by_type(self, field_type: FieldType) -> Dict[str, FieldDefinition]:
        """Get fields by type"""
        return {name: field for name, field in self._fields.items() if field.field_type == field_type}

    def get_required_fields(self) -> Dict[str, FieldDefinition]:
        """Get required fields"""
        return {name: field for name, field in self._fields.items() if field.required}

    def get_cantera_computable_fields(self) -> Dict[str, FieldDefinition]:
        """Get fields that can be computed with Cantera"""
        return {name: field for name, field in self._fields.items() if field.cantera_computable}


# Global field registry
field_registry = FieldRegistry()