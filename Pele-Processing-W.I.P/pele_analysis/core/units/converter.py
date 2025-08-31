"""
Professional Unit Conversion System

Robust unit converter that handles PELE's mixed CGS/SI conventions
with proper error handling, caching, and extensibility.
"""

import re
import warnings
from typing import Union, Dict, Optional, Tuple, Any
from functools import lru_cache
import numpy as np

from .definitions import (
    PELE_UNIT_DEFINITIONS,
    UnitDefinition,
    get_unit_info,
    _custom_units
)


class ConversionError(ValueError):
    """Exception raised for unit conversion errors"""

    def __init__(self, message: str, from_unit: str = None, to_unit: str = None):
        self.from_unit = from_unit
        self.to_unit = to_unit
        super().__init__(message)


class UnitConverter:
    """
    Professional unit converter with caching and robust error handling

    Handles the complex unit conversions needed for PELE data processing,
    including CGS to SI conversions and compound unit expressions.
    """

    def __init__(self, enable_caching: bool = True, cache_size: int = 256):
        """
        Initialize unit converter

        Args:
            enable_caching: Enable conversion factor caching for performance
            cache_size: Maximum number of cached conversion factors
        """
        self.enable_caching = enable_caching
        self.cache_size = cache_size

        # Conversion factor cache for performance
        self._conversion_cache: Dict[str, float] = {}

        # Statistics for monitoring
        self._stats = {
            'conversions': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }

    def convert_to_si(self, value: Union[float, np.ndarray],
                      from_unit: str) -> Union[float, np.ndarray]:
        """
        Convert value from any supported unit to SI

        Args:
            value: Value(s) to convert
            from_unit: Source unit (e.g., 'g/cm³', 'K', 'cm/s')

        Returns:
            Value(s) in SI units

        Raises:
            ConversionError: If conversion fails
        """
        if not from_unit or from_unit.strip() == '':
            return value  # Dimensionless

        try:
            factor = self.get_conversion_factor(from_unit, 'SI')
            self._stats['conversions'] += 1

            if isinstance(value, np.ndarray):
                return value * factor
            else:
                return value * factor

        except Exception as e:
            self._stats['errors'] += 1
            raise ConversionError(f"Failed to convert '{from_unit}' to SI: {e}", from_unit, 'SI')

    def convert(self, value: Union[float, np.ndarray],
                from_unit: str, to_unit: str) -> Union[float, np.ndarray]:
        """
        Convert value between arbitrary units

        Args:
            value: Value(s) to convert
            from_unit: Source unit
            to_unit: Target unit

        Returns:
            Converted value(s)

        Raises:
            ConversionError: If conversion fails
        """
        if from_unit == to_unit:
            return value

        try:
            factor = self.get_conversion_factor(from_unit, to_unit)
            self._stats['conversions'] += 1

            if isinstance(value, np.ndarray):
                return value * factor
            else:
                return value * factor

        except Exception as e:
            self._stats['errors'] += 1
            raise ConversionError(f"Failed to convert from '{from_unit}' to '{to_unit}': {e}",
                                  from_unit, to_unit)

    @lru_cache(maxsize=256)
    def get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """
        Get conversion factor between units with caching

        Args:
            from_unit: Source unit
            to_unit: Target unit ('SI' for SI units)

        Returns:
            Multiplication factor for conversion

        Raises:
            ConversionError: If units not found or incompatible
        """
        if from_unit == to_unit:
            return 1.0

        # Handle empty/dimensionless units
        if not from_unit or from_unit.strip() == '':
            if not to_unit or to_unit.strip() == '' or to_unit == 'SI':
                return 1.0
            else:
                raise ConversionError(f"Cannot convert dimensionless to '{to_unit}'")

        # Cache key for performance
        cache_key = f"{from_unit}→{to_unit}"

        if self.enable_caching and cache_key in self._conversion_cache:
            self._stats['cache_hits'] += 1
            return self._conversion_cache[cache_key]

        self._stats['cache_misses'] += 1

        try:
            if to_unit == 'SI':
                factor = self._get_to_si_factor(from_unit)
            else:
                # Convert via SI: from_unit → SI → to_unit
                to_si_factor = self._get_to_si_factor(from_unit)
                from_si_factor = 1.0 / self._get_to_si_factor(to_unit)
                factor = to_si_factor * from_si_factor

            # Cache result
            if self.enable_caching:
                if len(self._conversion_cache) >= self.cache_size:
                    # Simple cache eviction - remove oldest 25%
                    items_to_remove = self.cache_size // 4
                    for _ in range(items_to_remove):
                        self._conversion_cache.pop(next(iter(self._conversion_cache)))

                self._conversion_cache[cache_key] = factor

            return factor

        except Exception as e:
            raise ConversionError(f"Cannot determine conversion factor: {e}", from_unit, to_unit)

    def _get_to_si_factor(self, unit: str) -> float:
        """
        Get conversion factor from given unit to SI

        Args:
            unit: Unit string (e.g., 'g/cm³')

        Returns:
            Conversion factor to SI

        Raises:
            ConversionError: If unit not recognized
        """
        if not unit or unit.strip() == '':
            return 1.0

        # Direct lookup first
        unit_info = get_unit_info(unit)
        if unit_info:
            return unit_info.to_si_factor

        # Try parsing compound units
        try:
            return self._parse_compound_unit(unit)
        except:
            pass

        # Last resort: try cleaning the unit string
        cleaned_unit = self._clean_unit_string(unit)
        if cleaned_unit != unit:
            unit_info = get_unit_info(cleaned_unit)
            if unit_info:
                return unit_info.to_si_factor

        raise ConversionError(f"Unknown unit: '{unit}'")

    def _parse_compound_unit(self, unit_expr: str) -> float:
        """
        Parse compound unit expressions like 'g / cm / s^2'

        Args:
            unit_expr: Compound unit expression

        Returns:
            Conversion factor to SI

        Raises:
            ConversionError: If parsing fails
        """
        try:
            # Handle division - split by '/'
            if '/' in unit_expr:
                parts = unit_expr.split('/')
                numerator_part = parts[0].strip()
                denominator_parts = [p.strip() for p in parts[1:]]

                # Calculate numerator factor
                num_factor = self._parse_unit_part(numerator_part) if numerator_part else 1.0

                # Calculate denominator factor
                denom_factor = 1.0
                for part in denominator_parts:
                    if part:  # Skip empty parts
                        denom_factor *= self._parse_unit_part(part)

                return num_factor / denom_factor
            else:
                # Simple unit or product of units
                return self._parse_unit_part(unit_expr)

        except Exception as e:
            raise ConversionError(f"Failed to parse compound unit '{unit_expr}': {e}")

    def _parse_unit_part(self, unit_part: str) -> float:
        """
        Parse individual unit part with optional exponent

        Args:
            unit_part: Unit part (e.g., 'cm^2', 'g', 's^-1')

        Returns:
            Conversion factor contribution
        """
        if not unit_part or unit_part.strip() == '':
            return 1.0

        unit_part = unit_part.strip()

        # Handle multiplication within the part (like 'g·cm')
        if '·' in unit_part or '*' in unit_part:
            # Split by multiplication operators
            mult_parts = re.split('[·*]', unit_part)
            factor = 1.0
            for part in mult_parts:
                factor *= self._parse_single_unit(part.strip())
            return factor
        else:
            return self._parse_single_unit(unit_part)

    def _parse_single_unit(self, unit: str) -> float:
        """
        Parse single unit with optional exponent

        Args:
            unit: Single unit (e.g., 'cm^2', 'K', 's^-1')

        Returns:
            Conversion factor
        """
        if not unit:
            return 1.0

        # Extract exponent using regex
        match = re.match(r'^([a-zA-Z°]+)(?:\^(-?\d+(?:\.\d+)?))?$', unit)
        if not match:
            # Try alternative exponent notation
            match = re.match(r'^([a-zA-Z°]+)(\d+)?$', unit)
            if match:
                base_unit, exp_str = match.groups()
                exponent = int(exp_str) if exp_str else 1
            else:
                # No exponent found, treat as exponent 1
                base_unit = unit
                exponent = 1
        else:
            base_unit, exp_str = match.groups()
            exponent = float(exp_str) if exp_str else 1

        # Look up base unit
        unit_info = get_unit_info(base_unit)
        if not unit_info:
            raise ConversionError(f"Unknown base unit: '{base_unit}' in '{unit}'")

        # Apply exponent
        return unit_info.to_si_factor ** exponent

    def _clean_unit_string(self, unit: str) -> str:
        """
        Clean unit string for better matching

        Args:
            unit: Raw unit string

        Returns:
            Cleaned unit string
        """
        if not unit:
            return unit

        # Common cleaning operations
        cleaned = unit.strip()

        # Replace common alternative notations
        replacements = {
            ' ': '',  # Remove spaces
            '**': '^',  # Python power notation
            'deg': '°',  # Degree symbol
            'degC': '°C',  # Celsius
            'Pa*s': 'Pa·s',  # Viscosity notation
        }

        for old, new in replacements.items():
            cleaned = cleaned.replace(old, new)

        return cleaned

    def validate_unit(self, unit: str) -> bool:
        """
        Check if unit is recognized by the converter

        Args:
            unit: Unit string to validate

        Returns:
            True if unit is recognized, False otherwise
        """
        try:
            self._get_to_si_factor(unit)
            return True
        except ConversionError:
            return False

    def get_si_equivalent(self, unit: str) -> str:
        """
        Get the SI equivalent unit symbol

        Args:
            unit: Input unit

        Returns:
            SI equivalent unit symbol

        Raises:
            ConversionError: If unit not recognized
        """
        unit_info = get_unit_info(unit)
        if unit_info:
            return unit_info.si_equivalent

        raise ConversionError(f"Cannot determine SI equivalent for '{unit}'")

    def get_statistics(self) -> Dict[str, Any]:
        """Get converter usage statistics"""
        stats = self._stats.copy()
        stats.update({
            'cache_size': len(self._conversion_cache),
            'cache_hit_rate': (self._stats['cache_hits'] /
                               max(1, self._stats['cache_hits'] + self._stats['cache_misses'])),
            'registered_units': len(PELE_UNIT_DEFINITIONS) + len(_custom_units)
        })
        return stats

    def clear_cache(self):
        """Clear conversion factor cache"""
        self._conversion_cache.clear()
        self._stats['cache_hits'] = 0
        self._stats['cache_misses'] = 0


# ===================================================================
# USAGE EXAMPLES AND TESTING
# ===================================================================

def example_usage():
    """Examples of using the units system"""

    from .converter import UnitConverter
    from .definitions import get_unit_info, list_units_by_type, UnitType

    # Create converter
    converter = UnitConverter()

    # Example 1: Convert PELE density to SI
    pele_density = 0.0015  # g/cm³
    si_density = converter.convert_to_si(pele_density, 'g/cm³')
    print(f"PELE density: {pele_density} g/cm³ = {si_density} kg/m³")

    # Example 2: Convert PELE pressure to SI
    pele_pressure = 150000  # g/cm/s²
    si_pressure = converter.convert_to_si(pele_pressure, 'g/cm/s²')
    print(f"PELE pressure: {pele_pressure} g/cm/s² = {si_pressure} Pa")

    # Example 3: Convert between arbitrary units
    temp_k = converter.convert(25.0, '°C', 'K')
    print(f"Temperature: 25°C = {temp_k} K")

    # Example 4: Array conversion
    import numpy as np
    velocities_pele = np.array([1000, 1500, 2000])  # cm/s
    velocities_si = converter.convert_to_si(velocities_pele, 'cm/s')
    print(f"Velocities: {velocities_pele} cm/s = {velocities_si} m/s")

    # Example 5: Unit information
    unit_info = get_unit_info('g/cm³')
    print(f"Unit info: {unit_info.name} ({unit_info.description})")

    # Example 6: List units by type
    pressure_units = list_units_by_type(UnitType.PRESSURE)
    print(f"Available pressure units: {[u.symbol for u in pressure_units]}")

    # Example 7: Statistics
    print(f"Converter statistics: {converter.get_statistics()}")


def test_unit_conversions():
    """Test suite for unit conversions"""

    converter = UnitConverter()

    # Test cases: (value, from_unit, expected_si_value, tolerance)
    test_cases = [
        (1.0, 'g/cm³', 1000.0, 1e-10),  # Density
        (1.0, 'cm/s', 0.01, 1e-10),  # Velocity
        (1.0, 'g/cm/s²', 0.1, 1e-10),  # Pressure
        (1.0, 'K', 1.0, 1e-10),  # Temperature
        (1.0, 'cm', 0.01, 1e-10),  # Length
        (1000.0, 'g cm²/s³/cm³', 100000.0, 1e-6),  # Heat release rate
    ]

    passed = 0
    failed = 0

    for value, unit, expected, tolerance in test_cases:
        try:
            result = converter.convert_to_si(value, unit)
            if abs(result - expected) <= tolerance:
                print(f"✓ {value} {unit} → {result} (expected {expected})")
                passed += 1
            else:
                print(f"✗ {value} {unit} → {result} (expected {expected}, error: {abs(result - expected)})")
                failed += 1
        except Exception as e:
            print(f"✗ {value} {unit} → ERROR: {e}")
            failed += 1

    print(f"\nTest results: {passed} passed, {failed} failed")
    return failed == 0


if __name__ == "__main__":
    print("=== Unit System Examples ===")
    example_usage()

    print("\n=== Unit Conversion Tests ===")
    test_unit_conversions()