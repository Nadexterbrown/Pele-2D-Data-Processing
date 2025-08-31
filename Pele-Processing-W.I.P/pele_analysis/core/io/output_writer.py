"""
Enhanced Output Writer with Space-Separated Text Format

Adds support for space-separated text files with dual header rows,
matching your existing write_nested_dict_to_file format.
"""

import json
import lzma
import csv
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from enum import Enum
import numpy as np
import pandas as pd

class OutputFormat(Enum):
    """Supported output formats"""
    JSON = "json"
    JSON_COMPRESSED = "json_xz"
    CSV = "csv"
    PICKLE = "pkl"
    HDF5 = "h5"
    EXCEL = "xlsx"
    SPACE_SEPARATED_TXT = "txt"  # New format matching your existing code

class OutputWriter:
    """
    Professional output writer with multiple format support

    Now includes space-separated text format with dual headers
    that matches your existing write_nested_dict_to_file function.
    """

    def __init__(self, output_directory: Union[str, Path],
                 field_width: int = 65, logger=None):
        """
        Initialize output writer

        Args:
            output_directory: Base directory for output files
            field_width: Field width for space-separated text format
            logger: Optional logger instance
        """
        self.output_dir = Path(output_directory)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger or self._create_default_logger()

        # Configuration for space-separated text format
        self.field_width = field_width

        # Unit mappings for headers
        self.unit_mappings = {
            'Position': 'm',
            'Temperature': 'K',
            'Pressure': 'Pa',
            'Density': 'kg/m³',
            'Velocity': 'm/s',
            'Time': 's',
            'Heat_Release_Rate': 'W/m³',
            'Viscosity': 'Pa·s',
            'Conductivity': 'W/m·K',
            'Sound_Speed': 'm/s',
            'Cp': 'J/kg·K',
            'Cv': 'J/kg·K',
            'Surface_Length': 'm',
            'Consumption_Rate': 'kg/s',
            'Burning_Velocity': 'm/s',
            'Flame_Thickness': 'm'
        }

    def write_results(self, data: Union[List[Dict], Dict[str, Any]],
                     filename: Optional[str] = None,
                     formats: List[OutputFormat] = None,
                     metadata: Optional[Dict] = None) -> Dict[str, Path]:
        """
        Write results in multiple formats

        Args:
            data: Results data to write
            filename: Base filename (without extension)
            formats: List of output formats (defaults to JSON and TXT)
            metadata: Optional metadata to include

        Returns:
            Dictionary mapping format names to written file paths
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pele_results_{timestamp}"

        if formats is None:
            formats = [OutputFormat.SPACE_SEPARATED_TXT, OutputFormat.JSON_COMPRESSED]

        # Prepare data with metadata
        output_data = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'version': '1.0',
                'total_records': len(data) if isinstance(data, list) else 1,
                **(metadata or {})
            },
            'results': data
        }

        written_files = {}

        for format_type in formats:
            try:
                file_path = self._write_format(output_data, filename, format_type)
                written_files[format_type.value] = file_path
                self.logger.info(f"Wrote {format_type.value.upper()} output: {file_path.name}")
            except Exception as e:
                self.logger.error(f"Failed to write {format_type.value} format: {e}")
                continue

        # Write summary file
        try:
            summary_path = self._write_summary(output_data, filename)
            written_files['summary'] = summary_path
        except Exception as e:
            self.logger.error(f"Failed to write summary: {e}")

        return written_files

    def _write_format(self, data: Dict[str, Any], filename: str,
                     format_type: OutputFormat) -> Path:
        """Write data in specified format"""

        if format_type == OutputFormat.JSON:
            return self._write_json(data, filename)
        elif format_type == OutputFormat.JSON_COMPRESSED:
            return self._write_json_compressed(data, filename)
        elif format_type == OutputFormat.CSV:
            return self._write_csv(data['results'], filename)
        elif format_type == OutputFormat.PICKLE:
            return self._write_pickle(data, filename)
        elif format_type == OutputFormat.HDF5:
            return self._write_hdf5(data['results'], filename)
        elif format_type == OutputFormat.EXCEL:
            return self._write_excel(data['results'], filename)
        elif format_type == OutputFormat.SPACE_SEPARATED_TXT:
            return self._write_space_separated_txt(data['results'], filename)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def _write_space_separated_txt(self, data: Union[List[Dict], Dict],
                                  filename: str) -> Path:
        """
        Write space-separated text file with dual headers

        This matches your existing write_nested_dict_to_file format:
        - Index row (# 1 2 3 ...)
        - Header row (# variable names with units)
        - Data rows (indented, scientific notation)
        """
        file_path = self.output_dir / f"{filename}.txt"

        # Ensure data is a list
        if isinstance(data, dict):
            data_list = [data]
        else:
            data_list = data

        if not data_list:
            raise ValueError("No data to write")

        # Collect headers and flatten data
        headers_grouped = self._collect_headers_by_group(data_list)
        headers_flat = [h for group in headers_grouped.values() for h in group]
        data_rows = self._collect_flat_data(data_list)

        with open(file_path, 'w') as f:
            # Write index row
            index_line = "# " + "".join(
                f"{i + 1:<{self.field_width}d}"
                for i in range(len(headers_flat))
            )
            f.write(index_line + "\n")

            # Write header row with units
            header_line = "# " + "".join(
                f"{self._add_units_to_header(h):<{self.field_width}s}"
                for h in headers_flat
            )
            f.write(header_line + "\n")

            # Write data rows
            indent = "  "
            for row in data_rows:
                data_line = "".join(
                    f"{self._format_value(v):<{self.field_width}s}"
                    for v in row
                )
                f.write(indent + data_line + "\n")

        return file_path

    def _collect_headers_by_group(self, data_array: List[Dict]) -> Dict[str, List[str]]:
        """
        Collect headers organized by group from nested dictionaries

        This replicates the logic from your original function but with
        proper error handling and type safety.
        """
        if not data_array:
            return {}

        grouped_headers = {}
        sample = data_array[0]  # Use first entry as template

        for group_key, sub_dict in sample.items():
            headers = []

            def recurse_collect(sub_data, prefix=""):
                """Recursively collect headers from nested structure"""
                if isinstance(sub_data, dict):
                    for key, value in sub_data.items():
                        if isinstance(value, dict):
                            recurse_collect(value, prefix + key + " ")
                        else:
                            # Leaf value - create header
                            full_header = prefix + key
                            headers.append(full_header.strip())
                else:
                    # Non-dict value at top level
                    headers.append(group_key)

            if isinstance(sub_dict, dict):
                recurse_collect(sub_dict)
                # Add group prefix to each header
                grouped_headers[group_key] = [f"{group_key} {h}" for h in headers]
            else:
                # Simple value, not nested
                grouped_headers[group_key] = [group_key]

        return grouped_headers

    def _collect_flat_data(self, data_array: List[Dict]) -> List[List]:
        """
        Flatten nested dictionary data into rows for writing

        This matches the data extraction logic from your original function.
        """
        data_rows = []

        for entry in data_array:
            row = []

            def recurse_extract(sub_data):
                """Recursively extract values from nested structure"""
                if isinstance(sub_data, dict):
                    for value in sub_data.values():
                        if isinstance(value, dict):
                            recurse_extract(value)
                        else:
                            # Leaf value - add to row
                            if isinstance(value, (list, tuple, np.ndarray)):
                                row.extend(value)
                            else:
                                row.append(value)
                else:
                    # Non-dict value
                    if isinstance(sub_data, (list, tuple, np.ndarray)):
                        row.extend(sub_data)
                    else:
                        row.append(sub_data)

            for group_data in entry.values():
                if isinstance(group_data, dict):
                    recurse_extract(group_data)
                else:
                    row.append(group_data)

            data_rows.append(row)

        return data_rows

    def _add_units_to_header(self, header: str) -> str:
        """
        Add units to header names based on unit mappings

        Args:
            header: Header name (e.g., "Flame Temperature")

        Returns:
            Header with units (e.g., "Flame Temperature [K]")
        """
        # Try to find unit mapping
        for field_name, unit in self.unit_mappings.items():
            if field_name.lower() in header.lower():
                return f"{header} [{unit}]"

        # Check for common patterns
        if 'temperature' in header.lower():
            return f"{header} [K]"
        elif 'pressure' in header.lower():
            return f"{header} [Pa]"
        elif 'velocity' in header.lower():
            return f"{header} [m/s]"
        elif 'position' in header.lower():
            return f"{header} [m]"
        elif 'time' in header.lower():
            return f"{header} [s]"
        elif 'density' in header.lower():
            return f"{header} [kg/m³]"

        # No unit found, return as-is
        return header

    def _format_value(self, value) -> str:
        """
        Format a value for space-separated text output

        Uses scientific notation for floats, string representation for others.
        This matches your original formatting logic.
        """
        if isinstance(value, (int, float, np.floating)):
            if np.isnan(value) or np.isinf(value):
                return "NaN"
            return f"{float(value):.6e}"
        else:
            return str(value)

    # ... (keep all the other existing methods from the original output_writer.py)

    def _write_json(self, data: Dict[str, Any], filename: str) -> Path:
        """Write human-readable JSON"""
        file_path = self.output_dir / f"{filename}.json"

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=self._json_serializer)

        return file_path

    def _write_json_compressed(self, data: Dict[str, Any], filename: str) -> Path:
        """Write compressed JSON for efficiency"""
        file_path = self.output_dir / f"{filename}.json.xz"

        json_string = json.dumps(data, separators=(',', ':'), default=self._json_serializer)

        with lzma.open(file_path, 'wt', preset=1) as f:
            f.write(json_string)

        return file_path

    def _write_csv(self, data: Union[List[Dict], Dict], filename: str) -> Path:
        """Write CSV for analysis tools"""
        file_path = self.output_dir / f"{filename}.csv"

        if isinstance(data, dict):
            flattened = [self._flatten_dict(data)]
        else:
            flattened = [self._flatten_dict(item) for item in data]

        if not flattened:
            raise ValueError("No data to write to CSV")

        all_fields = set()
        for item in flattened:
            all_fields.update(item.keys())

        fieldnames = sorted(all_fields)

        with open(file_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)

        return file_path

    def _flatten_dict(self, nested_dict: Dict, parent_key: str = '',
                     separator: str = '_') -> Dict:
        """Flatten nested dictionary for tabular formats"""
        items = []

        for key, value in nested_dict.items():
            new_key = f"{parent_key}{separator}{key}" if parent_key else key

            if isinstance(value, dict):
                items.extend(self._flatten_dict(value, new_key, separator).items())
            elif isinstance(value, (list, tuple, np.ndarray)):
                if len(value) == 1:
                    items.append((new_key, value[0]))
                elif len(value) <= 5:
                    items.append((new_key, str(value)))
                else:
                    if isinstance(value, np.ndarray):
                        items.append((f"{new_key}_mean", float(np.mean(value))))
                        items.append((f"{new_key}_std", float(np.std(value))))
                        items.append((f"{new_key}_min", float(np.min(value))))
                        items.append((f"{new_key}_max", float(np.max(value))))
                    else:
                        items.append((new_key, f"Array[{len(value)}]"))
            else:
                items.append((new_key, value))

        return dict(items)

    def _json_serializer(self, obj):
        """Custom JSON serializer for numpy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif hasattr(obj, 'isoformat'):
            return obj.isoformat()
        return str(obj)

    def _write_summary(self, data: Dict[str, Any], filename: str) -> Path:
        """Write processing summary"""
        file_path = self.output_dir / f"{filename}_summary.json"

        results = data['results']
        if isinstance(results, list):
            num_results = len(results)
            sample_result = results[0] if results else {}
        else:
            num_results = 1
            sample_result = results

        summary = {
            'processing_summary': {
                'total_results': num_results,
                'sample_fields': list(self._flatten_dict(sample_result).keys()) if sample_result else [],
                'file_size_mb': sum(f.stat().st_size for f in self.output_dir.glob(f"{filename}.*")) / (1024*1024),
            },
            'metadata': data.get('metadata', {}),
            'created_at': datetime.now().isoformat()
        }

        with open(file_path, 'w') as f:
            json.dump(summary, f, indent=2)

        return file_path

    def _create_default_logger(self):
        """Create default logger"""
        import logging
        logger = logging.getLogger('OutputWriter')
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

# ===================================================================
# USAGE EXAMPLES
# ===================================================================

def example_space_separated_output():
    """Example of using the space-separated text format"""

    # Sample nested data (like your processing results)
    sample_data = [
        {
            'Time': 1.5e-6,
            'Flame': {
                'Position': 0.045,
                'Velocity': 1250.0,
                'Thermodynamic': {
                    'Temperature': 2800.0,
                    'Pressure': 15000.0,
                    'Density': 0.45
                }
            },
            'Shock': {
                'Position': 0.065,
                'Velocity': 2100.0
            }
        },
        {
            'Time': 1.6e-6,
            'Flame': {
                'Position': 0.047,
                'Velocity': 1280.0,
                'Thermodynamic': {
                    'Temperature': 2850.0,
                    'Pressure': 15200.0,
                    'Density': 0.46
                }
            },
            'Shock': {
                'Position': 0.068,
                'Velocity': 2120.0
            }
        }
    ]

    # Create writer with custom field width
    writer = OutputWriter("./test_output", field_width=50)

    # Write in space-separated format
    output_files = writer.write_results(
        sample_data,
        filename="flame_shock_analysis",
        formats=[OutputFormat.SPACE_SEPARATED_TXT]
    )

    print(f"Space-separated file written: {output_files['txt']}")

    # Show what the output looks like
    with open(output_files['txt'], 'r') as f:
        print("\nOutput preview:")
        print(f.read())

if __name__ == "__main__":
    example_space_separated_output()