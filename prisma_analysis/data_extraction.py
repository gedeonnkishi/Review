"""
Data extraction and standardization module
Extracts study characteristics, methods, and results
"""

import pandas as pd
import numpy as np
import re
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging

logger = logging.getLogger(__name__)

@dataclass
class StudyCharacteristics:
    """Standardized study characteristics data class"""
    study_id: str
    authors: str
    year: int
    title: str
    source: str  # Journal, conference, preprint
    doi: Optional[str] = None
    citation: Optional[str] = None
    
    # Methodology characteristics
    qml_approach: str  # QNN, QSVM, QELM, QRC
    quantum_level: str  # simulation, emulation, hardware
    implementation_details: Dict[str, Any] = None
    
    # Dataset characteristics
    dataset_name: str
    dataset_type: str  # synthetic, real-world
    dataset_domain: str  # energy, finance, healthcare
    sample_size: Optional[int] = None
    forecasting_horizon: Optional[str] = None
    
    # Performance metrics
    metrics: Dict[str, float] = None  # RMSE, MAE, MAPE, R²
    baseline_comparison: Dict[str, float] = None  # Ratios vs baselines
    statistical_significance: Optional[bool] = None
    
    # Quantum resources
    qubits_used: Optional[int] = None
    quantum_backend: Optional[str] = None
    circuit_depth: Optional[int] = None
    encoding_scheme: Optional[str] = None
    
    # Reproducibility
    code_availability: Optional[bool] = None
    data_availability: Optional[bool] = None
    hyperparameters: Optional[Dict] = None
    
    def to_dict(self):
        """Convert to dictionary"""
        return asdict(self)

class DataExtractor:
    """Extracts and standardizes data from included studies"""
    
    def __init__(self, extraction_template: str = "templates/extraction_form.json"):
        """
        Initialize data extractor
        
        Args:
            extraction_template: Path to extraction template JSON
        """
        self.template = self._load_template(extraction_template)
        self.extracted_data = []
        
    def _load_template(self, template_path: str) -> Dict:
        """Load extraction template"""
        with open(template_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_from_pdf(self, pdf_path: str) -> Dict:
        """
        Extract data from PDF (simplified - would use actual PDF parsing)
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Extracted data dictionary
        """
        # Note: In practice, use PyPDF2, pdfplumber, or GROBID
        # This is a simplified version
        
        extracted = {
            'metadata': self._extract_metadata(pdf_path),
            'methods': self._extract_methods(pdf_path),
            'results': self._extract_results(pdf_path),
            'discussion': self._extract_discussion(pdf_path)
        }
        
        return extracted
    
    def extract_from_structured_source(self, 
                                      source_type: str,
                                      source_data: Dict) -> StudyCharacteristics:
        """
        Extract from structured sources (CSV, database, API)
        
        Args:
            source_type: Type of source ('csv', 'api', 'database')
            source_data: Source-specific data
        
        Returns:
            Standardized study characteristics
        """
        if source_type == 'csv':
            return self._extract_from_csv(source_data)
        elif source_type == 'api':
            return self._extract_from_api(source_data)
        elif source_type == 'manual_entry':
            return self._extract_manual(source_data)
        else:
            raise ValueError(f"Unsupported source type: {source_type}")
    
    def standardize_metrics(self, raw_metrics: Dict) -> Dict:
        """
        Standardize performance metrics across studies
        
        Args:
            raw_metrics: Raw metrics from study
        
        Returns:
            Standardized metrics dictionary
        """
        standardized = {}
        
        # Define metric normalization rules
        metric_mapping = {
            'rmse': ['RMSE', 'Root Mean Square Error', 'root mean squared error'],
            'mae': ['MAE', 'Mean Absolute Error', 'mean absolute error'],
            'mape': ['MAPE', 'Mean Absolute Percentage Error'],
            'r2': ['R²', 'R2', 'R-squared', 'coefficient of determination'],
            'mse': ['MSE', 'Mean Squared Error']
        }
        
        for std_name, variants in metric_mapping.items():
            for variant in variants:
                if variant in raw_metrics:
                    standardized[std_name] = self._normalize_metric_value(
                        raw_metrics[variant], std_name
                    )
                    break
        
        # Calculate ratios if baseline metrics are provided
        if 'baseline_rmse' in raw_metrics and 'rmse' in standardized:
            standardized['rmse_ratio'] = standardized['rmse'] / raw_metrics['baseline_rmse']
        
        return standardized
    
    def validate_extraction(self, 
                          study_data: StudyCharacteristics,
                          validation_rules: Dict) -> Dict[str, List[str]]:
        """
        Validate extracted data against rules
        
        Args:
            study_data: Extracted study data
            validation_rules: Validation rules dictionary
        
        Returns:
            Dictionary of validation errors and warnings
        """
        errors = []
        warnings = []
        
        data_dict = study_data.to_dict()
        
        # Check required fields
        for field in validation_rules.get('required_fields', []):
            if field not in data_dict or data_dict[field] is None:
                errors.append(f"Missing required field: {field}")
        
        # Check data types
        for field, expected_type in validation_rules.get('type_checks', {}).items():
            if field in data_dict and data_dict[field] is not None:
                if not isinstance(data_dict[field], eval(expected_type)):
                    warnings.append(f"Field {field} has unexpected type")
        
        # Check value ranges for numeric fields
        for field, (min_val, max_val) in validation_rules.get('ranges', {}).items():
            if field in data_dict and isinstance(data_dict[field], (int, float)):
                if not (min_val <= data_dict[field] <= max_val):
                    warnings.append(f"Field {field} value {data_dict[field]} outside expected range [{min_val}, {max_val}]")
        
        # Check metric consistency
        if study_data.metrics:
            self._validate_metrics(study_data.metrics, errors, warnings)
        
        return {'errors': errors, 'warnings': warnings}
    
    def create_extraction_table(self, 
                               output_format: str = 'dataframe') -> Any:
        """
        Create final extraction table
        
        Args:
            output_format: 'dataframe', 'csv', 'excel', 'json'
        
        Returns:
            Extraction table in specified format
        """
        if not self.extracted_data:
            raise ValueError("No data extracted yet")
        
        # Convert to DataFrame
        df_data = []
        for study in self.extracted_data:
            if isinstance(study, StudyCharacteristics):
                df_data.append(study.to_dict())
            else:
                df_data.append(study)
        
        df = pd.DataFrame(df_data)
        
        # Apply standard column ordering
        column_order = self._get_standard_column_order()
        df = df.reindex(columns=column_order)
        
        # Export based on format
        if output_format == 'dataframe':
            return df
        elif output_format == 'csv':
            output_path = f"extraction_table_{pd.Timestamp.now():%Y%m%d}.csv"
            df.to_csv(output_path, index=False, encoding='utf-8')
            return output_path
        elif output_format == 'excel':
            output_path = f"extraction_table_{pd.Timestamp.now():%Y%m%d}.xlsx"
            df.to_excel(output_path, index=False)
            return output_path
        elif output_format == 'json':
            output_path = f"extraction_table_{pd.Timestamp.now():%Y%m%d}.json"
            df.to_json(output_path, orient='records', indent=2)
            return output_path
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    # Helper methods
    def _extract_metadata(self, pdf_path: str) -> Dict:
        """Extract metadata from PDF"""
        # Implementation would use actual PDF parsing
        pass
    
    def _extract_methods(self, pdf_path: str) -> Dict:
        """Extract methods section"""
        pass
    
    def _extract_results(self, pdf_path: str) -> Dict:
        """Extract results section"""
        pass
    
    def _extract_discussion(self, pdf_path: str) -> Dict:
        """Extract discussion section"""
        pass
    
    def _extract_from_csv(self, csv_data: Dict) -> StudyCharacteristics:
        """Extract from CSV file"""
        pass
    
    def _extract_from_api(self, api_data: Dict) -> StudyCharacteristics:
        """Extract from API response"""
        pass
    
    def _extract_manual(self, manual_data: Dict) -> StudyCharacteristics:
        """Extract from manual entry form"""
        pass
    
    def _normalize_metric_value(self, value: Any, metric_name: str) -> float:
        """Normalize metric value to standard format"""
        try:
            if isinstance(value, str):
                # Remove percentage signs, commas, etc.
                value = re.sub(r'[^\d\.\-]', '', value)
            return float(value)
        except ValueError:
            logger.warning(f"Could not normalize {metric_name} value: {value}")
            return np.nan
    
    def _validate_metrics(self, 
                         metrics: Dict[str, float],
                         errors: List,
                         warnings: List) -> None:
        """Validate metric values"""
        # Check for impossible values
        for metric, value in metrics.items():
            if metric in ['rmse', 'mae', 'mse'] and value < 0:
                errors.append(f"{metric} cannot be negative: {value}")
            elif metric == 'r2' and not (-1 <= value <= 1):
                warnings.append(f"R² value {value} outside typical range [-1, 1]")
            elif metric == 'mape' and value < 0:
                warnings.append(f"MAPE {value} is negative")
    
    def _get_standard_column_order(self) -> List[str]:
        """Get standard column ordering for extraction table"""
        return [
            'study_id', 'authors', 'year', 'title', 'source',
            'qml_approach', 'quantum_level', 'dataset_name',
            'dataset_domain', 'sample_size', 'forecasting_horizon',
            'rmse', 'mae', 'mape', 'r2', 'rmse_ratio',
            'qubits_used', 'quantum_backend', 'encoding_scheme',
            'code_availability', 'data_availability'
        ]
