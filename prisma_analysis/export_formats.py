"""
Export module for multiple output formats
"""

import pandas as pd
import json
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ExportManager:
    """Manages export to multiple formats"""
    
    def __init__(self, output_dir: str = "exports"):
        """
        Initialize export manager
        
        Args:
            output_dir: Base output directory
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
    
    def export_prisma_flow(self,
                          flow_data: Dict,
                          formats: List[str] = ['json', 'csv', 'html']) -> Dict:
        """
        Export PRISMA flow data
        
        Args:
            flow_data: PRISMA flow data
            formats: List of export formats
        
        Returns:
            Dictionary of exported file paths
        """
        exported = {}
        
        # Create timestamp for filenames
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in formats:
            if fmt == 'json':
                filepath = self.output_dir / f"prisma_flow_{timestamp}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(flow_data, f, indent=2, default=str)
                exported['json'] = str(filepath)
            
            elif fmt == 'csv':
                # Flatten nested structure for CSV
                flat_data = self._flatten_prisma_data(flow_data)
                filepath = self.output_dir / f"prisma_flow_{timestamp}.csv"
                flat_data.to_csv(filepath, index=False)
                exported['csv'] = str(filepath)
            
            elif fmt == 'html':
                filepath = self.output_dir / f"prisma_flow_{timestamp}.html"
                html_content = self._create_prisma_html(flow_data)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                exported['html'] = str(filepath)
            
            elif fmt == 'yaml':
                filepath = self.output_dir / f"prisma_flow_{timestamp}.yaml"
                with open(filepath, 'w', encoding='utf-8') as f:
                    yaml.dump(flow_data, f, default_flow_style=False)
                exported['yaml'] = str(filepath)
        
        logger.info(f"Exported PRISMA flow data in {len(exported)} format(s)")
        return exported
    
    def export_extraction_table(self,
                               table: pd.DataFrame,
                               formats: List[str] = ['csv', 'excel', 'json'],
                               include_metadata: bool = True) -> Dict:
        """
        Export extraction table
        
        Args:
            table: Extraction table DataFrame
            formats: List of export formats
            include_metadata: Whether to include metadata
        
        Returns:
            Dictionary of exported file paths
        """
        exported = {}
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        # Prepare metadata if requested
        if include_metadata:
            metadata = {
                'export_date': timestamp,
                'num_studies': len(table),
                'columns': table.columns.tolist(),
                'summary_stats': self._calculate_table_stats(table)
            }
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = self.output_dir / f"extraction_table_{timestamp}.csv"
                table.to_csv(filepath, index=False, encoding='utf-8')
                exported['csv'] = str(filepath)
                
                # Export metadata separately
                if include_metadata:
                    meta_path = self.output_dir / f"extraction_metadata_{timestamp}.json"
                    with open(meta_path, 'w', encoding='utf-8') as f:
                        json.dump(metadata, f, indent=2)
            
            elif fmt == 'excel':
                filepath = self.output_dir / f"extraction_table_{timestamp}.xlsx"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    table.to_excel(writer, sheet_name='Extraction Data', index=False)
                    
                    if include_metadata:
                        # Add metadata sheet
                        meta_df = pd.DataFrame([metadata])
                        meta_df.to_excel(writer, sheet_name='Metadata', index=False)
                
                exported['excel'] = str(filepath)
            
            elif fmt == 'json':
                filepath = self.output_dir / f"extraction_table_{timestamp}.json"
                # Export as records
                table.to_json(filepath, orient='records', indent=2)
                exported['json'] = str(filepath)
            
            elif fmt == 'latex':
                filepath = self.output_dir / f"extraction_table_{timestamp}.tex"
                latex_content = table.to_latex(
                    index=False,
                    caption='Systematic Review Extraction Table',
                    label='tab:extraction_table',
                    longtable=True
                )
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(latex_content)
                exported['latex'] = str(filepath)
        
        logger.info(f"Exported extraction table in {len(exported)} format(s)")
        return exported
    
    def export_meta_analysis_results(self,
                                    results: Dict,
                                    formats: List[str] = ['json', 'html', 'excel']) -> Dict:
        """
        Export meta-analysis results
        
        Args:
            results: Meta-analysis results dictionary
            formats: List of export formats
        
        Returns:
            Dictionary of exported file paths
        """
        exported = {}
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in formats:
            if fmt == 'json':
                filepath = self.output_dir / f"meta_analysis_{timestamp}.json"
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, default=self._json_serializer)
                exported['json'] = str(filepath)
            
            elif fmt == 'html':
                filepath = self.output_dir / f"meta_analysis_report_{timestamp}.html"
                html_content = self._create_meta_analysis_html(results)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                exported['html'] = str(filepath)
            
            elif fmt == 'excel':
                filepath = self.output_dir / f"meta_analysis_{timestamp}.xlsx"
                with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                    # Export each result type to separate sheet
                    for result_type, result_data in results.items():
                        if isinstance(result_data, dict):
                            # Flatten nested dictionaries
                            flat_data = self._flatten_dict(result_data)
                            df = pd.DataFrame([flat_data])
                            df.to_excel(writer, sheet_name=result_type[:31], index=False)
                
                exported['excel'] = str(filepath)
            
            elif fmt == 'r':
                # Export for R metafor package
                filepath = self.output_dir / f"meta_analysis_r_{timestamp}.csv"
                r_data = self._prepare_r_export(results)
                r_data.to_csv(filepath, index=False)
                exported['r'] = str(filepath)
        
        logger.info(f"Exported meta-analysis results in {len(exported)} format(s)")
        return exported
    
    def export_quality_assessment(self,
                                 assessment_data: pd.DataFrame,
                                 formats: List[str] = ['csv', 'html', 'json']) -> Dict:
        """
        Export quality assessment results
        
        Args:
            assessment_data: Quality assessment DataFrame
            formats: List of export formats
        
        Returns:
            Dictionary of exported file paths
        """
        exported = {}
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        for fmt in formats:
            if fmt == 'csv':
                filepath = self.output_dir / f"quality_assessment_{timestamp}.csv"
                assessment_data.to_csv(filepath, index=False)
                exported['csv'] = str(filepath)
            
            elif fmt == 'html':
                filepath = self.output_dir / f"quality_assessment_{timestamp}.html"
                html_content = self._create_quality_html(assessment_data)
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(html_content)
                exported['html'] = str(filepath)
            
            elif fmt == 'json':
                filepath = self.output_dir / f"quality_assessment_{timestamp}.json"
                assessment_data.to_json(filepath, orient='records', indent=2)
                exported['json'] = str(filepath)
            
            elif fmt == 'risk_of_bias_table':
                # Create risk of bias summary table
                filepath = self.output_dir / f"risk_of_bias_{timestamp}.csv"
                risk_table = self._create_risk_table(assessment_data)
                risk_table.to_csv(filepath, index=True)
                exported['risk_table'] = str(filepath)
        
        return exported
    
    def export_comprehensive_report(self,
                                   prisma_data: Dict,
                                   extraction_table: pd.DataFrame,
                                   meta_results: Dict,
                                   quality_data: pd.DataFrame,
                                   output_format: str = 'html') -> str:
        """
        Export comprehensive systematic review report
        
        Args:
            prisma_data: PRISMA flow data
            extraction_table: Extraction table
            meta_results: Meta-analysis results
            quality_data: Quality assessment data
            output_format: 'html' or 'pdf'
        
        Returns:
            Path to exported report
        """
        timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
        
        if output_format == 'html':
            filepath = self.output_dir / f"systematic_review_report_{timestamp}.html"
            html_content = self._create_comprehensive_html(
                prisma_data, extraction_table, meta_results, quality_data
            )
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Comprehensive report exported to: {filepath}")
            return str(filepath)
        
        elif output_format == 'pdf':
            # Would require additional libraries like weasyprint or pdfkit
            logger.warning("PDF export requires additional dependencies")
            return None
        
        else:
            raise ValueError(f"Unsupported report format: {output_format}")
    
    # Helper methods
    def _flatten_prisma_data(self, flow_data: Dict) -> pd.DataFrame:
        """Flatten nested PRISMA data for CSV export"""
        flat_records = []
        
        for phase, phase_data in flow_data.items():
            record = {'phase': phase}
            
            for key, value in phase_data.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        record[f"{key}_{subkey}"] = subvalue
                else:
                    record[key] = value
            
            flat_records.append(record)
        
        return pd.DataFrame(flat_records)
    
    def _create_prisma_html(self, flow_data: Dict) -> str:
        """Create HTML representation of PRISMA flow"""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>PRISMA 2020 Flow Diagram</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .prisma-container { display: flex; flex-direction: column; align-items: center; }
                .phase { border: 2px solid #333; border-radius: 5px; padding: 15px; margin: 10px; width: 80%; }
                .identification { background-color: #e3f2fd; }
                .screening { background-color: #e8f5e9; }
                .eligibility { background-color: #fff3e0; }
                .included { background-color: #ffebee; }
                .arrow { text-align: center; font-size: 24px; margin: 5px; }
                h2 { color: #333; }
                table { width: 100%; border-collapse: collapse; margin-top: 10px; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="prisma-container">
                <h1>PRISMA 2020 Flow Diagram</h1>
        """
        
        phases = {
            'identification': 'Identification',
            'screening': 'Screening',
            'eligibility': 'Eligibility',
            'included': 'Included'
        }
        
        for phase_key, phase_name in phases.items():
            phase_data = flow_data.get(phase_key, {})
            
            html += f"""
                <div class="phase {phase_key}">
                    <h2>{phase_name}</h2>
                    <table>
            """
            
            for key, value in phase_data.items():
                if isinstance(value, dict):
                    html += f'<tr><th colspan="2">{key}</th></tr>'
                    for subkey, subvalue in value.items():
                        html += f'<tr><td style="padding-left: 20px;">{subkey}</td><td>{subvalue}</td></tr>'
                else:
                    html += f'<tr><td>{key}</td><td>{value}</td></tr>'
            
            html += """
                    </table>
                </div>
            """
            
            if phase_key != 'included':
                html += '<div class="arrow">↓</div>'
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _calculate_table_stats(self, table: pd.DataFrame) -> Dict:
        """Calculate summary statistics for extraction table"""
        stats = {
            'total_studies': len(table),
            'years_covered': f"{table['year'].min()} - {table['year'].max()}",
            'methods_distribution': table['qml_approach'].value_counts().to_dict(),
            'avg_rmse_ratio': table['rmse_ratio'].mean() if 'rmse_ratio' in table.columns else None,
            'domains_covered': table['dataset_domain'].unique().tolist() if 'dataset_domain' in table.columns else []
        }
        
        return stats
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for non-serializable objects"""
        if isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return str(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif pd.isna(obj):
            return None
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
    
    def _create_meta_analysis_html(self, results: Dict) -> str:
        """Create HTML report for meta-analysis results"""
        # Simplified implementation
        return f"""
        <html>
        <body>
            <h1>Meta-Analysis Results</h1>
            <pre>{json.dumps(results, indent=2, default=self._json_serializer)}</pre>
        </body>
        </html>
        """
    
    def _flatten_dict(self, data: Dict, parent_key: str = '', sep: str = '_') -> Dict:
        """Flatten nested dictionary"""
        items = []
        for k, v in data.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def _prepare_r_export(self, results: Dict) -> pd.DataFrame:
        """Prepare data for R metafor package"""
        # Extract effect sizes for R
        if 'effect_sizes' in results and 'data' in results['effect_sizes']:
            effect_data = results['effect_sizes']['data']
            r_df = pd.DataFrame({
                'study': effect_data['study_id'],
                'yi': effect_data['effect_size'],  # Effect size
                'vi': effect_data['variance'],      # Variance
                'n1': effect_data.get('sample_size_qml', 30),
                'n2': effect_data.get('sample_size_baseline', 30)
            })
            return r_df
        return pd.DataFrame()
    
    def _create_quality_html(self, assessment_data: pd.DataFrame) -> str:
        """Create HTML report for quality assessment"""
        # Simplified implementation
        return assessment_data.to_html(index=False)
    
    def _create_risk_table(self, assessment_data: pd.DataFrame) -> pd.DataFrame:
        """Create risk of bias summary table"""
        # Count studies by risk level for each domain
        risk_columns = [col for col in assessment_data.columns 
                       if col.startswith('D') and col != 'Domain']
        
        risk_summary = pd.DataFrame(index=['Low', 'Some Concerns', 'High'])
        
        for domain in risk_columns:
            counts = assessment_data[domain].value_counts()
            for level in ['Low', 'Some Concerns', 'High']:
                risk_summary.loc[level, domain] = counts.get(level, 0)
        
        return risk_summary
    
    def _create_comprehensive_html(self,
                                  prisma_data: Dict,
                                  extraction_table: pd.DataFrame,
                                  meta_results: Dict,
                                  quality_data: pd.DataFrame) -> str:
        """Create comprehensive HTML report"""
        # This would be a full report with all sections
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Systematic Review Report - QML for Time Series Forecasting</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }
                .section { margin-bottom: 40px; padding: 20px; border: 1px solid #ddd; border-radius: 5px; }
                h1, h2, h3 { color: #2c3e50; }
                table { width: 100%; border-collapse: collapse; margin: 10px 0; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .highlight { background-color: #e8f4f8; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Systematic Review Report</h1>
            <h2>Quantum Machine Learning for Time Series Forecasting</h2>
            
            <div class="section">
                <h3>1. PRISMA Flow Summary</h3>
                <div class="highlight">
        """
        
        # Add PRISMA summary
        for phase, data in prisma_data.items():
            html += f"<h4>{phase.title()}</h4><ul>"
            for key, value in data.items():
                if not isinstance(value, dict):
                    html += f"<li><strong>{key}:</strong> {value}</li>"
            html += "</ul>"
        
        html += """
                </div>
            </div>
            
            <div class="section">
                <h3>2. Included Studies Summary</h3>
                <p>Total studies: {}</p>
        """.format(len(extraction_table))
        
        # Add study table (first 10 rows for readability)
        html += extraction_table.head(10).to_html(index=False)
        
        html += """
            </div>
            
            <div class="section">
                <h3>3. Meta-Analysis Results</h3>
        """
        
        # Add meta-analysis summary
        if 'random_effects' in meta_results:
            re = meta_results['random_effects']
            html += f"""
                <div class="highlight">
                    <p><strong>Random Effects Model:</strong></p>
                    <ul>
                        <li>Pooled Effect: {re.get('pooled_effect', 'N/A'):.3f}</li>
                        <li>95% CI: [{re.get('ci_95_lower', 'N/A'):.3f}, {re.get('ci_95_upper', 'N/A'):.3f}]</li>
                        <li>I²: {re.get('i_squared', 'N/A'):.1f}%</li>
                        <li>p-value: {re.get('p_value', 'N/A'):.4f}</li>
                    </ul>
                </div>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h3>4. Quality Assessment</h3>
        """
        
        # Add quality assessment summary
        if not quality_data.empty:
            html += quality_data.to_html(index=False)
        
        html += """
            </div>
            
            <div class="section">
                <h3>5. Conclusions</h3>
                <p>Based on the systematic review of {} studies, quantum machine learning 
                approaches show promising results for time series forecasting, with an 
                average RMSE ratio of {:.2f} compared to classical baselines.</p>
            </div>
        </body>
        </html>
        """.format(
            len(extraction_table),
            extraction_table['rmse_ratio'].mean() if 'rmse_ratio' in extraction_table.columns else 'N/A'
        )
        
        return html
