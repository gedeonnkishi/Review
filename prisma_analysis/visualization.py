"""
Visualization module for PRISMA diagrams and meta-analysis plots
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class PRISMAVisualizer:
    """Creates PRISMA 2020 flow diagrams and other visualizations"""
    
    def __init__(self, flow_data: Dict):
        """
        Initialize visualizer
        
        Args:
            flow_data: PRISMA flow data from selection process
        """
        self.flow_data = flow_data
        self.figures = {}
    
    def create_prisma_flow_diagram(self,
                                  style: str = 'classic',
                                  output_path: str = None,
                                  dpi: int = 300) -> plt.Figure:
        """
        Create PRISMA 2020 flow diagram
        
        Args:
            style: 'classic' or 'modern'
            output_path: Path to save figure
            dpi: Resolution for saved figure
        
        Returns:
            Matplotlib figure
        """
        if style == 'classic':
            return self._create_classic_prisma_diagram(output_path, dpi)
        elif style == 'modern':
            return self._create_modern_prisma_diagram(output_path, dpi)
        else:
            raise ValueError(f"Unknown style: {style}")
    
    def create_interactive_prisma(self,
                                 output_path: str = 'prisma_flow.html') -> go.Figure:
        """
        Create interactive PRISMA diagram using Plotly
        
        Args:
            output_path: Path to save HTML file
        
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=1, cols=1,
            subplot_titles=('PRISMA 2020 Flow Diagram',)
        )
        
        # Define positions for each box
        positions = {
            'identification': {'x': 0.5, 'y': 0.9, 'width': 0.4, 'height': 0.1},
            'screening': {'x': 0.5, 'y': 0.7, 'width': 0.4, 'height': 0.1},
            'eligibility': {'x': 0.5, 'y': 0.5, 'width': 0.4, 'height': 0.1},
            'included': {'x': 0.5, 'y': 0.3, 'width': 0.4, 'height': 0.1}
        }
        
        # Add boxes
        colors = {
            'identification': 'lightblue',
            'screening': 'lightgreen',
            'eligibility': 'lightyellow',
            'included': 'lightcoral'
        }
        
        for phase, pos in positions.items():
            fig.add_shape(
                type="rect",
                x0=pos['x'] - pos['width']/2,
                y0=pos['y'] - pos['height']/2,
                x1=pos['x'] + pos['width']/2,
                y1=pos['y'] + pos['height']/2,
                fillcolor=colors[phase],
                line=dict(color="black", width=2),
                opacity=0.7
            )
            
            # Add text
            phase_data = self.flow_data.get(phase, {})
            text = f"<b>{phase.title()}</b><br>"
            for key, value in phase_data.items():
                if isinstance(value, dict):
                    continue
                text += f"{key}: {value}<br>"
            
            fig.add_annotation(
                x=pos['x'],
                y=pos['y'],
                text=text,
                showarrow=False,
                font=dict(size=10)
            )
        
        # Add connecting arrows
        arrows = [
            ('identification', 'screening'),
            ('screening', 'eligibility'),
            ('eligibility', 'included')
        ]
        
        for start, end in arrows:
            fig.add_annotation(
                ax=positions[start]['x'],
                ay=positions[start]['y'] - positions[start]['height']/2,
                axref="x",
                ayref="y",
                x=positions[end]['x'],
                y=positions[end]['y'] + positions[end]['height']/2,
                xref="x",
                yref="y",
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=2,
                arrowcolor="black"
            )
        
        # Update layout
        fig.update_layout(
            title="PRISMA 2020 Flow Diagram - Interactive",
            showlegend=False,
            width=800,
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        # Save interactive version
        if output_path:
            fig.write_html(output_path)
        
        self.figures['interactive_prisma'] = fig
        return fig
    
    def create_performance_comparison_plot(self,
                                          meta_results: Dict,
                                          output_path: str = None) -> plt.Figure:
        """
        Create performance comparison plot
        
        Args:
            meta_results: Meta-analysis results
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Effect sizes by study
        ax1 = axes[0, 0]
        if 'effect_sizes' in meta_results:
            effect_data = meta_results['effect_sizes']['data']
            ax1.errorbar(
                effect_data['effect_size'],
                range(len(effect_data)),
                xerr=effect_data['se'] * 1.96,
                fmt='o',
                capsize=5
            )
            ax1.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax1.set_xlabel('Effect Size (Hedges\' g)')
            ax1.set_ylabel('Study')
            ax1.set_title('Effect Sizes by Study')
        
        # 2. Subgroup analysis
        ax2 = axes[0, 1]
        subgroup_keys = [k for k in meta_results.keys() if k.startswith('subgroup_')]
        if subgroup_keys:
            for key in subgroup_keys[:3]:  # Plot first 3 subgroups
                subgroup = meta_results[key]['subgroups']
                for name, data in subgroup.items():
                    ax2.errorbar(
                        data['pooled_effect'],
                        name,
                        xerr=[data['pooled_effect'] - data['ci_95_lower'],
                              data['ci_95_upper'] - data['pooled_effect']],
                        fmt='o',
                        label=name
                    )
            ax2.axvline(0, color='red', linestyle='--', alpha=0.5)
            ax2.set_xlabel('Effect Size')
            ax2.set_title('Subgroup Analysis')
            ax2.legend()
        
        # 3. Publication bias funnel plot
        ax3 = axes[1, 0]
        if 'publication_bias' in meta_results:
            bias_data = meta_results['publication_bias'].get('funnel', {})
            if bias_data:
                ax3.scatter(
                    bias_data['effect_sizes'],
                    bias_data['precision'],
                    alpha=0.6
                )
                ax3.axvline(0, color='red', linestyle='--', alpha=0.5)
                ax3.set_xlabel('Effect Size')
                ax3.set_ylabel('Precision (1/SE)')
                ax3.set_title('Funnel Plot for Publication Bias')
        
        # 4. Sensitivity analysis
        ax4 = axes[1, 1]
        sensitivity_keys = [k for k in meta_results.keys() if 'sensitivity' in k]
        if sensitivity_keys:
            for key in sensitivity_keys:
                sens_data = meta_results[key]
                if 'results' in sens_data:
                    effects = [r['pooled_effect'] for r in sens_data['results']]
                    studies = [r['omitted_study'] for r in sens_data['results']]
                    ax4.plot(studies, effects, 'o-', alpha=0.7)
            
            ax4.axhline(
                meta_results.get('random_effects', {}).get('pooled_effect', 0),
                color='red',
                linestyle='--',
                label='Overall Effect'
            )
            ax4.set_xlabel('Omitted Study')
            ax4.set_ylabel('Pooled Effect')
            ax4.set_title('Leave-One-Out Sensitivity Analysis')
            ax4.tick_params(axis='x', rotation=45)
            ax4.legend()
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.figures['performance_comparison'] = fig
        return fig
    
    def create_temporal_trend_plot(self,
                                  extraction_table: pd.DataFrame,
                                  output_path: str = None) -> plt.Figure:
        """
        Create temporal trend plot of studies over years
        
        Args:
            extraction_table: DataFrame with study data
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Studies per year
        ax1 = axes[0, 0]
        yearly_counts = extraction_table['year'].value_counts().sort_index()
        ax1.bar(yearly_counts.index, yearly_counts.values)
        ax1.set_xlabel('Year')
        ax1.set_ylabel('Number of Studies')
        ax1.set_title('Studies Published per Year')
        ax1.tick_params(axis='x', rotation=45)
        
        # 2. Cumulative studies
        ax2 = axes[0, 1]
        cumulative = yearly_counts.sort_index().cumsum()
        ax2.plot(cumulative.index, cumulative.values, 'o-', linewidth=2)
        ax2.set_xlabel('Year')
        ax2.set_ylabel('Cumulative Number of Studies')
        ax2.set_title('Cumulative Growth of QML-TSF Research')
        ax2.grid(True, alpha=0.3)
        
        # 3. Methods evolution
        ax3 = axes[1, 0]
        if 'qml_approach' in extraction_table.columns:
            method_years = extraction_table.groupby(['year', 'qml_approach']).size().unstack()
            method_years.plot(kind='area', ax=ax3, alpha=0.7)
            ax3.set_xlabel('Year')
            ax3.set_ylabel('Number of Studies')
            ax3.set_title('Evolution of QML Approaches')
            ax3.legend(title='Method')
        
        # 4. Performance trends
        ax4 = axes[1, 1]
        if 'rmse_ratio' in extraction_table.columns:
            yearly_perf = extraction_table.groupby('year')['rmse_ratio'].agg(['mean', 'std', 'count'])
            ax4.errorbar(
                yearly_perf.index,
                yearly_perf['mean'],
                yerr=yearly_perf['std'],
                fmt='o-',
                capsize=5
            )
            ax4.axhline(1.0, color='red', linestyle='--', alpha=0.5, label='Baseline')
            ax4.set_xlabel('Year')
            ax4.set_ylabel('RMSE Ratio (QML/Classical)')
            ax4.set_title('Performance Trends Over Time')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
        
        self.figures['temporal_trends'] = fig
        return fig
    
    def create_network_visualization(self,
                                    extraction_table: pd.DataFrame,
                                    output_path: str = None) -> plt.Figure:
        """
        Create network visualization of study relationships
        
        Args:
            extraction_table: DataFrame with study data
            output_path: Path to save figure
        
        Returns:
            Matplotlib figure
        """
        try:
            import networkx as nx
            
            # Create graph
            G = nx.Graph()
            
            # Add nodes (studies)
            for _, row in extraction_table.iterrows():
                G.add_node(
                    row['study_id'],
                    year=row.get('year'),
                    method=row.get('qml_approach'),
                    performance=row.get('rmse_ratio', 0)
                )
            
            # Add edges based on shared characteristics
            # This is a simplified example - in practice would use citation data
            methods = extraction_table['qml_approach'].unique()
            for method in methods:
                method_studies = extraction_table[
                    extraction_table['qml_approach'] == method
                ]['study_id'].tolist()
                
                # Connect studies using same method
                for i in range(len(method_studies)):
                    for j in range(i+1, len(method_studies)):
                        G.add_edge(method_studies[i], method_studies[j], weight=0.5)
            
            # Create layout
            pos = nx.spring_layout(G, seed=42)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 10))
            
            # Draw nodes
            node_colors = []
            node_sizes = []
            
            for node in G.nodes():
                node_data = G.nodes[node]
                # Color by method
                method = node_data.get('method', 'Unknown')
                if method == 'QNN':
                    node_colors.append('red')
                elif method == 'QELM':
                    node_colors.append('blue')
                elif method == 'QRC':
                    node_colors.append('green')
                else:
                    node_colors.append('gray')
                
                # Size by performance (inverse of RMSE ratio)
                perf = node_data.get('performance', 1)
                node_sizes.append(300 * (2 - min(perf, 1.5)))
            
            nx.draw_networkx_nodes(
                G, pos,
                node_color=node_colors,
                node_size=node_sizes,
                alpha=0.7,
                ax=ax
            )
            
            # Draw edges
            nx.draw_networkx_edges(
                G, pos,
                alpha=0.2,
                width=0.5,
                ax=ax
            )
            
            # Draw labels
            nx.draw_networkx_labels(
                G, pos,
                font_size=8,
                ax=ax
            )
            
            ax.set_title('Network Visualization of QML-TSF Studies')
            ax.axis('off')
            
            # Add legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='red', alpha=0.7, label='QNN'),
                Patch(facecolor='blue', alpha=0.7, label='QELM'),
                Patch(facecolor='green', alpha=0.7, label='QRC'),
                Patch(facecolor='gray', alpha=0.7, label='Other')
            ]
            ax.legend(handles=legend_elements, loc='upper right')
            
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
            
            self.figures['network'] = fig
            return fig
            
        except ImportError:
            print("NetworkX not installed. Install with: pip install networkx")
            return None
    
    # Helper methods for PRISMA diagrams
    def _create_classic_prisma_diagram(self,
                                      output_path: str = None,
                                      dpi: int = 300) -> plt.Figure:
        """Create classic PRISMA flow diagram"""
        fig, ax = plt.subplots(figsize=(10, 12))
        
        # Define box positions and sizes
        boxes = {
            'identification': {
                'xy': (0.3, 0.9),
                'width': 0.4,
                'height': 0.08,
                'color': 'lightblue'
            },
            'screening': {
                'xy': (0.3, 0.7),
                'width': 0.4,
                'height': 0.08,
                'color': 'lightgreen'
            },
            'eligibility': {
                'xy': (0.3, 0.5),
                'width': 0.4,
                'height': 0.08,
                'color': 'lightyellow'
            },
            'included': {
                'xy': (0.3, 0.3),
                'width': 0.4,
                'height': 0.08,
                'color': 'lightcoral'
            }
        }
        
        # Draw boxes
        for phase, box in boxes.items():
            rect = patches.Rectangle(
                box['xy'],
                box['width'],
                box['height'],
                linewidth=2,
                edgecolor='black',
                facecolor=box['color'],
                alpha=0.8
            )
            ax.add_patch(rect)
            
            # Add text
            phase_data = self.flow_data.get(phase, {})
            text = f"{phase.title()}\n"
            for key, value in phase_data.items():
                if not isinstance(value, dict):
                    text += f"{key}: {value}\n"
            
            ax.text(
                box['xy'][0] + box['width']/2,
                box['xy'][1] + box['height']/2,
                text,
                ha='center',
                va='center',
                fontsize=9
            )
        
        # Draw arrows
        arrows = [
            (boxes['identification'], boxes['screening']),
            (boxes['screening'], boxes['eligibility']),
            (boxes['eligibility'], boxes['included'])
        ]
        
        for start, end in arrows:
            ax.annotate(
                '',
                xy=(end['xy'][0] + end['width']/2, end['xy'][1] + end['height']),
                xytext=(start['xy'][0] + start['width']/2, start['xy'][1]),
                arrowprops=dict(
                    arrowstyle='->',
                    lw=2,
                    color='black'
                )
            )
        
        # Set limits and remove axes
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        ax.set_title('PRISMA 2020 Flow Diagram', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        
        self.figures['classic_prisma'] = fig
        return fig
    
    def _create_modern_prisma_diagram(self,
                                     output_path: str = None,
                                     dpi: int = 300) -> plt.Figure:
        """Create modern styled PRISMA diagram"""
        # Similar to classic but with rounded corners, shadows, etc.
        # Implementation would be more elaborate
        return self._create_classic_prisma_diagram(output_path, dpi)
    
    def export_all_figures(self,
                          output_dir: str = 'figures',
                          formats: List[str] = ['png', 'pdf']) -> Dict:
        """
        Export all generated figures
        
        Args:
            output_dir: Directory to save figures
            formats: List of formats to export
        
        Returns:
            Dictionary of saved file paths
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        saved_files = {}
        
        for fig_name, fig in self.figures.items():
            for fmt in formats:
                if fmt == 'html' and hasattr(fig, 'write_html'):
                    filepath = os.path.join(output_dir, f"{fig_name}.html")
                    fig.write_html(filepath)
                    saved_files.setdefault(fig_name, []).append(filepath)
                elif isinstance(fig, plt.Figure):
                    filepath = os.path.join(output_dir, f"{fig_name}.{fmt}")
                    fig.savefig(filepath, dpi=300, bbox_inches='tight')
                    saved_files.setdefault(fig_name, []).append(filepath)
        
        return saved_files
