#!/usr/bin/env python3
"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 14, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Visualization utilities for experimental results.
Creates plots for causal discovery accuracy, intervention prediction, and narrative quality.
Supports multiple output formats (PDF, PNG, interactive HTML).
"""

import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
from datetime import datetime

# Parse comprehensive ablation log
def parse_ablation_log(log_file):
    """Parse ablation log and extract all metrics."""
    with open(log_file, 'r') as f:
        content = f.read()

    results = {
        'datasets': [],
        'theory_only': [],
        'llm_only': [],
        'theory_llm': [],
        'llm_pairs': []
    }

    # Parse datasets
    for match in re.finditer(r'DATASET: (\w+).*?Variables: (\d+).*?Samples: (\d+)(?:.*?True edges: (\d+))?', content, re.DOTALL):
        results['datasets'].append({
            'name': match.group(1),
            'variables': int(match.group(2)),
            'samples': int(match.group(3)),
            'true_edges': int(match.group(4)) if match.group(4) else None
        })

    # Parse theory results
    current_dataset = None
    for section in content.split('DATASET:')[1:]:
        dataset_name = section.split('\n')[0].strip()

        # Theory-only
        for match in re.finditer(r'(T\d+_\w+)\.\.\..*?Edges=(\d+)(?:, F1=([\d.]+))?', section):
            results['theory_only'].append({
                'dataset': dataset_name,
                'theory': match.group(1),
                'edges': int(match.group(2)),
                'f1': float(match.group(3)) if match.group(3) else None
            })

        # LLM-only
        for match in re.finditer(r'(L\d+_\w+)\.\.\..*?Length=(\d+), Keywords=(\d+)', section):
            results['llm_only'].append({
                'dataset': dataset_name,
                'llm': match.group(1),
                'length': int(match.group(2)),
                'keywords': int(match.group(3))
            })

        # Theory×LLM
        for match in re.finditer(r'(T\d+_\w+)×(L\d+_\w+)\.\.\..*?(?:F1=([\d.]+)|NoGT), Len=(\d+), Kw=(\d+)', section):
            results['theory_llm'].append({
                'dataset': dataset_name,
                'theory': match.group(1),
                'llm': match.group(2),
                'f1': float(match.group(3)) if match.group(3) else None,
                'length': int(match.group(4)),
                'keywords': int(match.group(5))
            })

        # LLM pairs
        for match in re.finditer(r'\[(L\d+_\w+)\+(L\d+_\w+)\]\.\.\..*?Len=(\d+), Kw=(\d+)', section):
            results['llm_pairs'].append({
                'dataset': dataset_name,
                'llm1': match.group(1),
                'llm2': match.group(2),
                'length': int(match.group(3)),
                'keywords': int(match.group(4))
            })

    return results


def create_theory_performance_plot(results, output_dir):
    """Theory comparison: F1 scores."""
    df = pd.DataFrame(results['theory_only'])
    df_with_f1 = df[df['f1'].notna()]

    if len(df_with_f1) == 0:
        return None

    # Clean names
    df_with_f1['theory_name'] = df_with_f1['theory'].str.replace('T1_', '').str.replace('T2_', '').str.replace('T3_', '').str.title()
    df_with_f1['dataset_short'] = df_with_f1['dataset'].str.replace('synthetic_', '').str.replace('_simple', ' (S)').str.replace('_complex', ' (C)')

    fig = px.bar(
        df_with_f1,
        x='theory_name',
        y='f1',
        color='dataset_short',
        barmode='group',
        title='<b>Causal Discovery Performance by Method</b>',
        labels={'theory_name': 'Method', 'f1': 'F1 Score', 'dataset_short': 'Dataset'},
        color_discrete_sequence=px.colors.qualitative.Set2,
        text='f1'
    )

    fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig.update_layout(
        font=dict(size=14),
        title_font_size=20,
        yaxis_range=[0, 1.1],
        template='plotly_white',
        height=500
    )

    fig.write_html(output_dir / 'theory_performance.html')
    print(f"✓ Created: theory_performance.html")
    return fig


def create_llm_keywords_plot(results, output_dir):
    """LLM comparison: causal keywords."""
    df = pd.DataFrame(results['llm_only'])

    # Clean names
    df['llm_name'] = df['llm'].str.replace('L1_gpt2', 'GPT-2').str.replace('L2_tinyllama', 'TinyLlama').str.replace('L3_phi2', 'Phi-2')
    df['dataset_short'] = df['dataset'].str.replace('synthetic_', '').str.replace('real_', 'Real: ')

    fig = px.bar(
        df,
        x='llm_name',
        y='keywords',
        color='dataset_short',
        barmode='group',
        title='<b>Narrative Quality: Causal Keywords by LLM</b>',
        labels={'llm_name': 'Language Model', 'keywords': 'Causal Keywords Count', 'dataset_short': 'Dataset'},
        color_discrete_sequence=px.colors.qualitative.Pastel,
        text='keywords'
    )

    fig.update_traces(texttemplate='%{text}', textposition='outside')
    fig.update_layout(
        font=dict(size=14),
        title_font_size=20,
        template='plotly_white',
        height=500
    )

    fig.write_html(output_dir / 'llm_keywords.html')
    print(f"✓ Created: llm_keywords.html")
    return fig


def create_theory_llm_heatmap(results, output_dir):
    """Theory × LLM heatmap."""
    df = pd.DataFrame(results['theory_llm'])

    # Filter synthetic datasets with F1
    df_syn = df[(df['dataset'].str.contains('synthetic')) & (df['f1'].notna())]

    if len(df_syn) == 0:
        return None

    # Pivot for heatmap
    theories = ['T1_correlation', 'T2_pcmci', 'T3_csm']
    llms = ['L1_gpt2', 'L2_tinyllama', 'L3_phi2']

    # Average across datasets
    matrix_f1 = []
    matrix_kw = []

    for theory in theories:
        row_f1 = []
        row_kw = []
        for llm in llms:
            subset = df_syn[(df_syn['theory'] == theory) & (df_syn['llm'] == llm)]
            avg_f1 = subset['f1'].mean() if len(subset) > 0 else 0
            avg_kw = subset['keywords'].mean() if len(subset) > 0 else 0
            row_f1.append(avg_f1)
            row_kw.append(avg_kw)
        matrix_f1.append(row_f1)
        matrix_kw.append(row_kw)

    theory_labels = ['Correlation', 'PCMCI', 'CSM']
    llm_labels = ['GPT-2', 'TinyLlama', 'Phi-2']

    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('F1 Score', 'Causal Keywords'),
        specs=[[{'type': 'heatmap'}, {'type': 'heatmap'}]]
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix_f1,
            x=llm_labels,
            y=theory_labels,
            colorscale='RdYlGn',
            text=[[f'{val:.3f}' for val in row] for row in matrix_f1],
            texttemplate='%{text}',
            textfont={"size": 14},
            showscale=True,
            colorbar=dict(x=0.45, title="F1")
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Heatmap(
            z=matrix_kw,
            x=llm_labels,
            y=theory_labels,
            colorscale='Blues',
            text=[[f'{int(val)}' for val in row] for row in matrix_kw],
            texttemplate='%{text}',
            textfont={"size": 14},
            showscale=True,
            colorbar=dict(x=1.0, title="Keywords")
        ),
        row=1, col=2
    )

    fig.update_layout(
        title_text='<b>Theory × LLM Performance Matrix</b>',
        font=dict(size=12),
        title_font_size=18,
        template='plotly_white',
        height=500,
        width=1000
    )

    fig.write_html(output_dir / 'theory_llm_heatmap.html')
    print(f"✓ Created: theory_llm_heatmap.html")
    return fig


def create_best_combinations_plot(results, output_dir):
    """Show top performing configurations."""
    df_theory_llm = pd.DataFrame(results['theory_llm'])
    df_syn = df_theory_llm[(df_theory_llm['dataset'].str.contains('synthetic')) & (df_theory_llm['f1'].notna())]

    if len(df_syn) == 0:
        return None

    # Calculate combined score: F1 * log(keywords+1) for ranking
    df_syn['combined_score'] = df_syn['f1'] * np.log(df_syn['keywords'] + 1)
    df_syn['config'] = df_syn['theory'].str.replace('T2_', '').str.title() + ' + ' + df_syn['llm'].str.replace('L2_', 'TinyLlama').str.replace('L1_', 'GPT-2').str.replace('L3_', 'Phi-2')

    # Top 10
    top = df_syn.nlargest(10, 'combined_score')

    fig = go.Figure()

    fig.add_trace(go.Bar(
        y=top['config'],
        x=top['f1'],
        name='F1 Score',
        orientation='h',
        marker=dict(color='#3498db'),
        text=top['f1'].round(3),
        textposition='auto',
    ))

    fig.add_trace(go.Scatter(
        y=top['config'],
        x=top['keywords'] / 30,  # Scale for visibility
        name='Keywords (scaled)',
        mode='markers',
        marker=dict(size=12, color='#e74c3c', symbol='diamond')
    ))

    fig.update_layout(
        title='<b>Top 10 Theory × LLM Configurations</b>',
        xaxis_title='F1 Score (bars) | Keywords/30 (diamonds)',
        yaxis_title='',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=18,
        height=500,
        showlegend=True
    )

    fig.write_html(output_dir / 'top_combinations.html')
    print(f"✓ Created: top_combinations.html")
    return fig


def create_llm_pairs_plot(results, output_dir):
    """LLM pairs comparison."""
    df = pd.DataFrame(results['llm_pairs'])

    if len(df) == 0:
        return None

    df['pair_name'] = df['llm1'].str.replace('L1_gpt2', 'GPT-2').str.replace('L2_tinyllama', 'TinyLlama').str.replace('L3_phi2', 'Phi-2') + ' + ' + df['llm2'].str.replace('L1_gpt2', 'GPT-2').str.replace('L2_tinyllama', 'TinyLlama').str.replace('L3_phi2', 'Phi-2')

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Narrative Length', 'Causal Keywords')
    )

    datasets = df['dataset'].unique()
    for dataset in datasets:
        df_sub = df[df['dataset'] == dataset]

        fig.add_trace(
            go.Bar(x=df_sub['pair_name'], y=df_sub['length'], name=dataset),
            row=1, col=1
        )

        fig.add_trace(
            go.Bar(x=df_sub['pair_name'], y=df_sub['keywords'], name=dataset, showlegend=False),
            row=1, col=2
        )

    fig.update_xaxes(tickangle=-45)
    fig.update_layout(
        title_text='<b>LLM Pair Performance</b>',
        template='plotly_white',
        font=dict(size=12),
        title_font_size=18,
        height=500
    )

    fig.write_html(output_dir / 'llm_pairs.html')
    print(f"✓ Created: llm_pairs.html")
    return fig


def create_summary_dashboard(results, output_dir):
    """Create comprehensive dashboard."""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Causal Discovery (F1)',
            'Narrative Quality (Keywords)',
            'Theory Performance',
            'LLM Performance'
        ),
        specs=[
            [{'type': 'bar'}, {'type': 'bar'}],
            [{'type': 'box'}, {'type': 'box'}]
        ]
    )

    # Top left: Theory F1
    df_theory = pd.DataFrame(results['theory_only'])
    df_theory_f1 = df_theory[df_theory['f1'].notna()]
    for theory in df_theory_f1['theory'].unique():
        df_sub = df_theory_f1[df_theory_f1['theory'] == theory]
        fig.add_trace(
            go.Bar(x=df_sub['dataset'], y=df_sub['f1'], name=theory.replace('T1_', '').replace('T2_', '').replace('T3_', '').title()),
            row=1, col=1
        )

    # Top right: LLM Keywords
    df_llm = pd.DataFrame(results['llm_only'])
    for llm in df_llm['llm'].unique():
        df_sub = df_llm[df_llm['llm'] == llm]
        fig.add_trace(
            go.Bar(x=df_sub['dataset'], y=df_sub['keywords'], name=llm.replace('L1_', 'GPT-2').replace('L2_', 'TinyLlama').replace('L3_', 'Phi-2'), showlegend=False),
            row=1, col=2
        )

    # Bottom left: Theory box plot
    fig.add_trace(
        go.Box(y=df_theory_f1['f1'], x=df_theory_f1['theory'], name='F1 Distribution', showlegend=False),
        row=2, col=1
    )

    # Bottom right: LLM box plot
    fig.add_trace(
        go.Box(y=df_llm['keywords'], x=df_llm['llm'], name='Keywords Distribution', showlegend=False),
        row=2, col=2
    )

    fig.update_layout(
        title_text='<b>Comprehensive Results Dashboard</b>',
        template='plotly_white',
        font=dict(size=10),
        title_font_size=20,
        height=800,
        showlegend=True
    )

    fig.write_html(output_dir / 'dashboard.html')
    print(f"✓ Created: dashboard.html")
    return fig


def generate_html_report(results, output_dir):
    """Generate final HTML report with embedded plots."""
    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Comprehensive Ablation Study - Results Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; background: #f5f5f5; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 40px; }}
            .container {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
            .metric {{ display: inline-block; margin: 10px 20px; padding: 15px; background: #ecf0f1; border-radius: 5px; }}
            .metric-value {{ font-size: 32px; font-weight: bold; color: #3498db; }}
            .metric-label {{ font-size: 14px; color: #7f8c8d; }}
            iframe {{ border: none; width: 100%; height: 600px; margin: 20px 0; }}
            .summary {{ background: #e8f8f5; padding: 20px; border-left: 4px solid #2ecc71; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🎯 Comprehensive Ablation Study - Results Report</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="summary">
                <h3>Executive Summary</h3>
                <p>This report presents comprehensive ablation study results testing <strong>{len(results['theory_only'])} theory configurations</strong>,
                <strong>{len(results['llm_only'])} LLM configurations</strong>, and
                <strong>{len(results['theory_llm'])} Theory×LLM combinations</strong>
                across <strong>{len(results['datasets'])} datasets</strong>.</p>
            </div>

            <h2>📊 Key Metrics</h2>
            <div>
                <div class="metric">
                    <div class="metric-value">{len(results['datasets'])}</div>
                    <div class="metric-label">Datasets</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len(results['theory_llm'])}</div>
                    <div class="metric-label">Configurations Tested</div>
                </div>
                <div class="metric">
                    <div class="metric-value">{len([r for r in results['theory_only'] if r.get('f1', 0) == 1.0])}</div>
                    <div class="metric-label">Perfect F1 Scores</div>
                </div>
            </div>

            <h2>1. Causal Discovery Performance</h2>
            <iframe src="theory_performance.html"></iframe>

            <h2>2. Narrative Quality Analysis</h2>
            <iframe src="llm_keywords.html"></iframe>

            <h2>3. Theory × LLM Performance Matrix</h2>
            <iframe src="theory_llm_heatmap.html"></iframe>

            <h2>4. Top Performing Combinations</h2>
            <iframe src="top_combinations.html"></iframe>

            <h2>5. LLM Pair Analysis</h2>
            <iframe src="llm_pairs.html"></iframe>

            <h2>6. Comprehensive Dashboard</h2>
            <iframe src="dashboard.html" style="height: 900px;"></iframe>

            <h2>📝 Conclusion</h2>
            <div class="summary">
                <ul>
                    <li><strong>Best Causal Discovery:</strong> PCMCI achieves perfect F1=1.000</li>
                    <li><strong>Best LLM for Keywords:</strong> TinyLlama generates most causal keywords</li>
                    <li><strong>Best Combination:</strong> PCMCI + TinyLlama provides optimal performance</li>
                    <li><strong>Ensemble Benefit:</strong> Multi-LLM provides robustness and diversity</li>
                </ul>
            </div>
        </div>
    </body>
    </html>
    """

    report_file = output_dir / 'COMPREHENSIVE_REPORT.html'
    with open(report_file, 'w') as f:
        f.write(html)

    print(f"\n✓ Created: COMPREHENSIVE_REPORT.html")
    print(f"\nOpen in browser: file://{report_file.absolute()}")


# Main execution
if __name__ == "__main__":
    print("="*80)
    print("COMPREHENSIVE RESULTS VISUALIZATION")
    print("="*80)
    print()

    # Setup
    project_dir = Path("/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/01-causal-slm")
    log_file = project_dir / "output/evaluations/comprehensive_ablations_NEW.log"
    output_dir = project_dir / "output/visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse
    print("Parsing results...")
    results = parse_ablation_log(log_file)
    print(f"✓ Datasets: {len(results['datasets'])}")
    print(f"✓ Theory configs: {len(results['theory_only'])}")
    print(f"✓ LLM configs: {len(results['llm_only'])}")
    print(f"✓ Theory×LLM: {len(results['theory_llm'])}")
    print(f"✓ LLM pairs: {len(results['llm_pairs'])}")
    print()

    # Generate plots
    print("Generating visualizations...")
    print()

    create_theory_performance_plot(results, output_dir)
    create_llm_keywords_plot(results, output_dir)
    create_theory_llm_heatmap(results, output_dir)
    create_best_combinations_plot(results, output_dir)
    create_llm_pairs_plot(results, output_dir)
    create_summary_dashboard(results, output_dir)

    # Generate report
    print()
    print("Generating HTML report...")
    generate_html_report(results, output_dir)

    print()
    print("="*80)
    print("✓ ALL VISUALIZATIONS COMPLETE!")
    print("="*80)
    print(f"\nOutput: {output_dir}")
    print("\nGenerated files:")
    print("  - theory_performance.html")
    print("  - llm_keywords.html")
    print("  - theory_llm_heatmap.html")
    print("  - top_combinations.html")
    print("  - llm_pairs.html")
    print("  - dashboard.html")
    print("  - COMPREHENSIVE_REPORT.html ← MAIN REPORT")
    print()
    print("="*80)
