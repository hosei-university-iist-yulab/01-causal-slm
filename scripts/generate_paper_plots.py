#!/usr/bin/env python3
"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 14, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Generates all plots and figures for paper submission.
Creates publication-quality PDF and PNG outputs.
Includes performance comparisons, ablation results, and architecture diagrams.
"""

import json
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path
import re
from datetime import datetime

# Parse log file to extract results
def parse_ablation_log(log_file):
    """Parse comprehensive ablation log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {
        'datasets': [],
        'ablations': {}
    }
    
    # Extract dataset sections
    dataset_pattern = r'DATASET: (\w+)\n.*?Variables: (\d+).*?Samples: (\d+)'
    for match in re.finditer(dataset_pattern, content, re.DOTALL):
        results['datasets'].append({
            'name': match.group(1),
            'variables': int(match.group(2)),
            'samples': int(match.group(3))
        })
    
    # Extract theory results (Part 1.1)
    theory_pattern = r'(T\d+_\w+)\.\.\..*?Edges=(\d+)(?:, F1=([\d.]+))?'
    for match in re.finditer(theory_pattern, content):
        theory = match.group(1)
        edges = int(match.group(2))
        f1 = float(match.group(3)) if match.group(3) else None
        
        if theory not in results['ablations']:
            results['ablations'][theory] = []
        results['ablations'][theory].append({
            'edges': edges,
            'f1': f1
        })
    
    # Extract LLM results (Part 1.2)
    llm_pattern = r'(L\d+_\w+)\.\.\..*?Length=(\d+), Keywords=(\d+)'
    for match in re.finditer(llm_pattern, content):
        llm = match.group(1)
        length = int(match.group(2))
        keywords = int(match.group(3))
        
        if llm not in results['ablations']:
            results['ablations'][llm] = []
        results['ablations'][llm].append({
            'length': length,
            'keywords': keywords
        })
    
    # Extract cross-testing results (Part 2)
    cross_pattern = r'(T\d+_\w+)×(L\d+_\w+)\.\.\..*?(?:F1=([\d.]+)|NoGT), Len=(\d+), Kw=(\d+)'
    for match in re.finditer(cross_pattern, content):
        combo = f"{match.group(1)}×{match.group(2)}"
        f1 = float(match.group(3)) if match.group(3) else None
        length = int(match.group(4))
        keywords = int(match.group(5))
        
        if combo not in results['ablations']:
            results['ablations'][combo] = []
        results['ablations'][combo].append({
            'f1': f1,
            'length': length,
            'keywords': keywords
        })
    
    # Extract LLM pair results (Part 3.2)
    pair_pattern = r'\[(L\d+_\w+)\+(L\d+_\w+)\]\.\.\..*?Len=(\d+), Kw=(\d+)'
    for match in re.finditer(pair_pattern, content):
        pair = f"[{match.group(1)}+{match.group(2)}]"
        length = int(match.group(3))
        keywords = int(match.group(4))
        
        if pair not in results['ablations']:
            results['ablations'][pair] = []
        results['ablations'][pair].append({
            'length': length,
            'keywords': keywords
        })
    
    return results

# Create visualizations
def create_f1_comparison(results, output_dir):
    """Create F1 score comparison bar chart."""
    # Extract F1 scores for theories
    theories = ['T1_correlation', 'T2_pcmci', 'T3_csm']
    datasets = results['datasets']
    
    data = []
    for theory in theories:
        if theory in results['ablations']:
            for i, result in enumerate(results['ablations'][theory]):
                if result.get('f1') is not None:
                    data.append({
                        'Theory': theory.replace('_', ' ').title(),
                        'Dataset': datasets[i]['name'] if i < len(datasets) else f'Dataset {i+1}',
                        'F1 Score': result['f1']
                    })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='Theory',
        y='F1 Score',
        color='Dataset',
        barmode='group',
        title='Causal Discovery Performance: F1 Scores by Method',
        labels={'F1 Score': 'F1 Score (higher is better)'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        font=dict(size=14),
        title_font_size=18,
        yaxis_range=[0, 1.1],
        template='plotly_white'
    )
    
    # Save
    fig.write_html(output_dir / 'f1_comparison.html')
    #fig.write_image(output_dir / 'f1_comparison.png', width=1000, height=600)
    
    return fig

def create_keyword_comparison(results, output_dir):
    """Create causal keywords comparison chart."""
    # Extract keyword counts for LLMs
    llms = ['L1_gpt2', 'L2_tinyllama', 'L3_phi2']
    datasets = results['datasets']
    
    data = []
    for llm in llms:
        if llm in results['ablations']:
            for i, result in enumerate(results['ablations'][llm]):
                data.append({
                    'LLM': llm.replace('_', ' ').title().replace('L1', 'GPT-2').replace('L2', 'TinyLlama').replace('L3', 'Phi-2'),
                    'Dataset': datasets[i]['name'] if i < len(datasets) else f'Dataset {i+1}',
                    'Causal Keywords': result.get('keywords', 0)
                })
    
    df = pd.DataFrame(data)
    
    fig = px.bar(
        df,
        x='LLM',
        y='Causal Keywords',
        color='Dataset',
        barmode='group',
        title='Narrative Quality: Causal Keywords by LLM',
        labels={'Causal Keywords': 'Number of Causal Keywords'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    
    fig.update_layout(
        font=dict(size=14),
        title_font_size=18,
        template='plotly_white'
    )
    
    # Save
    fig.write_html(output_dir / 'keywords_comparison.html')
    #fig.write_image(output_dir / 'keywords_comparison.png', width=1000, height=600)
    
    return fig

def create_theory_llm_heatmap(results, output_dir):
    """Create heatmap of Theory × LLM combinations."""
    theories = ['T1_correlation', 'T2_pcmci', 'T3_csm']
    llms = ['L1_gpt2', 'L2_tinyllama', 'L3_phi2']
    
    # Average F1 scores across datasets
    f1_matrix = []
    for theory in theories:
        row = []
        for llm in llms:
            combo = f"{theory}×{llm}"
            if combo in results['ablations']:
                f1_values = [r.get('f1', 0) for r in results['ablations'][combo] if r.get('f1') is not None]
                avg_f1 = sum(f1_values) / len(f1_values) if f1_values else 0
                row.append(avg_f1)
            else:
                row.append(0)
        f1_matrix.append(row)
    
    theory_labels = ['Correlation', 'PCMCI', 'CSM']
    llm_labels = ['GPT-2', 'TinyLlama', 'Phi-2']
    
    fig = go.Figure(data=go.Heatmap(
        z=f1_matrix,
        x=llm_labels,
        y=theory_labels,
        colorscale='RdYlGn',
        text=[[f'{val:.3f}' for val in row] for row in f1_matrix],
        texttemplate='%{text}',
        textfont={"size": 16},
        colorbar=dict(title="F1 Score")
    ))
    
    fig.update_layout(
        title='Theory × LLM Performance Matrix (Average F1 Score)',
        xaxis_title='Language Model',
        yaxis_title='Causal Discovery Method',
        font=dict(size=14),
        title_font_size=18,
        template='plotly_white'
    )
    
    # Save
    fig.write_html(output_dir / 'theory_llm_heatmap.html')
    #fig.write_image(output_dir / 'theory_llm_heatmap.png', width=800, height=600)
    
    return fig

def create_narrative_length_chart(results, output_dir):
    """Create narrative length comparison."""
    # LLM pairs
    pairs = [k for k in results['ablations'].keys() if '[L' in k]
    
    data = []
    for pair in pairs:
        for i, result in enumerate(results['ablations'][pair]):
            data.append({
                'Configuration': pair.replace('[', '').replace(']', '').replace('_gpt2', ' GPT-2').replace('_tinyllama', ' TinyLlama').replace('_phi2', ' Phi-2').replace('L1', '').replace('L2', '').replace('L3', ''),
                'Dataset': f'Dataset {i+1}',
                'Length': result.get('length', 0),
                'Keywords': result.get('keywords', 0)
            })
    
    if not data:
        return None
    
    df = pd.DataFrame(data)
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Narrative Length', 'Causal Keywords'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    for dataset in df['Dataset'].unique():
        df_subset = df[df['Dataset'] == dataset]
        fig.add_trace(
            go.Bar(
                x=df_subset['Configuration'],
                y=df_subset['Length'],
                name=dataset,
                showlegend=True
            ),
            row=1, col=1
        )
        fig.add_trace(
            go.Bar(
                x=df_subset['Configuration'],
                y=df_subset['Keywords'],
                name=dataset,
                showlegend=False
            ),
            row=1, col=2
        )
    
    fig.update_xaxes(title_text="LLM Pair", row=1, col=1)
    fig.update_xaxes(title_text="LLM Pair", row=1, col=2)
    fig.update_yaxes(title_text="Characters", row=1, col=1)
    fig.update_yaxes(title_text="Keyword Count", row=1, col=2)
    
    fig.update_layout(
        title_text='LLM Pair Performance: Narrative Quality Metrics',
        font=dict(size=12),
        title_font_size=16,
        template='plotly_white',
        height=500
    )
    
    # Save
    fig.write_html(output_dir / 'llm_pairs_comparison.html')
    #fig.write_image(output_dir / 'llm_pairs_comparison.png', width=1200, height=500)
    
    return fig

def generate_latex_table(results, output_file):
    """Generate LaTeX table for paper."""
    theories = ['T1_correlation', 'T2_pcmci', 'T3_csm']
    
    latex = []
    latex.append(r'\begin{table}[h]')
    latex.append(r'\centering')
    latex.append(r'\caption{Causal Discovery Performance Comparison}')
    latex.append(r'\begin{tabular}{lccc}')
    latex.append(r'\toprule')
    latex.append(r'Method & Dataset 1 & Dataset 2 & Dataset 3 & Average \\')
    latex.append(r'\midrule')
    
    for theory in theories:
        if theory in results['ablations']:
            name = theory.replace('T1_', '').replace('T2_', '').replace('T3_', '').replace('_', ' ').title()
            f1_values = [r.get('f1', 0) for r in results['ablations'][theory] if r.get('f1') is not None]
            
            if len(f1_values) >= 2:
                avg_f1 = sum(f1_values) / len(f1_values)
                f1_str = ' & '.join([f'{f:.3f}' for f in f1_values[:3]])
                latex.append(f'{name} & {f1_str} & \\textbf{{{avg_f1:.3f}}} \\\\')
    
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\label{tab:causal_discovery}')
    latex.append(r'\end{table}')
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(latex))
    
    print(f"✓ LaTeX table saved to: {output_file}")

def generate_markdown_report(results, figures, output_file):
    """Generate comprehensive Markdown report."""
    md = []
    md.append('# Comprehensive Ablation Study - Results Report\n')
    md.append(f'**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
    md.append('---\n')
    
    # Dataset info
    md.append('## Datasets\n')
    for i, ds in enumerate(results['datasets'], 1):
        md.append(f"{i}. **{ds['name']}**: {ds['variables']} variables, {ds['samples']} samples")
    md.append('')
    
    # Key findings
    md.append('## Key Findings\n')
    md.append('### Causal Discovery Performance\n')
    
    theories = ['T2_pcmci', 'T1_correlation', 'T3_csm']
    for theory in theories:
        if theory in results['ablations']:
            name = theory.replace('_', ' ').title()
            f1_values = [r.get('f1') for r in results['ablations'][theory] if r.get('f1') is not None]
            if f1_values:
                avg_f1 = sum(f1_values) / len(f1_values)
                max_f1 = max(f1_values)
                md.append(f'- **{name}**: Average F1 = {avg_f1:.3f}, Max F1 = {max_f1:.3f}')
    
    md.append('\n### Narrative Quality\n')
    llms = ['L2_tinyllama', 'L3_phi2', 'L1_gpt2']
    for llm in llms:
        if llm in results['ablations']:
            name = llm.replace('L1_', 'GPT-2').replace('L2_', 'TinyLlama').replace('L3_', 'Phi-2')
            keyword_counts = [r.get('keywords', 0) for r in results['ablations'][llm]]
            if keyword_counts:
                avg_kw = sum(keyword_counts) / len(keyword_counts)
                md.append(f'- **{name}**: Average {avg_kw:.1f} causal keywords per narrative')
    
    # Best combinations
    md.append('\n### Top Performing Combinations\n')
    
    # Find best Theory×LLM combo
    best_combo = None
    best_f1 = 0
    for key in results['ablations']:
        if '×' in key:
            f1_values = [r.get('f1') for r in results['ablations'][key] if r.get('f1') is not None]
            if f1_values:
                avg_f1 = sum(f1_values) / len(f1_values)
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_combo = key
    
    if best_combo:
        keywords = [r.get('keywords', 0) for r in results['ablations'][best_combo]]
        avg_kw = sum(keywords) / len(keywords) if keywords else 0
        md.append(f'1. **Best Theory×LLM**: {best_combo} (F1 = {best_f1:.3f}, {avg_kw:.1f} keywords)')
    
    # Visualizations
    md.append('\n## Visualizations\n')
    md.append('Interactive plots available in HTML format:\n')
    md.append('- [F1 Score Comparison](f1_comparison.html)')
    md.append('- [Keyword Comparison](keywords_comparison.html)')
    md.append('- [Theory×LLM Heatmap](theory_llm_heatmap.html)')
    md.append('- [LLM Pairs Comparison](llm_pairs_comparison.html)')
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(md))
    
    print(f"✓ Markdown report saved to: {output_file}")

# Main execution
if __name__ == "__main__":
    print("="*80)
    print("GENERATING PAPER-READY PLOTS AND REPORTS")
    print("="*80)
    print()
    
    # Setup paths
    project_dir = Path("/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/01-causal-slm")
    log_file = project_dir / "output/evaluations/comprehensive_ablations_NEW.log"
    output_dir = project_dir / "output/plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Parse results
    print("Parsing ablation results from log file...")
    results = parse_ablation_log(log_file)
    print(f"✓ Found {len(results['datasets'])} datasets")
    print(f"✓ Found {len(results['ablations'])} ablation configurations\n")
    
    # Generate plots
    print("Generating visualizations...")
    
    figures = {}
    
    print("  Creating F1 comparison plot...")
    figures['f1'] = create_f1_comparison(results, output_dir)
    
    print("  Creating keyword comparison plot...")
    figures['keywords'] = create_keyword_comparison(results, output_dir)
    
    print("  Creating Theory×LLM heatmap...")
    figures['heatmap'] = create_theory_llm_heatmap(results, output_dir)
    
    print("  Creating LLM pairs comparison...")
    figures['pairs'] = create_narrative_length_chart(results, output_dir)
    
    print("\n✓ All plots generated!\n")
    
    # Generate tables
    print("Generating LaTeX table...")
    generate_latex_table(results, output_dir / "ablation_table.tex")
    
    # Generate report
    print("Generating Markdown report...")
    generate_markdown_report(results, figures, output_dir / "ABLATION_REPORT.md")
    
    print("\n" + "="*80)
    print("✓ ALL VISUALIZATIONS AND REPORTS GENERATED!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print("\nGenerated files:")
    print("  - f1_comparison.html / .png")
    print("  - keywords_comparison.html / .png")
    print("  - theory_llm_heatmap.html / .png")
    print("  - llm_pairs_comparison.html / .png")
    print("  - ablation_table.tex (LaTeX)")
    print("  - ABLATION_REPORT.md (Markdown)")
    print("\n" + "="*80)
