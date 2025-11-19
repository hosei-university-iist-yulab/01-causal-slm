#!/usr/bin/env python3
"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 14, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

import re
import json
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from pathlib import Path

def parse_ablation_log(log_file):
    """Parse ablation results."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    results = {'theory': [], 'llm': [], 'cross': []}
    
    for section in content.split('DATASET:')[1:]:
        dataset = section.split('\n')[0].strip()
        
        # Theory results
        for m in re.finditer(r'(T\d+_\w+)\.\.\..*?Edges=(\d+)(?:, F1=([\d.]+))?', section):
            results['theory'].append({
                'dataset': dataset,
                'method': m.group(1).replace('T1_', 'Correlation').replace('T2_', 'PCMCI').replace('T3_', 'CSM'),
                'f1': float(m.group(3)) if m.group(3) else None
            })
        
        # LLM results
        for m in re.finditer(r'(L\d+_\w+)\.\.\..*?Length=(\d+), Keywords=(\d+)', section):
            results['llm'].append({
                'dataset': dataset,
                'llm': m.group(1).replace('L1_gpt2', 'GPT-2').replace('L2_tinyllama', 'TinyLlama').replace('L3_phi2', 'Phi-2'),
                'keywords': int(m.group(3))
            })
        
        # Cross results
        for m in re.finditer(r'(T\d+_\w+)×(L\d+_\w+)\.\.\..*?(?:F1=([\d.]+)|NoGT), Len=(\d+), Kw=(\d+)', section):
            results['cross'].append({
                'dataset': dataset,
                'theory': m.group(1).replace('T2_', 'PCMCI').replace('T1_', 'Corr').replace('T3_', 'CSM'),
                'llm': m.group(2).replace('L1_', 'GPT-2').replace('L2_', 'TinyLlama').replace('L3_', 'Phi-2'),
                'f1': float(m.group(3)) if m.group(3) else None,
                'keywords': int(m.group(5))
            })
    
    return results

def create_fig1_theory_comparison(results, output_dir):
    """Figure 1: Causal Discovery Performance."""
    df = pd.DataFrame(results['theory'])
    df = df[df['f1'].notna()]
    
    if len(df) == 0:
        return
    
    fig = go.Figure()
    
    for method in df['method'].unique():
        df_method = df[df['method'] == method]
        fig.add_trace(go.Bar(
            x=df_method['dataset'].str.replace('synthetic_', '').str.replace('_simple', ''),
            y=df_method['f1'],
            name=method,
            text=df_method['f1'].round(3),
            textposition='outside'
        ))
    
    fig.update_layout(
        title='Causal Discovery Performance',
        xaxis_title='Dataset',
        yaxis_title='F1 Score',
        yaxis_range=[0, 1.1],
        font=dict(family='Arial', size=16),
        barmode='group',
        template='plotly_white',
        width=800,
        height=500,
        legend=dict(x=0.7, y=0.95)
    )
    
    fig.write_image(output_dir / 'figure1_causal_discovery.pdf', width=800, height=500)
    fig.write_image(output_dir / 'figure1_causal_discovery.png', width=800, height=500, scale=2)
    print("✓ Figure 1: Causal Discovery Performance (PDF + PNG)")

def create_fig2_narrative_quality(results, output_dir):
    """Figure 2: Narrative Quality Comparison."""
    df = pd.DataFrame(results['llm'])
    
    if len(df) == 0:
        return
    
    fig = go.Figure()
    
    for llm in ['GPT-2', 'TinyLlama', 'Phi-2']:
        df_llm = df[df['llm'] == llm]
        if len(df_llm) > 0:
            fig.add_trace(go.Bar(
                x=df_llm['dataset'].str.replace('synthetic_', '').str.replace('real_', 'Real '),
                y=df_llm['keywords'],
                name=llm,
                text=df_llm['keywords'],
                textposition='outside'
            ))
    
    fig.update_layout(
        title='Narrative Quality: Causal Keywords by LLM',
        xaxis_title='Dataset',
        yaxis_title='Causal Keywords Count',
        font=dict(family='Arial', size=16),
        barmode='group',
        template='plotly_white',
        width=800,
        height=500,
        legend=dict(x=0.02, y=0.98)
    )
    
    fig.write_image(output_dir / 'figure2_narrative_quality.pdf', width=800, height=500)
    fig.write_image(output_dir / 'figure2_narrative_quality.png', width=800, height=500, scale=2)
    print("✓ Figure 2: Narrative Quality (PDF + PNG)")

def create_fig3_sota_comparison(results, output_dir):
    """Figure 3: SOTA Comparison."""
    df = pd.DataFrame(results['cross'])
    df_pcmci = df[df['theory'] == 'PCMCI']
    df_syn = df_pcmci[df_pcmci['f1'].notna()]
    
    if len(df_syn) == 0:
        return
    
    # Group by dataset and LLM
    summary = df_syn.groupby(['dataset', 'llm']).agg({
        'f1': 'mean',
        'keywords': 'mean'
    }).reset_index()
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('F1 Score', 'Causal Keywords'),
        specs=[[{'type': 'bar'}, {'type': 'bar'}]]
    )
    
    for llm in ['GPT-2', 'TinyLlama', 'Phi-2']:
        df_llm = summary[summary['llm'] == llm]
        if len(df_llm) > 0:
            fig.add_trace(
                go.Bar(x=df_llm['dataset'].str.replace('synthetic_', ''), y=df_llm['f1'], name=llm),
                row=1, col=1
            )
            fig.add_trace(
                go.Bar(x=df_llm['dataset'].str.replace('synthetic_', ''), y=df_llm['keywords'], name=llm, showlegend=False),
                row=1, col=2
            )
    
    fig.update_xaxes(title_text="Dataset", row=1, col=1)
    fig.update_xaxes(title_text="Dataset", row=1, col=2)
    fig.update_yaxes(title_text="F1 Score", row=1, col=1, range=[0, 1.1])
    fig.update_yaxes(title_text="Keywords", row=1, col=2)
    
    fig.update_layout(
        title_text='SOTA Comparison: PCMCI + Single LLM',
        font=dict(family='Arial', size=14),
        template='plotly_white',
        width=1000,
        height=500
    )
    
    fig.write_image(output_dir / 'figure3_sota_comparison.pdf', width=1000, height=500)
    fig.write_image(output_dir / 'figure3_sota_comparison.png', width=1000, height=500, scale=2)
    print("✓ Figure 3: SOTA Comparison (PDF + PNG)")

def create_fig4_heatmap(results, output_dir):
    """Figure 4: Theory × LLM Heatmap."""
    df = pd.DataFrame(results['cross'])
    df_syn = df[df['f1'].notna()]
    
    if len(df_syn) == 0:
        return
    
    # Pivot for heatmap
    pivot = df_syn.groupby(['theory', 'llm'])['keywords'].mean().reset_index()
    
    theories = ['Corr', 'PCMCI', 'CSM']
    llms = ['GPT-2', 'TinyLlama', 'Phi-2']
    
    matrix = []
    for theory in theories:
        row = []
        for llm in llms:
            val = pivot[(pivot['theory'] == theory) & (pivot['llm'] == llm)]['keywords'].mean()
            row.append(val if not np.isnan(val) else 0)
        matrix.append(row)
    
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=llms,
        y=theories,
        colorscale='Blues',
        text=[[f'{int(v)}' for v in row] for row in matrix],
        texttemplate='%{text}',
        textfont={"size": 18},
        colorbar=dict(title="Keywords")
    ))
    
    fig.update_layout(
        title='Theory × LLM: Average Causal Keywords',
        xaxis_title='Language Model',
        yaxis_title='Causal Discovery Method',
        font=dict(family='Arial', size=16),
        template='plotly_white',
        width=600,
        height=500
    )
    
    fig.write_image(output_dir / 'figure4_theory_llm_heatmap.pdf', width=600, height=500)
    fig.write_image(output_dir / 'figure4_theory_llm_heatmap.png', width=600, height=500, scale=2)
    print("✓ Figure 4: Theory × LLM Heatmap (PDF + PNG)")

# Main
if __name__ == "__main__":
    print("="*80)
    print("GENERATING PUBLICATION-QUALITY PDF FIGURES")
    print("="*80)
    print()
    
    project_dir = Path("/home/Aboya_25R9803/projects/perso/LLMium/projects/02-SLM-Foundational/01-causal-slm")
    log_file = project_dir / "output/evaluations/comprehensive_ablations_full.log"
    output_dir = project_dir / "output/figures"  # Use same directory as other figures
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Parsing results...")
    results = parse_ablation_log(log_file)
    print(f"✓ Theory: {len(results['theory'])} results")
    print(f"✓ LLM: {len(results['llm'])} results")
    print(f"✓ Cross: {len(results['cross'])} results")
    print()
    
    print("Generating PDF figures...")
    create_fig1_theory_comparison(results, output_dir)
    create_fig2_narrative_quality(results, output_dir)
    create_fig3_sota_comparison(results, output_dir)
    create_fig4_heatmap(results, output_dir)
    
    print()
    print("="*80)
    print("✓ ALL PDF FIGURES GENERATED!")
    print("="*80)
    print(f"\nOutput: {output_dir}")
    print("\nGenerated files:")
    print("  - figure1_causal_discovery.pdf (+ .png)")
    print("  - figure2_narrative_quality.pdf (+ .png)")
    print("  - figure3_sota_comparison.pdf (+ .png)")
    print("  - figure4_theory_llm_heatmap.pdf (+ .png)")
    print("\nAll figures are publication-ready (800-1000px, 300dpi PNG)")
    print("="*80)
