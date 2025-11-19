"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 1, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Utilities for tracking experiment performance metrics.
Logs F1 scores, MAE, SHD, and other evaluation metrics.
Supports visualization and statistical analysis.
"""

import time
import psutil
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import subprocess

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from codecarbon import EmissionsTracker
    CODECARBON_AVAILABLE = True
except ImportError:
    CODECARBON_AVAILABLE = False
    print("Warning: codecarbon not installed. CO2 tracking disabled.")
    print("Install with: pip install codecarbon")


class PerformanceTracker:
    """
    Comprehensive performance tracking for ML experiments.

    Tracks runtime, memory (RAM + GPU), and CO2 emissions.
    """

    def __init__(
        self,
        experiment_name: str,
        track_co2: bool = True,
        gpu_ids: Optional[List[int]] = None,
        output_dir: Optional[str] = None
    ):
        """
        Initialize performance tracker.

        Args:
            experiment_name: Name of experiment for reporting
            track_co2: Whether to track CO2 emissions
            gpu_ids: List of GPU IDs to monitor (None = all available)
            output_dir: Directory to save reports (None = no auto-save)
        """
        self.experiment_name = experiment_name
        self.track_co2 = track_co2 and CODECARBON_AVAILABLE
        self.gpu_ids = gpu_ids
        self.output_dir = Path(output_dir) if output_dir else None

        # Timing
        self.start_time = None
        self.end_time = None
        self.cpu_start = None
        self.cpu_end = None

        # Memory
        self.process = psutil.Process(os.getpid())
        self.peak_ram_mb = 0
        self.start_ram_mb = 0
        self.peak_gpu_memory_mb = {}

        # CO2 tracker
        self.emissions_tracker = None
        self.co2_kg = 0.0
        self.energy_kwh = 0.0

        # GPU info
        self.gpu_info = self._get_gpu_info()

        # Checkpoints for intermediate measurements
        self.checkpoints = []

    def _get_gpu_info(self) -> Dict:
        """Get GPU information."""
        gpu_info = {
            'available': TORCH_AVAILABLE and torch.cuda.is_available(),
            'count': 0,
            'devices': []
        }

        if gpu_info['available']:
            gpu_info['count'] = torch.cuda.device_count()
            for i in range(gpu_info['count']):
                if self.gpu_ids is None or i in self.gpu_ids:
                    props = torch.cuda.get_device_properties(i)
                    gpu_info['devices'].append({
                        'id': i,
                        'name': props.name,
                        'total_memory_gb': props.total_memory / 1024**3,
                        'compute_capability': f"{props.major}.{props.minor}"
                    })

        return gpu_info

    def _get_gpu_memory_usage(self) -> Dict[int, float]:
        """Get current GPU memory usage in MB."""
        gpu_memory = {}

        if TORCH_AVAILABLE and torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                if self.gpu_ids is None or i in self.gpu_ids:
                    gpu_memory[i] = torch.cuda.memory_allocated(i) / 1024**2

        return gpu_memory

    def _get_ram_usage(self) -> float:
        """Get current RAM usage in MB."""
        return self.process.memory_info().rss / 1024**2

    def start(self):
        """Start tracking performance."""
        print(f"\n{'='*80}")
        print(f"Performance Tracking: {self.experiment_name}")
        print(f"{'='*80}")

        # Start timing
        self.start_time = time.time()
        self.cpu_start = time.process_time()

        # Record starting memory
        self.start_ram_mb = self._get_ram_usage()
        self.peak_ram_mb = self.start_ram_mb

        # Start CO2 tracking
        if self.track_co2:
            try:
                self.emissions_tracker = EmissionsTracker(
                    project_name=self.experiment_name,
                    output_dir=str(self.output_dir) if self.output_dir else ".",
                    log_level="warning"
                )
                self.emissions_tracker.start()
                print("✓ CO2 tracking enabled (codecarbon)")
            except Exception as e:
                print(f"⚠ CO2 tracking failed: {e}")
                self.track_co2 = False

        # Print GPU info
        if self.gpu_info['available']:
            print(f"✓ GPU tracking enabled ({self.gpu_info['count']} devices)")
            for dev in self.gpu_info['devices']:
                print(f"  GPU {dev['id']}: {dev['name']} ({dev['total_memory_gb']:.1f} GB)")

        print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

    def checkpoint(self, name: str):
        """Record a checkpoint with current metrics."""
        current_time = time.time() - self.start_time
        current_ram = self._get_ram_usage()
        current_gpu = self._get_gpu_memory_usage()

        # Update peak memory
        self.peak_ram_mb = max(self.peak_ram_mb, current_ram)
        for gpu_id, mem in current_gpu.items():
            if gpu_id not in self.peak_gpu_memory_mb:
                self.peak_gpu_memory_mb[gpu_id] = 0
            self.peak_gpu_memory_mb[gpu_id] = max(
                self.peak_gpu_memory_mb[gpu_id], mem
            )

        checkpoint = {
            'name': name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time_s': current_time,
            'ram_mb': current_ram,
            'gpu_memory_mb': current_gpu
        }

        self.checkpoints.append(checkpoint)

        print(f"Checkpoint '{name}': {current_time:.1f}s, RAM: {current_ram:.1f} MB")
        if current_gpu:
            gpu_str = ", ".join([f"GPU{i}: {m:.1f}MB" for i, m in current_gpu.items()])
            print(f"  {gpu_str}")

    def stop(self):
        """Stop tracking and finalize measurements."""
        # Stop timing
        self.end_time = time.time()
        self.cpu_end = time.process_time()

        # Final memory check
        final_ram = self._get_ram_usage()
        self.peak_ram_mb = max(self.peak_ram_mb, final_ram)

        final_gpu = self._get_gpu_memory_usage()
        for gpu_id, mem in final_gpu.items():
            if gpu_id not in self.peak_gpu_memory_mb:
                self.peak_gpu_memory_mb[gpu_id] = 0
            self.peak_gpu_memory_mb[gpu_id] = max(
                self.peak_gpu_memory_mb[gpu_id], mem
            )

        # Stop CO2 tracking
        if self.track_co2 and self.emissions_tracker:
            try:
                self.emissions_tracker.stop()
                # Read emissions file
                emissions_file = Path(".") / "emissions.csv"
                if emissions_file.exists():
                    import csv
                    with open(emissions_file, 'r') as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        if rows:
                            last_row = rows[-1]
                            self.co2_kg = float(last_row.get('emissions', 0))
                            self.energy_kwh = float(last_row.get('energy_consumed', 0))
            except Exception as e:
                print(f"⚠ Failed to read CO2 data: {e}")

        # Print summary
        self._print_summary()

    def _print_summary(self):
        """Print performance summary."""
        wall_time = self.end_time - self.start_time
        cpu_time = self.cpu_end - self.cpu_start

        print(f"\n{'='*80}")
        print(f"Performance Summary: {self.experiment_name}")
        print(f"{'='*80}")
        print(f"Wall time:      {wall_time:.2f} seconds ({wall_time/60:.2f} minutes)")
        print(f"CPU time:       {cpu_time:.2f} seconds")
        print(f"CPU efficiency: {(cpu_time/wall_time)*100:.1f}%")
        print()
        print(f"RAM usage:")
        print(f"  Start:        {self.start_ram_mb:.1f} MB")
        print(f"  Peak:         {self.peak_ram_mb:.1f} MB")
        print(f"  Delta:        {self.peak_ram_mb - self.start_ram_mb:.1f} MB")

        if self.peak_gpu_memory_mb:
            print()
            print(f"GPU memory (peak):")
            for gpu_id, mem_mb in sorted(self.peak_gpu_memory_mb.items()):
                print(f"  GPU {gpu_id}:       {mem_mb:.1f} MB ({mem_mb/1024:.2f} GB)")

        if self.track_co2:
            print()
            print(f"Carbon footprint:")
            print(f"  CO2 emissions: {self.co2_kg:.6f} kg ({self.co2_kg*1000:.3f} g)")
            print(f"  Energy used:   {self.energy_kwh:.6f} kWh")

            # Equivalents for context
            if self.co2_kg > 0:
                km_driving = self.co2_kg * 5.5  # avg car: 180g CO2/km
                print(f"  Equivalent to: {km_driving:.2f} km of driving")

        print(f"{'='*80}\n")

    def get_report(self) -> Dict:
        """
        Get comprehensive performance report.

        Returns:
            Dictionary with all performance metrics
        """
        wall_time = self.end_time - self.start_time if self.end_time else 0
        cpu_time = self.cpu_end - self.cpu_start if self.cpu_end else 0

        report = {
            'experiment_name': self.experiment_name,
            'timestamp': datetime.now().isoformat(),

            'timing': {
                'wall_time_s': wall_time,
                'wall_time_m': wall_time / 60,
                'wall_time_h': wall_time / 3600,
                'cpu_time_s': cpu_time,
                'cpu_efficiency_percent': (cpu_time / wall_time * 100) if wall_time > 0 else 0
            },

            'memory': {
                'ram_start_mb': self.start_ram_mb,
                'ram_peak_mb': self.peak_ram_mb,
                'ram_delta_mb': self.peak_ram_mb - self.start_ram_mb,
                'gpu_peak_mb': self.peak_gpu_memory_mb,
                'gpu_peak_gb': {k: v/1024 for k, v in self.peak_gpu_memory_mb.items()}
            },

            'co2': {
                'tracked': self.track_co2,
                'emissions_kg': self.co2_kg,
                'emissions_g': self.co2_kg * 1000,
                'energy_kwh': self.energy_kwh,
                'equivalent_km_driving': self.co2_kg * 5.5
            },

            'gpu_info': self.gpu_info,

            'checkpoints': self.checkpoints
        }

        return report

    def save_report(self, filepath: Optional[str] = None):
        """
        Save performance report to JSON file.

        Args:
            filepath: Path to save report. If None, uses output_dir/experiment_name.json
        """
        if filepath is None:
            if self.output_dir is None:
                raise ValueError("No filepath provided and no output_dir set")
            self.output_dir.mkdir(parents=True, exist_ok=True)
            filepath = self.output_dir / f"{self.experiment_name}_performance.json"

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        report = self.get_report()

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)

        print(f"✓ Performance report saved to: {filepath}")

        return filepath


def format_performance_table(reports: List[Dict]) -> str:
    """
    Format multiple performance reports as a comparison table.

    Args:
        reports: List of performance report dictionaries

    Returns:
        Formatted table string
    """
    header = f"{'Experiment':<30} {'Time (m)':<12} {'RAM (MB)':<12} {'GPU (GB)':<12} {'CO2 (g)':<10}"
    separator = "-" * 80

    lines = [separator, header, separator]

    for report in reports:
        name = report['experiment_name'][:28]
        time_m = report['timing']['wall_time_m']
        ram_mb = report['memory']['ram_peak_mb']

        # Sum GPU memory across all GPUs
        gpu_gb = sum(report['memory']['gpu_peak_gb'].values()) if report['memory']['gpu_peak_gb'] else 0

        co2_g = report['co2']['emissions_g']

        line = f"{name:<30} {time_m:<12.2f} {ram_mb:<12.1f} {gpu_gb:<12.2f} {co2_g:<10.3f}"
        lines.append(line)

    lines.append(separator)

    # Add totals
    total_time = sum(r['timing']['wall_time_m'] for r in reports)
    total_ram = sum(r['memory']['ram_peak_mb'] for r in reports)
    total_gpu = sum(sum(r['memory']['gpu_peak_gb'].values()) for r in reports if r['memory']['gpu_peak_gb'])
    total_co2 = sum(r['co2']['emissions_g'] for r in reports)

    lines.append(f"{'TOTAL':<30} {total_time:<12.2f} {total_ram:<12.1f} {total_gpu:<12.2f} {total_co2:<10.3f}")
    lines.append(separator)

    return "\n".join(lines)


if __name__ == "__main__":
    # Example usage
    print("Example: Performance Tracking\n")

    tracker = PerformanceTracker(
        experiment_name="example_experiment",
        track_co2=True,
        gpu_ids=[0, 1],
        output_dir="output/performance"
    )

    tracker.start()

    # Simulate work
    import numpy as np
    for i in range(3):
        time.sleep(1)
        _ = np.random.randn(1000, 1000) @ np.random.randn(1000, 1000)
        tracker.checkpoint(f"iteration_{i+1}")

    tracker.stop()

    # Save report
    tracker.save_report()

    # Print report
    report = tracker.get_report()
    print("\nFull Report:")
    print(json.dumps(report, indent=2))
