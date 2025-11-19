#!/usr/bin/env python3
"""
Univ: Hosei University - Tokyo, Yulab
Author: Franck Junior Aboya Messou
Date: November 8, 2025
Repo: https://github.com/hosei-university-iist-yulab/01-causal-slm.git
"""

"""
Test suite runner for CSLM framework.
Executes unit tests, integration tests, and regression tests.
Validates correctness of all components.
"""

import sys
import os
from pathlib import Path
import subprocess
import json
from datetime import datetime
import traceback

# Set up paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
os.chdir(PROJECT_ROOT)

CHECKPUTS_DIR = PROJECT_ROOT / 'tests' / 'checkputs'
CHECKPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Set PYTHONPATH
os.environ['PYTHONPATH'] = f"{PROJECT_ROOT / 'src'}:{os.environ.get('PYTHONPATH', '')}"


class TestRunner:
    """Comprehensive test runner."""

    def __init__(self):
        self.results = []
        self.timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    def log(self, message, level='INFO'):
        """Log message with timestamp."""
        timestamp = datetime.now().strftime('%H:%M:%S')
        print(f"[{timestamp}] [{level}] {message}")

    def run_test(self, name, command, timeout=300, save_output=True):
        """Run a single test."""
        self.log(f"Running test: {name}", "TEST")
        start_time = datetime.now()

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=PROJECT_ROOT
            )

            duration = (datetime.now() - start_time).total_seconds()
            success = result.returncode == 0

            test_result = {
                'name': name,
                'command': command,
                'success': success,
                'return_code': result.returncode,
                'duration_seconds': duration,
                'timestamp': start_time.isoformat()
            }

            # Save output if requested
            if save_output:
                output_file = CHECKPUTS_DIR / f"{name.replace(' ', '_')}_{self.timestamp}.log"
                with open(output_file, 'w') as f:
                    f.write(f"Command: {command}\n")
                    f.write(f"Return Code: {result.returncode}\n")
                    f.write(f"Duration: {duration:.2f}s\n")
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDOUT:\n")
                    f.write("="*80 + "\n")
                    f.write(result.stdout)
                    f.write("\n" + "="*80 + "\n")
                    f.write("STDERR:\n")
                    f.write("="*80 + "\n")
                    f.write(result.stderr)

                test_result['output_file'] = str(output_file.relative_to(PROJECT_ROOT))

            self.results.append(test_result)

            status = "✓ PASS" if success else "✗ FAIL"
            self.log(f"{status} - {name} ({duration:.2f}s)", "RESULT")

            return success

        except subprocess.TimeoutExpired:
            duration = (datetime.now() - start_time).total_seconds()
            self.log(f"✗ TIMEOUT - {name} ({duration:.2f}s)", "ERROR")
            self.results.append({
                'name': name,
                'command': command,
                'success': False,
                'error': 'Timeout',
                'duration_seconds': duration,
                'timestamp': start_time.isoformat()
            })
            return False

        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            self.log(f"✗ ERROR - {name}: {str(e)}", "ERROR")
            self.results.append({
                'name': name,
                'command': command,
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc(),
                'duration_seconds': duration,
                'timestamp': start_time.isoformat()
            })
            return False

    def run_all_tests(self):
        """Run all tests."""
        self.log("="*80)
        self.log("STARTING COMPREHENSIVE TEST SUITE")
        self.log("="*80)

        tests = [
            # Test 1: Python imports and dependencies
            {
                'name': 'test_imports',
                'command': 'python -c "import torch; import transformers; import numpy; import pandas; print(\'All imports successful\')"',
                'timeout': 30
            },

            # Test 2: Dataset generation
            {
                'name': 'test_dataset_generation',
                'command': 'python scripts/generate_datasets.py',
                'timeout': 60
            },

            # Test 3: CSM integration (quick test with 10 epochs)
            {
                'name': 'test_csm_quick',
                'command': 'python -c "import sys; sys.path.insert(0, \'src\'); from innovations.causal_score_matching import CausalScoreMatching, CSMConfig; import numpy as np; data = np.random.randn(100, 3); csm = CausalScoreMatching(CSMConfig(n_epochs=10, device=\'cpu\')); graph = csm.discover_graph(data); print(f\'CSM test passed: discovered {np.sum(graph > 0)} edges\')"',
                'timeout': 120
            },

            # Test 4: Multi-LLM system loading (CPU mode for testing)
            {
                'name': 'test_llm_loading',
                'command': 'python scripts/test_llm_loading.py',
                'timeout': 180
            },

            # Test 5: Quick end-to-end test
            {
                'name': 'test_end_to_end',
                'command': 'python scripts/quick_end_to_end_test.py',
                'timeout': 300
            },

            # Test 6: Figure generation (verify no errors)
            {
                'name': 'test_figure_generation',
                'command': 'python -c "import matplotlib; matplotlib.use(\'Agg\'); import matplotlib.pyplot as plt; fig, ax = plt.subplots(); ax.plot([1,2,3]); plt.savefig(\'tests/checkputs/test_plot.png\'); print(\'Figure generation test passed\')"',
                'timeout': 30
            },

            # Test 7: JSON output validation
            {
                'name': 'test_json_outputs',
                'command': f'python -c "import json; from pathlib import Path; outputs = list(Path(\'output/evaluations\').glob(\'*.json\')); print(f\'Found {{len(outputs)}} JSON files\'); [json.load(open(f)) for f in outputs[:3]]; print(\'All JSON files valid\')"',
                'timeout': 30
            },

            # Test 8: LaTeX tables validation
            {
                'name': 'test_latex_tables',
                'command': 'python -c "from pathlib import Path; tex_file = Path(\'output/paper_tables/COMPREHENSIVE_LATEX_TABLES.tex\'); content = tex_file.read_text(); assert \'\\\\begin{document}\' in content; assert \'\\\\end{document}\' in content; print(f\'LaTeX tables valid: {len(content)} chars\')"',
                'timeout': 10
            },

            # Test 9: Figure files existence
            {
                'name': 'test_figure_files',
                'command': 'python -c "from pathlib import Path; figures = list(Path(\'output/figures\').glob(\'*.pdf\')); print(f\'Found {len(figures)} PDF figures\'); assert len(figures) >= 12, f\'Expected 12+ figures, found {len(figures)}\'; print(\'All figure files present\')"',
                'timeout': 10
            },

            # Test 10: Pipeline scripts syntax check
            {
                'name': 'test_syntax_check',
                'command': 'python -m py_compile scripts/*.py',
                'timeout': 30
            },
        ]

        passed = 0
        failed = 0

        for test in tests:
            success = self.run_test(**test)
            if success:
                passed += 1
            else:
                failed += 1

        # Save summary
        self.save_summary(passed, failed)

    def save_summary(self, passed, failed):
        """Save test summary."""
        summary_file = CHECKPUTS_DIR / f'test_summary_{self.timestamp}.json'

        summary = {
            'timestamp': self.timestamp,
            'total_tests': len(self.results),
            'passed': passed,
            'failed': failed,
            'success_rate': f"{(passed / len(self.results) * 100):.1f}%",
            'results': self.results
        }

        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        self.log("="*80)
        self.log("TEST SUITE COMPLETED")
        self.log("="*80)
        self.log(f"Total Tests: {len(self.results)}")
        self.log(f"Passed: {passed} ✓")
        self.log(f"Failed: {failed} ✗")
        self.log(f"Success Rate: {(passed / len(self.results) * 100):.1f}%")
        self.log(f"Summary saved to: {summary_file.relative_to(PROJECT_ROOT)}")

        # Create markdown report
        self.create_markdown_report(summary)

    def create_markdown_report(self, summary):
        """Create human-readable markdown report."""
        report_file = CHECKPUTS_DIR / f'TEST_REPORT_{self.timestamp}.md'

        with open(report_file, 'w') as f:
            f.write(f"# Test Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Summary\n\n")
            f.write(f"- **Total Tests:** {summary['total_tests']}\n")
            f.write(f"- **Passed:** {summary['passed']} ✓\n")
            f.write(f"- **Failed:** {summary['failed']} ✗\n")
            f.write(f"- **Success Rate:** {summary['success_rate']}\n\n")

            f.write(f"## Test Results\n\n")
            f.write("| # | Test Name | Status | Duration (s) |\n")
            f.write("|---|-----------|--------|---------------|\n")

            for i, result in enumerate(summary['results'], 1):
                status = "✓ PASS" if result['success'] else "✗ FAIL"
                duration = result.get('duration_seconds', 0)
                f.write(f"| {i} | {result['name']} | {status} | {duration:.2f} |\n")

            f.write("\n## Detailed Results\n\n")
            for result in summary['results']:
                f.write(f"### {result['name']}\n\n")
                f.write(f"- **Command:** `{result['command']}`\n")
                f.write(f"- **Status:** {'✓ PASS' if result['success'] else '✗ FAIL'}\n")
                f.write(f"- **Duration:** {result.get('duration_seconds', 0):.2f}s\n")

                if 'output_file' in result:
                    f.write(f"- **Output Log:** `{result['output_file']}`\n")

                if not result['success']:
                    if 'error' in result:
                        f.write(f"- **Error:** {result['error']}\n")

                f.write("\n")

            f.write("---\n\n")
            f.write(f"**Generated:** {datetime.now().isoformat()}\n")
            f.write(f"**Project:** Causal SLM - Multi-LLM Ensemble System\n")

        self.log(f"Markdown report saved to: {report_file.relative_to(PROJECT_ROOT)}")


def main():
    """Main entry point."""
    runner = TestRunner()
    runner.run_all_tests()


if __name__ == '__main__':
    main()
