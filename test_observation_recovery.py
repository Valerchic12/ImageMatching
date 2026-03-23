"""
Test script for recover_observations_after_pose_refinement feature.

Verifies observation recovery functionality with Hyundai i20 test data.

Usage:
    python test_observation_recovery.py [--debug] [--suite]
    
Options:
    --debug : Enable debug logging
    --suite : Run full regression suite (slower but more comprehensive)
"""

import sys
import os
import subprocess
from pathlib import Path


def run_single_test(debug=False):
    """Run a single Hyundai i20 calibration test."""
    print("=" * 80)
    print("Testing Observation Recovery Feature (Hyundai i20 - Single Test)")
    print("=" * 80)
    
    # Build command to run the existing test calibration script
    cmd = [
        sys.executable,
        'test_calibration_logic.py',
        '--width', '1600',
        '--height', '1200',
        '--focal', '2222.22',
        '--summary-json', 'test_observation_recovery_result.json',
        '--visualization-svg', 'test_observation_recovery_visualization.svg',
        '--diagnostic-dir', 'test_observation_recovery_diagnostics',
    ]
    
    if debug:
        cmd.append('--debug-logging')
    
    print(f"\nRunning test command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Test PASSED: Calibration completed successfully")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("Test FAILED: Calibration encountered errors")
        print("=" * 80)
        return False


def run_regression_suite(debug=False):
    """Run the full regression test suite."""
    print("=" * 80)
    print("Testing Observation Recovery Feature (Full Regression Suite)")
    print("=" * 80)
    
    # Build command to run the regression suite
    cmd = [
        sys.executable,
        'test_calibration_logic.py',
        '--regression-suite',
        '--width', '1600',
        '--height', '1200',
        '--focal', '2222.22',
        '--suite-output-dir', 'test_calibration_suite_recovery',
        '--suite-summary-json', 'calibration_test_suite_recovery_summary.json',
    ]
    
    if debug:
        cmd.append('--debug-logging')
    
    print(f"\nRunning test command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=str(Path(__file__).parent))
    
    if result.returncode == 0:
        print("\n" + "=" * 80)
        print("Regression Suite PASSED")
        print("=" * 80)
        return True
    else:
        print("\n" + "=" * 80)
        print("Regression Suite FAILED")
        print("=" * 80)
        return False


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test observation recovery feature')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--suite', action='store_true', help='Run full regression suite')
    args = parser.parse_args()
    
    os.chdir(Path(__file__).parent)
    
    if args.suite:
        success = run_regression_suite(debug=args.debug)
    else:
        success = run_single_test(debug=args.debug)
    
    sys.exit(0 if success else 1)

