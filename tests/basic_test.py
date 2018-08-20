#!/usr/bin/env python

# Python imports.
from __future__ import print_function
import subprocess
import os
import sys

def run_example(path_to_example_file):
    '''
    Args:
        path_to_example_file (str)

    Returns:
        (bool): True if pass, Fail if error.
    '''
    try:
        fnull = open(os.devnull, 'w')
        subprocess.check_call(["python", path_to_example_file, "no_plot"], stdout=fnull)
        return True
    except subprocess.CalledProcessError:
        return False

def main():
    # Add examples to path.
    parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    sys.path.insert(0, parent_dir)

    # Grab all example files.
    example_dir = os.path.join(os.getcwd(), "..", "examples")
    example_files = [f for f in os.listdir(example_dir) if os.path.isfile(os.path.join(example_dir, f)) and f.split(".")[-1] == "py"]

    # Remove non-tests.
    non_tests = ["init", "viz_exam", "blank", "gym", "srl_example", "grid_from_file", "belief", "brtdp"]
    for phrase in non_tests:
        for test_file in example_files:
            if phrase in test_file:
                example_files.remove(test_file)

    # Prints.
    print("\n" + "="*32)
    print("== Running", len(example_files), "simple_rl tests ==")
    print("="*32 + "\n")
    total_passed = 0

    # Run each test.
    for i, ex in enumerate(example_files):
        print("\t [Test", str(i + 1) + "] ", ex + ": ",)
        result = run_example(os.path.join(example_dir, ex))
        if result:
            total_passed += 1
            print("\t\tPASS.")
        else:
            print("\t\tFAIL.")

    # Results.
    print("\nResults:", total_passed, "/", len(example_files), "passed.")

if __name__ == "__main__":
    main()
