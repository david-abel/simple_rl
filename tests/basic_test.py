#!/usr/bin/env python

# Python imports.
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
		FNULL = open(os.devnull, 'w')
  		subprocess.check_call(["python", path_to_example_file, "no_plot"], stdout=FNULL)
  		return True
	except subprocess.CalledProcessError:
		return False

def main():
	# Add examples to path.
	parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
	sys.path.insert(0, parent_dir)

	# Grab all example files.
	example_dir = os.path.join(os.getcwd(), "..", "examples")
	example_files = [f for f in os.listdir(example_dir) if os.path.isfile(os.path.join(example_dir, f)) and "py" == f.split(".")[-1] and "init" not in f and "viz_exam" not in f]

	print "\n" + "="*32
	print "== Running", len(example_files), "simple_rl tests =="
	print "="*32 + "\n"
	total_passed = 0

	for i, ex in enumerate(example_files):
		print "\t [Test", str(i + 1) + "] ", ex + ": ",
		result = run_example(os.path.join(example_dir, ex))
		if result:
			total_passed += 1
			print "PASS."
		else:
			print "FAIL."
	print "\nResults:", total_passed, "/", len(example_files), "passed."

if __name__ == "__main__":
	main()
