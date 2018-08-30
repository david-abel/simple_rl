# Other imports
import os

this_dir = os.path.dirname(os.path.realpath(__file__))

dir_fixes = {"_modules":"modules", "_static":"static", "_sources":"sources"}

def rename_directories():

	# Get all subdirectories.
	sub_dirs = dir_fixes.keys() #[f for f in os.listdir(this_dir) if os.path.isdir(f)]



	# Rename them.
	for subdir in sub_dirs:

		if os.path.exists(subdir):
			no_underscore_dir_name = subdir.replace("_", "")
			os.rename(os.path.join(this_dir, subdir), os.path.join(this_dir, no_underscore_dir_name))


def get_all_html_file_names():
	'''
	Returns:
		(list): Contains any file that ends in .html in the simple_rl/docs folder (excluding simple_rl/docs/sphinx).
	'''

	# Get the non sphinx sub directories.
	all_non_sphinx_sub_dirs = [x for x in os.walk(this_dir) if "sphinx" not in x[0]]

	# Loop through and find the html files.
	all_html_full_paths = []
	for sub_dir, root, files in all_non_sphinx_sub_dirs:
		for f in files:
			if ".html" in f:
				all_html_full_paths.append(os.path.join(sub_dir, f))

	return all_html_full_paths
		

def replace_underscored_module_names(path_to_file):
	'''
	Args:
		paths_to_files (str)

	Summary:
		Opens the file @path_to_file and replaces all uses of the _static with static,
			for all key-val pairs in the global dictionary @dir_fixes.
	'''
	html_file = open(path_to_file, "r")
	all_words = html_file.read()

	# Replace all uses of wrong directory name.
	result_text = all_words
	for dir_to_fix in dir_fixes.keys():
		result_text = result_text.replace(dir_to_fix, dir_fixes[dir_to_fix])

	# Close the file, open it for reading, overwrite with the newly replaced text.
	html_file.close()
	html_file = open(path_to_file, "w+")
	html_file.write(result_text)
	html_file.close()

def main():

	# Rename the underscored directories.
	rename_directories()

	# Grab all relevant html files.
	all_html_files_to_fix = get_all_html_file_names()

	for html_file in all_html_files_to_fix:
		replace_underscored_module_names(html_file)


if __name__ == "__main__":
	main()