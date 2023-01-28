"""CmdStanCache caches Stan MCMC runs."""
import os
import glob
import joblib
import numpy as np
import hashlib
import re
import tempfile
import shutil
import warnings
import collections

import cmdstanpy

__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '1.2.0'

path = os.path.expanduser("~/.stan_cache")
mem = joblib.Memory(path, verbose=False)

__all__ = ["mem", "get_path", "clear", "run_stan"]


def get_path():
	"""Get path of the cache."""
	return path


def clear():
	"""Clear cache of models, data and runs."""
	# clear joblib memory
	mem.clear()
	# glob all the cache files
	files = glob.glob(os.path.join(get_path(), "*.json"))
	files += glob.glob(os.path.join(get_path(), "*.stan"))
	for f in files:
		os.remove(f)


def trim_model_code(code):
	"""Strip white space, empty lines and comments from stan code.

	Parameters
	----------
	code: str
		Stan code

	Returns
	-------
	code: str
		Trimmed, normalised code
	"""
	lines = code.split("\n")
	code_lines = [re.sub('//.*$', '', line).strip() for line in lines]
	code_lines_singlespace = [
		line.replace('    ', ' ').replace('  ', ' ').replace('  ', ' ')
		for line in code_lines if len(line) > 0]

	slimcode = '\n'.join(code_lines_singlespace).strip()
	slimbytes = slimcode.encode(encoding="ascii", errors="ignore")
	slimcode_ascii = slimbytes.decode(encoding="ascii")
	return slimcode_ascii


def hash_model_code(code):
	"""Get a hash for stan code.

	Parameters
	----------
	code: str
		Stan code

	Returns
	-------
	hash: str
		md5 hash
	"""
	slimbytes = code.encode(encoding="ascii")
	hexcode = hashlib.md5(slimbytes).hexdigest()
	return hexcode


def hash_data(datafile):
	"""Get a hash for a data json file.

	Parameters
	----------
	datafile: str
		Path to a text file

	Returns
	-------
	hash: str
		md5 hash
	"""
	BUF_SIZE = 1024 * 1024
	md5 = hashlib.md5()

	with open(datafile, 'rb') as f:
		while True:
			data = f.read(BUF_SIZE)
			if not data:
				break
			md5.update(data)

	return md5.hexdigest()


def get_formatted_code(code):
	"""Get reasonably readable formatted code from trimmed code.

	Parameters
	----------
	code: str
		Stan code

	Returns
	-------
	formatted_code: str
		Stan code
	"""
	formatted_code_lines = []
	indent = 0
	for i, line in enumerate(code.split("\n")):
		indent -= line.count('}')
		formatted_code_lines.append('%3d: %s%s' % (i + 1, '  ' * indent, line))
		indent += line.count('{')
	return '\n'.join(formatted_code_lines)


@mem.cache(ignore=['verbose'])
def cached_run_stan(code, datafile, verbose=True, **kwargs):
	"""Run MCMC with the given code and data file.

	Parameters
	----------
	code: str
		Stan model code
	datafile: str
		Path to data file
	verbose: bool
		whether to print the code being compiled, posterior summaries and diagnostics.
	**kwargs: dict
		arguments passed on to `cmdstanpy.CmdStanModel.sample`

	Returns
	-------
	stan_variables: object
		stan_variables returned by fit object
	method_variables: object
		method_variables returned by fit object
	"""
	if verbose:
		print("Code")
		print("----")
		print(get_formatted_code(code))
	code_hash = hash_model_code(code)
	codefile = os.path.join(path, code_hash + '.stan')
	if not os.path.exists(codefile):
		with open(codefile, 'w') as f:
			f.write(code)
	model = cmdstanpy.CmdStanModel(stan_file=codefile)
	assert model.code() == code, (model.code, code)

	fit = model.sample(data=datafile, **kwargs)
	if verbose:
		print("Summary")
		print("-------")
		print(fit.summary())
		print("Diagnostics")
		print("-----------")
		print(fit.diagnose())

	return fit.stan_variables(), fit.method_variables()


def run_stan(code, data, verbose=True, **kwargs):
	"""Run MCMC with the given code and data.

	Parameters
	----------
	code: str
		Stan model code
	data: dict
		Model data
	verbose: bool
		whether to print the code being compiled, summaries of the
		input data and posterior, and convergence diagnostics.
	**kwargs: dict
		arguments passed on to `cmdstanpy.CmdStanModel.sample`

	Returns
	-------
	stan_variables: object
		stan_variables returned by fit object
	method_variables: object
		method_variables returned by fit object
	"""
	simple_model_code = trim_model_code(code)

	if verbose:
		print()
		print("Data")
		print("----")
		for k, v in data.items():
			shape = np.shape(v)
			if shape == ():
				print('  %-10s: %s' % (k, v))
			elif shape[0] == 0:
				print('  %-10s: shape %s' % (k, shape))
			else:
				print('  %-10s: shape %s [%s ... %s]' % (k, shape, np.min(v), np.max(v)))
			del k, v

	with tempfile.NamedTemporaryFile(suffix='.json') as f:
		fname = f.name
		data_sorted = {k: data[k] for k in sorted(data.keys())}
		cmdstanpy.write_stan_json(fname, data_sorted)
		del data_sorted
		data_hash = hash_data(fname)

		datafile = os.path.join(path, data_hash + '.json')
		shutil.copy(fname, datafile)

	results = cached_run_stan(simple_model_code, datafile, verbose=verbose, **kwargs)

	try:
		os.unlink(datafile)
	except IOError as e:
		warnings.warn('Cleaning up stan input data file failed: %s' % e)

	return results


def remove_stuck_chains(stan_variables, method_variables):
	"""
	Return posteriors chains.

	Parameters
	----------
	stan_variables: object
		stan_variables returned by fit object
	method_variables: object
		method_variables returned by fit object

	Returns
	-------
	stan_variables: dict
		Dictionary of variables with their posterior chains.
	"""
	# identify the likelihood range of each chain
	# select the chain with the highest likelihood
	# include all chains which cross into this likelihood range
	# Chains stuck with very poor solutions will be removed
	lp = method_variables['lp__']

	top_chain_index = np.argmax(lp.max(axis=0))
	top_chain_min_lp = lp[:, top_chain_index].min()
	chain_mask = lp.max(axis=0) > top_chain_min_lp
	if not chain_mask.all():
		warnings.warn("Ignoring these stuck chains: %s" % (np.where(~chain_mask)[0] + 1))

	chain_length, num_chains = lp.shape
	mask = np.zeros(lp.size, dtype=bool)
	for i in np.where(chain_mask)[0]:
		mask[i * chain_length: (i + 1) * chain_length] = True

	filtered_variables = collections.OrderedDict()
	for k, v in stan_variables.items():
		filtered_variables[k] = v[mask, ...]

	return filtered_variables


def plot_corner(stan_variables, max_array_size=20, verbose=True, **kwargs):
	"""
	Make a simple corner plot, based on samples extracted from fit.

	Only scalar variables are included.
	If there is only one model variable, and it is smaller than max_array_size,
	it is used instead.

	log-variables are preferred (logfoo is plotted instead of foo if both exist).

	Parameters
	----------
	stan_variables: object
		stan_variables returned by fit object
	max_array_size: int
		largest array variable size to include in the plot. Very large arrays give illegible plots.
	verbose: bool
		whether to print posterior mean and std
	kwargs: dict
		arguments passed to `corner.corner`.

	Returns
	-------
	plot: object
		whatever `corner.corner` returns.
	"""
	la = stan_variables
	samples = []
	paramnames = []
	badlist = [k for k in la.keys() if 'log' in k and k.replace('log', '') in la.keys()]

	for k, v in la.items():
		if verbose:
			print('%20s: %.4f +- %.4f' % (k, v.mean(), v.std()))
		if k not in badlist and v.ndim == 1:
			samples.append(la[k])
			paramnames.append(k)
		del k

	if len(samples) == 0:
		del samples, paramnames
		arrays = [k for k in la.keys() if la[k].ndim == 2 and la[k].shape[1] <= max_array_size and k not in badlist]
		if len(arrays) != 1:
			warnings.warn("no scalar variables found")
			return

		key = arrays[0]
		# flatten across chains and column for each variable
		final_samples = la[key]
		paramnames = ['%s[%d]' % (key, i + 1) for i in range(la[key].shape[1])]
	else:
		final_samples = np.transpose(samples)

	import corner
	if 'plot_density' not in kwargs:
		kwargs['plot_density'] = False
		kwargs['plot_datapoints'] = False
	return corner.corner(final_samples, labels=paramnames, **kwargs)
