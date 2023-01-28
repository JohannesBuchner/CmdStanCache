"""CmdStanCache caches Stan MCMC runs."""
import os
import glob
import joblib
import numpy
import hashlib
import re
import tempfile
import shutil

import cmdstanpy

__author__ = """Johannes Buchner"""
__email__ = 'johannes.buchner.acad@gmx.com'
__version__ = '1.0.1'

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


@mem.cache
def cached_run_stan(code, datafile, **kwargs):
	"""Run MCMC with the given code and data file.

	Parameters
	----------
	code: str
		Stan model code
	datafile: str
		Path to data file
	**kwargs: dict
		arguments passed on to `cmdstanpy.CmdStanModel.sample`

	Returns
	-------
	stan_variables: object
		stan_variables returned by fit object
	method_variables: object
		method_variables returned by fit object
	"""
	print("Slimmed Code")
	print("------------")
	print(code)
	code_hash = hash_model_code(code)
	codefile = os.path.join(path, code_hash + '.stan')
	if not os.path.exists(codefile):
		with open(codefile, 'w') as f:
			f.write(code)
	model = cmdstanpy.CmdStanModel(stan_file=codefile)
	assert model.code() == code, (model.code, code)

	fit = model.sample(data=datafile, **kwargs)
	print("Summary")
	print("-------")
	print(fit.summary())
	print("Diagnostics")
	print("-----------")
	print(fit.diagnose())
	return fit.stan_variables(), fit.method_variables()


def run_stan(code, data, **kwargs):
	"""Run MCMC with the given code and data.

	Parameters
	----------
	code: str
		Stan model code
	data: dict
		Model data
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

	print()
	print("Data")
	print("----")
	for k, v in data.items():
		if numpy.shape(v) == ():
			print('  %-10s: %s' % (k, v))
		elif numpy.shape(v)[0] == 0:
			print('  %-10s: shape %s' % (k, numpy.shape(v)))
		else:
			print('  %-10s: shape %s [%s ... %s]' % (k, numpy.shape(v), numpy.min(v), numpy.max(v)))

	data2 = {k: data[k] for k in sorted(data.keys())}
	f = tempfile.NamedTemporaryFile(suffix='.json')
	fname = f.name
	cmdstanpy.write_stan_json(fname, data2)
	data_hash = hash_data(fname)

	datafile = os.path.join(path, data_hash + '.json')
	shutil.copy(fname, datafile)

	return cached_run_stan(simple_model_code, datafile, **kwargs)
