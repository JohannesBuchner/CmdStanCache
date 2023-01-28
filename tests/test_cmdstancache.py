import os
import numpy as np
import glob
import matplotlib.pyplot as plt
from cmdstancache import clear, run_stan, get_path, plot_corner, remove_stuck_chains

def test_cache():
	print("clearing entire cache ...")
	clear()
	print("clearing entire cache ... done")
	run_stan("""
data {
  int N;
}
parameters {
  real<lower=-10.0, upper=10.0> x[N];
}
model {
  for (i in 1:N-1) {
	 target += -2 * (100 * square(x[i+1] - square(x[i])) + square(1 - x[i]));
  }
}
""", dict(N=2))
	data_files = glob.glob(os.path.join(get_path(), "*.json"))
	assert len(data_files) == 1
	code_files = glob.glob(os.path.join(get_path(), "*.stan"))
	assert len(code_files) == 1
	del data_files, code_files

	run_stan("""
data {
  // dimensionality:
  int N;
}
parameters {
  real<lower=-10.0, upper=10.0> x[N];
}
model {
  for (i in 1:N-1) {
	 target += -2 * (100 * square(x[i+1] - square(x[i])) + square(1 - x[i]));
  }
}
""", dict(N=2))

	code_files = glob.glob(os.path.join(get_path(), "*.stan"))
	assert len(code_files) == 1
	data_files = glob.glob(os.path.join(get_path(), "*.json"))
	assert len(data_files) == 1
	del data_files, code_files

	run_stan("""
data {
  // dimensionality:
  int N;
}
parameters {
  real<lower=-10.0, upper=10.0> x[N];
}
model {
  for (i in 1:N-1) {
	 target += -2 * (100 * square(x[i+1] - square(x[i])) + square(1 - x[i]));
  }
}
""", dict(N=3))

	code_files = glob.glob(os.path.join(get_path(), "*.stan"))
	assert len(code_files) == 1
	data_files = glob.glob(os.path.join(get_path(), "*.json"))
	assert len(data_files) == 2
	del data_files, code_files

def test_plot1var():
	stan_variables, method_variables = run_stan("""
data {
  int N;
}
parameters {
  real<lower=-10.0, upper=10.0> x[N];
}
model {
  for (i in 1:N-1) {
	 target += -2 * (100 * square(x[i+1] - square(x[i])) + square(1 - x[i]));
  }
}
""", dict(N=2))
	assert set(stan_variables.keys()) == set(['x']), stan_variables.keys()
	plot = plot_corner(stan_variables)
	plot.savefig('test_rosen2.pdf')
	plt.close()

def test_clean_stuck_chains():
	method_variables = dict(lp__ = np.random.normal(size=(1000,4)))
	method_variables['lp__'][:,2] = np.random.normal(-20000, 1, size=1000)
	stan_variables = dict(myvar1 = np.random.normal(size=4000), myvar2 = np.random.normal(size=4000))

	cleaned_variables = remove_stuck_chains(stan_variables, method_variables)

	expected_mask = np.transpose(method_variables['lp__']).flatten() > -1000
	assert cleaned_variables.keys() == set(['myvar1', 'myvar2'])
	np.testing.assert_equal(cleaned_variables['myvar1'], stan_variables['myvar1'][expected_mask])
	np.testing.assert_equal(cleaned_variables['myvar2'], stan_variables['myvar2'][expected_mask])

def test_plot_cleaned():
	stan_variables, method_variables = run_stan("""
parameters {
  real x;
  real<lower=0> y;
  real bigarray[100];
}
transformed parameters {
  real z = x * y;
}
model {
  x ~ normal(3, 3);
  10 ~ poisson(y);
  bigarray ~ normal(3, 3);
}
""", dict(N=2))

	print("method_variables:", method_variables)
	cleaned_variables = remove_stuck_chains(stan_variables, method_variables)
	print("cleaned_variables:", cleaned_variables)
	for k in 'xyz':
		np.testing.assert_equal(cleaned_variables[k], stan_variables[k])
	
	plot = plot_corner(cleaned_variables)
	plot.savefig('test_degeneracy.pdf')
	plt.close()
