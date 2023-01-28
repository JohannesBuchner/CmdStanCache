import os
import glob
from cmdstancache import clear, run_stan, get_path

def test_cache():
	clear()
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
