CmdStanCache
=============

Quicker model iterations and enhanced productivity for Stan MCMC by

* caching model compilation in a smart way
* caching sampling results in a smart way

Install 
-------

::

	$ pip install cmdstancache

Usage
-----
::

	model = """
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
	"""
	data = dict(N=2)

	import cmdstancache

	stan_variables, method_variables = cmdstancache.run_stan(
		model,
		data=data, 
		# any other sample() parameters go here
		seed=42
	)

**Now comes the trick**:

* If you run this code twice, the second time the stored result is read.

* If you add or modify a code comment, the same result is returned without having to rerun.

How it works
-------------

cmdstancache keeps a cache of code and data that has previously been used for MCMC sampling.
If it already has the results, it returns it from the cache.

Here are the details:

1. The code is normalised (stripped of comments and indents)
2. A hash of the normalised code is computed
3. The model code is stored in ~/.stan_cache/<codehash>.stan
4. The model is compiled, if it is not already there
5. The data are sorted by key, exported to json, and a hash computed
6. The data are stored in ~/.stan_cache/<datahash>.json
7. cmdstanpy MCMC is run with code=<codehash>.stan and data=<datahash>.json
8. fit.stan_variables() and fit.method_variables() are returned
9. joblib memoizes steps 7 and 8, avoiding resampling when the same data and code hash are seen.


Contributors
-------------

* @JohannesBuchner

Contributions are welcome.
