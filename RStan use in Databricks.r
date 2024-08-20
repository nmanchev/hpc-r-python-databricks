# Databricks notebook source
# MAGIC %md
# MAGIC #RStan use in Databricks
# MAGIC
# MAGIC RStan is an R package that provides an interface to the Stan modeling language, which is a probabilistic programming language for Bayesian inference. Stan (also known as Stan Modeling Language) is a C++ library for Bayesian inference, and RStan is a wrapper around this library that allows users to fit Bayesian models using R.
# MAGIC
# MAGIC In this notebook we demonstrate the use of the `rstan` package on Databricks to compile and run some Stan code. 
# MAGIC
# MAGIC First, we load the `rstan` package (or install it on the fly if it's not currently available)

# COMMAND ----------

if(!require("rstan")) install.packages("rstan")


# COMMAND ----------

# MAGIC %md
# MAGIC Next, we set the number of cores available for parallel computing in R. We also set the RStan option `auto_write` to `TRUE`. This will automatically write the compiled model to a file with a `.stan` extension, which can be reused in future runs. This can save time and reduce the memory requirements for large models.

# COMMAND ----------

options(mc.cores = parallel::detectCores())
rstan_options(auto_write = TRUE)

# COMMAND ----------

# MAGIC %md
# MAGIC Below we have a Stan program, which defines a Bayesian hierarchical model for estimating the treatment effects of a set of schools. The model assumes that the treatment effects for each school are normally distributed with a population mean `mu` and standard deviation `tau`. The estimated treatment effects `y` are modeled as normally distributed with mean `theta` (the true treatment effect for each school) and standard error `sigma`.

# COMMAND ----------

file.show("/dbfs/Users/nikolay.manchev@databricks.com/schools.stan")

# COMMAND ----------

# MAGIC %md
# MAGIC Next, let's create some sample schoold data. `J` contains the number of school, and we also provide the `y` and `sigma` for each school.

# COMMAND ----------

schools_dat <- list(J = 8, 
                    y = c(28,  8, -3,  7, -1,  1, 18, 12),
                    sigma = c(15, 10, 16, 11,  9, 11, 10, 18))


# COMMAND ----------

# MAGIC %md
# MAGIC Now let's run Stan to fit the Bayesian model.

# COMMAND ----------

fit <- stan(file = '/dbfs/Users/nikolay.manchev@databricks.com/schools.stan', data = schools_dat)


# COMMAND ----------

# MAGIC %md
# MAGIC Finally, let's explore the fitted model and its coefficients.

# COMMAND ----------

print(fit)

