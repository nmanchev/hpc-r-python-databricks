# Databricks notebook source
# MAGIC %md
# MAGIC # Running R code in parallel on Databricks
# MAGIC
# MAGIC This notebook demonstrates how we can execute R code in parallel using Databricks backend compute. Feel free to change the attached compute and observe the change in execution times.
# MAGIC
# MAGIC First, let's load the libraries required for running the code below. If some of the libraries are missing we can install them on the fly.

# COMMAND ----------

if(!require("ranger")) install.packages("ranger")
if(!require("doParallel")) install.packages("doParallel")
if(!require("brms")) install.packages("brms")

# COMMAND ----------

# MAGIC %md
# MAGIC Just for clarity, let's inspect the number of CPUs/cores available from the currently attached compute. Note that the cell below is marked as Python. In Databricks a single notebook can contain a mixture of R, Python, Scala, SQL, and Markdown cells.

# COMMAND ----------

# MAGIC %python
# MAGIC import psutil
# MAGIC
# MAGIC cores = psutil.cpu_count()
# MAGIC threads_count = psutil.cpu_count() / psutil.cpu_count(logical=False)
# MAGIC
# MAGIC print("Number of cores : ", cores)
# MAGIC print("Threads per core: ", threads_count)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's store the number of cores and threads in variables that we can pass to the relevant R functions.

# COMMAND ----------

library(parallel)

ncores <- detectCores(logical = FALSE)
nthreads <- ncores * 2

cat(ncores)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's execute the R code below. It generates random data, fits a random forest model using the `ranger` package, and extracts the top 10 feature importances in parallel using the `foreach` and `doParallel` packages.
# MAGIC
# MAGIC For comparison, using a single node with 4 cores x 2 threads each (AMD EPYC 7763 64-Core Processor) should take about 12 minutes.

# COMMAND ----------

library(ranger)
library(foreach)
library(doParallel)

generate_data <- function(n = 1000) {
  d <- as.data.frame(matrix(rnorm(n * n), nrow = n))
  d$y <- as.factor(sample(0:1, size = n, replace = TRUE))
  d
}

registerDoParallel(cores = ncores) # forking

n <- 400

start.time <- Sys.time()

res <- foreach(i = 1:n) %dopar% {
  d <- generate_data()
  fit <- ranger(y ~ ., data = d, importance = "impurity", num.threads = nthreads)
  importance(fit)[1:10]
}

end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)

cat("Elapsed minutes:", time.taken)


# COMMAND ----------

# MAGIC %md
# MAGIC Next, the R code below fits a Bayesian linear mixed-effects model using the `brms` package to analyze the kidney dataset. The number of parallel chains to run is specified via the `cores` argument.
# MAGIC
# MAGIC For comparison, using a single node with 4 cores x 2 threads each (AMD EPYC 7763 64-Core Processor) should take about 3 minutes.
# MAGIC

# COMMAND ----------

library(brms)

data("kidney", package = "brms")

start.time <- Sys.time()

fit <- brm(
  formula = time | cens(censored) ~ age * sex + disease + (1 + age|patient),
  data = kidney,
  family = lognormal(),
  prior = c(
    set_prior("normal(0,5)", class = "b"),
    set_prior("cauchy(0,2)", class = "sd"),
    set_prior("lkj(2)", class = "cor")
  ),
  warmup = 1000,
  iter = 2000,
  chains = 32,
  cores = nthreads,
  control = list(adapt_delta = 0.95)
  
)

end.time <- Sys.time()
time.taken <- round(end.time - start.time,2)

summary(fit, waic = TRUE)

cat("Elapsed minutes:", time.taken)
