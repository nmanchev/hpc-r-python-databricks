# Databricks notebook source
# MAGIC %md
# MAGIC #Ray on Databricks
# MAGIC
# MAGIC Ray is a high-performance distributed computing framework for Python. It allows you to scale your existing Python code to run on multiple machines, making it ideal for large-scale data processing, machine learning, and scientific computing tasks.
# MAGIC
# MAGIC **Key Features**
# MAGIC
# MAGIC * Distributed Computing: Ray enables you to run your Python code on multiple machines, allowing you to scale your computations horizontally.
# MAGIC * Task Parallelism: Ray provides a high-level API for parallelizing tasks, making it easy to write concurrent code.
# MAGIC * Actor Model: Ray implements the actor model, which allows you to write concurrent code using actors that communicate with each other.
# MAGIC * Dynamic Task Graphs: Ray allows you to create dynamic task graphs, which enable you to execute tasks in a flexible and efficient manner.
# MAGIC * Integration with Popular Libraries: Ray integrates well with popular Python libraries like NumPy, pandas, scikit-learn, and TensorFlow.
# MAGIC
# MAGIC Running Ray on Databricks allows you to leverage the breadth of the Databricks ecosystem, enhancing data processing and machine learning workflows with services and integrations that are not available in open source Ray. In any Databricks notebook that is attached to a Databricks cluster, you can run the `setup_ray_cluster` command to start a fixed-size or an autoscaling Ray cluster:

# COMMAND ----------

from ray.util.spark import setup_ray_cluster, MAX_NUM_WORKER_NODES, shutdown_ray_cluster

help(setup_ray_cluster)

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's spin up a 5 node Ray cluster (1 head node and 4 worker nodes). Note, that this is a *local* Ray cluster, which times out after being idle for 30 minutes. Databricks also support the provisioning of *global* Ray cluster. A global mode Ray cluster allows all users attached to the Databricks cluster to also use the Ray cluster. This mode of running a Ray cluster doesn’t have the active timeout functionality that a single-user cluster has when running a single-user Ray cluster instance.

# COMMAND ----------

import ray

setup_ray_cluster(max_worker_nodes=4,
                  num_cpus_head_node=1, 
                  num_cpus_worker_node=4, 
                  num_gpus_worker_node=0, 
                  num_gpus_head_node=0
                  )

# Pass any custom configuration to ray.init
ray.init(ignore_reinit_error=True)

# COMMAND ----------

# MAGIC %md
# MAGIC We can inspect the resources available on the Ray cluster by calling `ray.cluster_resources()`:

# COMMAND ----------

print(ray.cluster_resources())

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Now let's define a sample function, which can be used to simulate computationally heavy workloads. The `monte_carlo_pi_sampler` function below uses the Monte Carlo method to estimate Pi by generating random points within a square and counting the proportion of points that fall within a quarter of a circle inscribed within the square. It takes two arguments: `n_samples`, the number of random samples to generate, and `debug`, a boolean flag that enables or disables debug printing.

# COMMAND ----------

import random

def monte_carlo_pi_sampler(n_samples, debug=False):
    n_inside_quadrant = 0
    if debug:
        print(f"monte_carlo_pi_sampler getting ready to do {n_samples} samples")
    for _ in range(n_samples):
        x = random.uniform(0,1)
        y = random.uniform(0,1)
        r = (x**2 + y**2)**0.5
        if r <= 1:
            n_inside_quadrant += 1
    if debug:
        print(f"monte_carlo_pi_sampler found {n_inside_quadrant} inside a unit circle")
    return n_inside_quadrant

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC Let's see how long it takes for our function to process a dataset of 30,000,000 points (3 x 10^7)

# COMMAND ----------

import time

start = time.time()

# Try experimenting with different sample sizes, and turning on debug mode
n_total = 3*(10**7)
n_inside = monte_carlo_pi_sampler(n_total)
pi = 4*n_inside/n_total

time_elapsed = time.time() - start
print(f'Pi is approximately {pi}.')
print(f'It took {time_elapsed:.2f}s with {n_total} total samples')

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's see what happens when we use Ray to parallelize the execution.

# COMMAND ----------

# First, we define a wrapper function that calls monte_carlo_pi_sampler. All that's required for Ray to be able to parallelize the function is to decorate it with @ray.remote. Note that instead of creating a wrapper we could decorate the original function, but having a separate wrapper allows us to call both the sequential and parallelized execution for the purposes of comparison. 
@ray.remote
def monte_carlo_pi_sampler_on_ray(n_samples, debug=False):
    return monte_carlo_pi_sampler(n_samples, debug=debug)

# This is a helper function that accepts the size of the dataset, and calls the parallelized function for each batch of points. It also prints the elapsed time after submitting the batch and the elapsed time it takes to process each batch.
def ray_approximate_pi(sample_batch_sizes, debug=False):
    start = time.time()
    total_samples = sum(sample_batch_sizes)
    n_inside_futures = []
    for i,n in enumerate(sample_batch_sizes):
        n_inside_futures.append(monte_carlo_pi_sampler_on_ray.remote(n, debug=debug))
        print(f'Time check after batch {i+1} submit: {time.time()-start:.2f}s')
    total_inside = 0
    for j, future in enumerate(n_inside_futures):
        total_inside += ray.get(future)
        print(f'Time check after batch {j+1} result: {time.time()-start:.2f}s')
    pi = 4*total_inside/total_samples
    print(f'Pi is approximately {pi}.')
    print(f'It took {time.time()-start:.2f}s with batches: {sample_batch_sizes}')

# Now let's see the execution time for 30,000,000 points
ray_approximate_pi(3*[10**7], debug=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ![Ray Image](images/ray.png)
# MAGIC
# MAGIC Ray is more than just a high-performance distributed computing framework - it's a thriving ecosystem that encompasses a wide range of tools and libraries designed to simplify and accelerate the development of scalable applications. At the heart of the Ray ecosystem lies a suite of key projects, including:
# MAGIC
# MAGIC * **Ray Tune**: a hyperparameter tuning library that enables efficient optimization of machine learning models
# MAGIC * **Ray SGD**: a distributed stochastic gradient descent library for scalable deep learning
# MAGIC * **Ray RLlib**: a reinforcement learning library that provides a unified interface for training and deploying RL models
# MAGIC * **Ray Datasets**: a library for efficient data processing and loading
# MAGIC * **Ray Serve**: a scalable model serving library for deploying machine learning models in production
# MAGIC * **Ray Workflows**: a library for managing and orchestrating complex workflows and pipelines
# MAGIC
# MAGIC Additionally, the Ray ecosystem also includes integrations with popular libraries like **TensorFlow**, **PyTorch**, and **scikit-learn**, as well as a growing community of developers and users who contribute to the ecosystem's growth and adoption. 
# MAGIC
# MAGIC The code below illustrates how we can use **Ray Tune** to parallelize the minimization of an objective function.

# COMMAND ----------

from ray import train, tune


def objective(config): 
    score = config["a"] ** 2 + config["b"]
    return {"score": score}


search_space = { 
    "a": tune.grid_search([0.001, 0.01, 0.1, 1.0]),
    "b": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(objective, param_space=search_space)  

results = tuner.fit()
print(results.get_best_result(metric="score", mode="min").config)

# COMMAND ----------

# MAGIC %md
# MAGIC Other frameworks can also leverage Ray clusters to speed up traditional machine learning tasks like model training and serving. **XGBoost**, for example, can be used to distribute the training of models across multiple nodes, enabling faster training times and larger model sizes. This integration allows users to seamlessly scale their **XGBoost** workloads using Ray's distributed computing capabilities. 
# MAGIC
# MAGIC The code below uses our Ray cluster to pre-process and train an XGBoost model against the freely available Breast Cancer dataset. Inspect the resulting output to determine the number of models attempted and to find the coefficients of the best performing model.

# COMMAND ----------

from typing import Tuple

import ray
from ray.data import Dataset, Preprocessor
from ray.data.preprocessors import StandardScaler
from ray.train.xgboost import XGBoostTrainer
from ray.train import Result, ScalingConfig
import xgboost

def prepare_data() -> Tuple[Dataset, Dataset, Dataset]:
    dataset = ray.data.read_csv("s3://anonymous@air-example-data/breast_cancer.csv")
    train_dataset, valid_dataset = dataset.train_test_split(test_size=0.3)
    test_dataset = valid_dataset.drop_columns(["target"])
    return train_dataset, valid_dataset, test_dataset
  
def train_xgboost(num_workers: int, use_gpu: bool = False) -> Result:
    train_dataset, valid_dataset, _ = prepare_data()

    # Scale some random columns
    columns_to_scale = ["mean radius", "mean texture"]
    preprocessor = StandardScaler(columns=columns_to_scale)
    train_dataset = preprocessor.fit_transform(train_dataset)
    valid_dataset = preprocessor.transform(valid_dataset)

    # XGBoost specific params
    params = {
        "tree_method": "approx",
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "error"],
    }

    trainer = XGBoostTrainer(
        scaling_config=ScalingConfig(num_workers=num_workers, use_gpu=use_gpu),
        label_column="target",
        params=params,
        datasets={"train": train_dataset, "valid": valid_dataset},
        num_boost_round=100,
        metadata = {"preprocessor_pkl": preprocessor.serialize()},
        run_config=ray.train.RunConfig(name="xgboost_trainer",
                                    storage_path="/dbfs/tmp/ray_results")
    )
    result = trainer.fit()
    print(result.metrics)

    return result
  
result = train_xgboost(num_workers=4, use_gpu=False)

# COMMAND ----------

# MAGIC %md
# MAGIC Finally, it is a best practice to shutdown the Ray cluster if no longer needed.

# COMMAND ----------

ray.util.spark.shutdown_ray_cluster()
