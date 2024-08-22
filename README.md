# hpc-r-python-databricks
A set of simple notebooks, illustrating the parallelization of R and Python code in Databricks

* Parallel code - This R script demonstrates parallel computing on a Databricks cluster using the `foreach` package, showcasing the benefits of parallel processing for computationally intensive tasks. The script generates a large dataset, performs serial and parallel computations on it, and compares the execution times to highlight the advantages of parallelization.

* R Mlflow MLOps - This R script demonstrates how to use MLflow, a machine learning lifecycle management platform, to train, log, and deploy a simple linear regression model on a Databricks cluster. The script showcases MLOps best practices, including model versioning, parameter tracking, and model serving, using MLflow's R API.

* Ray on Databricks - This Python script demonstrates how to use Ray, a high-performance distributed computing framework, on a Databricks cluster to perform distributed computing tasks, such as parallelizing computations and data processing. The script showcases the integration of Ray with Databricks, enabling scalable and efficient distributed computing on large datasets.

* RStan use in Databricks - This R script demonstrates how to use RStan, a Bayesian modeling library, on a Databricks cluster to perform Bayesian linear regression and model inference. The script showcases the integration of RStan with Databricks, enabling scalable and distributed Bayesian modeling and analysis on large datasets.

* Using sparklyr with Databricks Spark - This R script demonstrates how to use `sparklyr`, an R interface to Apache Spark, to interact with a Databricks Spark cluster and perform data manipulation and analysis tasks. The script showcases the use of `sparklyr` to read and write data, perform data transformations, and execute Spark SQL queries on a Databricks cluster.