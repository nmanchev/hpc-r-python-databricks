# Workbench Example Scripts

This directory contains scripts shows how to work with Databricks from Posit Workbench

To use these scripts:

1. Load in the example data set

```
CREATE CATALOG IF NOT EXISTS posit_demo;
CREATE TABLE posit_demo.default.lending_club CLONE PARQUET.`dbfs:/databricks-datasets/samples/lending_club/parquet/`;
```

2. Start a Posit Workbench session with Databricks credentials (chosen when launching a session)
3. Fill in your cluster ID when running `databricks_loan_analysis_spark.qmd` and `databricks_model_training.qmd`. Fill in HTTPPath when running `databricks_loan_analysis_sql.qmd`
4. Run in your Console `renv::restore()` to install the packages used in this project
5. Install pysparklyr with `install.packages("pysparklyr")` to use the `databricks_connect` function.
6. Install the `mlflow` [Python package](https://github.com/mlflow/mlflow/tree/master/mlflow/R/mlflow#prerequisites) to use `databricks_model_training.qmd`
