# Databricks notebook source
# MAGIC %md
# MAGIC # R integration with MLflow and MLOps demo
# MAGIC
# MAGIC MLflow is an open-source platform for managing the end-to-end machine learning lifecycle. It was developed by Databricks and is now a part of the Linux Foundation. MLflow helps data scientists and machine learning engineers to track, reproduce, and deploy machine learning models in a scalable and efficient manner.
# MAGIC
# MAGIC Key Features of MLflow:
# MAGIC * **Model Management**: MLflow allows you to track and manage your machine learning models, including their versions, parameters, and performance metrics.
# MAGIC * **Experiment Tracking**: MLflow provides a centralized repository for tracking experiments, including hyperparameters, metrics, and artifacts such as models, images, and data.
# MAGIC * **Reproducibility**: MLflow enables reproducibility by storing the entire experimentation process, including data, code, and environment, allowing you to reproduce results and iterate on existing work.
# MAGIC * **Model Serving**: MLflow provides a simple way to deploy models to various environments, such as cloud, on-premises, or edge devices, using a variety of serving engines.
# MAGIC * **Collaboration**: MLflow supports collaboration among data scientists and engineers by providing a shared workspace for managing and tracking experiments and models.
# MAGIC * **Extensibility**: MLflow is extensible and integrates with popular machine learning frameworks and libraries, such as TensorFlow, PyTorch, Scikit-learn, and R.
# MAGIC
# MAGIC In this notebook we'll train a number of simple R models and we'll register them with MLflow. We'll then select the best performing model and we'll show how it can be exposed for inference via the [Databricks Execution API](https://docs.databricks.com/api/workspace/commandexecution).
# MAGIC
# MAGIC We start by loading all the libraries required for the model training and registration. We also fetch the sample dataset ([Wine Quality](https://archive.ics.uci.edu/dataset/186/wine+quality) from the UCI ML Repository) we'll be using for the demo.

# COMMAND ----------

library(carrier)
library(curl)
library(mlflow)
library(httr)
library(SparkR)
library(glmnet)

reds <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep = ";")
whites <- read.csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep = ";")
 
wine_quality <- rbind(reds, whites)
 
head(wine_quality)

# COMMAND ----------

# MAGIC %md
# MAGIC Next, we define a training function - `train_wine_quality`. The function splits the dataset into training and test sets, trains a model with the provided hyperparameters (`alpha` and `lambda`), and registers the trained model with MLflow. Note that the function also registers the model performance metrics (`RMSE`, `MAE`, and `R^2`) in MLflow, and it produces a cross-validation plot that is stored as artifact in MLflow as well.

# COMMAND ----------

train_wine_quality <- function(data, alpha, lambda, model_name = "model") {
 
    # Split the data into training and test sets. (0.75, 0.25) split.
    sampled <- base::sample(1:nrow(data), 0.75 * nrow(data))
    train <- data[sampled, ]
    test <- data[-sampled, ]
    
    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x <- as.matrix(train[, !(names(train) == "quality")])
    test_x <- as.matrix(test[, !(names(train) == "quality")])
    train_y <- train[, "quality"]
    test_y <- test[, "quality"]
    
    ## Define the parameters used in each MLflow run
    alpha <- mlflow_param("alpha", alpha, "numeric")
    lambda <- mlflow_param("lambda", lambda, "numeric")

    with(mlflow_start_run(), {
        model <- glmnet(train_x, train_y, alpha = alpha, lambda = lambda, family= "gaussian", standardize = FALSE)
        l1se <- cv.glmnet(train_x, train_y, alpha = alpha)$lambda.1se
        predictor <- carrier::crate(~ glmnet::predict.glmnet(!!model, as.matrix(.x)), !!model, s = l1se)
    
        predicted <- predictor(test_x)
    
        rmse <- sqrt(mean((predicted - test_y) ^ 2))
        mae <- mean(abs(predicted - test_y))
        r2 <- as.numeric(cor(predicted, test_y) ^ 2)
    
        message("Elasticnet model (alpha=", alpha, ", lambda=", lambda, "):")
        message("  RMSE: ", rmse)
        message("  MAE: ", mae)
        message("  R2: ", mean(r2, na.rm = TRUE))
    
        ## Log the parameters associated with this run
        mlflow_log_param("alpha", alpha)
        mlflow_log_param("lambda", lambda)
    
        ## Log metrics we define from this run
        mlflow_log_metric("rmse", rmse)
        mlflow_log_metric("r2", mean(r2, na.rm = TRUE))
        mlflow_log_metric("mae", mae)
    
        # Save plot to disk
        png(filename = "ElasticNet-CrossValidation.png")
        plot(cv.glmnet(train_x, train_y, alpha = alpha), label = TRUE)
        dev.off()
    
        ## Log that plot as an artifact
        mlflow_log_artifact("ElasticNet-CrossValidation.png")

        mlflow_log_model(predictor, model_name)
    
    })

}

# COMMAND ----------

# MAGIC %md
# MAGIC Now let's call `train_wine_quality` to train three models (using different hyperparameters).

# COMMAND ----------

set.seed(1234)
 
model_name = "nm-wine-model"
 
## Run 1
train_wine_quality(data = wine_quality, alpha = 0.03, lambda = 0.98, model_name)
 
## Run 2
train_wine_quality(data = wine_quality, alpha = 0.14, lambda = 0.4, model_name)
 
## Run 3
train_wine_quality(data = wine_quality, alpha = 0.20, lambda = 0.99, model_name)


# COMMAND ----------

# MAGIC %md
# MAGIC We can now use the MLflow UI (Experiments) to browse the experiments and the artifacts stored in MLflow. We can also query MLflow programmatically to pull the best performing model (accoridng to a specified metric) and get its URI. 

# COMMAND ----------

runs <- mlflow_search_runs(order_by = "metrics.r2 DESC")

# Assuming the first run is the best model according to r^2
best_run_id <- runs$run_id[1]

# Construct model URI
model_uri <- paste0("runs:/", best_run_id, "/model")

# Construct model URI
message(c("run_id        : ", best_run_id))
message(c("experiment_id : ", runs$experiment_id[1]))

model_uri <- paste(runs$artifact_uri[1], model_name, sep = "/")
message(c("Model URI.    : ", model_uri, "\n"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Consuming models from MLflow
# MAGIC
# MAGIC Now that we have the best performing model (identifiable by the model's URI), let's see how we can load it from MLflow and use it to generate some predictions.

# COMMAND ----------

best_model <- mlflow_load_model(model_uri = model_uri)
 
## Generate prediction on 5 rows of data 
predictions <- data.frame(mlflow_predict(best_model, data = wine_quality[1:5, !(names(wine_quality) == "quality")]))
                          
names(predictions) <- "wine_quality_pred"
 
## Take a look
display(predictions)

# COMMAND ----------

# MAGIC %md
# MAGIC We don't necessarily need to run inference from within the notebook. We can connect to Databricks and execute R code remotely, using the [Databricks Execution API](https://docs.databricks.com/api/workspace/commandexecution). All we need to be able to do this is the unique ID of the Databricks cluster (`clusterId`) and the workspace URL (`workspace_url`). Since this code is running within a Databricks notebook, we can even fetch these values automatically.
# MAGIC
# MAGIC In addition, as we are connecting to the cluster externally, we would need an API access token and a means to provide the specific model URI. These are set in this notebook via [Databricks widgets](https://docs.databricks.com/en/notebooks/widgets.html)

# COMMAND ----------

# MAGIC %python
# MAGIC clusterId = spark.conf.get("spark.databricks.clusterUsageTags.clusterId")
# MAGIC workspace_url = "https://" + spark.conf.get("spark.databricks.workspaceUrl")
# MAGIC api_url = workspace_url + "/api/1.2"
# MAGIC context_name = "my_execution_context" # Just a name to distinguish the new execution context we'll create 
# MAGIC
# MAGIC print("cluster_id        :", clusterId)
# MAGIC print("workspace_url     :", workspace_url)
# MAGIC print("api_url           :", api_url)
# MAGIC print("execution_context :", context_name)
# MAGIC

# COMMAND ----------

# MAGIC %python
# MAGIC import time
# MAGIC import requests
# MAGIC import json
# MAGIC
# MAGIC access_token = dbutils.widgets.get("access_token")
# MAGIC model_uri = dbutils.widgets.get("model_uri")
# MAGIC
# MAGIC # Check if access_token is empty
# MAGIC if not access_token:
# MAGIC     raise ValueError("access_token is empty. Stopping execution.")
# MAGIC
# MAGIC headers = {
# MAGIC     "Authorization": f"Bearer {access_token}",
# MAGIC     "Content-Type": "application/json"
# MAGIC }
# MAGIC
# MAGIC # Set the request body
# MAGIC body = {
# MAGIC     "language": "r",
# MAGIC     "clusterId": clusterId,
# MAGIC     "name": context_name
# MAGIC }
# MAGIC
# MAGIC # Execute the API call to create the context
# MAGIC response = requests.post(f"{api_url}/contexts/create", headers=headers, data=json.dumps(body))
# MAGIC
# MAGIC # Check the response status code
# MAGIC if response.status_code == 200:
# MAGIC     context_id = json.loads(response.text)["id"]
# MAGIC     print(f"Execution context created with ID: {context_id}")
# MAGIC else:
# MAGIC     print(f"Error creating execution context: {response.text}")
# MAGIC
# MAGIC def execute(clusterId, contextId, command):
# MAGIC   # Set the request body
# MAGIC   body = {
# MAGIC       "language": "r",
# MAGIC       "clusterId": clusterId,
# MAGIC       "command": command,
# MAGIC       "contextId": contextId
# MAGIC   }
# MAGIC
# MAGIC   # Execute the API call
# MAGIC   response = requests.post(f"{api_url}/commands/execute", headers=headers, data=json.dumps(body))
# MAGIC
# MAGIC   # Check the response status code
# MAGIC   if response.status_code == 200:
# MAGIC       run_id = json.loads(response.text)["id"]
# MAGIC       print(f"Command submitted with run ID: {run_id}")
# MAGIC   else:
# MAGIC       print(f"Error executing command: {response.text}")
# MAGIC   
# MAGIC   if run_id is not None:
# MAGIC     
# MAGIC     # Get the status of the command
# MAGIC     body.pop("command")
# MAGIC     body["commandId"] = run_id
# MAGIC     status = "Running"
# MAGIC
# MAGIC     while status == "Running" or status == "Queued":
# MAGIC       print("Command running...")
# MAGIC       time.sleep(1)
# MAGIC       # Execute the API call to get the command info
# MAGIC       response = requests.get(f"{api_url}/commands/status", headers=headers, params=body)
# MAGIC
# MAGIC       # Check the response status code
# MAGIC       if response.status_code == 200:
# MAGIC           command_info = json.loads(response.text)
# MAGIC           status = command_info["status"]
# MAGIC       else:
# MAGIC           print(f"Error retrieving command info: {response.text}")
# MAGIC
# MAGIC   return command_info
# MAGIC
# MAGIC code = f'''
# MAGIC library(mlflow)
# MAGIC # Create some test data
# MAGIC column_names <- c("fixed.acidity", "volatile.acidity", "citric.acid", "residual.sugar", "chlorides", "free.sulfur.dioxide", "total.sulfur.dioxide", "density", "pH", "sulphates", "alcohol")
# MAGIC data_matrix <- matrix(data =  c(7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4,
# MAGIC                                 7.8,0.88,0,2.6,0.098,25,67,0.9968,3.2,0.68,9.8,
# MAGIC                                 7.8,0.76,0.04,2.3,0.092,15,54,0.997,3.26,0.65,9.8,
# MAGIC                                 11.2,0.28,0.56,1.9,0.075,17,60,0.998,3.16,0.58,9.8,
# MAGIC                                 7.4,0.7,0,1.9,0.076,11,34,0.9978,3.51,0.56,9.4),
# MAGIC                                 nrow = 5, ncol = length(column_names))
# MAGIC colnames(data_matrix) <- column_names
# MAGIC df <- as.data.frame(data_matrix)
# MAGIC
# MAGIC # Connect to MLflow and pull the model. Inject the Databricks host & token for MLflow authentication
# MAGIC Sys.setenv(DATABRICKS_HOST = "{workspace_url}", "DATABRICKS_TOKEN" = "{access_token}")
# MAGIC
# MAGIC best_model <- mlflow_load_model(model_uri = "{model_uri}")
# MAGIC
# MAGIC # Score the data
# MAGIC predictions <- data.frame(mlflow_predict(best_model, data = df))          
# MAGIC toString(predictions[[1]])
# MAGIC '''
# MAGIC
# MAGIC model_uri = dbutils.widgets.get("model_uri")
# MAGIC
# MAGIC if not model_uri:
# MAGIC     print("model_uri not specified. Will not test calling an MLflow model.")
# MAGIC else:
# MAGIC   # Run the R code and print the predictions
# MAGIC   predictions = execute(clusterId, context_id, code)
# MAGIC   print(predictions["results"]["data"])
