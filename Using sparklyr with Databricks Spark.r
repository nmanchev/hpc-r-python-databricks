# Databricks notebook source
# MAGIC %md
# MAGIC #Using sparklyr with Databricks Spark
# MAGIC
# MAGIC `sparklyr` is an R package that allows users to work with Apache Spark, a popular open-source big data processing engine, from within R. `sparklyr` provides a set of APIs that enable R users to connect to Spark clusters, manipulate and analyze large-scale data, and leverage Spark's distributed computing capabilities. 
# MAGIC
# MAGIC With `sparklyr`, users can create and manage Spark DataFrames, which are similar to R data frames but can handle massive amounts of data. The package also supports a range of data operations, including filtering, grouping, and joining, as well as machine learning algorithms and data visualization tools. By integrating Spark with R, `sparklyr` enables data scientists and analysts to work with large datasets and perform complex computations without having to leave the familiar R environment.
# MAGIC
# MAGIC In this notebook we connect to the backend Databricks Spark cluster using `sparklyr` and execute one of the [TPC-H](https://www.tpc.org/tpch/) queries. Despite using a small subset of the original TPC-H dataset, this query still takes some time to complete as it is one of the heavy queries in the benchmark. We can experiment by changing the size of the attached Spark cluster and observe how the processing times change.

# COMMAND ----------

library(sparklyr)

# Connect to the Databricks cluster
sc <- sparklyr::spark_connect(method = "databricks")

# Store the Unity catalog, database, and table name in variables schema and table
schema <- "samples.tpch"

sparklyr::tbl_change_db(sc, schema)

sdf <- sparklyr::sdf_sql(sc, "
select
	nation,
	o_year,
	sum(amount) as sum_profit
from
	(
		select
			n_name as nation,
			extract(year from o_orderdate) as o_year,
			l_extendedprice * (1 - l_discount) - ps_supplycost * l_quantity as amount
		from
			samples.tpch.part,
			samples.tpch.supplier,
			samples.tpch.lineitem,
			samples.tpch.partsupp,
			samples.tpch.orders,
			samples.tpch.nation
		where
			s_suppkey = l_suppkey
			and ps_suppkey = l_suppkey
			and ps_partkey = l_partkey
			and p_partkey = l_partkey
			and o_orderkey = l_orderkey
			and s_nationkey = n_nationkey
			and (p_name like 'b%' or p_name like 'a%')
	) as profit
group by nation, o_year
order by nation, o_year desc
limit 1")
display(head(sdf))

# COMMAND ----------

# MAGIC %md
# MAGIC After setting up the sparklyr connection, you can use the sparklyr API. You can import and combine sparklyr with dplyr or MLlib.

# COMMAND ----------

library(dplyr)

iris_tbl <- copy_to(sc, iris)
src_tbls(sc)
iris_tbl %>% count

# COMMAND ----------

library(ggplot2)

# Change the default plot height 
options(repr.plot.height = 300)

iris_summary <- iris_tbl %>% 
  mutate(Sepal_Width = ROUND(Sepal_Width * 2) / 2) %>% # Bucketizing Sepal_Width
  group_by(Species, Sepal_Width) %>% 
  summarize(count = n(), Sepal_Length_Mean = mean(Sepal_Length), stdev = sd(Sepal_Length)) %>% collect
 
ggplot(iris_summary, aes(Sepal_Width, Sepal_Length_Mean, color = Species)) + 
  geom_line(size = 1.2) +
  geom_errorbar(aes(ymin = Sepal_Length_Mean - stdev, ymax = Sepal_Length_Mean + stdev), width = 0.05) +
  geom_text(aes(label = count), vjust = -0.2, hjust = 1.2, color = "black") +
  theme(legend.position="top")

