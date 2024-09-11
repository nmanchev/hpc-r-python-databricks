# Shiny Example App

This directory contains a Shiny app shown in the [Running bioinformatics workloads at scale with R](https://www.youtube.com/watch?v=UP6cgJqKcLQ) video during the Posit Workbench and Posit Connect section.

To use run this app on Posit Workbench:

1. Load in the example data set:

```
CREATE CATALOG IF NOT EXISTS posit_demo;
CREATE TABLE posit_demo.default.lending_club CLONE PARQUET.`dbfs:/databricks-datasets/samples/lending_club/parquet/`;
```

2. Start a Posit Workbench session with Databricks credentials (chosen when launching a session)
3. Fill in your HTTPPath at the top of the `app.R` script.
4. Run in your Console `renv::restore()` to install the packages used in this project
5. Click the `Run App` button

To deploy this app on Posit Connect:

1. In Workbench, click the [blue publish application button](https://docs.posit.co/connect/user/publishing-rstudio/#publishing-general) in the upper right corner of the code editor window.
2. In Connect, click into the deployed application and [add the Azure Databricks integration from the list of available integrations](https://docs.posit.co/connect/user/oauth-integrations/#adding-oauth-integrations-to-deployed-content).
3. In Connect, customize and share the access link with colleagues
