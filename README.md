# Capstone project

This project uses the dataset: **Forty soybean cultivars from subsequents harvests**.

It is intended to design a Machine learning model that could help predict the question stated.

## The question

This is a classification problem and we would like to predict what type of cultivar will be better to plant in a specific Season.

The main reasons needed to use an MLOps strategy are:

- Identify what season would be better to harvest the plants without the need of doing any manual work
- Automate the ML model deployments
- Train the models with new data

## Exploratory Data Analysis

From the dataset we can find the next variables:

|      Name      |Description                    |Type                         |
|----------------|-------------------------------|-----------------------------|
|PH				 |`Plant Height.` It is determined from the soil surface to the insertion of the last leaf using a milimeter ruler. | cm |
|IFP          	 |`Insertion of the first pod.` Determined from the soil surface to the insertion of the firts vegetable.  | cm |
|NS     	     |`Number of stems.` Through manual counting. |unit|
|NLP     	     |`Number of legumes per plant.` Through manual counting. |unit|
|NGP     	     |`Number of grains per plant.` Through manual counting. |unit|
|NGL     	     |`Number of grains per pod.` Through manual counting. |unit|
|MHG     	     |`Thousand seed weight.` According to the methodology described in Brasil. |g|
|GY     	     |`Grain yield.` Determined by harvesting the useful area of the plot and standardized to a grain moisture level of 13%. |kg ha^-1^|


### Target variable

There are different types of target variables:

- Categorical variables. These represent distinct classes or categories. They're often used in classification problems.
- Numerical variables. These can take on continuous values and are usually used in regression problems.
- Ordinal variables. These target variables have a specific order or rank.

This problem tries to solve a classification problem so we need a categorical variable. The main idea is to predict in which season we'll get better harvests.

According to the values shown in the data frame, the target variable could be the "Season" since it could only take the values 1 and 2.

It's also needed to analyze the correlations between variables so we can have a better understanding.

![diagram](images/target_variable_exploration.png)

### Analyzing the correlation of the variables

Let's see the correlation between the variables in the next graph. Some of the more meaningful correlations are NLP and NGP, NGP and NGL, and NS and NLP.

![diagram](images/correlations.png)

Checking the graphs for each corelation we have this:

- NLP and NGP

![diagram](images/nlp_ngp.png)

- NGP and NGL

![diagram](images/ngp_ngl.png)

- NS and NLP

![diagram](images/ns_nlp.png)

Using NS and NLP would help classify easier to know what season would be better to plant a cultivar.

> NOTE: We are also considering to reduce the overfitting in the model

## Architecture

We need an MLOps strategy so we can automate the model deployments. We would be able to test and simplify every release going through several environments like dev or prod.

At a first glance, this is the proposed architecture:

![diagram](images/architecture.png)

This architecture has all its services or most of them running in Kubernetes, this was choosen due to its capability to autoscale really quick. (not implemented)

As you can see on the left side, there's the Analytics section where the users can explore, analyze and validate the data and their operations.

Docker. There are 2 Docker containers:
- Container 1 is used to build, train and test the models (implemented)
- Container 2 is used to deploy or serve the versioned model (not implemented).

For CI/CD (Automation), Github actions would do the magic (not implemented with real configuration for this deployment).

![diagram](images/github_actions.png)

The part of the ETLs is proposed making use of Airflow to schedule the jobs for the data transformations. In the diagram the data would be stored through the different layers (bronze, silver and gold) and taken by the container from the datalake.

The model should take the data for training or testing depending on the operation. For this example the data is stored directly in the container.

To save the deployed model we could make use of DVC with S3 as blob storage behind, DVC will also allow us to version it (not implemented).

To monitor the models we need to choose the metrics we want to see; it can be used Prometheus for the metrics and Grafana to create dashboards for observability.

If we take a closer look at how the data and models are going to be processed, we can see those workflows as follows:

![diagram](images/pipelines.png)

## ML Model

Since we're trying to solve a classification problem and due to complexity, the more suitable models to predict the values with high accuracy would be a decision tree, a logistic regression or a random forest model.

Making a comparison of the models, we are getting a major accuracy with the Logistic Regression model. We would use it to classify the cultivars and decide on which season they should be planted according to the number of steams and legumes.

![diagram](images/models_comparison.png)

To check out the operations in details, see this [notebook](/jupyther-notebooks/Final_project.ipynb)
