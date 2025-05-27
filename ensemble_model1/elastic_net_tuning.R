# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# ELASTIC NET TUNING FOR ENSEMBLE MODEL 1


# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

# load package necessary for model engine
library(ranger)

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))


# Define model ----
elastic_net_model <-
  logistic_reg(
    mixture = tune(),
    penalty = tune(),
    mode = "classification"
  ) %>% 
  set_engine("glmnet")

# check tuning parameters
hardhat::extract_parameter_set_dials(elastic_net_model)

# set up tuning grid ----
elastic_net_params <- hardhat::extract_parameter_set_dials(elastic_net_model) 

# Define grid ----
elastic_net_grid <- grid_regular(elastic_net_params, levels = 5)

# workflow ----
elastic_net_workflow <- workflow() %>%
  add_model(elastic_net_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
elastic_net_res <- elastic_net_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = elastic_net_grid,
    control = control_stack_grid()
  )

# Write out results & workflow ----
save(elastic_net_res, file = "ensemble_model1/model_results/elastic_net_res.Rdata")
