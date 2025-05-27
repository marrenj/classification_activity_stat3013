# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# RANDOM FOREST TUNING FOR ENSEMBLE MODEL 1

# This model consists of knn, boosted tree, random forest, svm, neural net, and mars

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
rf_model <-
  rand_forest(
    mtry = tune(),
    min_n = tune(),
    mode = "classification"
  ) %>% 
  set_engine("ranger")

# check tuning parameters
hardhat::extract_parameter_set_dials(rf_model)

# set up tuning grid ----
rf_params <- hardhat::extract_parameter_set_dials(rf_model) %>% 
  update(mtry = mtry(c(1, 5)))

# Define grid ----
rf_grid <- grid_regular(rf_params, levels = 5)

# workflow ----
rf_workflow <- workflow() %>%
  add_model(rf_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
rf_res <- rf_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = rf_grid,
    control = control_stack_grid()
  )

# Write out results & workflow ----
save(rf_res, file = "ensemble_model1/model_results/rf_res.Rdata")


