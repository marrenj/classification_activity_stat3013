# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# MARS TUNING FOR ENSEMBLE MODEL 1

# This model consists of knn, boosted tree, random forest, svm, neural net, and mars

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

# load package necessary for model engine
library(earth)

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))


# Define model ----
mars_model <- 
  mars(num_terms = tune(),
       prod_degree = tune()) %>% 
  set_engine("earth") %>% 
  set_mode("classification")

# check tuning parameters
hardhat::extract_parameter_set_dials(mars_model)

# set up tuning grid ----
mars_params <- hardhat::extract_parameter_set_dials(mars_model)

# Define grid ----
mars_grid <- grid_regular(mars_params, levels = 5)

# workflow ----
mars_workflow <- workflow() %>%
  add_model(mars_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
mars_res <- mars_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = mars_grid,
    control = control_stack_grid()
  )

# Write out results & workflow ----
save(mars_res, file = "ensemble_model1/model_results/mars_res.Rdata")



