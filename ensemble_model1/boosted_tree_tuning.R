# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# BOOSTED TREE TUNING FOR ENSEMBLE MODEL 1

# This model consists of knn, boosted tree, random forest, svm, neural net, and mars

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

# load package necessary for model engine
library("xgboost")

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))


# Define model ----
boosted_tree_model <-
  boost_tree(
    mode = "classification",
    mtry = tune(),
    min_n = tune(),
    learn_rate = tune()
  ) %>% 
  set_engine("xgboost")

# check tuning parameters
hardhat::extract_parameter_set_dials(boosted_tree_model)

# set up tuning grid ----
boosted_tree_params <- hardhat::extract_parameter_set_dials(boosted_tree_model) %>% 
  update(mtry = mtry(c(1, 6)))

# define grid ----
boosted_tree_grid <- grid_regular(boosted_tree_params, levels = 5)

# workflow ----
boosted_tree_workflow <- workflow() %>%
  add_model(boosted_tree_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
boosted_tree_res <- boosted_tree_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = boosted_tree_grid,
    control = control_stack_grid()
  )

# Write out results & workflow ----
save(boosted_tree_res, file = "ensemble_model1/model_results/boosted_tree_res.Rdata")


