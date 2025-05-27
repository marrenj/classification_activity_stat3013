# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# KNN TUNING FOR ENSEMBLE MODEL 1

# This model consists of knn, boosted tree, random forest, svm, neural net, and mars

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)

# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))

# Knn tuning ----


# Define model ----
knn_model <- nearest_neighbor(
  mode = "classification",
  neighbors = tune()
) %>%
  set_engine("kknn")

#check tuning parameters
hardhat::extract_parameter_set_dials(knn_model)

# set-up tuning grid ----
knn_params <- hardhat::extract_parameter_set_dials(knn_model) %>%
  update(neighbors = neighbors(range = c(1,100)))

# define grid
knn_grid <- grid_regular(knn_params, levels = 5)

# workflow ----
knn_workflow <- workflow() %>%
  add_model(knn_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
knn_res <- knn_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = knn_grid,
    control = control_stack_grid()
  )

# Write out results & workflow
save(knn_res, file = "ensemble_model1/model_results/knn_res.Rdata")


