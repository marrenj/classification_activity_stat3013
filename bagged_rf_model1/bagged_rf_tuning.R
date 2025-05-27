# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# BAGGED RANDOM FOREST MODEL 1 

# This model consists of knn, boosted tree, random forest, svm, neural net, and mars

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(stacks)
library(doMC)
library(rpart)
library(baguette)


# Handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))


# Define model ----
bagged_rf_model <-
  bag_tree(
    tree_depth = tune(),
    min_n = tune()
  ) %>% 
  set_engine("rpart", times = 25) %>% 
  set_mode("classification") 

# check tuning parameters
hardhat::extract_parameter_set_dials(bagged_rf_model)

# set up tuning grid ----
bagged_rf_params <- hardhat::extract_parameter_set_dials(bagged_rf_model) 

# Define grid ----
bagged_rf_grid <- grid_regular(bagged_rf_params, levels = 5)

# workflow ----
bagged_rf_workflow <- workflow() %>%
  add_model(bagged_rf_model) %>%
  add_recipe(basic_classification_recipe)

# Tuning/fitting ----
bagged_rf_res <- bagged_rf_workflow %>%
  tune_grid(
    resamples = loans_folds,
    grid = bagged_rf_grid,
    control = keep_pred
  )

# Write out results & workflow ----
save(bagged_rf_res, file = "bagged_rf_model1/model_results/bagged_rf_res.Rdata")


# FITTING THE WINNING MODEL AND ITS OPTIMIZED HYPERPARAMETERS TO THE ENTIRE TRAINING SET ####

# finalize workflow
bagged_rf_tuned <- bagged_rf_workflow %>% 
  finalize_workflow(select_best(bagged_rf_res, metric = "accuracy"))

# train
bagged_rf_trained <- fit(bagged_rf_tuned, loans_train)

# FITTING THE MODEL TO THE TESTING SET ####

loans_test_pred <- 
  predict(bagged_rf_trained, new_data = loans_test) %>% 
  filter(!is.na(.pred_class)) %>% 
  rename("Category" = ".pred_class") %>% 
  mutate("Id" = row_number()) %>% 
  select(Id, Category)

# write out results
write_csv(loans_test_pred, file = "predictions_bagged_rf_model1.csv")


