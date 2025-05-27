# Penalized Regression tuning ----

# Load package(s) ----
library(tidyverse)
library(tidymodels)
library(doMC)

# load package necessary for model engine
library(glmnet)

# handle common conflicts
tidymodels_prefer()

# set seed
set.seed(3013)

# load required objects ----
load("data/classification_setup.Rdata")

# register cores for parallel processing ----
registerDoMC(cores = parallel::detectCores(logical = TRUE))

# Define model ----
penalized_reg_spec <-
  logistic_reg(mixture = tune(), 
             penalty = tune(),
             mode = "classification") %>% 
  set_engine("glmnet")

# set up tuning grid ----
penalized_reg_params <- parameters(penalized_reg_spec) 

penalized_reg_grid <- grid_regular(penalized_reg_params, levels = 5)


# workflow ----
penalized_reg_wflow <- workflow() %>% 
  add_model(penalized_reg_spec) %>% 
  add_recipe(basic_classification_recipe)


# Tuning/fitting ----

# tune model ----
penalized_reg_tuned <-
  penalized_reg_wflow %>% 
  tune_grid(
    resamples = loans_folds,
    grid = penalized_reg_grid
  )


# save out info ----
save(penalized_reg_spec, penalized_reg_wflow, penalized_reg_tuned, file = "penalized_reg/model_results/penalized_reg_tuned.Rdata")


