# load packages
library(tidyverse)
library(tidymodels)

# handle conflicts
tidymodels_prefer()

# set seed
set.seed(3013)


# load necessary objects
load("data/classification_setup.Rdata")
load("penalized_reg/model_results/penalized_reg_tuned.Rdata")

# check the best model results
show_best(penalized_reg_tuned, metric = "accuracy")

# FITTING THE WINNING MODEL AND ITS OPTIMIZED HYPERPARAMETERS TO THE ENTIRE TRAINING SET ####
penalized_reg_wflow_tuned <- penalized_reg_wflow %>% 
  finalize_workflow(select_best(penalized_reg_tuned, metric = "accuracy"))

penalized_reg_final_results <- fit(penalized_reg_wflow_tuned, loans_train)

# FITTING THE MODEL TO THE TESTING SET ####
penalized_reg_pred <- 
  predict(penalized_reg_final_results, new_data = loans_test) %>% 
  filter(!is.na(.pred_class)) %>% 
  rename("Category" = ".pred_class") %>% 
  mutate(id = row_number()) %>% 
  relocate(id, .before = Category)

# save out info ----
write_csv(penalized_reg_pred, file = "predictions_penalized_reg.csv")
