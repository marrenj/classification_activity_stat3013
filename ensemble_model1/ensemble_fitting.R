# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# ENSEMBLE MODEL 1 FITTING & ANALYSIS

# This model consists of knn, boosted tree, random forest, and mars models

# Load package(s) ----
library(tidymodels)
library(tidyverse)
library(stacks)

# Handle common conflicts
tidymodels_prefer()

# Load candidate model info ----
load("ensemble_model1/model_results/knn_res.Rdata")
load("ensemble_model1/model_results/boosted_tree_res.Rdata")
load("ensemble_model1/model_results/rf_res.Rdata")
load("ensemble_model1/model_results/mars_res.Rdata")
load("ensemble_model1/model_results/elastic_net_res.Rdata")

# Load data setup
load("data/classification_setup.Rdata")

# Create data stack ----
loans_data_stack <-
  stacks() %>% 
  add_candidates(knn_res) %>% 
  add_candidates(boosted_tree_res) %>% 
  add_candidates(rf_res) %>% 
  add_candidates(mars_res) %>% 
  add_candidates(elastic_net_res)

# Fit the stack ----
# penalty values for blending (set penalty argument when blending)
blend_penalty <- c(10^(-6:-1), 0.5, 1, 1.5, 2)

# Blend predictions using penalty defined above (tuning step, set seed) ----
set.seed(3013)

loans_model_stack <-
  loans_data_stack %>%
  blend_predictions(penalty = blend_penalty)

# Save blended model stack for reproducibility & easy reference (Rmd report) ----
save(loans_model_stack, file = "ensemble_model1/model_results/model_stack.Rdata")

# fit to ensemble to entire training set ----
loans_model_stack_trained <-
  loans_model_stack %>%
  fit_members()

# Save trained ensemble model for reproducibility & easy reference (Rmd report) ----
save(loans_model_stack_trained, file = "ensemble_model1/model_results/trained_model_stack.Rdata")

# Explore and assess trained ensemble model ----
collect_parameters(loans_model_stack_trained, "knn_res") %>% 
  filter(coef != 0) 

collect_parameters(loans_model_stack_trained, "boosted_tree_res") %>% 
  filter(coef != 0) 

collect_parameters(loans_model_stack_trained, "rf_res") %>% 
  filter(coef != 0)

collect_parameters(loans_model_stack_trained, "mars_res") %>% 
  filter(coef != 0)

collect_parameters(loans_model_stack_trained, "elastic_net_res") %>% 
  filter(coef != 0)

# fitting model to testing set ----
loans_test_pred <- 
  loans_test %>% 
  bind_cols(predict(loans_model_stack_trained, .)) %>% 
  filter(!is.na(.pred_class)) %>% 
  rename("Category" = ".pred_class") %>% 
  mutate("Id" = row_number()) %>% 
  select(Id, Category)


# saving predictions ----
write_csv(loans_test_pred, file = "predictions_ensemble_model1.csv", col_names = TRUE)





