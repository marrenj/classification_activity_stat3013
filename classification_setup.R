# MARREN JENKINS
# STAT 301-3 DATA SCIENCE 3 WITH R
# CLASSIFICATION ACTIVITY
# DATA SETUP


# Load package(s)
library(tidymodels)
library(tidyverse)
library(ggplot2)
library(lubridate)

# handle common conflicts
tidymodels_prefer()

# Seed
set.seed(3013)

## TRAINING DATA WRANGLING ----

loans_train <- read_csv("data/train.csv") %>%
  # clean names
  janitor::clean_names() %>% 
  mutate(
    # make application_type into a factor
    application_type = factor(application_type),
    # make earliest credit line into a numeric variable that is the number of days since opening earliest credit line
    earliest_cr_line = lubridate::my(earliest_cr_line),
    earliest_cr_line = as.numeric(today() - earliest_cr_line),
    # make employment length into a factor
    emp_length = factor(emp_length),
    emp_length = fct_recode(emp_length, "0" = "< 1 year",
                            "1" = "1 year",
                            "2" = "2 years",
                            "3" = "3 years",
                            "4" = "4 years",
                            "5" = "5 years",
                            "6" = "6 years",
                            "7" = "7 years",
                            "8" = "8 years",
                            "9" = "9 years",
                            "10" = "10+ years"),
    # make loan grade into a factor 
    grade = factor(grade),
    # make home ownership into a factor
    home_ownership = factor(home_ownership),
    # make factor
    initial_list_status = factor(initial_list_status),
    # make last_credit_pull_d into a numeric variable that is the number of days since opening earliest credit line
    last_credit_pull_d = lubridate::my(last_credit_pull_d),
    last_credit_pull_d = as.numeric(today() - last_credit_pull_d),
    # make loan sub-grade into a factor
    sub_grade = factor(sub_grade),
    # make term into a factor
    term = factor(term),
    # make into factor
    verification_status = factor(verification_status),
    # make into factor
    purpose = factor(purpose),
    acc_now_delinq = ifelse(acc_now_delinq == 0, 0, 1),
    acc_now_delinq = factor(acc_now_delinq),
    delinq_2yrs = ifelse(delinq_2yrs == 0, 0, 1),
    delinq_2yrs = factor(delinq_2yrs),
    hi_int_prncp_pd = factor(hi_int_prncp_pd)
  )


# inspect data
loans_train %>% 
  ggplot(aes(hi_int_prncp_pd)) +
  geom_boxplot()



## TESTING DATA WRANGLING

loans_test <- read_csv("data/test.csv") %>%
  # clean names
  janitor::clean_names() %>% 
  mutate(
    # make application_type into a factor
    application_type = factor(application_type),
    # make earliest credit line into a numeric variable that is the number of days since opening earliest credit line
    earliest_cr_line = lubridate::my(earliest_cr_line),
    earliest_cr_line = as.numeric(today() - earliest_cr_line),
    # make employment length into a factor
    emp_length = factor(emp_length),
    emp_length = fct_recode(emp_length, "0" = "< 1 year",
                            "1" = "1 year",
                            "2" = "2 years",
                            "3" = "3 years",
                            "4" = "4 years",
                            "5" = "5 years",
                            "6" = "6 years",
                            "7" = "7 years",
                            "8" = "8 years",
                            "9" = "9 years",
                            "10" = "10+ years"),
    # make loan grade into a factor 
    grade = factor(grade),
    # make home ownership into a factor
    home_ownership = factor(home_ownership),
    # make factor
    initial_list_status = factor(initial_list_status),
    # make last_credit_pull_d into a numeric variable that is the number of days since opening earliest credit line
    last_credit_pull_d = lubridate::my(last_credit_pull_d),
    last_credit_pull_d = as.numeric(today() - last_credit_pull_d),
    # make loan sub-grade into a factor
    sub_grade = factor(sub_grade),
    # make term into a factor
    term = factor(term),
    # make into factor
    verification_status = factor(verification_status),
    # make into factor
    purpose = factor(purpose),
    acc_now_delinq = ifelse(acc_now_delinq == 0, 0, 1),
    acc_now_delinq = factor(acc_now_delinq),
    delinq_2yrs = ifelse(delinq_2yrs == 0, 0, 1),
    delinq_2yrs = factor(delinq_2yrs)
  )


## FEATURE ENGINEERING ----

# create a variable for total debt and debt-to-loan ratio
loans_train <- loans_train %>% 
  mutate(debt = (dti * annual_inc),
         dtl = (debt / loan_amnt)) %>% 
  select(-c(acc_now_delinq,
            acc_open_past_24mths, 
            addr_state, 
            annual_inc,
            avg_cur_bal,
            bc_util,
            delinq_2yrs,
            delinq_amnt,
            dti,
            earliest_cr_line,
            emp_length,
            emp_title, 
            grade,
            last_credit_pull_d,
            mort_acc,
            num_sats,
            num_tl_120dpd_2m,
            num_tl_30dpd,
            num_tl_90g_dpd_24m,
            pub_rec,
            pub_rec_bankruptcies,
            purpose,
            sub_grade, 
            tot_coll_amt,
            tot_cur_bal,
            total_rec_late_fee,
            verification_status,
            id))

loans_test <- loans_test %>% 
  mutate(
    debt = (dti * annual_inc),
    dtl = (debt / loan_amnt)
  ) %>% 
  select(-c(acc_now_delinq,
            acc_open_past_24mths, 
            addr_state, 
            annual_inc,
            avg_cur_bal,
            bc_util,
            delinq_2yrs,
            delinq_amnt,
            dti,
            earliest_cr_line,
            emp_length,
            emp_title, 
            grade,
            last_credit_pull_d,
            mort_acc,
            num_sats,
            num_tl_120dpd_2m,
            num_tl_30dpd,
            num_tl_90g_dpd_24m,
            pub_rec,
            pub_rec_bankruptcies,
            purpose,
            sub_grade, 
            tot_coll_amt,
            tot_cur_bal,
            total_rec_late_fee,
            verification_status,
            id))



# SETTING RESAMPLING METHOD ####

# 5-fold cross-validation with 3 repeats
loans_folds <- 
  loans_train %>% 
  vfold_cv(v = 10, repeats = 5, strata = hi_int_prncp_pd)

# set resampling options
keep_pred <- control_grid(save_pred = TRUE, save_workflow= TRUE)


# CREATING A RECIPE ####
basic_classification_recipe <- recipe(hi_int_prncp_pd ~ term + int_rate + out_prncp_inv + loan_amnt + debt, data = loans_train) %>%
  step_normalize(all_numeric_predictors()) %>% 
  step_novel(all_nominal_predictors()) %>% 
  step_dummy(all_nominal_predictors()) %>% 
  step_nzv(all_predictors()) #%>% 
  #prep() %>% 
  #bake(new_data = NULL)




# SAVING INITIAL SETUP ####
save(loans_folds, 
     keep_pred, 
     loans_train, 
     loans_test,
     basic_classification_recipe, 
     file = "data/classification_setup.Rdata")

