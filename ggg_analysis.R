library(tidyverse)
library(tidymodels)
library(vroom)
library(themis)
library(embed)

test <- vroom("test.csv")
train <- vroom("train.csv")
na_df <- vroom("trainWithMissingValues.csv")


## Recipe for imputing null values
impute_recipe <- recipe(type~., data = na_df) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_impute_knn(hair_length, impute_with = imp_vars(all_predictors()), neighbors = 4) %>%
  step_impute_knn(rotting_flesh, impute_with = imp_vars(all_predictors()), neighbors = 4) %>%
  step_impute_knn(bone_length, impute_with = imp_vars(all_predictors()), neighbors = 4)

mean_impute <- recipe(type~., data = na_df) %>%
  step_impute_mean(all_numeric_predictors())

prep <- prep(mean_impute)
imputed_df <- bake(prep, new_data = na_df)

rmse_vec(train[is.na(na_df)], imputed_df[is.na(na_df)])

# Normal Analysis ---------------------------------------------------------

## Create Recipe

my_recipe <- recipe(type~., data=train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) #%>%
  #step_normalize(all_numeric_predictors()) %>%
  #step_lencode_glm(all_nominal_predictors(), outcome = vars(type))

prep <- prep(my_recipe)
baked <- bake(prep, new_data = train)

# Support Vector Machines -------------------------------------------------


# Radial SVM
svmRadial <- svm_rbf(rbf_sigma = tune(), cost = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab")

svm_workflow <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(svmRadial)

## Tune 

## Grid of tuning values
tuning_grid <- grid_regular(rbf_sigma(),
                            cost(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(train, v = 10, repeats = 1)

# run Cross validation
CV_results <- svm_workflow %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

# find best parameters
bestTune <- CV_results %>%
  select_best("accuracy")

final_svm_workflow <- svm_workflow %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
svm_preds <- predict(final_svm_workflow,
                     new_data = test,
                     type = "class")

final_svm_preds <- tibble(id = test$id,
                          type = svm_preds$.pred_class)

vroom_write(final_svm_preds, "svm_predictions.csv", delim = ",")








