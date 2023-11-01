library(tidyverse)
library(tidymodels)
library(vroom)

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
