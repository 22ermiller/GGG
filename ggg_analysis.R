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
  step_mutate_at(all_nominal_predictors(), fn = factor)#%>%
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


# Neural Networks ---------------------------------------------------------

nn_recipe <- recipe(formula = type~., data = train) %>%
  step_mutate_at(all_nominal_predictors(), fn = factor) %>%
  step_dummy(all_nominal_predictors()) %>%
  step_range(all_numeric_predictors(), min=0, max=1) # scale to [0,1]

nn_model <- mlp(hidden_units = tune(),
                epochs = 50#,
                #activation="relu"
                ) %>%
  set_engine("nnet") %>%
  #set_engine("keras", verbose = 0) %>% # verbose  = 0 prints less in console
  set_mode("classification")

nn_wf <- workflow() %>%
  add_recipe(nn_recipe) %>%
  add_model(nn_model)

nn_tuneGrid <- grid_regular(hidden_units(range=c(1, 60)),
                            levels=40)

# split data into folds
folds <- vfold_cv(train, v = 10, repeats = 1)

# run Cross validation
CV_results <- nn_wf %>%
  tune_grid(resamples = folds,
            grid = nn_tuneGrid,
            metrics = metric_set(roc_auc, accuracy))

# find best parameters
bestTune <- CV_results %>%
  select_best("accuracy")

# graph tuning parameters

CV_results %>% collect_metrics() %>%
  filter(.metric == "accuracy") %>%
  ggplot(aes(x=hidden_units, y=mean)) +
  geom_line()

final_nn_workflow <- nn_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
nn_preds <- predict(final_nn_workflow,
                     new_data = test,
                     type = "class")

final_nn_preds <- tibble(id = test$id,
                          type = nn_preds$.pred_class)

vroom_write(final_nn_preds, "nn_predictions.csv", delim = ",")


# Boosted Trees -----------------------------------------------------------

library(bonsai)
library(lightgbm)


boost_model <- boost_tree(tree_depth=tune(),
                          trees=tune(),
                          learn_rate=tune()) %>%
  set_engine("lightgbm") %>%
  set_mode("classification")

boost_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(boost_model)

tuning_grid <- grid_regular(tree_depth(),
                            trees(),
                            learn_rate(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(train, v = 10, repeats = 1)

# run Cross validation
CV_results <- boost_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc, accuracy))

# find best parameters
bestTune <- CV_results %>%
  select_best("accuracy")

final_boost_workflow <- boost_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
boost_preds <- predict(final_boost_workflow,
                     new_data = test,
                     type = "class")

final_boost_preds <- tibble(id = test$id,
                          type = boost_preds$.pred_class)

vroom_write(final_boost_preds, "boost_predictions.csv", delim = ",")


# BART --------------------------------------------------------------------


bart_model <- bart(trees=tune()) %>%
  set_engine("dbarts") %>%
  set_mode("classification")

bart_wf <- workflow() %>%
  add_recipe(my_recipe) %>%
  add_model(bart_model)

tuning_grid <- grid_regular(trees(),
                            levels = 5)

# split data into folds
folds <- vfold_cv(train, v = 10, repeats = 1)

# run Cross validation
CV_results <- bart_wf %>%
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# find best parameters
bestTune <- CV_results %>%
  select_best("accuracy")

final_bart_workflow <- bart_wf %>%
  finalize_workflow(bestTune) %>%
  fit(data = train)

# predict
bart_preds <- predict(final_bart_workflow,
                       new_data = test,
                       type = "class")

final_bart_preds <- tibble(id = test$id,
                            type = bart_preds$.pred_class)

vroom_write(final_bart_preds, "bart_predictions.csv", delim = ",")


