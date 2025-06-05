options(timeout = 120)

# Import dataset from the ICS database
initial_data <- "adult.zip"
if(!file.exists(initial_data))
  download.file("https://archive.ics.uci.edu/static/public/2/adult.zip", initial_data)

adult_data <- "adult.data"
if(!file.exists(adult_data))
  unzip(initial_data, adult_data)

# Load libraries
library(ggplot2)
library(nnet)
library(gbm)
library(tidyverse)
library(caret)
library(glmnet)
library(randomForest)
library(pROC)
library(xgboost)
library(reshape2)

# Load data and preprocess it
adult_income <- read.csv(adult_data, header = FALSE, strip.white = TRUE, stringsAsFactors = FALSE)

# Naming columns
colnames(adult_income) <- c("age", "workclass", "fnlwgt", "education", "education_num",
                            "marital_status", "occupation", "relationship", "race",
                            "sex", "capital_gain", "capital_loss", "hours_per_week",
                            "native_country", "income")

# Set non-numerical variables as factors
adult_income <- adult_income %>% drop_na() %>%
  mutate(across(c(workclass, education, marital_status, occupation,
                  relationship, race, sex, native_country, income), as.factor))

# Split data into a training and a test set
set.seed(123)
train_index <- createDataPartition(adult_income$income, p = 0.7, list = FALSE)
train_data <- adult_income[train_index, ]
test_data <- adult_income[-train_index, ]

# Conduct additional features engineering
convert_income <- function(data, target = "income", type = c("factor", "numeric")) {
  type <- match.arg(type)
  if (type == "factor") {
    data[[target]] <- factor(data[[target]], levels = c("<=50K", ">50K"), labels = c("low", "high"))
  } else {
    data[[target]] <- ifelse(data[[target]] == ">50K", 1, 0)
  }
  return(data)
}

train_data_factor <- convert_income(train_data, type = "factor")
test_data_factor <- convert_income(test_data, type = "factor")
train_data_numeric <- convert_income(train_data, type = "numeric")
test_data_numeric <- convert_income(test_data, type = "numeric")

ctrl <- trainControl(method = "cv", number = 5, classProbs = TRUE,
                     summaryFunction = twoClassSummary, savePredictions = TRUE)

# Create common dummyVars object
xform <- dummyVars(income ~ ., data = train_data_factor)
train_x <- predict(xform, newdata = train_data_factor) %>% as.matrix()
test_x <- predict(xform, newdata = test_data_factor) %>% as.matrix()
train_y <- train_data_factor$income
test_y <- test_data_factor$income

# Pre-stage the models' evaluation and comparison
evaluate_model <- function(pred_class, prob, true_labels, label) {
  cm <- confusionMatrix(pred_class, true_labels, positive = "high")
  roc_obj <- roc(true_labels, prob, levels = c("low", "high"), direction = ">")
  TP <- cm$table[2, 2]; FP <- cm$table[2, 1]; FN <- cm$table[1, 2]
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall == 0), NA,
               2 * precision * recall / (precision + recall))
  logloss <- logLoss(ifelse(true_labels == "high", 1, 0), prob)
  data.frame(Model = label, AUC = as.numeric(auc(roc_obj)), F1 = f1,
             Accuracy = cm$overall["Accuracy"],
             Sensitivity = cm$byClass["Sensitivity"],
             Specificity = cm$byClass["Specificity"],
             Precision = precision,
             LogLoss = logloss)
}

results <- list()

# Model fitting section
# Logistic Regression
m1 <- train(income ~ ., data = train_data_factor, method = "glm", family = "binomial", metric = "ROC", trControl = ctrl)
p1 <- predict(m1, test_data_factor)
p1_prob <- predict(m1, test_data_factor, type = "prob")$high
results[["Logistic Regression"]] <- evaluate_model(p1, p1_prob, test_data_factor$income, "Logistic Regression")

# Random Forest
m2 <- train(income ~ ., data = train_data_factor, method = "rf", metric = "ROC", trControl = ctrl)
p2 <- predict(m2, test_data_factor)
p2_prob <- predict(m2, test_data_factor, type = "prob")$high
results[["Random Forest"]] <- evaluate_model(p2, p2_prob, test_data_factor$income, "Random Forest")

# XGBoost
m3 <- train(income ~ ., data = train_data_factor, method = "xgbTree", metric = "ROC", trControl = ctrl)
p3 <- predict(m3, test_data_factor)
p3_prob <- predict(m3, test_data_factor, type = "prob")$high
results[["XGBoost"]] <- evaluate_model(p3, p3_prob, test_data_factor$income, "XGBoost")

# GBM
m4 <- train(income ~ ., data = train_data_factor, method = "gbm", metric = "ROC", trControl = ctrl, verbose = FALSE)
p4 <- predict(m4, test_data_factor)
p4_prob <- predict(m4, test_data_factor, type = "prob")$high
results[["GBM"]] <- evaluate_model(p4, p4_prob, test_data_factor$income, "GBM")

# Neural Net
m5 <- train(income ~ ., data = train_data_factor, method = "nnet", metric = "ROC", trControl = ctrl, trace = FALSE)
p5 <- predict(m5, test_data_factor)
p5_prob <- predict(m5, test_data_factor, type = "prob")$high
results[["Neural Net"]] <- evaluate_model(p5, p5_prob, test_data_factor$income, "Neural Net")

# GLMNET Tuned
set.seed(123)
glmnet_cv <- cv.glmnet(train_x, as.numeric(train_data_numeric$income), family = "binomial", type.measure = "auc", alpha = 0.5)
pred_prob <- predict(glmnet_cv, newx = test_x, s = "lambda.min", type = "response")
thresh_seq <- seq(0.1, 0.9, by = 0.05)
metrics <- sapply(thresh_seq, function(t) {
  pred_class <- ifelse(pred_prob >= t, 1, 0)
  cm <- confusionMatrix(factor(pred_class, levels = c(0, 1)), factor(test_data_numeric$income, levels = c(0, 1)))
  TP <- cm$table[2, 2]; FP <- cm$table[2, 1]; FN <- cm$table[1, 2]
  precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
  recall <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
  f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall == 0), NA,
               2 * precision * recall / (precision + recall))
  return(f1)
})
best_thresh <- thresh_seq[which.max(metrics)]
final_pred <- ifelse(pred_prob >= best_thresh, 1, 0)
roc_glmnet <- roc(test_data_numeric$income, pred_prob)
logloss_glmnet <- logLoss(test_data_numeric$income, pred_prob)
cm_glmnet <- confusionMatrix(factor(final_pred, levels = c(0, 1)), factor(test_data_numeric$income, levels = c(0, 1)))
TP <- cm_glmnet$table[2,2]; FP <- cm_glmnet$table[2,1]; FN <- cm_glmnet$table[1,2]
precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
recall <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
f1 <- ifelse(is.na(precision) | is.na(recall) | (precision + recall == 0), NA, 2 * precision * recall / (precision + recall))
results[["GLMNET Tuned"]] <- data.frame(Model = "GLMNET Tuned", AUC = as.numeric(auc(roc_glmnet)),
                                        F1 = f1,
                                        Accuracy = cm_glmnet$overall["Accuracy"],
                                        Sensitivity = cm_glmnet$byClass["Sensitivity"],
                                        Specificity = cm_glmnet$byClass["Specificity"],
                                        Precision = precision,
                                        LogLoss = logloss_glmnet)

# Combine and plot heatmap
final_results <- bind_rows(results)
final_results_rounded <- final_results %>% mutate(across(where(is.numeric), round, 3))
print(final_results_rounded)

# Heatmap
heat_data <- melt(final_results_rounded, id.vars = "Model")
ggplot(heat_data, aes(x = variable, y = Model, fill = value)) +
  geom_tile(color = "white") +
  scale_fill_gradient2(low = "red", high = "green", mid = "yellow", midpoint = 0.75, limit = c(0, 1), space = "Lab") +
  geom_text(aes(label = round(value, 2)), color = "black", size = 3.5) +
  theme_minimal() +
  labs(title = "Model Performance Heatmap", x = "Metric", y = "Model")

## Compare models using F1 scores across thresholds

# Define thresholds to evaluate
thresholds <- seq(0.1, 0.9, by = 0.01)

# Helper function to compute F1 score at different thresholds
compute_f1_curve <- function(probs, true_labels, model_name) {
  sapply(thresholds, function(t) {
    preds <- ifelse(probs >= t, "high", "low") %>% factor(levels = c("low", "high"))
    cm <- confusionMatrix(preds, true_labels, positive = "high")
    TP <- cm$table[2, 2]; FP <- cm$table[2, 1]; FN <- cm$table[1, 2]
    precision <- ifelse((TP + FP) == 0, NA, TP / (TP + FP))
    recall <- ifelse((TP + FN) == 0, NA, TP / (TP + FN))
    if (is.na(precision) | is.na(recall) | (precision + recall == 0)) return(NA)
    return(2 * precision * recall / (precision + recall))
  }) %>%
    data.frame(threshold = thresholds, F1 = ., model = model_name)
}

# Compute F1 score curves for each model
f1_curves <- bind_rows(
  compute_f1_curve(p1_prob, test_y, "Logistic Regression"),
  compute_f1_curve(p2_prob, test_y, "Random Forest"),
  compute_f1_curve(p3_prob, test_y, "XGBoost"),
  compute_f1_curve(p4_prob, test_y, "GBM"),
  compute_f1_curve(p5_prob, test_y, "Neural Net"),
  compute_f1_curve(as.vector(pred_prob), factor(test_data_numeric$income, levels = c(0, 1), labels = c("low", "high")), "GLMNET Tuned")
)

# Find optimal threshold points
opt_f1 <- f1_curves %>%
  group_by(model) %>%
  slice_max(F1, with_ties = FALSE)

# Plot
ggplot(f1_curves, aes(x = threshold, y = F1, color = model)) +
  geom_line(size = 1) +
  geom_point(data = opt_f1, aes(x = threshold, y = F1), size = 3, shape = 21, fill = "white") +
  geom_text(data = opt_f1, aes(label = paste0("Opt: ", round(threshold, 2))), vjust = -1, size = 3) +
  theme_minimal() +
  labs(title = "F1 Score vs. Threshold for All Models",
       x = "Threshold",
       y = "F1 Score",
       color = "Model")

## Tuned glmnet ROC curve

# Predict probabilities from tuned GLMNET model
pred_prob_glmnet <- predict(glmnet_cv, newx = test_x, s = "lambda.min", type = "response")

# Compute ROC
roc_glmnet <- roc(test_data_numeric$income, pred_prob_glmnet)

# Plot ROC
plot(roc_glmnet, col = "blue", lwd = 2,
     main = "ROC Curve - Tuned GLMNET",
     legacy.axes = TRUE)
abline(a = 0, b = 1, lty = 2, col = "gray")

# Add AUC text
auc_val <- auc(roc_glmnet)
legend("bottomright", legend = paste0("AUC = ", round(auc_val, 3)),
       col = "blue", lwd = 2, bty = "n")