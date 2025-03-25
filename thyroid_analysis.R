# Load necessary libraries
library(tidyverse)
library(caret)
library(randomForest)
library(class)
library(e1071)
library(pROC)
library(openxlsx)
library(kernlab)
library(class)

# Load data set
data <- read.csv("C:/Users/ASUS/Desktop/Duzce-CS/BM525-Veri Isleme/Classification/tyroid_ml_project_R/Thyroid_Diff.csv")

# Fix column names
colnames(data) <- gsub("[. ]", "_", colnames(data))

print("Data uploaded successfully.")

# Convert binary categorical variables (No Scaling Needed)
binary_cols <- c("Gender", "Smoking", "Hx_Smoking", "Hx_Radiothreapy", "Recurred")
for (col in binary_cols) {
  data[[col]] <- ifelse(data[[col]] %in% c("Yes", "F"), 1, 0)
}

# Ordinal Encoding
data$Risk <- as.numeric(factor(data$Risk, levels = c("Low", "Intermediate", "High"), ordered = TRUE))
data$Focality <- as.numeric(factor(data$Focality, levels = c("Uni-Focal", "Multi-Focal"), ordered = TRUE))
data$Stage <- as.numeric(factor(data$Stage, levels = c("I", "II", "III", "IVA", "IVB"), ordered = TRUE))
data$T <- as.numeric(factor(data$T, levels = c("T1a", "T1b", "T2", "T3a", "T3b", "T4a", "T4b"), ordered = TRUE))
data$N <- as.numeric(factor(data$N, levels = c("N0", "N1a", "N1b"), ordered = TRUE))
data$M <- as.numeric(factor(data$M, levels = c("M0", "M1"), ordered = TRUE))


# One-Hot Encoding
one_hot_cols <- c("Thyroid_Function", "Physical_Examination", "Pathology", "Response", "Adenopathy")
data <- dummyVars(~ ., data = data) %>% predict(data) %>% as.data.frame()

print("Encoding completed successfully.")


# Train-test split
set.seed(42)
train_index <- sample(1:nrow(data), 283, replace = FALSE)  
train_data <- data[train_index, ]
test_data <- data[-train_index, ]

print(paste("Train size:", nrow(train_data), "Test size:", nrow(test_data)))

# Split into features and target (corrected column names)
X_train <- subset(train_data, select = -Recurred)
X_test <- subset(test_data, select = -Recurred)

y_train <- train_data$Recurred
y_test <- test_data$Recurred
y_train <- as.factor(y_train)
y_test <- as.factor(y_test)

# Scaling
X_train <- scale(X_train)
X_test <- scale(X_test, center = attr(X_train, "scaled:center"), scale = attr(X_train, "scaled:scale"))

print("Data preprocessing completed successfully.")


## Train models
rf_model <- randomForest(X_train, y_train, ntree = 100)
knn_pred <- knn(train = X_train, test = X_test, cl = y_train, k = 5, prob = TRUE)
svm_model <- svm(X_train, y_train, kernel = "radial", probability = TRUE)

print("Models trained successfully.")

# Model evaluation function (includes PPV and NPV)
evaluate_model <- function(y_true, y_pred, y_proba = NULL) {
  cm <- confusionMatrix(y_pred, y_true, positive = "1")
  
  accuracy <- round(cm$overall["Accuracy"], 4)
  sensitivity <- round(cm$byClass["Sensitivity"], 4)  # Recall (True Positive Rate)
  specificity <- round(cm$byClass["Specificity"], 4)  # True Negative Rate
  ppv <- round(cm$byClass["Pos Pred Value"], 4)       # Precision
  npv <- round(cm$byClass["Neg Pred Value"], 4)       # Negative Predictive Value
  
  auc <- if (!is.null(y_proba)) {
    roc_curve <- roc(as.numeric(as.character(y_true)), y_proba)
    round(as.numeric(auc(roc_curve)), 4)
  } else {
    NA
  }
  
  list(
    Accuracy = accuracy,
    Sensitivity = sensitivity,
    Specificity = specificity,
    PPV = ppv,  # Precision
    NPV = npv,
    AUC = auc
  )
}

# RF probabilities
rf_probs <- predict(rf_model, X_test, type = "prob")[, 2]
rf_results <- evaluate_model(y_test, predict(rf_model, X_test), rf_probs)

# KNN probabilities
knn_probs <- ifelse(knn_pred == 1, attr(knn_pred, "prob"), 1 - attr(knn_pred, "prob"))
knn_results <- evaluate_model(y_test, knn_pred, knn_probs)

# SVM probabilities
svm_pred <- predict(svm_model, X_test, probability = TRUE)
svm_probs <- attr(svm_pred, "probabilities")[, "1"]
svm_results <- evaluate_model(y_test, svm_pred, svm_probs)

print("Model evaluation completed.")

# Print results
print("Random Forest Results:")
print(rf_results)

print("KNN Results:")
print(knn_results)

print("SVM Results:")
print(svm_results)

# Save results to Excel
results <- data.frame(
  Model = c("Random Forest", "KNN", "SVM"),
  Accuracy = c(rf_results$Accuracy, knn_results$Accuracy, svm_results$Accuracy),
  Sensitivity = c(rf_results$Sensitivity, knn_results$Sensitivity, svm_results$Sensitivity),
  Specificity = c(rf_results$Specificity, knn_results$Specificity, svm_results$Specificity),
  PPV = c(rf_results$PPV, knn_results$PPV, svm_results$PPV),
  NPV = c(rf_results$NPV, knn_results$NPV, svm_results$NPV),
  AUC = c(rf_results$AUC, knn_results$AUC, svm_results$AUC)
)

write.xlsx(results, "C:/Users/ASUS/Desktop/Duzce-CS/BM525-Veri Isleme/Classification/tyroid_ml_project_R/results.xlsx")
print("Results saved to results.xlsx.")

