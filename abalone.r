# Load necessary libraries
library(caret)
library(e1071)

# Set random seed for reproducibility
set.seed(123)

# Define column names for the dataset
column_names <- c("Sex", "Length", "Diameter", "Height", 
                  "WholeWeight", "ShuckedWeight", 
                  "VisceraWeight", "ShellWeight", "Rings")

# Read the dataset into R
abalone_data <- read.csv('Abalone/abalone/abalone.data', header = FALSE, col.names = column_names)

# Check for missing values
cat("Number of missing values:", sum(is.na(abalone_data)), "\n")

# Convert 'Sex' to a factor
abalone_data$Sex <- as.factor(abalone_data$Sex)

# Split data into training (80%) and test (20%) sets
train_index <- createDataPartition(abalone_data$Rings, p = 0.8, list = FALSE)
train_data <- abalone_data[train_index, ]
test_data <- abalone_data[-train_index, ]

# Scale the numeric features in training data
numeric_cols <- setdiff(names(train_data), "Sex")  # Exclude the 'Sex' column
train_data_scaled <- train_data
train_data_scaled[numeric_cols] <- scale(train_data[numeric_cols])

# Scale the numeric features in test data using training data's parameters
test_data_scaled <- test_data
test_data_scaled[numeric_cols] <- scale(test_data[numeric_cols], 
                                        center = attr(scale(train_data[numeric_cols]), "scaled:center"),
                                        scale = attr(scale(train_data[numeric_cols]), "scaled:scale"))

# Dummy variable encoding for 'Sex' in training data
dummy_vars_train <- dummyVars(~ Sex, data = train_data_scaled)
dummy_sex_train <- predict(dummy_vars_train, newdata = train_data_scaled)
train_data_final <- cbind(dummy_sex_train, train_data_scaled[, -which(names(train_data_scaled) == "Sex")])

# Dummy variable encoding for 'Sex' in test data
dummy_vars_test <- dummyVars(~ Sex, data = test_data_scaled)
dummy_sex_test <- predict(dummy_vars_test, newdata = test_data_scaled)
test_data_final <- cbind(dummy_sex_test, test_data_scaled[, -which(names(test_data_scaled) == "Sex")])

# Define training control for cross-validation
train_control <- trainControl(method = "cv", number = 5)  # 5-fold cross-validation

### 1. SVR with Linear Kernel
set.seed(123)
svr_linear_model <- train(
  Rings ~ ., 
  data = train_data_final, 
  method = "svmLinear",
  trControl = train_control,
  tuneGrid = expand.grid(C = seq(0.1, 2, by = 0.2))  # Grid search for 'C'
)

# Best hyperparameters for linear kernel
cat("Best parameters for linear kernel:\n")
print(svr_linear_model$bestTune)

# look at the model summary
svr_linear_model

# Make predictions and evaluate performance
linear_predictions <- predict(svr_linear_model, newdata = test_data_final)
linear_rmse <- sqrt(mean((linear_predictions - test_data_final$Rings)^2))
cat("Linear Kernel RMSE:", linear_rmse, "\n")


### 2. SVR with RBF (Radial Basis Function) Kernel
set.seed(123)
svr_rbf_model <- train(
  Rings ~ ., 
  data = train_data_final, 
  method = "svmRadial",
  trControl = train_control,
  tuneGrid = expand.grid(
    sigma = c(0.01, 0.05, 0.1),  # Grid search for 'sigma'
    C = seq(0.1, 2, by = 0.2)    # Grid search for 'C'
  )
)

# look at the model summary
svr_rbf_model

# Best hyperparameters for RBF kernel
cat("Best parameters for RBF kernel:\n")
print(svr_rbf_model$bestTune)

# Make predictions and evaluate performance
rbf_predictions <- predict(svr_rbf_model, newdata = test_data_final)
rbf_rmse <- sqrt(mean((rbf_predictions - test_data_final$Rings)^2))
cat("RBF Kernel RMSE:", rbf_rmse, "\n")


# These results indicate that the RBF kernel (non-linear) performs better than the 
# linear kernel for predicting the target variable Rings
# as evidenced by its lower Root Mean Squared Error (RMSE) of 0.6304323 compared to 0.6880048 Of Linear kernel
