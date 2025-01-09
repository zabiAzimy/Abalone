# Load necessary library
library(randomForest)
library(glmnet)
library(ggplot2)


# Install the library if required
install.packages("caret")
library(caret)

# Column names as described in the .names file
column_names <- c("Sex", "Length", "Diameter", "Height", 
                  "WholeWeight", "ShuckedWeight", 
                  "VisceraWeight", "ShellWeight", "Rings")

# Read the dataset into R
abalone_data <- read.csv("F:/BHT/2nd_winter_semester/ML2/project-ml/project-ml/abalone/abalone.data")

# dimension of the dataset
dim(abalone_data)

# Check the first few rows
head(abalone_data)

# View the structure of the data
str(abalone_data)

# Prepare data
set.seed(123)  # For reproducibility
abalone_data$Sex <- as.factor(abalone_data$Sex)

# look at the first rows of the dataset
head(abalone_data)
str(abalone_data)

summary(abalone_data)
# Split data into training (80%), and test (20%) sets
train_index <- createDataPartition(abalone_data$Rings, p = 0.8, list = FALSE)
train_data <- abalone_data[train_index, ]
test_data <- abalone_data[-train_index, ]


# Check the sizes of each set
cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
#############################################
# Final Report: Ridge and Lasso Regression on Abalone Dataset (with PCA and Polynomial Regression)

# Load necessary libraries
library(caret)
library(glmnet)
library(ggplot2)
library(corrplot)

setwd("F:/BHT/2nd_winter_semester/ML2/project-ml/project-ml/abalone")

# Column names for the abalone dataset
column_names <- c("Sex", "Length", "Diameter", "Height", 
                  "WholeWeight", "ShuckedWeight", 
                  "VisceraWeight", "ShellWeight", "Rings")

# Load the dataset
abalone_data <- read.csv("abalone.data", header = FALSE, col.names = column_names)

# Convert 'Sex' to factor and apply one-hot encoding
abalone_data$Sex <- as.factor(abalone_data$Sex)
abalone_data <- cbind(model.matrix(~Sex - 1, data=abalone_data), abalone_data[,-1])

# Correlation heatmap
cor_matrix <- cor(abalone_data[, -ncol(abalone_data)])
corrplot(cor_matrix, method = "color")

# Split the data into training (80%) and test (20%) sets
set.seed(123)
train_index <- createDataPartition(abalone_data$Rings, p = 0.8, list = FALSE)
train_data <- abalone_data[train_index, ]
test_data <- abalone_data[-train_index, ]

# Prepare data for Ridge and Lasso Regression
x_train <- as.matrix(train_data[,-ncol(train_data)])
y_train <- train_data$Rings
x_test <- as.matrix(test_data[,-ncol(test_data)])
y_test <- test_data$Rings

# Perform PCA
pca_model <- prcomp(x_train, scale. = TRUE)
x_train_pca <- pca_model$x[, 1:5]
x_test_pca <- predict(pca_model, newdata = x_test)[, 1:5]

# Hyperparameter tuning and cross-validation for Ridge Regression with PCA
ridge_grid <- expand.grid(alpha = 0, lambda = seq(0.001, 1, length = 50))
ridge_cv_pca <- train(x_train_pca, y_train, method = "glmnet", tuneGrid = ridge_grid, trControl = trainControl(method = "cv", number = 10))
ridge_pca_rmse <- sqrt(mean((predict(ridge_cv_pca, newdata = x_test_pca) - y_test)^2))

# Lasso Regression with PCA
lasso_grid <- expand.grid(alpha = 1, lambda = seq(0.001, 1, length = 50))
lasso_cv_pca <- train(x_train_pca, y_train, method = "glmnet", tuneGrid = lasso_grid, trControl = trainControl(method = "cv", number = 10))
lasso_pca_rmse <- sqrt(mean((predict(lasso_cv_pca, newdata = x_test_pca) - y_test)^2))

# Polynomial Regression
x_poly <- model.matrix(Rings ~ poly(Length, 2) + poly(Diameter, 2) + poly(Height, 2) +
                         poly(WholeWeight, 2) + poly(ShuckedWeight, 2) +
                         poly(VisceraWeight, 2) + poly(ShellWeight, 2), data = abalone_data)[, -1]

y <- abalone_data$Rings

# Split the data for polynomial regression
train_poly_index <- createDataPartition(y, p = 0.8, list = FALSE)
x_train_poly <- x_poly[train_poly_index, ]
y_train_poly <- y[train_poly_index]
x_test_poly <- x_poly[-train_poly_index, ]
y_test_poly <- y[-train_poly_index]

ridge_poly_model <- train(x_train_poly, y_train_poly, method = "glmnet", tuneGrid = ridge_grid, trControl = trainControl(method = "cv", number = 10))
ridge_poly_rmse <- sqrt(mean((predict(ridge_poly_model, newdata = x_test_poly) - y_test_poly)^2))

# Final RMSE Results Comparison
rmse_results <- data.frame(Model = c("Ridge with PCA", "Lasso with PCA", "Ridge with Polynomial"),
                           RMSE = c(ridge_pca_rmse, lasso_pca_rmse, ridge_poly_rmse))

ggplot(rmse_results, aes(x = Model, y = RMSE, fill = Model)) +
  geom_bar(stat = "identity") +
  theme_minimal() +
  ggtitle("RMSE Comparison Across Models")


