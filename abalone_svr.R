# Load necessary library
library(randomForest)
library(caret)

# Install the library if required
install.packages("caret")
# Load the necessary library for SVR
install.packages("e1071")
library(e1071)


# Column names as described in the .names file
column_names <- c("Sex", "Length", "Diameter", "Height", 
                  "WholeWeight", "ShuckedWeight", 
                  "VisceraWeight", "ShellWeight", "Rings")

# Read the dataset into R
abalone_data <- read.csv('abalone/abalone.data', header = FALSE, col.names = column_names)

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
# Split data into training (80%) and test (20%) sets
train_index <- createDataPartition(abalone_data$Rings, p = 0.8, list = FALSE)
train_data <- abalone_data[train_index, ]
test_data <- abalone_data[-train_index, ]


# Check the sizes of each set
cat("Training set size:", nrow(train_data), "\n")
cat("Test set size:", nrow(test_data), "\n")


# Train the SVR model
set.seed(123)  # For reproducibility
svr_model <- svm(Rings ~ ., data = train_data, kernel = "radial", cost = 1, gamma = 0.1)

# Summarize the model
summary(svr_model)

# Predict on the test set
svr_predictions <- predict(svr_model, test_data)

# Evaluate model performance
MAE <- mean(abs(svr_predictions - test_data$Rings))
RMSE <- sqrt(mean((svr_predictions - test_data$Rings)^2))

cat("Mean Absolute Error (MAE):", MAE, "\n")
cat("Root Mean Square Error (RMSE):", RMSE, "\n")


# Train the SVR model with a linear kernel
set.seed(123)  # For reproducibility
svr_model_linear <- svm(Rings ~ ., data = train_data, kernel = "linear", cost = 1)

# Summarize the model
summary(svr_model_linear)

# Predict on the test set
svr_predictions_linear <- predict(svr_model_linear, test_data)

# Evaluate model performance
MAE_linear <- mean(abs(svr_predictions_linear - test_data$Rings))
RMSE_linear <- sqrt(mean((svr_predictions_linear - test_data$Rings)^2))

cat("Mean Absolute Error (MAE) with Linear Kernel:", MAE_linear, "\n")
cat("Root Mean Square Error (RMSE) with Linear Kernel:", RMSE_linear, "\n")
