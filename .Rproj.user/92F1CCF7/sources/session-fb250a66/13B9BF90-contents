# Load necessary library
library(randomForest)

# Install the library if required
install.packages("caret")
library(caret)

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
# Split data into training (60%), validation (20%), and test (20%) sets
train_index <- createDataPartition(abalone_data$Rings, p = 0.6, list = FALSE)
train_data <- abalone_data[train_index, ]
remaining_data <- abalone_data[-train_index, ]

# Further split the remaining data into validation and test sets
validation_index <- createDataPartition(remaining_data$Rings, p = 0.5, list = FALSE)
validation_data <- remaining_data[validation_index, ]
test_data <- remaining_data[-validation_index, ]

# Check the sizes of each set
cat("Training set size:", nrow(train_data), "\n")
cat("Validation set size:", nrow(validation_data), "\n")
cat("Test set size:", nrow(test_data), "\n")
