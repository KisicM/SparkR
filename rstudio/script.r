library(sparklyr)
library(dplyr)

# Install
spark_install()

# Connect
sc <- sparklyr::spark_connect(master = "local")


datasetPath <- "/data/train.csv"
trainSet <- spark_read_csv(sc, name = "my_data", path = datasetPath, header = TRUE, infer_schema = TRUE)
trainSet <- na.omit(trainSet)

datasetPath <- "/data/test.csv"
testSet <- spark_read_csv(sc, name = "my_data", path = datasetPath, header = TRUE, infer_schema = TRUE)
testSet <- na.omit(testSet)

glimpse(trainSet)
head(trainSet)

# Extract column names
column_names <- colnames(trainSet)

# Select the relevant columns for prediction
selected_columns <- c("Gender", "Customer_Type", "Age", "Type_of_Travel", "Class", "Flight_Distance",
                      "Inflight_wifi_service", "DepartureArrival_time_convenient", "Ease_of_Online_booking",
                      "Food_and_drink", "Online_boarding", "Seat_comfort",
                      "Inflight_entertainment", "Onboard_service", "Leg_room_service", "Baggage_handling",
                      "Checkin_service", "Inflight_service", "Cleanliness", "Departure_Delay_in_Minutes",
                      "Arrival_Delay_in_Minutes", "satisfaction")

# Check if selected columns exist in the dataset
missing_columns <- setdiff(selected_columns, column_names)
if (length(missing_columns) > 0) {
  stop(paste("The following columns are missing in the dataset:", paste(missing_columns, collapse = ", ")))
}

# Select the relevant columns
trainSet <- trainSet %>% select(one_of(selected_columns))
testSet <- testSet %>% select(one_of(selected_columns))

# Convert categorical variables to factors
trainSet <- trainSet %>% mutate(across(everything(), as.factor))
testSet <- testSet %>% mutate(across(everything(), as.factor))

# Define a list of classification methods
methods <- c("Logistic Regression")

# Initialize a list to store evaluation results
eval_results <- list()

# Iterate over each method
for (method in methods) {
  cat("Running", method, "\n")
  
  # Train the model
  if (method == "Logistic Regression") {
    model <- ml_logistic_regression(trainSet, satisfaction ~ .)
  } else if (method == "Random Forest") {
    model <- ml_random_forest(trainSet, satisfaction ~ .)
  } else if (method == "Gradient-Boosted Trees") {
    model <- ml_gradient_boosted_trees(trainSet, satisfaction ~ .)
  } else {
    cat("Unsupported method:", method, "\n")
    next
  }
  # Evaluate the model and store the results
  predictions <- ml_predict(model, testSet)
  predictedLabels <- collect(select(predictions, prediction))
  trueLabels <- collect(select(testSet, satisfaction))
  accuracy <- sum(predictedLabels == trueLabels) / length(trueLabels)
  # precision <- precision(predictedLabels, trueLabels, positive = "label_value")
  # recall <- recall(predictedLabels, trueLabels, positive = "label_value")
  
  eval_results[[method]] <- list(
    accuracy = accuracy
    # precision = precision,
    # recall = recall
  )
}

# Print evaluation results
for (method in methods) {
  cat("Evaluation results for", method, ":\n")
  print(eval_results[[method]])
  cat("\n")
}

spark_disconnect(sc)
