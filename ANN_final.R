library(keras)
library(tensorflow)
library(rsample)
#-----------------------------------frequency data---------------------------------------------------

freq_data <- read.csv("/Users/ronypabraham/Downloads/freMTPL2freq.csv")

#-----------------------------------severity data-----------------------------------------------

sev_data <- read.csv("/Users/ronypabraham/Desktop/freMTPL2sev.csv")

str(freq_data)
str(sev_data)


#merging both data frames
merged_data_NN <- merge(freq_data, sev_data, by = "IDpol", all.x =  TRUE)


merged_data_NN$Exposure[merged_data_NN$Exposure > 1] <- 1

#Correcting na values to 0

merged_data_NN[is.na(merged_data_NN)] <- 0

#missing values 
sapply(merged_data_NN, function(x) {sum(is.na(x))})

#null values
is.null(merged_data_NN)


#------------------------outliers---------------------------------#

#capping vehicle age at 15

merged_data_NN$VehAge <- ifelse(merged_data_NN$VehAge > 15, 15, 
                                merged_data_NN$VehAge)

#capping driver age at 80

merged_data_NN$DrivAge <- pmin(merged_data_NN$DrivAge, 80)

#converting density to log scale

merged_data_NN$Density <- log(merged_data_NN$Density)

#bonusmalus

bonusmalus_table <- table(cut(merged_data_NN$BonusMalus, 
                              breaks = c(seq(50, 100, by = 5), Inf)))

#VehPower
merged_data_NN$VehPower <- pmin(merged_data_NN$VehPower, 10)


# Remove rows where ClaimNb > 4
filtered_data_NN <- merged_data_NN[merged_data_NN$ClaimNb <= 4, ]


# Remove rows with duplicate ClaimAmount for distinct IDpol

filtered_data_NN <- filtered_data_NN %>%
  group_by(IDpol) %>%
  filter(!duplicated(ClaimAmount))

# Remove rows with duplicate ClaimAmount for ClaimNb = 2 and unique IDpol
filtered_data_NN <- filtered_data_NN %>%
  group_by(IDpol) %>%
  mutate(ClaimAmount = ifelse(ClaimNb == 2 & ClaimAmount == 1128.12, 1128, ClaimAmount)) %>%
  distinct(IDpol, ClaimAmount, .keep_all = TRUE)


# Update ClaimNb to be the distinct count of rows after deleting duplicates
filtered_data_NN <- filtered_data_NN %>%
  group_by(IDpol) %>%
  mutate(ClaimNb = ifelse(ClaimAmount != 0, n_distinct(ClaimAmount), ClaimNb))


# Assuming your data frame is named 'data'
filtered_data_NN <- subset(filtered_data_NN, ClaimAmount != 0)


# Set the threshold as the 99.9th percentile
threshold <- quantile(filtered_data_NN$ClaimAmount, 0.9999)

# Replace values above the threshold with the threshold value
filtered_data_NN$ClaimAmount <- pmin(filtered_data_NN$ClaimAmount, threshold)

# Check the summary statistics after capping extreme values
summary(filtered_data_NN$ClaimAmount)

#----------------------------Pre processing for NN-----------------------------#

#Min-Max scaling for numerical features

# List of numerical features to be scaled
numerical_features <- c("Exposure", "VehPower", "VehAge", "DrivAge", "BonusMalus", "Density")

# Min-Max Scaling function
min_max_scaling <- function(x) {
  (x - min(x)) / (max(x) - min(x))
}

# Apply Min-Max Scaling to numerical features
filtered_data_NN[numerical_features] <- lapply(filtered_data_NN[numerical_features], min_max_scaling)

# Print the scaled data frame
print(filtered_data_NN)

# List of categorical features to be one-hot encoded
categorical_features <- c("Area", "VehGas", "VehBrand", "Region")

# Function to perform one-hot encoding on categorical variables
one_hot_encode <- function(data, feature) {
  encoded_col <- factor(data[[feature]])
  levels_encoded <- levels(encoded_col)
  encoded_values <- as.integer(encoded_col)
  
  data[[paste0(feature, "_encoded")]] <- encoded_values
  data <- data[, !(names(data) %in% feature)]
  
  return(data)
}

# Apply one-hot encoding to each categorical feature
for (feature in categorical_features) {
  filtered_data_NN <- one_hot_encode(filtered_data_NN, feature)
}

# Print the modified data frame with one-hot encoded categorical features
print(filtered_data_NN)




# Split data into train and test sets after pre-processing
set.seed(123) 

data_split <- initial_split(filtered_data_NN, prop = 0.7) 
train_data <- training(data_split)
test_data <- testing(data_split)

# Verify the sizes of train and test sets
nrow(train_data)
nrow(test_data)

# Predictor variables
X_train <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", "Region_encoded", "Area_encoded", "Density")]

X_test <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                        "VehBrand_encoded", "VehGas_encoded", "Region_encoded", "Area_encoded", "Density")]

# Target variables
y_train <- train_data$ClaimNb
y_test <- test_data$ClaimNb





#---------------------------Model 1----------------------------------------#


set.seed(123)

model_ann <- keras_model_sequential()
model_ann %>%
  layer_dense(units = 64, activation = "tanh", input_shape = 9, name = "dense_input1")
model_ann %>%
  layer_dense(units = 80, activation = "sigmoid")
model_ann %>%
  layer_dense(units = 88, activation = "relu")
model_ann %>%
  layer_dense(units = 1, activation = "linear")


# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate=0.01)

# Compile the model
model_ann$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)


# Train the model and evaluate on validation data
freq_ann <- model_ann %>% 
  fit(
    x = as.matrix(X_train),
    y = y_train,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on th test set
results_ann <- model_ann %>% evaluate(
  x = as.matrix(X_test),
  y = y_test
)


# Make predictions on the test data
predictions_ann <- model_ann %>% predict(as.matrix(X_test))

# Calculate the Mean Absolute Error (MAE)
mae_ann <- mean(abs(predictions_ann - y_test))

# Print the MAE
cat("Mean Absolute Error (MAE) of model 1:", mae_ann, "\n")



#-----------------------------Model2----------------------------------------#

# Predictor variables
X_train_2 <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                            "VehBrand_encoded", "VehGas_encoded", 
                            "Region_encoded", "Area_encoded")]

X_test_2 <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Area_encoded")]
# Target variables
y_train <- train_data$ClaimNb
y_test <- test_data$ClaimNb

model_ann_2 <- keras_model_sequential()
model_ann_2 %>%
  layer_dense(units = 64, activation = "tanh", input_shape = 8, name = "dense_input1")
model_ann_2 %>%
  layer_dense(units = 80, activation = "sigmoid")
model_ann_2 %>%
  layer_dense(units = 88, activation = "relu")
model_ann_2 %>%
  layer_dense(units = 1, activation = "linear")


# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate=0.01)

# Compile the model
model_ann_2$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)


# Train the model and evaluate on validation data
freq_ann_2 <- model_ann_2 %>% 
  fit(
    x = as.matrix(X_train_2),
    y = y_train,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on the test set
results_ann2 <- model_ann_2 %>% evaluate(
  x = as.matrix(X_test_2),
  y = y_test
)


# Make predictions on the test data
predictions_ann_2 <- model_ann_2 %>% predict(as.matrix(X_test_2))

# Calculate the Mean Absolute Error (MAE)
mae_ann_2 <- mean(abs(predictions_ann_2 - y_test))

# Print the MAE
cat("Mean Absolute Error (MAE) of model 2:", mae_ann_2, "\n")



#---------------------------Model 3----------------------------------------#

# Predictor variables
X_train_3 <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                            "VehBrand_encoded", "VehGas_encoded", 
                            "Region_encoded", "Density")]

X_test_3 <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Density")]
# Target variables
y_train <- train_data$ClaimNb
y_test <- test_data$ClaimNb


model_ann_3 <- keras_model_sequential()
model_ann_3 %>%
  layer_dense(units = 64, activation = "tanh", input_shape = 8, name = "dense_input1")
model_ann_3 %>%
  layer_dense(units = 80, activation = "sigmoid")
model_ann_3 %>%
  layer_dense(units = 88, activation = "relu")
model_ann_3 %>%
  layer_dense(units = 1, activation = "linear")


# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Nadam(learning_rate=0.01)

# Compile the model
model_ann_3$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)


# Train the model and evaluate on validation data
freq_ann_3 <- model_ann_3 %>% 
  fit(
    x = as.matrix(X_train_3),
    y = y_train,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on the test set
results_ann3 <- model_ann_3 %>% evaluate(
  x = as.matrix(X_test_3),
  y = y_test
)


# Make predictions on the test data
predictions_ann_3 <- model_ann_3 %>% predict(as.matrix(X_test_3))

# Calculate the Mean Absolute Error (MAE)
mae_ann_3 <- mean(abs(predictions_ann_3 - y_test))


# Print the MAE
cat("Mean Absolute Error (MAE) of model 1:", mae_ann, "\n")
cat("Mean Absolute Error (MAE) of model 1:", mae_ann_2, "\n")
cat("Mean Absolute Error (MAE) of model 3:", mae_ann_3, "\n")

#---------------------------Severity---------------------------------------#


# Predictor variables
X_train <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Area_encoded", "Density")]

X_test <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                        "VehBrand_encoded", "VehGas_encoded", 
                        "Region_encoded", "Area_encoded", "Density")]
# Target variables
y_train_sev <- train_data$ClaimAmount
y_test_sev<- test_data$ClaimAmount


model_ann_4 <- keras_model_sequential()
model_ann_4 %>%
  layer_dense(units = 256, activation = "sigmoid", input_shape = 9, name = "dense_input1")
model_ann_4 %>%
  layer_dense(units = 32, activation = "tanh")
model_ann_4 %>%
  layer_dense(units = 72, activation = "tanh")
model_ann_4 %>%
  layer_dense(units = 88, activation = "tanh")
model_ann_4 %>%
  layer_dense(units = 1, activation = "linear")


# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate=0.01)

# Compile the model
model_ann_4$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)






# Train the model and evaluate on validation data
sev_ann <- model_ann_4 %>% 
  fit(
    x = as.matrix(X_train),
    y = y_train_sev,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on the test set
test_resultssev <- model_ann_4 %>% evaluate(
  x = as.matrix(X_test),
  y = y_test_sev
)


# Make predictions on the test data
predictions_ann_sev <- model_ann_4 %>% predict(as.matrix(X_test))

# Calculate the Mean Absolute Error (MAE)
mae_ann_sev <- mean(abs(predictions_ann_sev - y_test_sev))

# Print the MAE
cat("Mean Absolute Error (MAE) of model 1:", mae_ann_sev, "\n")




#----------------------------Model 2--------------------------------------#


# Predictor variables
X_train_2 <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                            "VehBrand_encoded", "VehGas_encoded", 
                            "Region_encoded", "Area_encoded")]

X_test_2 <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Area_encoded")]
# Target variables
y_train_sev <- train_data$ClaimAmount
y_test_sev<- test_data$ClaimAmount

model_ann_5 <- keras_model_sequential()
model_ann_5 %>%
  layer_dense(units = 256, activation = "sigmoid", input_shape = 8, name = "dense_input1")
model_ann_5 %>%
  layer_dense(units = 32, activation = "tanh")
model_ann_5 %>%
  layer_dense(units = 72, activation = "tanh")
model_ann_5 %>%
  layer_dense(units = 88, activation = "tanh")
model_ann_5 %>%
  layer_dense(units = 1, activation = "linear")
# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam(learning_rate=0.01)


# Compile the model
model_ann_5$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Train the model and evaluate on validation data
sev_ann_2 <- model_ann_5 %>% 
  fit(
    x = as.matrix(X_train_2),
    y = y_train_sev,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on the test set
test_results2_sev <- model_ann_5 %>% evaluate(
  x = as.matrix(X_test_2),
  y = y_test_sev
)


# Make predictions on the test data
predictions_ann_sev_2 <- model_ann_5 %>% predict(as.matrix(X_test_2))

# Calculate the Mean Absolute Error (MAE)
mae_ann_sev_2 <- mean(abs(predictions_ann_sev_2 - y_test_sev))

# Print the MAE
cat("Mean Absolute Error (MAE) of model 2:", mae_ann_sev_2, "\n")




#----------------------------Model 3--------------------------------------#



# Predictor variables
X_train_3 <- train_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                            "VehBrand_encoded", "VehGas_encoded", 
                            "Region_encoded", "Density")]

X_test_3 <- test_data[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Density")]
# Target variables
y_train_sev <- train_data$ClaimAmount
y_test_sev<- test_data$ClaimAmount

model_ann_6 <- keras_model_sequential()
model_ann_6 %>%
  layer_dense(units = 256, activation = "sigmoid", input_shape = 8, name = "dense_input1")
model_ann_6 %>%
  layer_dense(units = 32, activation = "tanh")
model_ann_6 %>%
  layer_dense(units = 72, activation = "tanh")
model_ann_6 %>%
  layer_dense(units = 88, activation = "tanh")
model_ann_6 %>%
  layer_dense(units = 1, activation = "linear")

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Nadam(learning_rate=0.01)


# Compile the model
model_ann_6$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)


# Train the model and evaluate on validation data
sev_ann_3 <- model_ann_6 %>% 
  fit(
    x = as.matrix(X_train_3),
    y = y_train_sev,
    batch_size = 32,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


# Evaluate the model on the test set
test_results3_sev <- model %>% evaluate(
  x = as.matrix(X_test_3),
  y = y_test_sev
)


# Make predictions on the test data
predictions_ann_sev_3 <- model_ann_6 %>% predict(as.matrix(X_test_3))

# Calculate the Mean Absolute Error (MAE)
mae_ann_sev_3 <- mean(abs(predictions_ann_sev_3 - y_test_sev))

# Print the MAE
cat("Mean Absolute Error (MAE) of model 3:", mae_ann_sev_3, "\n")



# Print the MAE
cat("Mean Absolute Error (MAE) of model 1:", mae_ann_sev, "\n")
cat("Mean Absolute Error (MAE) of model 2:", mae_ann_sev_2, "\n")
cat("Mean Absolute Error (MAE) of model 3:", mae_ann_sev_3, "\n")




#------------------calculating loss cost-------------------------------#

#Calculating expected claims from best freq model
predictions_ann <- model_ann %>% predict(as.matrix(X_test))

#Calculating expected costs from best sev model
predictions_ann_sev <- model_ann_4 %>% predict(as.matrix(X_test))

claim_amount_expected_ANN <- predictions_ann * predictions_ann_sev


actual_claim_amount_NN <- test_data$ClaimAmount * test$ClaimNb

# Calculate the loss for each observation
loss_cost_ANN <- abs(claim_amount_expected_ANN - actual_claim_amount_NN)

# Calculate the mean loss (loss cost) for all observations
mean_loss_cost_ANN <- mean(loss_cost_ANN)


#-----------------------------------AVE------------------------------------------------------#


# Sort the filtered actual claim amount in ascending order
sorted_actual_claim_amount <- sort(actual_claim_amount)

# Create a data frame for plotting
actual_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(sorted_actual_claim_amount)),
  ClaimAmount = cumsum(sorted_actual_claim_amount)) 

# Calculate cumulative sum of expected claim amounts of GLM
cumulative_expected_claim_amount <- cumsum(claim_amount_expected_GLM)

# Create a data frame for plotting
expected_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount)),
  ClaimAmount = cumulative_expected_claim_amount)

# Calculate cumulative sum of expected claim amounts of GBM 
cumulative_expected_claim_amount_GBM <- cumsum(claim_amount_expected_GBM)

# Create a data frame for plotting
expected_data_GBM <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_GBM)),
  ClaimAmount = cumulative_expected_claim_amount_GBM)


# Calculate cumulative sum of expected claim amounts of ANN
cumulative_expected_claim_amount_ANN <- cumsum(claim_amount_expected_ANN)

# Create a data frame for plotting
expected_data_ANN <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_ANN)),
  ClaimAmount = cumulative_expected_claim_amount_ANN)

# Calculate cumulative sum of expected claim amounts of CANN
cumulative_expected_claim_amount_ensemble <- cumsum(claim_amount_expected_ensemble)

# Create a data frame for plotting
expected_data_ensemble <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_ANN)),
  ClaimAmount = cumulative_expected_claim_amount_ensemble)

# Create a line chart with actual and expected claim amounts
ggplot() +
  geom_line(data = actual_data, aes(x = CumulativePercentage, y = ClaimAmount, color = "Actual Data"),linetype = "solid", size = 1) +
  geom_line(data = expected_data, aes(x = CumulativePercentage, y = ClaimAmount, color = "GLM") ,linetype = "solid", size = 1)+
  geom_line(data = expected_data_GBM, aes(x = CumulativePercentage, y = ClaimAmount, color = "GBM"), linetype = "solid", size = 1)+
  geom_line(data = expected_data_ANN, aes(x = CumulativePercentage, y = ClaimAmount, color = "ANN"), linetype = "solid", size = 1)+
  geom_line(data = expected_data_ensemble, aes(x = CumulativePercentage, y = ClaimAmount, color = "Ensemble Model"), linetype = "solid", size = 1)+
  scale_color_manual(values = c("Actual Data" = "blue", "GLM" = "green", "GBM" = "red", "ANN" = "purple","Ensemble Model" = "yellow")) +  # Modify the color value here
  labs(x = "Percentage of Cases", y = "Loss Cost", title = "Actual Vs Expected Plots") +
  theme_minimal()




