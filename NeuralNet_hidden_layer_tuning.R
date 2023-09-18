# train_model.R

library(keras)
library(tensorflow)
library(rsample)
library(dplyr)

# Set the seed for reproducibility
set.seed(123)
# Create a 10% subset
subset_ind_nn <- initial_split(filtered_data_NN, prop = 0.1)  # 10% for the subset
subset_data_nn <- training(subset_ind_nn)  #training portion for the subset
subset_test_nn <- testing(subset_ind_nn)
cat("The number of rows in the 10% subset is", nrow(subset_data_nn))

write.csv(subset_data_nn, file = "subset_train.csv", row.names = FALSE)
write.csv(subset_test_nn, file = "subset_test.csv", row.names = FALSE)

par(mfrow=c(1,2))

#Claim number Bar plot
barplot(table(filtered_data_NN$ClaimNb),
        main = "Original Data",
        xlab = "Claim Count",
        ylab = "Frequency",
        col = "green"
)

#Claim number Bar plot
barplot(table(subset_data_nn$ClaimNb),
        main = "Subset",
        xlab = "Claim Count",
        ylab = "Frequency",
        col = "green"
)



# Vehicle Age histogram
hist(filtered_data_NN$ClaimAmount, 
     main = "Original Data",
     xlab = "Claim Amount",
     ylab = "Count",
     col = "blue",
     border = "blue" 
     #breaks = "FD"  # Method for determining the number of bins (histogram breaks)
)

# Vehicle Age histogram
hist(subset_data_nn$ClaimAmount, 
     main = "Subset",
     xlab = "Claim Amount",
     ylab = "Count",
     col = "blue",
     border = "blue" 
     #breaks = "FD"  # Method for determining the number of bins (histogram breaks)
)


# Predictor variables
X_train_1 <- subset_data_nn[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                          "VehBrand_encoded", "VehGas_encoded", 
                          "Region_encoded", "Area_encoded", "Density")]

X_test_1 <- subset_test_nn[, c("VehPower", "VehAge", "DrivAge", "BonusMalus", 
                        "VehBrand_encoded", "VehGas_encoded", 
                        "Region_encoded", "Area_encoded", "Density")]
# Target variables
y_train_1 <- subset_data_nn$ClaimNb
y_test_1 <- subset_test_nn$ClaimNb

y_train_2 <- subset_data_nn$ClaimAmount
y_test_2 <- subset_test_nn$ClaimAmount




#---------------------------one hidden------------------------------------#

model <- keras_model_sequential()

model %>%
  layer_dense(units = 20, activation = "relu", input_shape = 9, name = "dense_17_input1")

model %>%
  layer_dense(units = 1, activation = "linear")

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_1,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )

#------------------------------two hidden --------------------------------#
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_17_input1")

model %>%
  layer_dense(units = 16, activation = "relu")

model %>%
  layer_dense(units = 1, activation = "linear")

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()  

# Compile the model
model$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_1,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )

#----------------------3 hidden layers-------------------------------------#


# Define the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_input")  # 32 neurons in the first hidden layer

model %>%
  layer_dense(units = 16, activation = "relu")  # 16 neurons in the second hidden layer

model %>%
  layer_dense(units = 8, activation = "relu")   # 8 neurons in the third hidden layer

model %>%
  layer_dense(units = 1, activation = "linear") # Output layer

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_1,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )




#----------------------4 hidden layers---------------------------------------#

# Define the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_input")  # 32 neurons in the first hidden layer

model %>%
  layer_dense(units = 16, activation = "relu")  # 16 neurons in the second hidden layer

model %>%
  layer_dense(units = 10, activation = "relu")   # 8 neurons in the third hidden layer

model %>%
  layer_dense(units = 8, activation = "relu")   # 4 neurons in the fourth hidden layer

model %>%
  layer_dense(units = 1, activation = "linear") # Output layer

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_squared_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_1,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )



#-------------------chose 3 hidden layers as best as it had low evaluation loss--------------

# Evaluate the model on the test data
evaluation <- model %>% evaluate(
  x = as.matrix(X_test_1),
  y = y_test_1,
  batch_size = 64,
  verbose = 1
)

# Print the evaluation results
cat("Test Loss:", evaluation[["loss"]], "\n")



#---------------------------one hidden------------------------------------#

model <- keras_model_sequential()

model %>%
  layer_dense(units = 20, activation = "relu", input_shape = 9, name = "dense_17_input1")

model %>%
  layer_dense(units = 1, activation = "linear")

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam() 

# Compile the model
model$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_2,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )

#------------------------------two hidden --------------------------------#
model <- keras_model_sequential()

model %>%
  layer_dense(units = 20, activation = "relu", input_shape = 9, name = "dense_17_input1")

model %>%
  layer_dense(units = 10, activation = "relu")

model %>%
  layer_dense(units = 1, activation = "linear")

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()  # You can change this to Adam

# Compile the model
model$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_2,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )


#----------------------3 hidden layers-------------------------------------#


# Define the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_input")  # 32 neurons in the first hidden layer

model %>%
  layer_dense(units = 16, activation = "relu")  # 16 neurons in the second hidden layer

model %>%
  layer_dense(units = 8, activation = "relu")   # 8 neurons in the third hidden layer

model %>%
  layer_dense(units = 1, activation = "linear") # Output layer

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_2,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )

plot(history)


#----------------------4 hidden layers---------------------------------------#

# Define the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_input")  # 32 neurons in the first hidden layer

model %>%
  layer_dense(units = 16, activation = "relu")  # 16 neurons in the second hidden layer

model %>%
  layer_dense(units = 10, activation = "relu")   # 8 neurons in the third hidden layer

model %>%
  layer_dense(units = 8, activation = "relu")   # 4 neurons in the fourth hidden layer

model %>%
  layer_dense(units = 1, activation = "linear") # Output layer

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_2,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )





#----------------------5 hidden layers---------------------------------------#

# Define the model
model <- keras_model_sequential()

model %>%
  layer_dense(units = 32, activation = "relu", input_shape = 9, name = "dense_input")  # 32 neurons in the first hidden layer

model %>%
  layer_dense(units = 16, activation = "relu")  # 16 neurons in the second hidden layer

model %>%
  layer_dense(units = 10, activation = "relu")   # 8 neurons in the third hidden layer

model %>%
  layer_dense(units = 8, activation = "relu")   # 4 neurons in the fourth hidden layer

model %>%
  layer_dense(units = 4, activation = "relu")   # 4 neurons in the fourth hidden layer

model %>%
  layer_dense(units = 1, activation = "linear") # Output layer

# Define the optimizer
optimizer <- tf$keras$optimizers$legacy$Adam()

# Compile the model
model$compile(
  loss = "mean_absolute_error",
  optimizer = optimizer
)

# Define the early stopping callback
early_stopping <- callback_early_stopping(
  monitor = "val_loss", patience = 10, restore_best_weights = TRUE
)

# Train the model and evaluate on validation data
history <- model %>% 
  fit(
    x = as.matrix(X_train_1),
    y = y_train_2,
    batch_size = 64,
    epochs = 50,
    verbose = 2,
    validation_split = 0.2
  )




# chose 4 as best