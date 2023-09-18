#install required libraries

install.packages("gbm")
install.packages("caret")
install.packages("Metrics")
install.packages("tweedie")
library(tweedie)
library(Metrics)
library(gbm)
library(caret)
library(dplyr)
#------------------------tuned hyper params-------------------------------------#



gbmtuned <- gbm(
  ClaimNb ~   VehPower + VehAge + DrivAge + BonusMalus
  + VehBrand + VehGas + Region + Density + offset(log(Exposure)) , data = train, distribution = "poisson", n.trees = 515, 
  shrinkage = 0.01, interaction.depth = 7, n.minobsinnode = 5)

summary(gbmtuned)

#---GBM Freq model 2

gbmtuned_2 =gbm(ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus
                + VehBrand + VehGas + Region + Area , data = train,
                distribution="poisson",n.trees= 515,shrinkage=0.01,interaction.depth=7,
                n.minobsinnode = 5 )

summary(gbmtuned_2)

#----GBM freq model 3

gbmtuned_3 =gbm(ClaimNb ~  VehPower + VehAge + DrivAge + BonusMalus
                + VehBrand + VehGas + Region + Density , data = train,
                distribution="poisson",n.trees=515,shrinkage=0.01,interaction.depth=7,
                n.minobsinnode = 5 )

summary(gbmtuned_3)


#Metrics Evaluation

# Predict on the test data
preds.gbm_1 <- predict(gbmtuned, newdata = test, n.trees =515, type = "response")
preds.gbm_2 <- predict(gbmtuned_2, newdata = test, n.trees = 515, type = "response")
preds.gbm_3 <- predict(gbmtuned_3, newdata = test, n.trees = 515, type = "response")

# Calculate Mean Absolute Error (MAE)
mae.gbm_1 <- mean(abs(test$ClaimNb - preds.gbm_1))
mae.gbm_2 <- mean(abs(test$ClaimNb - preds.gbm_2))
mae.gbm_3 <- mean(abs(test$ClaimNb - preds.gbm_3))


# Print the results
cat("MSE of model 1:", mae.gbm_1, "\n")
cat("MSE of model 2:", mae.gbm_2, "\n")
cat("MSE of model 3:", mae.gbm_3, "\n")





#--------------------------------severity--------------------------------------#

# Model 1 with all variables

gbmtuned.sev <- gbm(ClaimAmount ~  VehPower + VehAge + DrivAge + BonusMalus +
                      VehBrand + VehGas + Density + Region + Area + offset(log(ClaimNb)), data = train,
                    distribution = "laplace", n.trees = 278, shrinkage = 0.01,
                    interaction.depth = 5, n.minobsinnode = 5)

  # Summary of the GBM Claim Severity model
  summary(gbmtuned.sev)

# Model 2 with only area
gbmtuned.sev2 <- gbm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus +
                        VehBrand + VehGas + Region + Area, data = train,
                      distribution = "laplace", n.trees = 278, shrinkage = 0.01,
                      interaction.depth = 5, n.minobsinnode = 5)

# Summary of the GBM Claim Sev model 2
summary(gbmtuned.sev2)

# Model 3 with only density
gbmtuned.sev3 <- gbm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus +
                        VehBrand + VehGas + Region + Density, data = train,
                      distribution = "laplace", n.trees = 278, shrinkage = 0.01,
                      interaction.depth = 5, n.minobsinnode = 5)

# Summary of the GBM Claim Sev model 3
summary(gbmtuned.sev3)


# Predict on the test data
preds.gbm.sev1 <- predict(gbmtuned.sev, newdata = test, n.trees = 278, type = "response")
preds.gbm.sev2 <- predict(gbmtuned.sev2, newdata = test, n.trees = 278, type = "response")
preds.gbm.sev3 <- predict(gbmtuned.sev3, newdata = test, n.trees = 278, type = "response")


# Calculate Mean Absolute Error (MAE)
mae.gbm.sev1 <- mean(abs(test$ClaimAmount- preds.gbm.sev1))
mae.gbm.sev2 <- mean(abs(test$ClaimAmount - preds.gbm.sev2))
mae.gbm.sev3 <- mean(abs(test$ClaimAmount - preds.gbm.sev3))


# Print the results
cat("MSE of model 1:", mae.gbm.sev1, "\n")
cat("MSE of model 2:", mae.gbm.sev2, "\n")
cat("MSE of model 3:", mae.gbm.sev3, "\n")


#------------------calculating loss cost-------------------------------#

#Calculating expected claims from best freq model
pred_counts_GBM <- predict(gbmtuned, newdata = test, n.trees = 515, type = "response")

#calculating average claim cost from best severity model
pred_sev_GBM <- predict(gbmtuned.sev, newdata = test, n.trees = 278, type = "response")


claim_amount_expected_GBM <- pred_sev_GBM * pred_counts_GBM

actual_claim_amount <- test$ClaimAmount * test$ClaimNb

# Calculate the loss for each observation
loss_cost_GBM <- abs(claim_amount_expected_GBM - actual_claim_amount)

# Calculate the mean loss (loss cost) for all observations
mean_loss_cost_GBM <- mean(loss_cost_GBM)




#-------------------------------Lift-------------------------------


# Calculate the cumulative sums for predicted claim amounts
sorted_claim_amounts_GBM <- sort(claim_amount_expected_GBM)
cumulative_claim_amounts_GBM <- cumsum(sorted_claim_amounts_GBM)

# Calculate the baseline cumulative gains (evenly distributed)
num_cases_GLM <- length(actual_claim_amount)
cumulative_gains_baseline_GLM <- cumsum(rep(1/num_cases_GLM, num_cases_GLM))

# Create a data frame for the Lift Plot for GLM model
lift_data_GB <- data.frame(
  Percentile = seq(0, 100, length.out = length(cumulative_claim_amounts_GBM)),
  CumulativeClaimAmounts_GBM = cumulative_claim_amounts_GBM,
  CumulativeGainsBaseline_GBM = cumulative_gains_baseline_GLM
)

# Calculate the lift values for GLM model
lift_data_GB$Lift_GB <- lift_data_GB$CumulativeClaimAmounts_GBM / lift_data_GB$CumulativeGainsBaseline_GBM


# Plot the Lift Curve for GBM model (showing only positive part)
ggplot(lift_data_GB, aes(x = Percentile, y = Lift_GB)) +
  geom_line(color = "blue", linetype = "solid", size = 1) +
  xlab("Percentage of Cases") +
  ylab("Lift") +
  ggtitle("Lift curve: GBM Model") +
  theme_minimal() +
  theme(legend.position = "none") +
  scale_y_continuous(limits = c(0, max(lift_data_GB$Lift_GB)))




#------------------------------------AVE-------------------------------



# Sort the filtered actual claim amount in ascending order
sorted_actual_claim_amount <- sort(actual_claim_amount)

# Create a data frame for plotting
actual_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(sorted_actual_claim_amount)),
  ClaimAmount = cumsum(sorted_actual_claim_amount))  # Use cumulative sum

# Calculate cumulative sum of expected claim amounts
cumulative_expected_claim_amount <- cumsum(claim_amount_expected_GLM)

# Create a data frame for plotting
expected_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount)),
  ClaimAmount = cumulative_expected_claim_amount)

# Calculate cumulative sum of expected claim amounts
cumulative_expected_claim_amount_GBM <- cumsum(claim_amount_expected_GBM)

# Create a data frame for plotting
expected_data_GBM <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_GBM)),
  ClaimAmount = cumulative_expected_claim_amount_GBM)

# Create a line chart with both actual and expected claim amounts
ggplot() +
  geom_line(data = actual_data, aes(x = CumulativePercentage, y = ClaimAmount, color = "Actual Data"),linetype = "solid", size = 1) +
  geom_line(data = expected_data, aes(x = CumulativePercentage, y = ClaimAmount, color = "GLM") ,linetype = "solid", size = 1)+
  geom_line(data = expected_data_GBM, aes(x = CumulativePercentage, y = ClaimAmount, color = "GBM"), linetype = "solid", size = 1)+
  scale_color_manual(values = c("Actual Data" = "blue", "GLM" = "green", "GBM" = "red")) +  # Modify the color value here
  labs(x = "Percentage of Cases", y = "Loss Cost", title = "Actual Vs Expected Plots") +
  theme_minimal()





#-----------------------------Gain's-----------------------------------#


# Calculate the cumulative gains for the GLM model's predictions
sorted_predictions_GBM <- sort(claim_amount_expected_GBM, decreasing = TRUE)
cumulative_gains_GBM <- cumsum(sorted_predictions_GBM) / sum(claim_amount_expected_GBM)

# Calculate the baseline cumulative gains (evenly distributed)
num_cases <- length(loss_cost_GBM)
cumulative_gains_baseline_GBM <- cumsum(rep(1/num_cases, num_cases))

# Create a data frame for the Gains Plot
gains_data <- data.frame(
  Percentile = seq(0, 100, length.out = length(cumulative_gains_GBM)),
  CumulativeGains_GBM = cumulative_gains_GBM,
  CumulativeGains_Baseline = cumulative_gains_baseline_GBM
)

# Create the Gains Plot
ggplot(gains_data, aes(x = Percentile)) +
  geom_line(aes(y = CumulativeGains_GBM), color = "blue", linetype = "solid", label = "GBM Model") +
  geom_line(aes(y = CumulativeGains_Baseline), color = "red", linetype = "dashed", label = "Baseline Model") +
  xlab("Percentage of Cases") +
  ylab("Cumulative Gains") +
  ggtitle("Gains Plot: GBM Model vs Baseline Model") +
  scale_color_manual(values = c("blue", "red")) +
  theme_minimal() +
  theme(legend.position = "bottom")



#residual

# Calculate the predicted claim amounts using the GBM model
predicted_claim_amounts_GBM <- claim_amount_expected_GBM

# Calculate the residuals
residuals_GBM <- actual_claim_amount - predicted_claim_amounts_GBM

# Create a data frame for the Residual Plot
residual_data_GBM <- data.frame(
  Observed = claim_amount_expected_GBM,
  Residual = residuals_GBM
)

# Plot the Residuals for GBM model

ggplot(residual_data_GBM, aes(x = Observed, y = Residual)) +
  geom_point(color = "blue", alpha = 0.5) +
  geom_hline(yintercept = 0, color = "red", linetype = "dashed") +
  xlab("Observed Claim Amount") +
  ylab("Residual") +
  ggtitle("Residual Plot for GBM Model") +
  theme_minimal()


















# Sort the filtered actual claim amount in ascending order
sorted_actual_claim_amount <- sort(test$ClaimNb)

# Create a data frame for plotting
actual_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(sorted_actual_claim_amount)),
  ClaimAmount = cumsum(sorted_actual_claim_amount))  # Use cumulative sum


# Calculate cumulative sum of expected claim amounts
cumulative_expected_claim_amount_GBM <- cumsum(pred_counts_GBM)

# Create a data frame for plotting
expected_data_GBM <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_GBM)),
  ClaimAmount = cumulative_expected_claim_amount_GBM)

# Create a line chart with both actual and expected claim amounts
ggplot() +
  geom_line(data = actual_data, aes(x = CumulativePercentage, y = ClaimAmount), color = "blue") +
  geom_line(data = expected_data_GBM, aes(x = CumulativePercentage, y = ClaimAmount), color = "green") +
  labs(x = "Percentage of Cases", y = "Claim Amount", title = "AvE ; GLM") +
  theme_minimal()




# Sort the filtered actual claim amount in ascending order
sorted_actual_claim_amount <- sort(test$ClaimAmount)

# Create a data frame for plotting
actual_data <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(sorted_actual_claim_amount)),
  ClaimAmount = cumsum(sorted_actual_claim_amount))  # Use cumulative sum


# Calculate cumulative sum of expected claim amounts
cumulative_expected_claim_amount_GBM <- cumsum(pred_sev_GBM)

# Create a data frame for plotting
expected_data_GBM <- data.frame(
  CumulativePercentage = seq(0, 100, length.out = length(cumulative_expected_claim_amount_GBM)),
  ClaimAmount = cumulative_expected_claim_amount_GBM)

# Create a line chart with both actual and expected claim amounts
ggplot() +
  geom_line(data = actual_data, aes(x = CumulativePercentage, y = ClaimAmount), color = "blue") +
  geom_line(data = expected_data_GBM, aes(x = CumulativePercentage, y = ClaimAmount), color = "green") +
  labs(x = "Percentage of Cases", y = "Claim Amount", title = "AvE ; GLM") +
  theme_minimal()







