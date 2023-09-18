install.packages("reshape2")
install.packages("ggplot2")
install.packages("magrittr")
install.packages("dplyr")
install.packages("e1071")
install.packages("AER")
install.packages("MASS")
install.packages("countreg", repos="http://R-Forge.R-project.org")
install.packages("boot")
install.packages("caret")
install.packages("rpart")
install.packages("caTools")
install.packages("ISLR")
install.packages("pscl")
install.packages("glmmTMB")
install.packages('TMB', type = 'source')
install.packages("lmerTest") #anova models comparison glmmTB
install.packages("rsample")
install.packages("ineq")
library(ineq)
library(lmerTest)
library(glmmTMB)
library(ISLR)
library(pscl)
library(caTools)
library(caret)
library(rpart)
library(boot)
library(countreg)
library(MASS)
library(AER)
library(e1071)
library(dplyr) #groupby
library(reshape2)
library(ggplot2)
library(magrittr) #> pipe
library(rsample)
library(gbm)


#----------------------------frequency data---------------------------------------------------

freq_data <- read.csv("/Users/ronypabraham/Downloads/freMTPL2freq.csv")

#---------------------------severity data-----------------------------------------------

sev_data <- read.csv("/Users/ronypabraham/Desktop/freMTPL2sev.csv")

str(freq_data)
str(sev_data)


#merging both data frames
merged_data <- merge(freq_data, sev_data, by = "IDpol", all.x =  TRUE)

#writing the file
#write.csv(merged_data, file = "merged_final.csv", row.names = FALSE)

#Correcting exposures greater than 1 to 1

merged_data$Exposure[merged_data$Exposure > 1] <- 1

#Correcting na values to 0

merged_data[is.na(merged_data)] <- 0

#missing values 
sapply(merged_data, function(x) {sum(is.na(x))})

#null values
is.null(merged_data)


# Conerting to factor

# Convert columns
#merged_data$Area <- as.numeric(as.factor(merged_data$Area))
#merged_data$VehBrand <- as.numeric(as.factor(merged_data$VehBrand))
#merged_data$VehGas <- as.numeric(as.factor(merged_data$VehGas))
#merged_data$Region <- as.numeric(as.factor(merged_data$Region))
#merged_data$VehPower <- as.numeric(merged_data$VehPower)

merged_data$Area <- as.numeric(as.factor(merged_data$Area))
merged_data$VehBrand <- as.factor(merged_data$VehBrand)
merged_data$VehGas <- as.numeric(as.factor(merged_data$VehGas))
merged_data$Region <- as.factor(merged_data$Region)
merged_data$VehPower <- as.numeric(merged_data$VehPower)
merged_data$BonusMalus <- as.integer(merged_data$BonusMalus)
merged_data$Density <- as.numeric(merged_data$Density)

merged_data$ClaimNb <- as.double(merged_data$ClaimNb)




#------------------------outliers---------------------------------#

#Correcting exposures greater than 1 to 1

merged_data$Exposure[merged_data$Exposure > 1] <- 1

#capping vehicle age at 15

merged_data$VehAge <- ifelse(merged_data$VehAge > 15, 15, 
                             merged_data$VehAge)

#capping driver age at 80

merged_data$DrivAge <- pmin(merged_data$DrivAge, 80)

#capping bonusmalus at 100

#merged_data$BonusMalus <- pmin(merged_data$BonusMalus, 100)

#converting density to log scale

merged_data$Density <- log(merged_data$Density)

#bonusmalus

bonusmalus_table <- table(cut(merged_data$BonusMalus, 
                              breaks = c(seq(50, 100, by = 5), Inf)))




#VehPower
merged_data$VehPower <- pmin(merged_data$VehPower, 10)


#-----------------------------END---------------------------------#
# Remove rows where ClaimNb > 4
filtered_data <- merged_data[merged_data$ClaimNb <= 4, ]

#write.csv(filtered_data, file = "filtered.csv", row.names = FALSE)

# Remove rows with duplicate ClaimAmount for distinct IDpol

filtered_data <- filtered_data %>%
  group_by(IDpol) %>%
  filter(!duplicated(ClaimAmount))


#write.csv(filtered_data, file = "filtered_after.csv", row.names = FALSE)

# Remove rows with duplicate ClaimAmount for ClaimNb = 2 and unique IDpol
filtered_data <- filtered_data %>%
  group_by(IDpol) %>%
  mutate(ClaimAmount = ifelse(ClaimNb == 2 & ClaimAmount == 1128.12, 1128, ClaimAmount)) %>%
  distinct(IDpol, ClaimAmount, .keep_all = TRUE)


# Update ClaimNb to be the distinct count of rows after deleting duplicates
filtered_data <- filtered_data %>%
  group_by(IDpol) %>%
  mutate(ClaimNb = ifelse(ClaimAmount != 0, n_distinct(ClaimAmount), ClaimNb))


#write.csv(filtered_data, file = "merged_final.csv", row.names = FALSE)

#filtered_data$ClaimAmount <- log1p(filtered_data$ClaimAmount) 


filtered_data <- subset(filtered_data, ClaimAmount != 0)


# Set the threshold as the 99.9th percentile
threshold <- quantile(filtered_data$ClaimAmount, 0.9999)

# Replace values above the threshold with the threshold value
filtered_data$ClaimAmount <- pmin(filtered_data$ClaimAmount, threshold)

# Check the summary statistics after capping extreme values
summary(filtered_data$ClaimAmount)


write.csv(filtered_data, file = "filtered_final.csv", row.names = FALSE)

#--------------------------------------END--------------------------------#
#-----------------------------------GLM-----------------------------------------#




set.seed(123)


ind <- initial_split(filtered_data, prop = 0.7)  # 70% for training, 30% for testing
train <- training(ind)
test <- testing(ind)
cat("The number of training set is ", nrow(train) ,"\n")
cat("The number of test set is", nrow(test))




#---GLM model 1 with Area & Density

poissonglm <- glm(ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Area+ Density, 
                  data = train,
                  family = "poisson",
                  offset = log(Exposure))

GLM_summary <- summary(poissonglm)

AIC_GLM1 <- AIC(poissonglm)
BIC_GLM1 <- BIC(poissonglm)
deviance_GLM1 <- deviance(poissonglm)
anova(poissonglm, test = "Chisq")

# Print the results
cat("AIC:", AIC_GLM1, "\n")
cat("BIC:", BIC_GLM1, "\n")
cat("Deviance:", deviance_GLM1, "\n")

#---GLM Model 2--With only area

poissonglm2 <- glm(ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Area, 
                   data = train,
                   family = "poisson",
                   offset = log(Exposure))


GLM_summary2 <- summary(poissonglm2)

AIC_GLM2 <- AIC(poissonglm2)
BIC_GLM2 <- BIC(poissonglm2)
deviance_GLM2 <- deviance(poissonglm2)
anova(poissonglm2, test = "Chisq")

# Print the results
cat("AIC:", AIC_GLM2, "\n")
cat("BIC:", BIC_GLM2, "\n")
cat("Deviance:", deviance_GLM2, "\n")

#---GLM Model 3--With only density

poissonglm3 <- glm(ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Density, 
                   data = train,
                   family = "poisson",
                   offset = log(Exposure))


GLM_summary3 <- summary(poissonglm3)

AIC_GLM3 <- AIC(poissonglm3)
BIC_GLM3 <- BIC(poissonglm3)
deviance_GLM3 <- deviance(poissonglm3)
anova(poissonglm3, test = "Chisq")

# Print the results
cat("AIC:", AIC_GLM3, "\n")
cat("BIC:", BIC_GLM3, "\n")
cat("Deviance:", deviance_GLM3, "\n")
#cat("Chi-square:", chi_square_GLM3 , "\n")



#----Testing 


# Prediction on data
predicted_counts1 <- predict(poissonglm, newdata = test, type = "response")
predicted_counts2 <- predict(poissonglm2, newdata = test, type = "response")
predicted_counts3 <- predict(poissonglm3, newdata = test, type = "response")

actual_counts <- test$ClaimNb

MAE1 <- mean(abs(predicted_counts1 - actual_counts))
MAE2 <- mean(abs(predicted_counts2 - actual_counts))
MAE3 <- mean(abs(predicted_counts3 - actual_counts))



#on train data

# Prediction on data
predicted_counts5 <- predict(poissonglm, newdata = train, type = "response")
predicted_counts6 <- predict(poissonglm2, newdata = train, type = "response")
predicted_counts7 <- predict(poissonglm3, newdata = train, type = "response")

actual_counts2 <- train$ClaimNb

MAE5 <- mean(abs(predicted_counts5 - actual_counts2))
MAE6 <- mean(abs(predicted_counts6 - actual_counts2))
MAE7 <- mean(abs(predicted_counts7 - actual_counts2))



# Create a data frame with the model names and their respective MAE values
model_names <- c("Model 1", "Without Density", "Without Area", "Without Density & VehBrand")
mae_values <- c(MAE1, MAE2, MAE3, MAE4)

mae_data <- data.frame(Model = model_names, MAE = mae_values)

# Create the bar plot
ggplot(mae_data, aes(x = Model, y = MAE)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Mean Absolute Error (MAE) for Different Models",
       x = "Model",
       y = "Mean Absolute Error (MAE)") +
  theme_minimal()

cat("MAE- Model 1: ", MAE1, "\n")
cat("MAE- Model 2 ", MAE2, "\n")
cat("MAE- Model 3 ", MAE3, "\n")

cat("MAE- Model 5: ", MAE5, "\n")
cat("MAE- Model 6 ", MAE6, "\n")
cat("MAE- Model 7 ", MAE7, "\n")


#plotting train MAEs
# Create a data frame with the model names and their respective MAE values
model_names2 <- c("Model 1", "Without Area", "Without Density", "Without Density & VehBrand")
mae_values2 <- c(MAE5, MAE6, MAE7, MAE8)

mae_data2 <- data.frame(Model = model_names2, MAE = mae_values2)

# Create the bar plot
ggplot(mae_data2, aes(x = Model, y = MAE)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Mean Absolute Error (MAE) for Different Models",
       x = "Model",
       y = "Mean Absolute Error (MAE)") +
  theme_minimal()

anova(poissonglm,poissonglm2,poissonglm3, test = "Chisq")



#-------K-Cross Validation---------------------


# Set the value of k (number of folds) for cross-validation
k <- 10

# k-fold cross-validation on test data
cv_test_1 <- cv.glm(data = test,glmfit = poissonglm, K = k)
cv_test_2 <- cv.glm(data = test,glmfit = poissonglm2, K = k)
cv_test_3 <- cv.glm(data = test,glmfit = poissonglm3, K = k)


cv_dev_1 <- cv_test_1$delta
cv_dev_2 <- cv_test_2$delta
cv_dev_3 <- cv_test_3$delta


cv_pred_1 <- fitted(poissonglm) - cv_dev_1
cv_pred_2 <- fitted(poissonglm2) - cv_dev_2
cv_pred_3 <- fitted(poissonglm3) - cv_dev_3

mae_cv_1 <- mean(abs(actual_counts - cv_pred_1))
mae_cv_2 <- mean(abs(actual_counts - cv_pred_2))
mae_cv_3 <- mean(abs(actual_counts - cv_pred_3))


cat("MAE - With all response variables: ", mae_cv_1, "\n")
cat("MAE- With only Area : ", mae_cv_2, "\n")
cat("MAE- With only Density: ", mae_cv_3, "\n")


#-----------------------K cross on train data-----------------------

# Perform k-fold cross-validation
cv_train_1 <- cv.glm(data = train,glmfit = poissonglm, K = k)
cv_train_2 <- cv.glm(data = train,glmfit = poissonglm2, K = k)
cv_train_3 <- cv.glm(data = train,glmfit = poissonglm3, K = k)


cv_dev_tr_1 <- cv_train_1$delta
cv_dev_tr_2 <- cv_train_2$delta
cv_dev_tr_3 <- cv_train_3$delta


cv_pred_tr_1 <- fitted(poissonglm) - cv_dev_tr_1
cv_pred_tr_2 <- fitted(poissonglm2) - cv_dev_tr_2
cv_pred_tr_3 <- fitted(poissonglm3) - cv_dev_tr_3


mae_cv_tr_1 <- mean(abs(actual_counts2 - cv_pred_tr_1))
mae_cv_tr_2 <- mean(abs(actual_counts2 - cv_pred_tr_2))
mae_cv_tr_3 <- mean(abs(actual_counts2 - cv_pred_tr_3))


cat("MAE - With all response variables: ", mae_cv_tr_1, "\n")
cat("MAE- With only Area: ", mae_cv_tr_2, "\n")
cat("MAE- With only density: ", mae_cv_tr_3, "\n")



# Kcross test data prediction 
model_names <- c("Model 1", "With Area", "With Density")
mae_values3 <- c(mae_cv_1, mae_cv_2, mae_cv_3)

mae_data3 <- data.frame(Model = model_names, MAE = mae_values3)

# Create the bar plot
ggplot(mae_data3, aes(x = Model, y = MAE)) +
  geom_bar(stat = "identity", fill = "skyblue", width = 0.5) +
  labs(title = "Mean Absolute Error (MAE) for Different Models",
       x = "Model",
       y = "Mean Absolute Error (MAE)") +
  theme_minimal()





#---------------------------------Claim Severity ------------------------------------------#



gamma_model <- glm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region + Area,
                   data = train, family = Gamma(link = "log"),
                   offset = log(ClaimNb))

pred_gamma_train <- predict(gamma_model, newdata = train, type = "response")

#------------------------------Model 2 ----With Area


gamma_model2 <- glm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Region + Area,
                   data = train, family = Gamma(link = "log"),
                   offset = log(ClaimNb))


#-------------------------Model 3----With Density

gamma_model3 <- glm(ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus + VehBrand + VehGas + Density + Region,
                    data = train, family = Gamma(link = "log"),
                    offset = log(ClaimNb))


#-------K-Cross Validation---------------------


# Set the value of k (number of folds) for cross-validation
k <- 10

# k-fold cross-validation on test data
cv_test_g_1 <- cv.glm(data = test,glmfit = gamma_model, K = k)
cv_test_g_2 <- cv.glm(data = test,glmfit = gamma_model2, K = k)
cv_test_g_3 <- cv.glm(data = test,glmfit = gamma_model3, K = k)


cv_dev_g_1 <- cv_test_g_1$delta
cv_dev_g_2 <- cv_test_g_2$delta
cv_dev_g_3 <- cv_test_g_3$delta


cv_pred_g_1 <- fitted(gamma_model) - cv_dev_g_1
cv_pred_g_2 <- fitted(gamma_model2) - cv_dev_g_2
cv_pred_g_3 <- fitted(gamma_model3) - cv_dev_g_3

mae_cv_g_1 <- mean(abs(actual_counts - cv_pred_g_1))
mae_cv_g_2 <- mean(abs(actual_counts - cv_pred_g_2))
mae_cv_g_3 <- mean(abs(actual_counts - cv_pred_g_3))


cat("MAE - With all response variables: ", mae_cv_g_1, "\n")
cat("MAE- With only Area : ", mae_cv_g_2, "\n")
cat("MAE- With only Density: ", mae_cv_g_3, "\n")


#-----------------------K cross on train data-----------------------

# Set the value of k (number of folds) for cross-validation
k <- 10

# k-fold cross-validation on test data
cv_train_g_1 <- cv.glm(data = train,glmfit = gamma_model, K = k)
cv_train_g_2 <- cv.glm(data = train,glmfit = gamma_model2, K = k)
cv_train_g_3 <- cv.glm(data = train,glmfit = gamma_model3, K = k)


cv_dev_g_1_tr <- cv_train_g_1$delta
cv_dev_g_2_tr <- cv_train_g_2$delta
cv_dev_g_3_tr <- cv_train_g_3$delta


cv_pred_g_1_tr <- fitted(gamma_model) - cv_dev_g_1_tr
cv_pred_g_2_tr <- fitted(gamma_model2) - cv_dev_g_2_tr
cv_pred_g_3_tr <- fitted(gamma_model3) - cv_dev_g_3_tr

mae_cv_g_1_tr <- mean(abs(actual_counts2 - cv_pred_g_1))
mae_cv_g_2_tr <- mean(abs(actual_counts2 - cv_pred_g_2))
mae_cv_g_3_tr <- mean(abs(actual_counts2 - cv_pred_g_3))


cat("MAE - With all response variables: ", mae_cv_g_1_tr, "\n")
cat("MAE- With only Area : ", mae_cv_g_2_tr, "\n")
cat("MAE- With only Density: ", mae_cv_g_3_tr, "\n")




#------------------calculating loss cost-------------------------------#

#Calculating expected claims from best freq model
pred_counts_GLM <- predict(poissonglm, newdata = test, type = "response")

#calculating average claim cost from best severity model
pred_sev_GLM <- predict(gamma_model, newdata = test, type = "response")

#calculate expected claim amount by multiplying predictions from freq and sev model
claim_amount_expected_GLM <- pred_sev_GLM * pred_counts_GLM

#calculate actual claim amount
actual_claim_amount <- test$ClaimAmount * test$ClaimNb

# Calculate the loss for each observation
loss_cost_GLM <- abs(claim_amount_expected_GLM - actual_claim_amount)


# Calculate the mean loss (loss cost) for all observations
mean_loss_cost_GLM <- mean(loss_cost_GLM)


#----------------------------MAE final plot------------------------------#


# Load necessary libraries
library(ggplot2)

# Create a data frame with model names and corresponding MAE values

mae_loss_cost <- data.frame(Model = c("GLM", "GBM", "NN", "Ensemble Model"),
                            MAE = c(mean_loss_cost_GLM, mean_loss_cost_GBM, mean_loss_cost_ANN, mean_loss_cost_ensemble))

# Define custom colors
custom_colors <- c("blue", "green", "red", "orange") # Add more colors if needed

# Plot MAE values
ggplot(mae_loss_cost, aes(x = Model, y = MAE, fill = Model)) +
  geom_bar(stat = "identity") +
  labs(title = "Comparison of Mean Absolute Error (MAE) for Different Models",
       x = "Models", y = "Mean Absolute Error (MAE)") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1)) +
  scale_fill_manual(values = custom_colors)



#---------------------------------END-------------------------------------#


