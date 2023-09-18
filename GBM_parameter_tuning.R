

#--------------------------starts from here-------------------------------------------


# Create a 10% subset
subset_ind <- initial_split(filtered_data, prop = 0.1)  # 10% for the subset
subset_data <- training(subset_ind)  #training portion for the subset

cat("The number of rows in the 10% subset is", nrow(subset_data))


par(mfrow=c(1,2))

#Claim number Bar plot
barplot(table(filtered_data$ClaimNb),
        main = "Original Data",
        xlab = "Claim Count",
        ylab = "Frequency",
        col = "green"
)

#Claim number Bar plot
barplot(table(subset_data$ClaimNb),
        main = "Subset",
        xlab = "Claim Count",
        ylab = "Frequency",
        col = "green"
)



# Claim Amount
hist(filtered_data$ClaimAmount, 
     main = "Original Data",
     xlab = "Claim Amount",
     ylab = "Count",
     col = "blue",
     border = "blue" 
     #breaks = "FD"  # Method for determining the number of bins (histogram breaks)
)

# Claim Amount histogram
hist(subset_data$ClaimAmount, 
     main = "Subset",
     xlab = "Claim Amount",
     ylab = "Count",
     col = "blue",
     border = "blue" 
     #breaks = "FD"  # Method for determining the number of bins (histogram breaks)
)

#Running a basic model with 1500 trees
  
gbm1500 <- gbm(
    ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus
    + VehBrand + VehGas + Region + Density + Area, data = subset_data, distribution = "poisson", n.trees = 1500, 
    shrinkage = 0.1, interaction.depth = 3, n.minobsinnode = 10,cv.folds = 3
  )


best_1500 <- which.min(gbm1500$cv.error)
  
# get MSE and compute RMSE
sqrt(gbm1500$cv.error[best_1500])
  
  
gbm.perf(gbm1500, method = "cv") #the blue dashed line shows what is the optimal number of iterations according to the metric and validation procedure used


  
# create hyperparameter grid
  hyper_grid <- expand.grid(
    shrinkage = c(.01, .1, .3),
    interaction.depth = c(1, 3, 5),
    n.minobsinnode = c(5, 10, 15),
    optimal_trees = NA,               # a place to dump results
    min_RMSE = NA                     # a place to dump results
  )
  
  # total number of combinations
  nrow(hyper_grid)
  
  # randomize data
  random_index <- sample(1:nrow(subset_data), nrow(subset_data))
  random_data_train <- train[random_index, ]
  
  # grid search 
  for(i in 1:nrow(hyper_grid)) {
    
    # reproducibility
    set.seed(123)
    
    # train model
    gbm.tune <- gbm(
      formula = ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus
      + VehBrand + VehGas + Region + Density + Area,
      distribution = "poisson",
      data = random_data_train,
      n.trees = 1500,
      interaction.depth = hyper_grid$interaction.depth[i],
      shrinkage = hyper_grid$shrinkage[i],
      n.minobsinnode = hyper_grid$n.minobsinnode[i],
      cv.folds = 3
    )
    
    # add min training error and trees to grid
    hyper_grid$optimal_trees[i] <- which.min(gbm.tune$cv.error)
    hyper_grid$min_RMSE[i] <- sqrt(min(gbm.tune$cv.error))
  }
  
  hyper_grid %>% 
    dplyr::arrange(min_RMSE) %>%
    head(10)  
  
  
  
  
  # create hyperparameter grid
  hyper_grid_2 <- expand.grid(
    shrinkage = c(0.1, 0.05, 0.01),
    interaction.depth = c(3, 5, 7),
    n.minobsinnode = c(5, 10, 15),
    optimal_trees = NA,               # a place to dump results
    min_RMSE = NA                     # a place to dump results
  )
  
  
  
nrow(hyper_grid_2)




# grid search 
for(i in 1:nrow(hyper_grid_2)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune1 <- gbm(
    formula = ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus
    + VehBrand + VehGas + Region + Density + Area ,
    distribution = "poisson",
    data = random_data_train,
    n.trees = 1500,
    interaction.depth = hyper_grid_2$interaction.depth[i],
    shrinkage = hyper_grid_2$shrinkage[i],
    n.minobsinnode = hyper_grid_2$n.minobsinnode[i],
    cv.folds = 3
  )
  
  # add min training error and trees to grid
  hyper_grid_2$optimal_trees[i] <- which.min(gbm.tune1$cv.error)
  hyper_grid_2$min_RMSE[i] <- sqrt(min(gbm.tune1$cv.error))
}

hyper_grid_2 %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)  


# for reproducibility
set.seed(123)


# train GBM model
gbm.fit.final <- gbm(
  formula = ClaimNb ~ VehPower + VehAge + DrivAge + BonusMalus
  + VehBrand + VehGas + Region + Density + Area,
  distribution = "poisson",
  data = train,
  n.trees = 515,
  interaction.depth = 7,
  shrinkage = 0.01,
  n.minobsinnode = 5
)  



#---------------------------severity--------------------------------------#



  
#Running a basic model with 1500 trees

gbm1500.sev <- gbm(
  ClaimAmount ~  VehPower + VehAge + DrivAge + BonusMalus
  + VehBrand + VehGas + Region + Density + Area , data = subset_data, distribution = "laplace", n.trees = 1500, 
  shrinkage = 0.1, interaction.depth = 3, n.minobsinnode = 10,cv.folds = 3
)


best_1500.sev <- which.min(gbm1500.sev$cv.error)

# get MSE and compute RMSE
sqrt(gbm1500.sev$cv.error[best_1500.sev])

gbm.perf(gbm1500.sev, method = "cv") #the blue dashed line shows what is the optimal number of iterations according to the metric and validation procedure used

#optimal number of trees were 53


# create hyperparameter grid
hyper_grid.sev <- expand.grid(
  shrinkage = c(.01, .1, .3),
  interaction.depth = c(1, 3, 5),
  n.minobsinnode = c(5, 10, 15),
  optimal_trees = NA,               # a place to dump results
  min_RMSE = NA                     # a place to dump results
)

# total number of combinations
nrow(hyper_grid.sev)

# randomize data
random_index <- sample(1:nrow(subset_data), nrow(subset_data))
random_data_train <- train[random_index, ]

# grid search 
for(i in 1:nrow(hyper_grid.sev)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune.sev <- gbm(
    formula = ClaimAmount ~  VehPower + VehAge + DrivAge + BonusMalus
    + VehBrand + VehGas + Region + Density + Area,
    distribution = "laplace",
    data = random_data_train,
    n.trees = 1500,
    interaction.depth = hyper_grid.sev$interaction.depth[i],
    shrinkage = hyper_grid.sev$shrinkage[i],
    n.minobsinnode = hyper_grid.sev$n.minobsinnode[i],
    cv.folds = 3
  )
  
  # add min training error and trees to grid
  hyper_grid.sev$optimal_trees[i] <- which.min(gbm.tune.sev$cv.error)
  hyper_grid.sev$min_RMSE[i] <- sqrt(min(gbm.tune.sev$cv.error))
}

hyper_grid.sev %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)  




# create hyperparameter grid
hyper_grid_2.sev <- expand.grid(
  shrinkage = c(.01, .1, 0.05),
  interaction.depth = c(3, 5, 7),
  n.minobsinnode = c(5, 10, 15),
  optimal_trees = NA,               # a place to dump results
  min_RMSE = NA                     # a place to dump results
)


nrow(hyper_grid_2.sev)




# grid search 
for(i in 1:nrow(hyper_grid_2.sev)) {
  
  # reproducibility
  set.seed(123)
  
  # train model
  gbm.tune1.sev <- gbm(
    formula = ClaimAmount ~ VehPower + VehAge + DrivAge + BonusMalus
    + VehBrand + VehGas + Region + Density + Area,
    distribution = "laplace",
    data = random_data_train,
    n.trees = 1500,
    interaction.depth = hyper_grid_2.sev$interaction.depth[i],
    shrinkage = hyper_grid_2.sev$shrinkage[i],
    n.minobsinnode = hyper_grid_2.sev$n.minobsinnode[i],
    cv.folds = 3
  )
  
  # add min training error and trees to grid
  hyper_grid_2.sev$optimal_trees[i] <- which.min(gbm.tune1.sev$cv.error)
  hyper_grid_2.sev$min_RMSE[i] <- sqrt(min(gbm.tune1.sev$cv.error))
}

hyper_grid_2.sev %>% 
  dplyr::arrange(min_RMSE) %>%
  head(10)  


# for reproducibility
set.seed(123)

# train GBM model
gbm.fit.sev <- gbm(
  formula = ClaimAmount ~  VehPower + VehAge + DrivAge + BonusMalus
  + VehBrand + VehGas + Region + Density,
  distribution = "laplace",
  data = subset_data,
  n.trees = 278,
  interaction.depth = 5,
  shrinkage = 0.01,
  n.minobsinnode = 5
) 

# train GBM model
gbm.fit.sev.final <- gbm(
  formula = ClaimAmount ~  VehPower + VehAge + DrivAge + BonusMalus
  + VehBrand + VehGas + Region + Density,
  distribution = "laplace",
  data = train,
  n.trees = 278,
  interaction.depth = 5,
  shrinkage = 0.01,
  n.minobsinnode = 5
) 



