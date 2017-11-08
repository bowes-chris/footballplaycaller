setwd("/Users/johnrivas/Desktop/UHD/SCHOOL/Fall_2017/Predictive/Project/GitHub/yards.gained")
source("yards.gained.models.R")
training <- read.csv("yards.gained.train.csv")
testing <- read.csv("yards.gained.test.csv")
training <- training[,-1]
testing <- testing[,-1]
training <- training[1:500,] # i used to make sure that everything could run. I commented out once it all worked. Left just in case.
# # There were some errors that i think were due to not using all factor levels in model. I will need more computing power. 

########## Decision Tree
# Train model
dc.fit <- do.e071.DT(training)
save(dc.fit, file = "dc.fit.rda")

# Performance
1-dc.fit$best.performance 
# Best Tuned Model's Parameters
dc.fit$best.parameters
# Plot Accuracy
plot(sort(1-dc.fit$performances[,3]),type = "l", ylab = "Accuracy", xlab = "Worst to Best Parameter Combination", main = "Decision Tree")

# Testing
dc.Fit.Pred <- do.e1071.Predict(dc.fit,testing)
dc.Fit.Pred

########## Random Forest
# Train model
RF.fit <- do.RF(training)
save(RF.fit, file = "RF.fit.rda")
print(RF.fit)
cor(round(RF.fit$finalModel$predicted,digits = 0),training$Yards.Gained[1:500])

plot(RF.fit$results[,1],RF.fit$results[,2], type = "l", ylab = "Accurary", xlab = "mtry", main = "Random Forest")

# Testing
RF.fit.Pred <- do.CARET.Predict(RF.fit,testing)
RF.fit.Pred
# Accuracy :

########## SVM radial basis kernel
svm.radial.fit <- do.RadialKernelSVM(training)
print(svm.radial.fit)
svm.radial.fit$bestTune

plot(svm.radial.fit$results[,3],type = "l", ylab = "Accuracy", xlab = "Index of Parameter Combination", main = "SVM Radial")

svm.radial.fit$results
# 

# Testing
svm.radial.Pred <- do.CARET.Predict(svm.radial.fit,testing)
svm.radial.Pred
# Accuracy :

########## SVM polynomial kernel
svm.poly.Fit <- do.PolyKernelSVM(training)
print(svm.poly.Fit)
svm.poly.Fit$bestTune

plot(svm.poly.Fit$results[,4],type = "l", ylab = "Accuracy", xlab = "Index of Parameter Combination", main = "SVM Poly")

svm.poly.Fit$results
# Accuracy : 

# Testing
svm.poly.Pred <- do.CARET.Predict(svm.poly.Fit,testing)
svm.poly.Pred
# Accuracy : 

training <- read.csv("yards.gained.train.csv")
testing <- read.csv("yards.gained.test.csv")
training <- training[,-1]
testing <- testing[,-1]
training$Yards.Gained <- as.factor(training$Yards.Gained)
testing$Yards.Gained <- as.factor(testing$Yards.Gained)
training <- training[1:500,] # i used to make sure that everything could run. I commented out once it all worked. Left just in case.
# # There were some errors that i think were due to not using all factor levels in model. I will need more computing power. 

########## KNN
knn.fit <- do.KNN(training)
print(knn.fit)
knn.fit$bestTune

plot(knn.fit$results[,4],type = "l", ylab = "Accuracy", xlab = "Index of Parameter Combination", main = "KNN")
knn.fit$
knn.fit$results
# Accuracy :  

# Testing
knn.Pred <- do.CARET.Predict(knn.it,testing)
knn.Pred
# Accuracy :

########## Penalized logistic regression
# Create baseline regression model
fit <- lm(Yards.Gained ~ ., data = training)
cor(round(fit$fitted.values,digits = 0),training$Yards.Gained[1:500])
plot(fit$fitted.values,training$Yards.Gained[1:500], main = "Yards Gained - Linear Model",ylab = "Actuals", xlab = "Predicted Yards Gained")
abline(lm(training$Yards.Gained~fit$fitted.values))

# Testing
pred <- predict(fit, newdata = testing, interval = "prediction")
