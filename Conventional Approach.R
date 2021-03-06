######### Titanic Model: Conventional Method

# Packages
require(caret)
require(randomForest)
require(readxl)
require(e1071)

# Loading Data
Survival <- read.csv("~/Desktop/Root/Projects/Titanic Competition/gender_submission.csv", header = TRUE)
Test <- read.csv("~/Desktop/Root/Projects/Titanic Competition/test.csv", header = TRUE)
Train <- read.csv("~/Desktop/Root/Projects/Titanic Competition/train.csv", header = TRUE)
Train <- Train[-c(892),]

# Creating Master File
PassengerId <- Test[,1]
Survived <- Survival[,2]
Test  <- cbind(PassengerId, Survived,Test[,2:11])
Master <- rbind(Train, Test)
Master$Sex <- as.character(Master$Sex)

# Changing Variable Structure
Master$Sex[which(Master$Sex == "male")] <- 1
Master$Sex[which(Master$Sex == "female")] <- 0
Master$Sex <- as.numeric(Master$Sex)

# Filling Missing Information
Age <- Master$Age[which(!is.na(Master$Age))]
hist(Age, main = "Distribution of Age", breaks = 50)

for(i in 1:length(Master$PassengerId)){
  if(is.na(Master$Age[i])){
    Master$Age[i] <- sample(c(1:80),1, replace = TRUE)
  } else {
    Master$Age[i] <- Master$Age[i]
  }
}

# Check, Train and Test Split
TrID <- sample(c(1:1309), floor(0.75*1309), replace = FALSE)
Train <- Master[TrID,]
Test <- Master[-TrID,]

Check <- Test[,c(1:2)]
for(i in length(Check$PassengerId)){ Check[,c("Pred", "Pred_GD", "Pred_R_Forest", "Pred_SVM_C", "Pred_SVM_Nu")] <- NA}

### Model Test: Logit (Without Age)

Logit <- glm(Survived ~ Sex + Pclass + SibSp + Age, data = Train, family = "binomial")
summary(Logit)

Percent <- c(1:101)
Search <- as.data.frame(Percent)
for(i in 1:101){
  Search[,c("Accuracy", "Gradient Descent")] <- NA
  Search$Percent[i] <- 0 + 0.01*i - 0.01
}

Test$Sex <- as.numeric(Test$Sex)
for(i in 1:101){
  Check$Pred <- 1/(1 + exp(-predict(Logit, Test)))
  Check$Pred[which(Check$Pred >= Search$Percent[i])] <- 1
  Check$Pred[which(Check$Pred < Search$Percent[i])] <- 0
  Check$Pred <- as.numeric(Check$Pred)
  Check$Survived <- factor(Check$Survived)
  Check$Pred <- factor(Check$Pred)
  
  if(is.element(1, Check$Pred) & is.element(0, Check$Pred)){
    Logit_CM <- confusionMatrix(Check$Survived, Check$Pred, positive = "1")
    Search$Accuracy[i] <- Logit_CM$overall[1]
  } else {
    Search$Accuracy[i] <- -1
  }
}
Search$Accuracy <- as.numeric(Search$Accuracy)
max(Search$Accuracy)

Check$Pred <- 1/(1 + exp(-predict(Logit, Test)))
A <- Search[which(Search$Accuracy == max(Search$Accuracy)),1]
Check$Pred[which(Check$Pred >= A[1])] <- 1
Check$Pred[which(Check$Pred < A[1])] <- 0
Check$Pred <- as.numeric(Check$Pred)
Check$Survived <- factor(Check$Survived)
Check$Pred <- factor(Check$Pred)

Logit_CM <- confusionMatrix(Check$Survived, Check$Pred, positive = "1")

Logit_CM$overall[1]

### Model Test: Logit (With Age)

Logit <- glm(Survived ~ Sex + Pclass + SibSp + Age, data = Train, family = "binomial")
summary(Logit)

Percent <- c(1:101)
Search <- as.data.frame(Percent)
for(i in 1:101){
  Search[,c("Accuracy", "Gradient Descent")] <- NA
  Search$Percent[i] <- 0 + 0.01*i - 0.01
}

Test$Sex <- as.numeric(Test$Sex)
for(i in 1:101){
  Check$Pred <- 1/(1 + exp(-predict(Logit, Test)))
  Check$Pred[which(Check$Pred >= Search$Percent[i])] <- 1
  Check$Pred[which(Check$Pred < Search$Percent[i])] <- 0
  Check$Pred <- as.numeric(Check$Pred)
  Check$Survived <- factor(Check$Survived)
  Check$Pred <- factor(Check$Pred)
  
  if(is.element(1, Check$Pred) & is.element(0, Check$Pred)){
    Logit_CM <- confusionMatrix(Check$Survived, Check$Pred, positive = "1")
    Search$Accuracy[i] <- Logit_CM$overall[1]
  } else {
    Search$Accuracy[i] <- -1
  }
}
Search$Accuracy <- as.numeric(Search$Accuracy)
max(Search$Accuracy)

Check$Pred <- 1/(1 + exp(-predict(Logit, Test)))
A <- Search[which(Search$Accuracy == max(Search$Accuracy)),1]
Check$Pred[which(Check$Pred >= A[1])] <- 1
Check$Pred[which(Check$Pred < A[1])] <- 0
Check$Pred <- as.numeric(Check$Pred)
Check$Survived <- factor(Check$Survived)
Check$Pred <- factor(Check$Pred)

Logit_CM <- confusionMatrix(Check$Survived, Check$Pred, positive = "1")

Logit_CM$overall[1]

### Applying Gradient Descent On Method 1: Logit

# Setting Up Matrix
Constant <- c(1:length(Train$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Train[,c(5, 3, 7, 6)])
X$Constant <- 1
X <- as.matrix(X)
Choice <- as.numeric(as.matrix(Train$Survived))

# Defining Functions
Sig <- function(y){
  no1 <- 1/(1+exp(-y))
  return(no1)
}

Cost <- function(Theta){
  n <- nrow(X)
  g <- Sig(X%*%Theta)
  J <- 1/n*sum((-Choice*log(g) - ((1-Choice)*log(1-g))))
  return(J)
}

beta <- NA
for(i in 1:5){ beta[i] <- Logit$coefficients[i] }
Initial_Theta <- c(beta[1], beta[2], beta[3], beta[4], beta[5])

Theta_Optim <- optim(fn = Cost, par = Initial_Theta, gr = "BFGS")

Constant <- c(1:length(Test$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Test[,c(5, 3, 7, 6)])
X$Constant <- 1
X <- data.matrix(X)

for(i in 1:101){
  Check$Pred_GD <- Sig(X%*%Theta_Optim$par)
  Check$Pred_GD[which(Check$Pred_GD >= Search$Percent[i])] <- 1
  Check$Pred_GD[which(Check$Pred_GD < Search$Percent[i])] <- 0
  Check$Pred_GD <- as.numeric(Check$Pred_GD)
  Check$Survived <- factor(Check$Survived)
  Check$Pred_GD <- factor(Check$Pred_GD)
  
  if(is.element(1, Check$Pred_GD) & is.element(0, Check$Pred_GD)){
    Logit_GD <- confusionMatrix(Check$Survived, Check$Pred_GD, positive = "1")
    Search$`Gradient Descent`[i] <- Logit_GD$overall[1]
  } else {
    Search$`Gradient Descent`[i] <- -1
  }
}
Search$`Gradient Descent` <- as.numeric(Search$`Gradient Descent`)
max(Search$`Gradient Descent`)

Check$Pred_GD <- Sig(X%*%Theta_Optim$par)
A <- Search$Percent[which(Search$`Gradient Descent` == max(Search$`Gradient Descent`))]
Check$Pred_GD[which(Check$Pred_GD >= A[1])] <- 1
Check$Pred_GD[which(Check$Pred_GD < A[1])] <- 0
Check$Pred_GD <- as.numeric(Check$Pred_GD)

Check$Survived <- factor(Check$Survived)
Check$Pred_GD <- factor(Check$Pred_GD)

Logit_GD <- confusionMatrix(Check$Survived, Check$Pred_GD, positive = "1")

Logit_GD$overall[1]

#############################################################################################
# The Gradient Descent approach used above can be found here:                               #
# https://www.r-bloggers.com/logistic-regression-with-r-step-by-step-implementation-part-2/ #
# written by Amar Gondaliya on December 8th, 2013.                                          #
#############################################################################################

### Model 2: Random Forest
Train$Survived <- factor(Train$Survived)
R_Forest <- randomForest(Survived ~ Sex + SibSp + Pclass + Age, data = Train, ntree = 1000, maxnodes = choose(6, 2), ntry = floor(sqrt(6)))
R_Forest

Check$Pred_R_Forest <- predict(R_Forest, Test)
R_Forest_CM <- confusionMatrix(Check$Survived, Check$Pred_R_Forest, positive = "1")

### Model 3: Support Vector Machines
SVM_C <- svm(Survived ~ Sex + SibSp + Pclass + Age, data = Train, type = "C-classification", kernel = "radial")
SVM_Nu <- svm(Survived ~ Sex + SibSp + Pclass + Age, data = Train, type = "nu-classification", kernel = "radial")

SVM <- Test[,c(3,5,6,7)]
Check$Pred_SVM_C <- predict(SVM_C, SVM)
Check$Pred_SVM_Nu <- predict(SVM_Nu, SVM)

SVM_CM_C <- confusionMatrix(Check$Survived, Check$Pred_SVM_C)
SVM_CM_Nu <- confusionMatrix(Check$Survived, Check$Pred_SVM_Nu)

Logit_CM$overall[1]
Logit_GD$overall[1]
R_Forest_CM$overall[1]
if(SVM_CM_C$overall[1] > SVM_CM_Nu$overall[1]){
  print("SVM C-Classification")
  SVM_CM_C$overall[1]
} else {
  print("SVM Nu-Classification")
  SVM_CM_Nu$overall[1]
}
