######### Titanic Model: Using Split Data By Kaggle

# Packages
require(caret)
require(randomForest)
require(readxl)
require(e1071)

# Loading Data
Survival <- read.csv("~/Desktop/Root/Projects/Titanic Competition/gender_submission.csv", header = TRUE)
Testing_Set <- read.csv("~/Desktop/Root/Projects/Titanic Competition/test.csv", header = TRUE)
Training_Set <- read.csv("~/Desktop/Root/Projects/Titanic Competition/train.csv", header = TRUE)
Training_Set <- Training_Set[-892,]

### Model 1: Logit

# Transforming Data Structures
Training_Set$Sex <- as.character(Training_Set$Sex)
Training_Set$Sex[which(Training_Set$Sex == "male")] <- 1
Training_Set$Sex[which(Training_Set$Sex == "female")] <- 0
Training_Set$Sex <- as.numeric(Training_Set$Sex)

# Filling Missing Information
Samp_Size <- length(which(is.na(Training_Set$Age)))
Missing_RID <- sample(which(!is.na(Training_Set$Age)), Samp_Size, replace = FALSE)
Training_Set$Age[which(is.na(Training_Set$Age))] <- Training_Set$Age[Missing_RID]

Samp_Size_Test <- length(which(is.na(Testing_Set$Age)))
Missing_RID_Test <- sample(which(!is.na(Testing_Set$Age)), Samp_Size_Test, replace = FALSE)
Testing_Set$Age[which(is.na(Testing_Set$Age))] <- Testing_Set$Age[Missing_RID_Test]

Age_Random <- Training_Set$Age
hist(Age_Random, main = "Distribution of Randomly Assigned Age", xlab = "Age Bins", 100)

################################################################
# Distribution of Age seems to not have changed. As such, this #
# appears to be a viable solution.                             #
################################################################

### Model Test: Logit (With Age)

for(i in length(Survival$PassengerId)){
  Survival[, c("Pred", "Pred_GD", "Pred_R_Forest", "Pred_SVM_C", "Pred_SVM_Nu")] <- NA
}

Logit <- glm(Survived ~ Sex + Pclass + SibSp + Age, data = Training_Set, family = "binomial")
summary(Logit)

Percent <- c(1:101)
Search <- as.data.frame(Percent)
for(i in 1:101){
  Search[,c("Accuracy", "Gradient Descent")] <- NA
  Search$Percent[i] <- 0 + 0.01*i - 0.01
}

Testing_Set$Sex <- as.character(Testing_Set$Sex)
Testing_Set$Sex[which(Testing_Set$Sex == "female")] <- 0
Testing_Set$Sex[which(Testing_Set$Sex == "male")] <- 1
Testing_Set$Sex <- as.numeric(Testing_Set$Sex)

for(i in 1:101){
  Survival$Pred <- 1/(1 + exp(-predict(Logit, Testing_Set)))
  Survival$Pred[which(Survival$Pred >= Search$Percent[i])] <- 1
  Survival$Pred[which(Survival$Pred < Search$Percent[i])] <- 0
  Survival$Pred <- as.numeric(Survival$Pred)
  Survival$Survived <- factor(Survival$Survived)
  Survival$Pred <- factor(Survival$Pred)
  
  if(is.element(1, Survival$Pred) & is.element(0, Survival$Pred)){
    Logit_CM <- confusionMatrix(Survival$Survived, Survival$Pred, positive = "1")
    Search$Accuracy[i] <- Logit_CM$overall[1]
  } else {
    Search$Accuracy[i] <- -1
  }
}
Search$Accuracy <- as.numeric(Search$Accuracy)
max(Search$Accuracy)

Survival$Pred <- 1/(1 + exp(-predict(Logit, Testing_Set)))
A <- Search[which(Search$Accuracy == max(Search$Accuracy)),1]
Survival$Pred[which(Survival$Pred >= A[1])] <- 1
Survival$Pred[which(Survival$Pred < A[1])] <- 0
Survival$Pred <- as.numeric(Survival$Pred)
Survival$Survived <- factor(Survival$Survived)
Survival$Pred <- factor(Survival$Pred)

Logit_CM <- confusionMatrix(Survival$Survived, Survival$Pred, positive = "1")

Logit_CM$overall[1]

### Model Test: Logit (Without Age)

for(i in length(Survival$PassengerId)){
  Survival[, c("Pred", "Pred_GD", "Pred_R_Forest", "Pred_SVM_C", "Pred_SVM_Nu")] <- NA
}

Logit <- glm(Survived ~ Sex + Pclass + SibSp, data = Training_Set, family = "binomial")
summary(Logit)

Percent <- c(1:101)
Search <- as.data.frame(Percent)
for(i in 1:101){
  Search[,c("Accuracy", "Gradient Descent")] <- NA
  Search$Percent[i] <- 0 + 0.01*i - 0.01
}

Testing_Set$Sex <- as.character(Testing_Set$Sex)
Testing_Set$Sex[which(Testing_Set$Sex == "female")] <- 0
Testing_Set$Sex[which(Testing_Set$Sex == "male")] <- 1
Testing_Set$Sex <- as.numeric(Testing_Set$Sex)

for(i in 1:101){
  Survival$Pred <- 1/(1 + exp(-predict(Logit, Testing_Set)))
  Survival$Pred[which(Survival$Pred >= Search$Percent[i])] <- 1
  Survival$Pred[which(Survival$Pred < Search$Percent[i])] <- 0
  Survival$Pred <- as.numeric(Survival$Pred)
  Survival$Survived <- factor(Survival$Survived)
  Survival$Pred <- factor(Survival$Pred)
  
  if(is.element(1, Survival$Pred) & is.element(0, Survival$Pred)){
    Logit_CM <- confusionMatrix(Survival$Survived, Survival$Pred, positive = "1")
    Search$Accuracy[i] <- Logit_CM$overall[1]
  } else {
    Search$Accuracy[i] <- -1
  }
}
Search$Accuracy <- as.numeric(Search$Accuracy)
max(Search$Accuracy)

Survival$Pred <- 1/(1 + exp(-predict(Logit, Testing_Set)))
A <- Search[which(Search$Accuracy == max(Search$Accuracy)),1]
Survival$Pred[which(Survival$Pred >= A[1])] <- 1
Survival$Pred[which(Survival$Pred < A[1])] <- 0
Survival$Pred <- as.numeric(Survival$Pred)
Survival$Survived <- factor(Survival$Survived)
Survival$Pred <- factor(Survival$Pred)

Logit_CM <- confusionMatrix(Survival$Survived, Survival$Pred, positive = "1")

Logit_CM$overall[1]

### Applying Gradient Descent On Method 1: Logit

# Setting Up Matrix
Constant <- c(1:length(Training_Set$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Training_Set[,c(5, 3, 7)])
X$Constant <- 1
X <- as.matrix(X)
Choice <- as.numeric(as.matrix(Training_Set$Survived))

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

Initial_Theta <- NA
for(i in 1:4){ Initial_Theta[i] <- Logit$coefficients[i] }

Theta_Optim <- optim(fn = Cost, par = Initial_Theta)

Constant <- c(1:length(Testing_Set$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Testing_Set[,c(4, 2, 6)])
X$Constant <- 1
X <- data.matrix(X)

for(i in 1:101){
  Survival$Pred_GD <- Sig(X%*%Theta_Optim$par)
  Survival$Pred_GD[which(Survival$Pred_GD >= Search$Percent[i])] <- 1
  Survival$Pred_GD[which(Survival$Pred_GD < Search$Percent[i])] <- 0
  Survival$Pred_GD <- as.numeric(Survival$Pred_GD)
  Survival$Survived <- factor(Survival$Survived)
  Survival$Pred_GD <- factor(Survival$Pred_GD)
  
  if(is.element(1, Survival$Pred_GD) & is.element(0, Survival$Pred_GD)){
    Logit_GD <- confusionMatrix(Survival$Survived, Survival$Pred_GD, positive = "1")
    Search$`Gradient Descent`[i] <- Logit_GD$overall[1]
  } else {
    Search$`Gradient Descent`[i] <- -1
  }
}
Search$`Gradient Descent` <- as.numeric(Search$`Gradient Descent`)
max(Search$`Gradient Descent`)

Survival$Pred_GD <- Sig(X%*%Theta_Optim$par)
A <- Search$Percent[which(Search$`Gradient Descent` == max(Search$`Gradient Descent`))]
Survival$Pred_GD[which(Survival$Pred_GD >= A[1])] <- 1
Survival$Pred_GD[which(Survival$Pred_GD < A[1])] <- 0
Survival$Pred_GD <- as.numeric(Survival$Pred_GD)

Survival$Survived <- factor(Survival$Survived)
Survival$Pred_GD <- factor(Survival$Pred_GD)

Logit_GD <- confusionMatrix(Survival$Survived, Survival$Pred_GD, positive = "1")

#############################################################################################
# The Gradient Descent approach used above can be found here:                               #
# https://www.r-bloggers.com/logistic-regression-with-r-step-by-step-implementation-part-2/ #
# written by Amar Gondaliya on December 8th, 2013.                                          #
#############################################################################################

### Model 2: Random Forest
Training_Set$Survived <- factor(Training_Set$Survived)
R_Forest <- randomForest(Survived ~ Sex + Pclass + SibSp + Age, data = Training_Set, ntree = 1000, maxnodes = choose(6, 2), ntry = floor(sqrt(6)))

Survival$Pred_R_Forest <- predict(R_Forest, Testing_Set)
R_Forest_CM <- confusionMatrix(Survival$Survived, Survival$Pred_R_Forest, positive = "1")

### Model 3: Support Vector Machines
SVM_C <- svm(Survived ~ Sex + Pclass + SibSp + Age, data = Training_Set, type = "C-classification", kernel = "radial")
SVM_Nu <- svm(Survived ~ Sex + Pclass + SibSp + Age, data = Training_Set, type = "nu-classification", kernel = "radial")

Test <- Testing_Set[,c(4, 2, 6, 5)]
Survival$Pred_SVM_C <- predict(SVM_C, Test)
Survival$Pred_SVM_Nu <- predict(SVM_Nu, Test)

SVM_CM_C <- confusionMatrix(Survival$Survived, Survival$Pred_SVM_C)
SVM_CM_Nu <- confusionMatrix(Survival$Survived, Survival$Pred_SVM_Nu)

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
