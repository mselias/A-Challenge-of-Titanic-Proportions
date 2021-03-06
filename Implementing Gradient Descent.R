######### Implementation of Gradient Descent Without Optim

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

### Applying Gradient Descent: On Randomly Sampled Coefficients

# Setting Up Matrix
Constant <- c(1:length(Train$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Train[,c(5, 3, 7, 6)])
X$Constant <- 1
X <- as.matrix(X)
Choice <- as.numeric(as.matrix(Train$Survived))

Theta <- NA
for(i in 1:5){ Theta[i] <- sample(c(-5:5),1) }

A <- Cost(Theta)

while(if(is.element(NaN,A)){TRUE}else{A >= 0.85}){
  alpha <- 0.001
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
  
  Gradient <- function(Theta,k){
    n <- nrow(X)
    g <- Sig(X[,k]*Theta)
    Grad <- g - Choice
    return(Grad)
  }
  
  for(k in 1:5){
    n <- nrow(X)
    Theta[k] <- Theta[k] - alpha*(1/n)*sum(X[,k]%*%Gradient(Theta[k],k))
  }
  A <- Cost(Theta)
  print(A)
}

Theta_Optim <- Theta

Constant <- c(1:length(Test$PassengerId))
X <- as.data.frame(Constant)
X <- cbind(X,Test[,c(5, 3, 7, 6)])
X$Constant <- 1
X <- data.matrix(X)

Percent <- c(1:101)
Search <- as.data.frame(Percent)
for(i in 1:101){
  Search[,c("Accuracy", "Gradient Descent")] <- NA
  Search$Percent[i] <- 0 + 0.01*i - 0.01
}

for(i in 1:101){
  Check$Pred_GD <- Sig(X%*%Theta_Optim)
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

Check$Pred_GD <- Sig(X%*%Theta_Optim)
A <- Search$Percent[which(Search$`Gradient Descent` == max(Search$`Gradient Descent`))]
Check$Pred_GD[which(Check$Pred_GD >= A[1])] <- 1
Check$Pred_GD[which(Check$Pred_GD < A[1])] <- 0
Check$Pred_GD <- as.numeric(Check$Pred_GD)

Check$Survived <- factor(Check$Survived)
Check$Pred_GD <- factor(Check$Pred_GD)

Logit_GD <- confusionMatrix(Check$Survived, Check$Pred_GD, positive = "1")
Logit_GD$overall[1]
