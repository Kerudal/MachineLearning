### SOURCE 
source("classifieurs.R")

options(warn=-1)

############################################################
# Library 
############################################################
library(MASS)
library(pROC)
library(mlbench)
library(caret)# confusion matrix
library(glmnet)
library(klaR)
#library(e1071)
library(corrplot)
library(nnet) # multinomial log linear models  
library(FNN)
library(naivebayes)
library(mda)

############################################################
# Load Data for classification 
############################################################
data <- read.table('../data/TPN1_a20_clas_app.txt')
names(data)[51] <- "Y"
data$Y <- as.factor(data$Y)

# mix up observations randomly
data <- data[sample(nrow(data)), ]

############################################################
# Nested and Non-nested Cross validation
############################################################
# creation de groupes pour la validation croisée 
n <-  nrow(data)
group.outer <- rep((1:10), (n/10)+1)[1:n]
idx.test.outer <- list()
idx.test.inner <- list()
rs.data.inner <- list()
for(i in 1:10){
  index.cv <- which(group.outer==i)
  idx.test.outer[[i]] <- index.cv
  n.inner <- n - length(index.cv)
  rs.data.inner[[i]] <- sample(n.inner)
  group.inner <- rep((1:10), (n.inner/10)+1)[1:n.inner]
  for(j in 1:10){
    index.cv <- which(group.inner==j)
    idx.test.inner[[j]] <- index.cv
  }
}

################################################################
## Classification
################################################################
err <-  matrix(0, 10, 13)
colnames(err) <- c("LR", "Ridge", "Lasso", "ElNet" , "RLM" , "RLN" ,"KNN", "NB", "QDA","LDA" , "MDA" , "FDA" , "RDA")

# ======== Regression Logistic Normal - RLN ============ # 
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # inner cross-valition
  X <- as.matrix(data.inner[, -51])
  y <- as.factor(data.inner[, 51])
  model.rln <- glmnet(X, y, family = "multinomial", alpha = 1, lambda = 0)
  
  # validation
  pred <- predict(model.rln, newx=as.matrix(data.validation[, -51]), type="response")
  confusion <- table(data.validation[, 51], max.col(pred[,,]))
  err[i, c("RLN")] <- 1 - sum(diag(confusion))/nrow(data.validation)
}

# ======== KNN classification ============ #
Kmax <- 15
K.sequences <- seq(1, Kmax, by = 2)
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne (inner)
  knn.mse.kmin <- rep(0, length(K.sequences))
  for(k in 1:length(K.sequences)){
    for(j in 1:10){
      index.inner.cv <- idx.test.inner[[j]]
      data.train.x <- data.inner[-index.inner.cv, -51]
      data.train.y <- data.inner[-index.inner.cv, 51]
      data.test.x <- data.inner[index.inner.cv, -51]
      data.test.y <- data.inner[index.inner.cv, 51]
      model.reg <- knn(train=data.train.x, test=data.test.x, 
                       cl=data.train.y, k=k)
      errc <- 1 - sum(diag(table(data.test.y, model.reg)))/nrow(data.inner)
      knn.mse.kmin[k] <- knn.mse.kmin[k] + errc
    }
  } 
  idx.kmin <- which(min(knn.mse.kmin) == knn.mse.kmin)
  best.kmin <- K.sequences[idx.kmin]
  
  # validation our model with best model 
  data.train.x <- data.inner[, -51]
  data.train.y <- data.inner[, 51]
  data.test.x <- data.validation[, -51]
  data.test.y <- data.validation[, 51]
  model.knn <- knn(train=data.train.x, test=data.test.x, cl=data.train.y, k=best.kmin)
  n <- nrow(data.validation)
  err[i, c("KNN")] <- 1 - sum(diag(table(data.test.y, model.knn)))/n
}

# ======== Naive Bayes ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # inner cross-valition
  X <- as.matrix(data.inner[, -51])
  y <- as.factor(data.inner[, 51])
  model.naiveBayes <- naive_bayes(x=X,y=y)
  
  # validation
  #naiveBayes.pred.prob <- predict(naiveBayes.model,newdata=data.test[,-51],type="prob")
  err[i,c("NB")] <- testError(data.validation,model.naiveBayes)
}

# ======== Multinomial Logistic Linear - RLM  (neural networks) ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]

  # inner cross-valition
  model.multinom <- multinom(Y~.,data=data.inner)

  # validation
  # multinom.pred.prob <- predict(multinom.model,newdata=data.test[,-51],type="prob")
  err[i,c("RLM")] <- testError(test=data.validation,model=model.multinom)
}


# ======== Regularized Discriminant Analysis RDA  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # inner cross-validation
  # grid search 
  cv_5_grid = trainControl(method = "cv", number = 10)
  fit_rda_grid = train(Y ~ ., data = data.inner, method = "rda", trControl = cv_5_grid)
  # random search 
  cv_5_rand = trainControl(method = "cv", number = 10, search = "random")
  fit_rda_rand = train(Y ~ ., data = data.inner, method = "rda",trControl = cv_5_rand, tuneLength = 9)
  if ( max(fit_rda_grid$results$Accuracy) > max(fit_rda_rand$results$Accuracy) ) {
    bestLambda = fit_rda_grid$bestTune[1,2]
    bestGamma  = fit_rda_grid$bestTune[1,1]
  }
  else {
    bestLambda = fit_rda_rand$bestTune[1,2]
    bestGamma  = fit_rda_rand$bestTune[1,1]
  }
  
  # validation
  model.rda<-rda(Y~.,data.inner,lambda = bestLambda , gamma=bestGamma)   
  err[i,c("RDA")] <- testError(test=data.validation,model=model.rda,type="rda") 
}


# ======== Quadratic Discriminant Analysis QDA  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne 
  model.qda<-qda(Y~.,data.inner) 
  
  # validation
  err[i,c("QDA")] <- testError(test=data.validation,model=model.qda,type="rda") 
}

# ======== Linear Discriminant Analysis LDA  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne 
  model.lda<-lda(Y~.,data.inner) 
  
  # validation
  err[i,c("LDA")] <- testError(test=data.validation,model=model.lda,type="rda") 
}

# ======== Mixture Discriminant Analysis MDA  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne 
  model.mda<-mda(Y~.,data.inner) 
  
  # validation
  err[i,c("MDA")] <- testError(test=data.validation,model=model.mda) 
}


# ======== Flexible Discriminant Analysis FDA  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne 
  model.fda<-fda(Y~.,data.inner) 
  
  # validation
  err[i,c("FDA")] <- testError(test=data.validation,model=model.fda) 
}

# ======== Linear Regression LR  ============ #
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne 
  model.lm<-lm(Y~.,data.inner) 
  
  # validation
  err[i,c("LR")] <- testError(test=data.validation,model=model.lm) 
}



# =================================================================== #
#                     LOGISTIC REGRESSION                             # 
# =================================================================== #
# http://www.sthda.com/english/articles/36-classification-methods-essentials/149-penalized-logistic-regression-essentials-in-r-ridge-lasso-and-elastic-net/ 
# https://daviddalpiaz.github.io/r4sl/regularization.html#ridge-regression
# LASSO // RIDGE // ELASTIC NET ? 

err_regression <-  matrix(0, 10, 3)
colnames(err_regression) <- c("Ridge Regression", "Lasso Regression", "ElasticNet Reg.") 

# Ridge Regression 
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  # Standardization
  #data.validation <- merge(scale(data.validation[,-51]),data.validation[,51])
  #data.inner      <- merge(scale(data.inner[,-51]),data.inner[,51])
  
  # cross-validation interne (inner)
  x <- as.matrix(data.inner[, -51])
  #x <- scale(x)
  y <- as.factor(data.inner[, 51])
  fit_ridge = glmnet(x, y, alpha = 0, family = "multinomial")
    #plot(fit_ridge)
    #plot(fit_ridge, xvar = "lambda", label = TRUE)
  # Inner Cross-Validation with 10 folds by default using cv.glmnet 
  fit_ridge_cv = cv.glmnet(x, y, alpha = 0,family = "multinomial" )
    #plot(fit_ridge_cv)
  #cv_fit <- cv.glmnet(x,y,nfolds = 5,type.measure = "class", alpha = 0, grouped = FALSE,family = "multinomial")
  #plot(cv_fit)
  #fit <- glmnet(x,y, alpha = 0, family = "multinomial")
  #predict(fit, X.test , s = cv_fit$lambda.min, type = "class")
  best.lambda = fit_ridge_cv$lambda.min
  
  # validation model 
  X.train <- as.matrix(data.inner[, -51])
  #X.train <- scale(X.train)
  y.train <- data.inner[, 51]
  X.test <- as.matrix(data.validation[, -51])
  #X.test <- scale(X.test)
  y.test <- data.validation[, 51]
  # Fit the final model on the training data
  model.ridge <- glmnet(X.train, y.train, alpha = 0 , family = "multinomial", lambda = best.lambda)
  pred <- predict(model.ridge,X.test, type="class")
  confusion <- table(y.test,pred)
  err_regression[i,c("Ridge Regression")] <-1-sum(diag(confusion))/nrow(X.test)
  err[i,c("Ridge")] <-1-sum(diag(confusion))/nrow(X.test)
}

# Lasso Regression 
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne (inner)
  x <- as.matrix(data.inner[, -51])
  y <- as.factor(data.inner[, 51])
  fit_lasso = glmnet(x, y, alpha = 1, family = "multinomial")
  #plot(fit_lasso)
  #plot(fit_lasso, xvar = "lambda", label = TRUE)
  # Inner Cross-Validation with 10 folds by default using cv.glmnet 
  fit_lasso_cv = cv.glmnet(x, y, alpha = 1,family = "multinomial" )
  #plot(fit_lasso_cv)
  best.lambda = fit_lasso_cv$lambda.min
  
  # validation model 
  X.train <- as.matrix(data.inner[, -51])
  y.train <- data.inner[, 51]
  X.test <- as.matrix(data.validation[, -51])
  y.test <- data.validation[, 51]
  # Fit the final model on the training data
  model.lasso <- glmnet(X.train, y.train, alpha = 1 , family = "multinomial", lambda = best.lambda)
  pred <- predict(model.lasso,X.test, type="class")
  confusion <- table(y.test,pred)
  err_regression[i,c("Lasso Regression")] <-1-sum(diag(confusion))/nrow(X.test)
  err[i,c("Lasso")] <-1-sum(diag(confusion))/nrow(X.test)
}

# Elastic Net Regression 
for(i in 1:10){
  index.outer.cv <- idx.test.outer[[i]]
  data.inner <- data[-index.outer.cv,]
  data.validation <- data[index.outer.cv,]
  # re-sampling (inner cross validation)
  data.inner <- data.inner[rs.data.inner[[i]], ]
  
  # cross-validation interne (inner)
  x <- as.matrix(data.inner[, -51])
  y <- as.factor(data.inner[, 51])
  cv_5 = trainControl(method = "cv", number = 5)
  fit_elnet_cv = train(Y ~ ., data = data.inner, method = "glmnet", trControl = cv_5 )
  best.alpha <- fit_elnet_cv$bestTune[1,1]
  best.lambda <- fit_elnet_cv$bestTune[1,2]
  
  # validation model 
  X.train <- as.matrix(data.inner[, -51])
  y.train <- data.inner[, 51]
  X.test <- as.matrix(data.validation[, -51])
  y.test <- data.validation[, 51]
  # Fit the final model on the training data
  model.elnet <- glmnet(X.train, y.train, family = "multinomial", alpha = best.alpha , lambda = best.lambda)
  pred <- predict(model.elnet,X.test, type="class")
  confusion <- table(y.test,pred)
  err_regression[i,c("ElasticNet Reg.")] <-1-sum(diag(confusion))/nrow(X.test)
  err[i,c("ElNet")] <-1-sum(diag(confusion))/nrow(X.test)
}


plot.cv.error(err_regression,c("Ridge Regression", "Lasso Regression", "ElasticNet Reg.") )
boxplot(err_regression)

plot.cv.error(err, c("Régression Logistique", "Régression Ridge", "Régression Lasso", "Régression Elastic Net" , "Régression logistique multinomiale", "Régression logistique normale","K plus proches voisins", "Näive Bayes", "Analyse discriminante quadratique","Analyse discriminante linéaire", "Analyse discriminante mixte", "Analyse discriminante flexible", "Analyse discriminante régularisée"))
boxplot(err,main="Répartition des probabilités d'erreurs")

best.modelClass<- model.rda
save(best.modelClass,file="env.RData")

