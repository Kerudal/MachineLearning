# file : classifieurs.R 
# best applied directly in code 

library(naivebayes)
library(nnet)
library(MASS)
library(klaR)

# =================================== #
#   Regression Logistique Multinomial #
# =================================== #
model.multinom <- function(data=data){
  model <- multinom(Y~.,data)
  return(model)
}
# =============================== #
#             Naif Bayes          #
# =============================== #
# Independent predictors 
# Use of Bayes rule 
model.naiveBayes <- function(data=data){
  model<-naive_bayes(Y~.,data) 
  return(model) 
}
# ============================================== #
#   Regularized Disriminant Analysis RDA         #
# ============================================== #
model.rda <- function(data=data,lambda=1,gamma=1){
  model<-rda(Y~.,data,lambda = lambda , gamma=gamma)
}
# ============================================== #
#    Factorial Discriminant Analysis  AFD        #
# ============================================== #

# ============================================== #
#    Linear Discriminant Analysis  ADC - LDA     #
# ============================================== #

