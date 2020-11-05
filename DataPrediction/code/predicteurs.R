regresseur = function(dataset) {
  load("env.RData")
  pred.test = predict(model.reg, newdata = dataset)
  return(pred.test)
}


classifieur = function(dataset) {
  load("env.RData")
  library(klaR)
  library(MASS)
  prediction <- predict(best.modelClass,newdata = dataset)
  return(prediction$class)
}


