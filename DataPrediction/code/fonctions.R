library(ggplot2)

# Erreur Test 
testError <- function(test,model,type="other"){
  pred <-predict(model,newdata=test[,-51])
  if (type == "other")
    confusion <- table(test$Y,pred)
  else 
    confusion <- table(test$Y,pred$class)
  err<-1-sum(diag(confusion))/nrow(test) # erreur : 37% biaisé 
  cat("\nErreur: ",err)
  return(err)
}

# Partition data in simply two parts 
validationSet<- function(data=data,percentage=2/3){
  N <- nrow(data)
  n.train <- as.integer(percentage*N)
  idx.train <- sample(N,n.train)
  data.train <- data[idx.train,]
  data.test <- data[-idx.train,]
  data <- list("train"=data.train,"test"=data.test)
  return(data)
}

# Partition data in simply three parts with validation error rate unbiased 
holdOutApproach <- function(data=data,learning=2/3,validation=1/6,test=1/6){
  N <- nrow(data)
  n.train      <- as.integer(learning*N)
  n.test       <- as.integer(test*N)
  n.validation <- as.integer(validation*N)
  idx.train <- sample(N,n.train)
  idx.test  <- sample(N,n.test)
  idx.validation <- sample(N,n.validation)
  data.train <- data[idx.train]
  data.test <- data[idx.test]
  data.validation <- data[idx.validation]
  data <- list("train"=data.train,"test"=data.test,"validation"=data.validation)
  return(data)
}

# Plot interval errors 
plot.cv.error <- function(data, x.title="x"){
  ic.error.bar <- function(x, lower, upper, length=0.1){ 
    arrows(x, upper, x, lower, angle=90, code=3, length=length, col='red')
  }
  stderr <- function(x) sd(x)/sqrt(length(x))
  # calculer les erreurs moyennes et l'erreur type (standard error)
  means.errs <- colMeans(data)
  std.errs <- apply(data, 2, stderr)
  # plotting  
  x.values <- 1:ncol(data)
  
 ggplot(data.frame(model=x.title, mean=as.vector(means.errs)),aes(x=model, y=mean))+ # New DF with the means
    geom_point(aes(x=model, y=mean))+
    ggtitle("Intervalle de confiance des erreurs") +
    xlab("Modèles")+
    geom_text(aes(label=mean, y=0.2),position = position_dodge(0.9), size = 3.5, angle = 90,  color = "black", hjust = 'left')+
    geom_errorbar(aes(ymin=mean - 1.6*std.errs, ymax= mean + 1.6*std.errs), width=.2,
                 position=position_dodge(.9)) +
    theme(plot.title = element_text(hjust = 0.5),axis.text.x=element_text(angle=60, hjust=1))
}
