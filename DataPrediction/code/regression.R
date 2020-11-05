library(FNN)
library(Matrix)
library(glmnet)
library(lmtest)
library(leaps)
library(olsrr)
library(performance)


# random selection
set.seed(17)

############################################################
# Load Data for regression 
############################################################
data <- read.table('../data/TPN1_a20_reg_app.txt',header = TRUE)
summary(data)

############################################################
# Analyse Exploratoire 
############################################################

# Tester la signification de régression linéaire
model.reg = lm(y ~., data = data)
model.reg$coefficients
summary(model.reg)
# p-value: < 2.2e-16
# La regression est donc significatif.

# prdicteurs les plus significatifs:
# X2, X5, X9, X11, X16, X17, X20, X22, X24, X26, X27, X28, X29, X38, X39, X42, X44, X45, X47, X49, X50
# X51, X53, X56, X57, X58, X59, X62, X63, X64, X66, X71, X72, X73, X75, X79, X82, X83, X86, X87, X89, X91, X92, X94, X96, X97

confint(model.reg) 

# Les valeurs prédites versus les vrais valeurs.
plot(data$y, fitted(model.reg), main = "Les valeurs prédites versus les vrais valeurs")
abline(0,1)

# Les résidus standardisés
plot(data$y, rstandard(model.reg), main = "Résidus standardisés")
abline(h = 0)

# Pour tester la normalité de notre échantillon des résidus
# tracer qqnorm
qqnorm(resid(model.reg))
qqline(resid(model.reg))

# tracer histogramme
hist(resid(model.reg), freq = FALSE)
eps = seq(-40, 40, 0.01)
lines(eps, dnorm(eps, mean = 0, sd = sd(resid(model.reg))))
 
# Le test de Shapiro-Wilk
# Hypothèse nulle : l'échantillon suit une loi normale. 
# Par conséquent si la p-value du test est significative, l'échantillon ne suit pas une loi normale.

shapiro.test(resid(model.reg))
## Shapiro-Wilk normality test
## data:  resid(model.reg)
## W = 0.99562, p-value = 0.1751

# Le degré de signification vaut 0.1751 et il n'est pas faible (<0.05), il ne faut donc pas rejeter l'hypothèse nulle de normalité à tort.

# test de Durbin-Watson
dwtest(model.reg)
## DW = 1.9689, p-value = 0.3588
# H0: l'independance; H1: les residus sont non independants et suivent un prosessus autorergressif d'ordre 1
# On peut pas rejeter l'hypothese H0

plot(model.reg, which = 4, cook.levels = c(0, 0.1))
plot(model.reg, which = 5, cook.levels = c(0, 0.1))


############################################################
# Modèles 
############################################################

models = c("Régression Linéaire", "Linéaire qudratique", "Ridge Regression", "Lasso Regression")
trial = 50

models.error = as.data.frame(matrix(0, nrow=trial, ncol=length(models)))
colnames(models.error) = models

n = dim(data)[1]
p = dim(data)[2]


# 1. Régression Linéaire avec les variables significatives

for (i in 1:trial)
{
  n_folds = 5
  folds = sample(rep(1:n_folds, length.out = n))
  error = 0
  # Boucle de la cross-validation
  
  for(k in 1:n_folds) 
  {
    data.train <- data[folds != k, ]
    data.test <- data[folds == k, ]
    model.reg = lm(y~X2 + X5 + X9 + X11 + X16 + X17 + X20 + X22 + X24 + X26 + X27 + X28 + X29 + X38 + 
                     X39 + X42 + X44 + X45 + X47 + X49 + X50 + X51 + X53 + X56 + X57 + X58 + X59 + 
                     X62 + X63 + X64 + X66 + X71 + X72 + X73 + X75 + X79 + X82 + X83 + X86 + X87 + 
                     X89 + X91 + X92 + X94 + X96 + X97, data = data.train)
    pred.test = predict(model.reg, newdata = data.test)
    error = error + mean((data.test[, c('y')] - pred.test)^2)
  }
  models.error[i, "Régression Linéaire"] = error/n_folds
}
print(mean(models.error[, "Régression Linéaire"]))



# 2. Regression quadratique linear avec les variables significatives

for (i in 1:trial)
{
  n_folds = 5
  folds = sample(rep(1:n_folds, length.out = n))
  error = 0
  # Boucle de la cross-validation
  
  for(k in 1:n_folds) 
  {
    data.train <- data[folds != k, ]
    data.test <- data[folds == k, ]
    model.reg = lm(y~X5 + X9 + X16 + X17 + X22 + X24 + X26 + X27 + X28 + X29 + X38 + X39 
                   + X42 + X44 + X45 + X47 + X49 + X50 + X51 + X56 + X57 + X59 + X62 + X63
                   + poly(X64, 2) + X66 + X72 + X75 + X79 + X82 + X83 + X86 + X87 + X91
                   + X94 + X96, data = data.train)
    pred.test = predict(model.reg, newdata = data.test)
    error = error + mean((data.test[, c('y')] - pred.test)^2)
  }
  models.error[i, "Linéaire qudratique"] = error/n_folds
}
print(mean(models.error[, "Linéaire qudratique"]))


# 3. Ridge regression

for (i in 1:trial)
{
  n_folds = 5
  folds = sample(rep(1:n_folds, length.out = n))
  error = 0
  # Boucle de la cross-validation
  
  for(k in 1:n_folds) 
  {
    x = model.matrix(y~., data = data)
    y = data$y
    
    x.app = x[folds != k, ]
    y.app = y[folds != k]
    x.tst = x[folds == k, ]
    y.tst = y[folds == k]
    
    cv.out = cv.glmnet(x = x.app , y = y.app, alpha = 0)
    # plot(cv.out)
    
    fit = glmnet(x = x.app, y = y.app, lambda=cv.out$lambda.min, alpha=0)
    ridge.pred = predict(fit, s = cv.out$lambda.min, newx = x.tst)
    error = error + mean((y.tst-ridge.pred)^2)
  }
  models.error[i, "Ridge Regression"] = error/n_folds
}
print(mean(models.error[, "Ridge Regression"]))





# 4. Lasso regression

for (i in 1:trial)
{
  n_folds = 5
  folds = sample(rep(1:n_folds, length.out = n))
  error = 0
  # Boucle de la cross-validation
  
  for(k in 1:n_folds) 
  {
    x = model.matrix(y~., data = data)
    y = data$y
    
    x.app = x[folds != k, ]
    y.app = y[folds != k]
    x.tst = x[folds == k, ]
    y.tst = y[folds == k]
    
    cv.out = cv.glmnet(x = x.app , y = y.app, alpha = 1)
    # plot(cv.out)
    
    fit.lasso = glmnet(x = x.app, y = y.app, lambda=cv.out$lambda.min, alpha = 1)
    lasso.pred = predict(fit.lasso, s = cv.out$lambda.min, newx = x.tst)
    error = error + mean((y.tst-lasso.pred)^2)
  }
  models.error[i, "Lasso Regression"] = error/n_folds
}
print(mean(models.error[, "Lasso Regression"]))


rres.model = fit.lasso$residuals
rstd.model = rstandard(fit.lasso)
rstu.model = rstudent(model.fit)


fit.lasso = glmnet(x = x.app, y = y.app, lambda=cv.out$lambda.min, alpha = 1)
plot(data$y, fitted(fit.lasso))
abline(0, 1)



boxplot(models.error, main="Moyenne des erreurs pour chaque modèle")

plot.cv.error(models.error,c("Régression Linéaire", "Linéaire qudratique", "Ridge Regression", "Lasso Regression"))

for (i in 1:4)
{
  print(mean(models.error[, i]))
}





