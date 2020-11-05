library(glmnet)

data = read.table(file = 'F:/OneDrive/UV/GI04/SY19/Projet 1/TPN1_a20_reg_app.txt', header = TRUE)


x = model.matrix(y~., data = data)
y = data$y


cv.out = cv.glmnet(x = x, y = y, alpha = 1)
plot(cv.out)

fit.lasso = glmnet(x = x, y = y, lambda=cv.out$lambda.min, alpha = 1)
# lasso.pred = predict(fit.lasso, s = cv.out$lambda.min, newx = x.tst)
# error = error + mean((y.tst-lasso.pred)^2)
