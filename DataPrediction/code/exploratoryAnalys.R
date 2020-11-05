data <- read.table('../data/TPN1_a20_clas_app.txt')
names(data)[51] <- "Y"
data$Y <- as.factor(data$Y)

#library(tidyverse)
library(ggplot2)
library(reshape2)
library(corrplot)
library(FactoMineR)
library(factoextra)
library(MASS)
library(lmtest)

head(data,10)
nrow(data)
ncol(data) 
str(data)
summary(data) 

# ========================================= #
#                 CORR PLOT                 #
# ========================================= #
corrplot(cor(data[,-51]), type = "upper", order = "hclust", tl.col = "black", tl.srt = 45)

# ========================================= #
#                  BAR PLOT                 #
# ========================================= #
boxplot(data)
#heatmap(data[,-51])

boxplot(data[data$Y==1,], col="red",xlab="X axes",ylab="values") 
boxplot(data[data$Y==2,], col="blue",add=TRUE) 
boxplot(data[data$Y==3,], col="green",add=TRUE) 

# ========================================= #
#               Multiple Boxplot            #
# ========================================= #
reshapeData <- melt(data,id.var="Y") 

ggplot(data = reshapeData , aes(x=variable, y=value)) + geom_boxplot(aes(fill=Y))

p <- ggplot(data = reshapeData, aes(x=variable, y=value)) + 
             geom_boxplot(aes(fill=Y))
p + facet_wrap( ~ variable, scales="free")

# ========================================= #
#                  PCA == ACP               #
# ========================================= #
model.pca <- PCA(data[,-51], graph = FALSE)
print(model.pca)

eig.val <- get_eigenvalue(model.pca)
fviz_eig(model.pca, addlabels = TRUE, ylim = c(0, 50),main="Méthode du coude avec l'ACP")

var <- get_pca_var(model.pca)
# Coordinates
head(var$coord)
# Cos2: quality on the factore map
head(var$cos2)
# Contributions to the principal components
head(var$contrib)
# Coordinates of variables
head(var$coord, 4)

#Plot variables : 
fviz_pca_var(model.pca, col.var = "black")
# You can visualize the cos2 of variables on all the dimensions using the corrplot package:
corrplot(var$cos2, is.corr=FALSE)
# Total cos2 of variables on Dim.1 and Dim.2
fviz_cos2(model.pca, choice = "var", axes = 1:2)


# ========================================= #
#                  FDA == LDA               #
# ========================================= #
model.lda <- lda(Y~. ,data=data)
D1 <- model.lda$scaling[,1] # vecteur 1
D2 <- model.lda$scaling[,2] # vecteur 2
# Calcul de projeté des individus sur D1 et D2
xy <- as.matrix(data[,-51])%*%as.matrix(cbind(D1,D2))
plot(xy)
color <-  as.character(data[,51]) ;
color[color=="1"] <- "black";
color[color=="2"] <- "red";
color[color=="3"] <- "green"; 
plot(xy,col=color,pch=16, main="Visualisation des données avec ACF")


# ========================================= #
#                  Durbin-Watson            #
# ========================================= #
# not possible for classification ! 


