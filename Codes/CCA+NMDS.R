require(ggplot2)
require(GGally)
require(CCA)
require(CCP)
require(vegan)

df = read.csv("CleanDatav3.csv")
na.omit(df)

summary(df)

x = df[,c("Height", "Behavior", 
          "Mass", "Body.Length", "OPE", "Rain", "S_fuel", "S_hour", "S_hr.pp", 
          "S_peeps", "flights", "Temp", "Luna", "Elev", "Long", "Lat", 
          "Northings", "Eastings")]
y = df[,c("Genus", "Distance", "Location.Type")]

x[,"Behavior"] = as.numeric(factor(x[,"Behavior"]))
y[,"Genus"] = as.numeric(factor(y[,"Genus"]))
y[,"Location.Type"] = as.numeric(factor(y[,"Location.Type"]))

correlations = matcor(x, y)
img.matcor(correlations, type = 2)

x = scale(x)
y = scale(y)
#cc3 <- cca(x1,y1)

cc1 <- cc(x, y)
# display the canonical correlations
cc1$cor #correlations between two data matrices

barplot(cc1$cor, main = "Canonical correlations for 'cancor()'", col = "gray")

# raw canonical coefficients
cc1[3:4] # like regression coefficient 


# compute canonical loadings
cc2 <- comput(x, y, cc1) #correlations between variables and the canonical variates.

# display canonical loadings
cc2[3:6]
#corr.X.xscores Correlation bewteen X and X canonical variates
#corr.Y.xscores Correlation bewteen Y and X canonical variates
#corr.X.yscores Correlation bewteen X and Y canonical variates
#corr.Y.yscores Correlation bewteen Y and Y canonical variates

#The idea is to display maximized correlations between transformed variables 
#of the dataset x and the dataset y.
plt.cc(cc1, var.label = TRUE, ind.names = df[,1])


# tests of canonical dimensions: number of significant dimensions
rho <- cc1$cor
## Define number of observations, number of variables in first set, and number of variables in the second set.
n <- dim(x)[1]
p <- length(x)
q <- length(y)

## Calculate p-values using the F-approximations of different test statistics:
p.asym(rho, n, p, q, tstat = "Wilks")


######################### NMDS ######################### 

x1 = df[,c("Height", "Behavior", "Location..m.", "Family","Genus","Species",
           "Location.Type", "Mass", "Body.Length", "Tail.Length")]

# 
# x1 = df[df$PhaseNumber == 1 ,c("PhaseNumber", "Height", "Behavior",
#            "Mass", "Body.Length", "Rain", "Temp", "Luna", "Elev", "Species",
#            "Distance", "Location.Type", "Genus")]


x1[,"Behavior"] = as.numeric(factor(x1[,"Behavior"]))
x1[,"Genus"] = as.numeric(factor(x1[,"Genus"]))
x1[,"Location.Type"] = as.numeric(factor(x1[,"Location.Type"]))
x1[,"Species"] = as.numeric(factor(x1[,"Species"]))
x1[,"Family"] = as.numeric(factor(x1[,"Family"]))

x1[is.na(x1)] <- 0

x2 = df[, c("Phase", "Distance")]
nmds = metaMDS(x1, distance = "bray", k = 2)
nmds


#phase
co=c("red","blue", "yellow", "green", "purple")
par(mar=c(5.1, 4.1, 4.1, 8.1),xpd=TRUE)
plot(nmds$points, col=co[x2$Phase], 
     main="Vegetation community composition",  xlab = "axis 1", ylab = "axis 2")
legend("topright",inset=c(-0.3,0), legend=c("AB", "DR", "EC", "LC", "RI"),  xpd = TRUE,horiz = FALSE, 
       col=c("red","blue", "yellow", "green", "purple"), pch=1, bty = "n")
#ordispider(nmds, groups = x2$Phase,  label = TRUE)

#distance
x2$Distance = as.factor(x2$Distance)
co=c("red","blue", "yellow", "green", "purple")
par(mar=c(5.1, 4.1, 4.1, 8.1),xpd=TRUE)
plot(nmds$points, col=co[x2$Distance], 
     main="Vegetation community composition",  xlab = "axis 1", ylab = "axis 2")
legend("topright", inset = c(-0.4,0), legend=c("50", "150", "250", "500", "1000"),  xpd = TRUE,horiz = FALSE, 
       col=c("red","blue", "yellow", "green", "purple"), pch=1, bty = "n")
