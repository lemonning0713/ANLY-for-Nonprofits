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

x1 = df[,c("Phase", "Height", "Behavior",
          "Mass", "Body.Length", "OPE", "Rain", "S_fuel", "S_hour", "S_hr.pp",
          "S_peeps", "flights", "Temp", "Luna", "Elev", "Species",
          "Northings", "Eastings", "Distance", "Location.Type", "Genus")]


# x1 = df[,c("Phase", "Height", "Behavior", 
#            "Mass", "Body.Length", "Rain", "Temp", "Luna", "Elev", "Species", 
#            "Distance", "Location.Type", "Genus")]

x1[,"Behavior"] = as.numeric(factor(x1[,"Behavior"]))
x1[,"Genus"] = as.numeric(factor(x1[,"Genus"]))
x1[,"Location.Type"] = as.numeric(factor(x1[,"Location.Type"]))
x1[,"Species"] = as.numeric(factor(x1[,"Species"]))

x1 = na.omit(x1)
#m_x1 = scale(x1[,2:12])
m_x1 = as.matrix(x1)
m_x1 = as.numeric(m_x1[,2:21])
m_x1 = na.omit(m_x1)

#21
nmds = metaMDS(m_x1, distance = "bray")
nmds

data.scores = as.data.frame(scores(nmds))

#add columns to data frame 
data.scores$Phase = m_x1[,1]

head(data.scores)


library(ggplot2)

xx = ggplot(data.scores, aes(x = NMDS1, y = NMDS2)) + 
  geom_point(size = 9, aes( colour = Phase))+ 
  theme(axis.text.y = element_text(colour = "black", size = 12, face = "bold"), 
        axis.text.x = element_text(colour = "black", face = "bold", size = 12), 
        legend.text = element_text(size = 12, face ="bold", colour ="black"), 
        legend.position = "right", axis.title.y = element_text(face = "bold", size = 14), 
        axis.title.x = element_text(face = "bold", size = 14, colour = "black"), 
        legend.title = element_text(size = 14, colour = "black", face = "bold"), 
        panel.background = element_blank(), panel.border = element_rect(colour = "black", fill = NA, size = 1.2),
        legend.key=element_blank()) + 
  labs(x = "NMDS1", colour = "Time", y = "NMDS2", shape = "Type")  + 
  scale_colour_manual(values = c("#009E73", "#E69F00", "red", "black", "blue")) 

xx

