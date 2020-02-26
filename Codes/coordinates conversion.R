library(rgdal)
library(tidyverse)
library(leaflet)

dat <- read.csv("data/cleaned data/CleanDatav2.csv")

head(dat)

# switch lat long:
# any number that is 7 digits long and begins with “8” should be the northing
# the 6 digit numbers beginning in “2” are eastings
# Zone - 19L


mapdata <- dat[c("Lat", "Long")]
names(mapdata) <- c("Long", "Lat")

#Create a UTM matrix
utms <- SpatialPoints(mapdata, proj4string=CRS("+proj=utm +zone=19L +south +ellps=WGS84 +datum=WGS84"))

#Convert to long.lat
longlats <- spTransform(utms, CRS("+proj=longlat +datum=WGS84"))


# Validate the coordinates on a map
vis <- longlats %>%leaflet() %>%addTiles() %>%addCircles() 

vis


# Update the orginial dataset
dat = dat[-1]
names(dat)[c(40,41)] <- c("Eastings", "Northings")
dat$Long <- longlats$Long
dat$Lat <- longlats$Lat
head(dat)

write.csv(dat, 'data/cleaned data/CleanDatav3.csv')
