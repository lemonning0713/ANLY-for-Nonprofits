#################################################################################################
## This file tranforms the original dataset to one that is more suitable for geo visualization ##
#################################################################################################

### Generate random location within r meters of a plot
# x0: longitude, east-west
# y0: latitude, north-south

rand_pos <- function(x0, y0, d = 5){
  
  options(digits = 10)
  #x0 = -71.00875339
  #y0 = -12.99351888
  # r = 5 # set r = 5 m
  r = d / 111300 # convert meters to degree - there are about 111,300 meters in a degree
  
  # generate u and v between 0 and 1
  u = runif(1)
  v = runif(1)
  
  w = r * sqrt(u)
  t = 2 * pi * v
  x = w * cos(t) 
  y = w * sin(t)
  
  x = x / cos(y0)
  
  y1 = y0 + y
  x1 = x0 + x
  return (list(x1, y1))
}




### function to generate new coordinate columns

gen_new_xy_col <- function(df, d = 5){
  n = nrow(df)
  df$vis.Long <- rep(NA, n)
  df$vis.Lat <- rep(NA, n)
  for(i in (1:n)){
    x0 = df$Long[i]
    y0 = df$Lat[i]
    new_xy <- rand_pos(x0, y0, d)
    df$vis.Long[i] <- new_xy[[1]]
    df$vis.Lat[i] <- new_xy[[2]]
  }
  return(df)
}


# generate new columns

vis_dat <- gen_new_xy_col(vis_dat, d = 30)

# write to data

write.csv(vis_dat, "data/cleaned data/vis_dat_v1.csv")





