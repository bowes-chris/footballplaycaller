############################################################################################################
############ This is a demo of the proposed application for a football coaching play-caller aid.
############ The intent of this demo is to demonstrate the architecture and main idea for this project
############ A description of the application is the file "coaching aid - pilot version.model.R".
############ Make sure the file metioned above is in the same directory as this file before running. 
############################################################################################################
# Predict on the situation below ####
Drive <- 2
qtr <- 1
down <- 2
TimeSecs <- 3300
ydstogo <- 6

source("coaching aid - pilot version.model.R")
print(results)