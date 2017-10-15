# Stuff ####
library(nflscrapR)
game <- nflscrapR::game_play_by_play(2013091509)
nflscrapR::

game$PlayType.RP <- paste(game$PlayType,".",game$PassLength,".",game$PassLocation,".",game$RunLocation)
game$PlayType.RP <- gsub("NA", "", game$PlayType.RP, fixed = TRUE)
game$PlayType.RP <- gsub(" ", "", game$PlayType.RP, fixed = TRUE)
game$PlayType.RP <- gsub("...", ".", game$PlayType.RP, fixed = TRUE)
test <- subset(game, PlayType == "Run" | PlayType == "Pass" )
test$PlayType.RP <-  as.factor(test$PlayType.RP)
str(test$PlayType.RP)
play.types <- test$PlayType.RP
play.types <- levels(play.types)
test <- test[,c(3,4,5,8,12,62,20,15,22)]    

fit.yds <- lm(Yards.Gained~., data = test[,-c(8,9)]) 
summary(fit)
fit.fd <- lm(FirstDown~., data = test[,-c(7,9)]) 
summary(fit.fd)
fit.td <- lm(Touchdown~., data = test[,-c(7,8)]) 
summary(fit.td)

# ####

Drive <- 3
qtr <- 1
down <- 2
TimeSecs <- 3300
ydstogo <- 4

# More Stuff#####
PlayType.RP <- NA
Pred.Yds <- NA
prop.fd <- NA
prop.td <- NA
new.situation <- cbind(Drive,qtr,down,TimeSecs,ydstogo,PlayType.RP,Pred.Yds,prop.fd,prop.td)
new.situation <- rbind(new.situation,new.situation,new.situation,new.situation,
                       new.situation,new.situation,new.situation,new.situation,
                       new.situation,new.situation,new.situation)

for(i in 1:length(play.types)){
  new.situation <- as.data.frame(new.situation)
  new.situation$PlayType.RP[i] <- play.types[i]
  new.situation$Pred.Yds[i] <- predict(fit.yds,new.situation[i,-c(7,8,9)])
  new.situation$prop.fd[i] <- round(predict(fit.fd,new.situation[i,-c(7,8,9)]),2)
  new.situation$prop.td[i] <- round(predict(fit.td,new.situation[i,-c(7,8,9)]),2)
}


# Predicted Yards, First Down Probability and Touchdown Probability  ####
new.situation[,6:9]
