training <- read.csv('yards.gained.train.csv', header = TRUE)
testing <- read.csv('yards.gained.test.csv', header = TRUE)
training.one.hot <- training
test.one.hot <- testing

#Quarters are more factor/categorical than numeric, need to one hot
quarts <- unique(factor(training$qtr))
quarts <- sort(quarts)

x = 1
qtr <- ifelse(training.one.hot$qtr == quarts[x], 1, 0)
tqtr <- ifelse(test.one.hot$qtr == quarts[x], 1, 0)
quarter <- qtr
tquarter <- tqtr

for (x in 2:length(quarts)) {
    qtr <- ifelse(training.one.hot$qtr == quarts[x], 1, 0)
    tqtr <- ifelse(test.one.hot$qtr == quarts[x], 1, 0)
    quarter <- cbind(quarter, qtr)
    tquarter <- cbind(tquarter, tqtr)
}
colnames(quarter) <- paste('Quarter', quarts, sep = '_')
colnames(tquarter) <- paste('Quarter', quarts, sep = '_')

#downs are more factor/categorical than numeric, need to one hot
downs <- unique(factor(training$down))
downs <- sort(downs)

x = 1
dn <- ifelse(training.one.hot$down == downs[x], 1, 0)
tdn <- ifelse(test.one.hot$down == downs[x], 1, 0)
down <- dn
tdown <- tdn

for (x in 2:length(downs)) {
    dn <- ifelse(training.one.hot$down == downs[x], 1, 0)
    tdn <- ifelse(test.one.hot$down == downs[x], 1, 0)
    down <- cbind(down, dn)
    tdown <- cbind(tdown, tdn)
}
colnames(down) <- paste('Down', downs, sep = '_')
colnames(tdown) <- paste('Down', downs, sep = '_')


#side of field
teams <- unique(training$SideofField)
teams <- sort(teams)

#sideoffield <- data.frame()
x = 1
side <- ifelse(training.one.hot$SideofField == teams[x], 1, 0)
tside <- ifelse(test.one.hot$SideofField == teams[x], 1, 0)
sideoffield <- side
tsideoffield <- tside

for (x in 2:length(teams)) {
    side <- ifelse(training.one.hot$SideofField == teams[x], 1, 0)
    tside <- ifelse(test.one.hot$SideofField == teams[x], 1, 0)
    sideoffield <- cbind(sideoffield, side)
    tsideoffield <- cbind(tsideoffield, tside)
}
colnames(sideoffield) <- paste('SoF', teams, sep = '_')
colnames(tsideoffield) <- paste('SoF', teams, sep = '_')

#posession team
teams <- unique(training$posteam)
teams <- sort(teams)

#posteam <- data.frame()
x = 1
pos <- ifelse(training.one.hot$posteam == teams[x], 1, 0)
tpos <- ifelse(test.one.hot$posteam == teams[x], 1, 0)
posteam <- pos
tposteam <- tpos

for (x in 2:length(teams)) {
    pos <- ifelse(training.one.hot$posteam == teams[x], 1, 0)
    tpos <- ifelse(test.one.hot$posteam == teams[x], 1, 0)
    posteam <- cbind(posteam, pos)
    tposteam <- cbind(tposteam, tpos)
}
colnames(posteam) <- paste('Pos', teams, sep = '_')
colnames(tposteam) <- paste('Pos', teams, sep = '_')

#defensive team
teams <- unique(training$DefensiveTeam)
teams <- sort(teams)

#defteam <- data.frame()
x = 1
def <- ifelse(training.one.hot$DefensiveTeam == teams[x], 1, 0)
tdef <- ifelse(test.one.hot$DefensiveTeam == teams[x], 1, 0)
defteam <- def
tdefteam <- tdef

for (x in 2:length(teams)) {
    def <- ifelse(training.one.hot$DefensiveTeam == teams[x], 1, 0)
    tdef <- ifelse(test.one.hot$DefensiveTeam == teams[x], 1, 0)
    defteam <- cbind(defteam, def)
    tdefteam <- cbind(tdefteam, tdef)
}
colnames(defteam) <- paste('Def', teams, sep = '_')
colnames(tdefteam) <- paste('Def', teams, sep = '_')

#Playtype 
ptypes <- unique(training$PlayType2)
ptypes <- sort(ptypes)

#playtype <- data.frame()
x = 1
play <- ifelse(training.one.hot$PlayType2 == ptypes[x], 1, 0)
tplay <- ifelse(test.one.hot$PlayType2 == ptypes[x], 1, 0)
playtype <- play
tplaytype <- tplay

for (x in 2:length(ptypes)) {
    play <- ifelse(training.one.hot$PlayType2 == ptypes[x], 1, 0)
    tplay <- ifelse(test.one.hot$PlayType2 == ptypes[x], 1, 0)
    playtype <- cbind(playtype, play)
    tplaytype <- cbind(tplaytype, tplay)
}
colnames(playtype) <- ptypes
colnames(tplaytype) <- ptypes

training.one.hot.out <- cbind(training.one.hot[, c(-1, -2, -6, -10, -11, -14)], quarter, down, sideoffield, posteam, defteam, playtype, Yards_Gained = training$Yards.Gained)
test.one.hot.out <- cbind(test.one.hot[, c(-1, -2, -6, -10, -11, -14)], tquarter, tdown, tsideoffield, tposteam, tdefteam, tplaytype, Yards_Gained = testing$Yards.Gained)

#normalize the remaining numeric features, using max normalization
training.one.hot.out$time <- training.one.hot.out$time / max(training.one.hot.out$time)
training.one.hot.out$TimeUnder <- training.one.hot.out$TimeUnder / max(training.one.hot.out$TimeUnder)
training.one.hot.out$TimeSecs <- training.one.hot.out$TimeSecs / max(training.one.hot.out$TimeSecs)
training.one.hot.out$yrdln <- training.one.hot.out$yrdln / max(training.one.hot.out$yrdln)
training.one.hot.out$ydstogo <- training.one.hot.out$ydstogo / max(training.one.hot.out$ydstogo)
training.one.hot.out$DefTeamScore <- training.one.hot.out$DefTeamScore / max(training.one.hot.out$DefTeamScore)
training.one.hot.out$ScoreDiff <- training.one.hot.out$ScoreDiff / max(training.one.hot.out$ScoreDiff)
#training.one.hot.out$AbsScoreDiff <- training.one.hot.out$AbsScoreDiff / max(training.one.hot.out$AbsScoreDiff)

test.one.hot.out$time <- test.one.hot.out$time / max(test.one.hot.out$time)
test.one.hot.out$TimeUnder <- test.one.hot.out$TimeUnder / max(test.one.hot.out$TimeUnder)
test.one.hot.out$TimeSecs <- test.one.hot.out$TimeSecs / max(test.one.hot.out$TimeSecs)
test.one.hot.out$yrdln <- test.one.hot.out$yrdln / max(test.one.hot.out$yrdln)
test.one.hot.out$ydstogo <- test.one.hot.out$ydstogo / max(test.one.hot.out$ydstogo)
test.one.hot.out$DefTeamScore <- test.one.hot.out$DefTeamScore / max(test.one.hot.out$DefTeamScore)
test.one.hot.out$ScoreDiff <- test.one.hot.out$ScoreDiff / max(test.one.hot.out$ScoreDiff)
#test.one.hot.out$AbsScoreDiff <- test.one.hot.out$AbsScoreDiff / max(test.one.hot.out$AbsScoreDiff)

write.csv(training.one.hot.out, "yards.gained.train.one.hot.csv", row.names = FALSE)
write.csv(test.one.hot.out, "yards.gained.test.one.hot.csv", row.names = FALSE)