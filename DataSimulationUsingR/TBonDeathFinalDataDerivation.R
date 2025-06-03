library("ipw")
data("basdat")
data("timedat")
basdat[1:10,]
timedat[1:10,]
table(duplicated(timedat[, c("id", "fuptime")]))
timedat$cd4.sqrt <- sqrt(timedat$cd4count)
timedat <- merge(timedat, basdat[,c("id","Ttb")], by = "id", all.x = TRUE)
timedat[1:10,]
timedat$tb.lag <- ifelse(with(timedat, !is.na(Ttb) & fuptime > Ttb), 1, 0)
timedat[1:10,]
library(nlme)
cd4.lme <- lme(cd4.sqrt ~ fuptime + tb.lag, random = ~ fuptime | id, data = timedat)
summary(cd4.lme)
times <- sort(unique(c(basdat$Ttb, basdat$Tend)))
startstop <- data.frame(id = rep(basdat$id, each = length(times)),
                        fuptime = rep(times, nrow(basdat)))
startstop <- merge(startstop, basdat, by = "id", all.x = TRUE)
startstop <- startstop[with(startstop, fuptime <= Tend), ]
startstop
startstop$tstart <- tstartfun(id, fuptime, startstop)
startstop


startstop$tb <- ifelse(with(startstop, !is.na(Ttb) & fuptime >= Ttb), 1, 0)
startstop
startstop$tb.lag <- ifelse(with(startstop, !is.na(Ttb) & fuptime > Ttb), 1, 0)
startstop
startstop$event <- ifelse(with(startstop, !is.na(Tdeath) & fuptime >= Tdeath), 1, 0)
startstop
startstop$cd4.sqrt <- predict(cd4.lme, newdata = data.frame(id = startstop$id, 
                                                            fuptime = startstop$fuptime, tb.lag = startstop$tb.lag))
startstop


