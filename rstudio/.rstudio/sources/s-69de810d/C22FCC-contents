install.packages("gridExtra", repos="https://cran.rstudio.com/")
library(sparklyr)
library(dplyr)
library(ggplot2)
library(cowplot)

# Install
spark_install()

# Connect
sc <- sparklyr::spark_connect(master = "local")


datasetPath <- "/data/train.csv"
trainSet <- spark_read_csv(sc, name = "train", path = datasetPath, header = TRUE, infer_schema = TRUE)
trainSet <- na.omit(trainSet)

datasetPath <- "/data/test.csv"
testSet <- spark_read_csv(sc, name = "test", path = datasetPath, header = TRUE, infer_schema = TRUE)
testSet <- na.omit(testSet)

glimpse(trainSet)
head(trainSet)

# Select columns:
trainFiltered <- trainSet %>% select(Gender, Age, Type_of_Travel, Flight_Distance, Inflight_service, satisfaction)
head(trainFiltered)
trainFiltered <- trainFiltered %>% mutate(satisfaction = switch(satisfaction,
                                                                         "satisfied"=1,
                                                                         "neutral or dissatisfied"=0))
testFiltered <- testSet %>% select(Gender, Age, Type_of_Travel, Flight_Distance, Inflight_service, satisfaction)
head(testFiltered)
testFiltered <- testFiltered %>% mutate(satisfaction = switch(satisfaction,
                                                                "satisfied"=1,
                                                                "neutral or dissatisfied"=0))
head(testFiltered)

formula <- satisfaction ~ Gender + Age + Type_of_Travel + Flight_Distance + Inflight_service
samples <- c(1:3)
max_iterations <- samples * 6
log.reg.weighted.precision <- samples
log.reg.weighted.recall <- samples
log.reg.weighted.f.measure <- samples
log.reg.area.under.roc <- samples
log.reg.accuracy <- samples

for(i in samples){
  logreg <- ml_logistic_regression(trainFiltered,
                                   formula,
                                   max_iter = max_iterations[i],
                                   family = "binomial")
  evaluation <- ml_evaluate(logreg, dataset=testFiltered)
  log.reg.weighted.precision[i] <- evaluation$weighted_precision()
  log.reg.weighted.recall[i] <- evaluation$weighted_recall()
  log.reg.weighted.f.measure[i] <- evaluation$weighted_f_measure()
  log.reg.area.under.roc[i] <- evaluation$area_under_roc()
  log.reg.accuracy[i] <- evaluation$accuracy()
}

df <- data.frame(i=max_iterations,
                 wp=log.reg.weighted.precision,
                 wr=log.reg.weighted.recall,
                 wf=log.reg.weighted.f.measure,
                 aur=log.reg.area.under.roc,
                 a=log.reg.accuracy)
p1 <- df %>%
  ggplot(aes(i, wp, color=wp)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=log.reg.weighted.precision) +
  theme(text = element_text(size = 16)) +
  labs(x="Maksimalni broj iteracija", y="Preciznost", title = "a) Zavisnost preciznosti od maksimalnog broja iteracija")

p2 <- df %>%
  ggplot(aes(i, wr, color=wr)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=log.reg.weighted.recall) +
  theme(text = element_text(size = 16)) +
  labs(x="Maksimalni broj iteracija", y="Osetljivost", title = "b) Zavisnost osetljivosti od maksimalnog broja iteracija")

p3 <- df %>%
  ggplot(aes(i, a, color=wf)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=log.reg.accuracy) +
  theme(text = element_text(size = 16)) +
  labs(x="Maksimalni broj iteracija", y="F1", title = "c) Zavisnost F1 mere od maksimalnog broja iteracija")

p4 <- df %>%
  ggplot(aes(i, aur, color=aur)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=log.reg.area.under.roc) +
  theme(text = element_text(size = 16)) +
  labs(x="Mkaismalni broj iteracija", y="Površina ispod ROC krive", title = "d) Zavisnost površine ispod ROC krive od maksimalnog broja iteracija")
p1
p2
p3
p4

spark_disconnect(sc)
