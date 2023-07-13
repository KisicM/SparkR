install.packages("tidyverse", repos = "https://cran.rstudio.com")
install.packages("rlang", repos = "https://cran.rstudio.com") 
install.packages("dplyr", repos = "https://cran.rstudio.com") 
install.packages("dbplyr", repos = "https://cran.rstudio.com") 
install.packages("sparklyr", repos = "https://cran.rstudio.com") 

library(tidyr)
library(dbplyr)
library(sparklyr)
library(dplyr)
library(ggplot2)


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

#1
formula <- satisfaction ~ Gender + Age + Type_of_Travel + Flight_Distance + Inflight_service
samples <- c(1:4)
max_iterations <- samples * 4
weighted_precision <- samples
weighted_recall <- samples
weighted_f_measure <- samples
area_under_roc <- samples
accuracy <- samples

for(i in samples){
  model <- ml_logistic_regression(trainFiltered,
                                   formula,
                                   max_iter = max_iterations[i],
                                   family = "binomial")
  evaluation <- ml_evaluate(model, dataset=testFiltered)
  weighted_precision[i] <- evaluation$weighted_precision()
  weighted_recall[i] <- evaluation$weighted_recall()
  weighted_f_measure[i] <- evaluation$weighted_f_measure()
  area_under_roc[i] <- evaluation$area_under_roc()
  accuracy[i] <- evaluation$accuracy()
}

df <- data.frame(i=max_iterations,
                 wp=weighted_precision,
                 wr=weighted_recall,
                 wf=weighted_f_measure,
                 aur=area_under_roc,
                 a=accuracy)
print(df)
p1 <- df %>%
  ggplot(aes(i, wp, color=wp)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=weighted_precision) +
  theme(text = element_text(size = 16)) +
  labs(x="Iterations", y="Precision")

p2 <- df %>%
  ggplot(aes(i, wr, color=wr)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=weighted_recall) +
  theme(text = element_text(size = 16)) +
  labs(x="Iterations", y="Recall")

p3 <- df %>%
  ggplot(aes(i, a, color=a)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=accuracy) +
  theme(text = element_text(size = 16)) +
  labs(x="Iterations", y="Accuracy")

p4 <- df %>%
  ggplot(aes(i, aur, color=aur)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=area_under_roc) +
  theme(text = element_text(size = 16)) +
  labs(x="Iterations", y="Area under ROC")

p5 <- df %>%
  ggplot(aes(i, wf, color=wf)) +
  geom_line(size=2) +
  scale_x_continuous(breaks=max_iterations) +
  scale_y_continuous(breaks=weighted_f_measure) +
  theme(text = element_text(size = 16)) +
  labs(x="Iterations", y="F measure")

p1
p2
p3
p4
p5

#2
bayes_model <- ml_naive_bayes(trainFiltered, formula)

svc_model <- ml_linear_svc(trainFiltered, formula)

dt_model <- ml_decision_tree_classifier(trainFiltered, formula)

bayes_accuracy <- ml_evaluate(bayes_model, dataset=testFiltered)$Accuracy
svc_accuracy <- ml_evaluate(svc_model, dataset=testFiltered)$Accuracy
dt_accuracy <- ml_evaluate(dt_model, dataset=testFiltered)$Accuracy

k_cross_fold_validation <- function(dataset, model, formula){
  dataset <- dataset %>%
    sdf_random_split(seed=1,
                     s1=0.25,
                     s2=0.25,
                     s3=0.25,
                     s4=0.25)
  training <- list(
    s1 = sdf_bind_rows(dataset$s2, dataset$s3, dataset$s4),
    s2 = sdf_bind_rows(dataset$s1, dataset$s3, dataset$s4),
    s3 = sdf_bind_rows(dataset$s1, dataset$s2, dataset$s4),
    s4 = sdf_bind_rows(dataset$s1, dataset$s2, dataset$s3)
  )
  
  trained = list(s1=model(training$s1, formula),
                 s2=model(training$s2, formula),
                 s3=model(training$s3, formula),
                 s4=model(training$s4, formula)
  )
  
  model.accuracy <- (ml_evaluate(trained$s1, dataset$s1)$Accuracy +
                       ml_evaluate(trained$s2, dataset$s2)$Accuracy +
                       ml_evaluate(trained$s3, dataset$s3)$Accuracy +
                       ml_evaluate(trained$s4, dataset$s4)$Accuracy
  ) / 4
}

bayes_k_cross_fold_accuracy <- k_cross_fold_validation(trainFiltered, ml_naive_bayes, formula)
svc_k_cross_fold_accuracy <- k_cross_fold_validation(trainFiltered, ml_linear_svc, formula)
dt_k_cross_fold_accuracy <- k_cross_fold_validation(trainFiltered, ml_decision_tree_classifier, formula)

df <- data.frame(bayes_accuracy=bayes_accuracy,
                 bayes_k_cross_fold_accuracy=bayes_k_cross_fold_accuracy,
                 svc_accuracy=svc_accuracy,
                 svc_k_cross_fold_accuracy=svc_k_cross_fold_accuracy,
                 dt_accuracy=dt_accuracy,
                 dt_k_cross_fold_accuracy=dt_k_cross_fold_accuracy)
print(df)

#3
df_clusterization <- trainSet %>% select(Flight_Distance, Inflight_service, Class)
print(df_clusterization)

# New Formula
cluster_formula <- Class ~ Flight_Distance + Inflight_service

# K-means 5
model_3 <- ml_kmeans(df_clusterization, cluster_formula, seed = 1, k = 3)

p1 <- model_3$centers %>%
  ggplot(aes(Flight_Distance, Inflight_service, color=Flight_Distance)) +
  geom_point(size=5) +
  theme(text = element_text(size=16)) +
  labs(x="Flight distance", y="Inflight service", title = "K=3")

model_5 <- ml_kmeans(df_clusterization, cluster_formula, seed = 1, k = 5)

p2 <- model_5$centers %>%
  ggplot(aes(Flight_Distance, Inflight_service, color=Inflight_service)) +
  geom_point(size=5) +
  theme(text = element_text(size=16)) +
  labs(x="Flight distance", y="Inflight service", title = "K=5")

model_10 <- ml_kmeans(df_clusterization, cluster_formula, seed = 1, k = 10)

p3 <- model_10$centers %>%
  ggplot(aes(Flight_Distance, Inflight_service, color=Inflight_service)) +
  geom_point(size=5) +
  theme(text = element_text(size=16)) +
  labs(x="Flight distance", y="Inflight service", title = "K=10")

p1
p2
p3

spark_disconnect(sc)
