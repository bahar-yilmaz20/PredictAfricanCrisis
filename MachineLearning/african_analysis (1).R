library(caret)
library(ROSE)
library(rpart)
require(tree)
library(class)
library(ggplot2)
library(naivebayes)

africa <- read.csv("C:\\Users\\Asus\\Desktop\\machine_learning\\african_crises.csv",header = TRUE)

colnames(africa)
#---DATA PREPARATION---

# 1 - Remove the duplicated rows if exists
isDuplicated <- duplicated(africa) #find duplicated index results are TRUE or FALSE
duplicated_values <- africa[isDuplicated,] #find the duplicated values which has TRUE index
# it's clearly seen that there is no duplicated rows

# 2 - Remove the missing values if exists
apply(is.na(africa),2,sum)
#there is no null values

# 3 - drop the unnecessary columns which do not effect on prediction
df_africa <- subset(africa, select = -c(country,cc3))

# 4 - convert categorical values into numeric values
#df_africa$country <- as.numeric(df_africa$country)

#5 - normalize data
summary(df_africa)
range(df_africa$inflation_annual_cpi)
preproc <- preProcess(df_africa[,c(4,8)], method = c("center", "scale")) # pre-process the data
cleaned_africa <- predict(preproc, df_africa) #performs normalization
summary(cleaned_africa)

#---UNDERSAMPLING---

#Firstly check the distribution of target feature
target_distribution <- table(cleaned_africa$banking_crisis) 
barplot(target_distribution,ylim = c(0,1000),main = "Target Feature Distribution",xlab = "Target Feature")  #Creates bar plot of distribution
prop.table(table(cleaned_africa$banking_crisis))
#It is clearly seen that the data is imbalanced.

#To prevent the possible problems occureed while the predictions, two possible methods can be used.
#africa_df <- cleaned_africa
#ind_crisis <- which(africa_df$banking_crisis == "crisis")
#ind_not_crisis <- which(africa_df$banking_crisis == "no_crisis")
#selected_ind_not_crisis <- subset(x = africa_df[ind_not_crisis,],)

balanced_africa <- ovun.sample(banking_crisis~., data=cleaned_africa, method="under",seed=1)$data
summary(balanced_africa$banking_crisis)
length(balanced_africa)

#---ML ALGORITHMS---

best_tree <- tree(banking_crisis~.,data = balanced_africa)#the best decision tree will be stored
accuracy_metric <-matrix(0,3,4)
precision_metric <-matrix(0,3,4)
recall_metric <-matrix(0,3,4)
f1score_metric <-matrix(0,3,4)
run_time_matrix <- matrix(0,3,4)

max_acc_decision_tree <- 0
max_acc_naive_bayes <- 0
best_i_decision_tree <- numeric(0)
best_i_naive_bayes <- numeric(0)
error_decision_tree <- numeric(0)
error_naive_bayes <- numeric(0)
error_knn <- numeric(0)
best_k_each_fold <- matrix(0,1,3)

for(i in 1:3) #cross validation
{
  cat(paste(i,". Fold Cross Validation :", "\n",sep=" "))
  #take subset of data %75 training purpose and %25 test purpose
  africa_train_idx <- sample(1:nrow(balanced_africa), size =nrow(balanced_africa)*0.75) #the indexex of training subset
  train_set <- balanced_africa[africa_train_idx,]#training subset
  test_set <- balanced_africa[-africa_train_idx,]#test subset
  
  #---DECISION TREE---
  cat(paste("DECISION TREE STARTS...\n"))
  start_time <- Sys.time()
  tree_africa <- tree(banking_crisis~., data = balanced_africa, subset = africa_train_idx)
  end_time <- Sys.time()
  total_time <- end_time - start_time
  cat(paste("Total Time :",total_time, "\n",sep=" "))
  summary(tree_africa)
  plot(tree_africa)
  text(tree_africa)
  
  dt_test_predict <- table(predict(tree_africa, test_set, type = "class"), test_set$banking_crisis) #confusion matrix of decision tree

  accuracy_metric[1,i] <- sum(diag(dt_test_predict)) / sum(dt_test_predict)
  precision_metric[1,i] <- dt_test_predict[1,1] / sum(dt_test_predict[,1])
  recall_metric[1,i] <- dt_test_predict[1,1] / sum(dt_test_predict[1,])
  f1score_metric[1,i] <- 2 * (precision_metric[1,i]*recall_metric[1,i]) / (precision_metric[1,i]+recall_metric[1,i])
  run_time_matrix[1,i] <- total_time
  
  if(accuracy_metric[1,i] > max_acc_decision_tree){
    max_acc_decision_tree <- accuracy_metric[1,i]
    best_tree <-tree_africa
    best_i <- i
  }
  #find the error rate 
  error_dt <- 1.0 - (dt_test_predict[1,1]+dt_test_predict[2,2])/ sum(dt_test_predict)
  error_decision_tree <- rbind(error_decision_tree,error_dt)
  decision_tree_metrics <- c( accuracy_metric[1,i], precision_metric[1,i],recall_metric[1,i],f1score_metric[1,i])
  barplot(decision_tree_metrics,xlab = "Evaluation Metrics",ylab = "Values",ylim = c(0,1),names.arg= c("Acc","Pre","Recall","fScore"),col=blues9)

  cat(paste("DECISION TREE ENDS...\n"))
  
  #---NAIVE BAYES----
  
  cat(paste("NAIVE BAYES STARTS...\n"))
  
  #the same data is used to compare the algorithms 
  #3 columns are ignored to improve the accuracy of the naive bayes algorithm according to the heat map
  train_subset_naive_bayes <- subset(balanced_africa[africa_train_idx,], select = -c(gdp_weighted_default,
                                                      currency_crises,
                                                      independence
  ))
  
  test_subset_naive_bayes <- subset(balanced_africa[-africa_train_idx,], select = -c(gdp_weighted_default,
                                                                                     currency_crises,
                                                                                     independence
  ))
  start_time <- Sys.time()
  model_naive_bayes <- naive_bayes(banking_crisis~.,  usekernel=T, data=train_subset_naive_bayes)
  end_time <- Sys.time()
  total_time <- end_time - start_time
  cat(paste("Total Time :",total_time, "\n",sep=" "))
  
  nb_test_predict <- table(predict(model_naive_bayes, test_subset_naive_bayes), test_subset_naive_bayes$banking_crisis)

  accuracy_metric[2,i] <- sum(diag(nb_test_predict)) / sum(nb_test_predict)
  precision_metric[2,i] <- nb_test_predict[1,1] / sum(nb_test_predict[,1])
  recall_metric[2,i] <- nb_test_predict[1,1] / sum(nb_test_predict[1,])
  f1score_metric[2,i] <- 2 * (precision_metric[2,i]*recall_metric[2,i]) / (precision_metric[2,i]+recall_metric[2,i])
  run_time_matrix[2,i] <- total_time
  
  if(accuracy_metric[2,i] > max_acc_naive_bayes){
    max_acc_naive_bayes <- accuracy_metric[2,i]
    best_i_naive_bayes <- i
  }
  error_nb <- 1.0 - (nb_test_predict[1,1]+nb_test_predict[2,2])/ sum(nb_test_predict)
  error_naive_bayes <- rbind(error_naive_bayes,error_nb)
  
  cat(paste("NAIVE BAYES ENDS\n"))
  
  
  #---KNN---
  cat(paste("KNN STARTS...\n"))
  
  #we need some variables to find the best k value in the for loop
  max_accuracy <- -1
  max_precision <- -1
  max_recall <- -1
  max_f1score <- -1
  max_run_time <- -1
  best_k <- 0  
  for(k_val in 1:15)
  {
    
    start_time <- Sys.time()
    model_knn <- knn(train = train_set[1:11], test = test_set[1:11], cl = train_set$banking_crisis, k=k_val)
    end_time <- Sys.time()
    total_time <- end_time - start_time

    knn_test_predict <- table(model_knn, test_set$banking_crisis)
    
    if(sum(diag(knn_test_predict)) / sum(knn_test_predict) > max_accuracy)
    {
      max_accuracy <- sum(diag(knn_test_predict)) / sum(knn_test_predict)
      max_precision <- knn_test_predict[1,1] / sum(knn_test_predict[,1])
      max_recall <- knn_test_predict[1,1] / sum(knn_test_predict[1,])
      max_f1score <- 2 * (max_precision*max_recall) / (max_precision+max_recall)
      max_run_time <- total_time
      best_k <- k_val
    }
  }
  cat(paste("Total Time :",max_run_time, "\n",sep=" "))
  accuracy_metric[3,i] <- max_accuracy
  precision_metric[3,i] <- max_precision
  recall_metric[3,i] <- max_recall
  f1score_metric[3,i] <- max_f1score
  run_time_matrix[3,i] <- max_run_time
  best_k_each_fold[1,i] <- best_k
  error <- 1.0 - max_accuracy
  error_knn <- rbind(error_knn,error)
  
  cat(paste("KNN ENDS...\n"))
}

#print the error rate and the mean of the all error
cat("DECISION TREE RESULTS \n")
cat("--------------------- \n")
cat(paste("The Mean of the Errors :", mean(error_decision_tree),"\n"))
cat("NAIVE BAYES RESULTS : \n")
cat("--------------------- \n")
cat(paste("The Mean of the Errors :", mean(error_naive_bayes),"\n"))
cat("KNN RESULTS : \n")
cat("--------------------- \n")
cat(paste("The Mean of the Errors :", mean(error_knn),"\n"))

#draw the best decision tree
summary(best_tree)
plot(best_tree)
text(best_tree)

#change columns and rownames of the matrixes
rownames(accuracy_metric) <- c("Decision Tree","Naive Bayes","KNN")
rownames(precision_metric) <- c("Decision Tree","Naive Bayes","KNN")
rownames(recall_metric) <- c("Decision Tree","Naive Bayes","KNN")
rownames(f1score_metric) <- c("Decision Tree","Naive Bayes","KNN")
rownames(run_time_matrix) <- c("Decision Tree","Naive Bayes","KNN")
colnames(accuracy_metric) <- c("Fold 1","Fold 2","Fold 3","Average")
colnames(precision_metric) <- c("Fold 1","Fold 2","Fold 3","Average")
colnames(recall_metric) <- c("Fold 1","Fold 2","Fold 3","Average")
colnames(f1score_metric) <- c("Fold 1","Fold 2","Fold 3","Average")
colnames(run_time_matrix) <- c("Fold 1","Fold 2","Fold 3","Average")
rownames(best_k_each_fold) <- c("Best K Value")
colnames(best_k_each_fold) <- c("Fold 1","Fold 2","Fold 3")

#fill the average column of the matrixes
for(i in 1:3)
{
  accuracy_metric[i,4] <- mean(accuracy_metric[i,1:3])
  precision_metric[i,4] <- mean(precision_metric[i,1:3])
  recall_metric[i,4] <- mean(recall_metric[i,1:3])
  f1score_metric[i,4] <- mean(f1score_metric[i,1:3])
  run_time_matrix[i,4] <- mean(run_time_matrix[i,1:3])
}

#print the all matrixes
cat("Accuracy Matrix\n")
print(accuracy_metric)
cat("Precision Matrix\n")
print(precision_metric)
cat("Recall Matrix\n")
print(recall_metric)
cat("F1 Score Matrix\n")
print(f1score_metric)
cat("Run-time of the Algorithms in each fold\n")
print(run_time_matrix)
cat("Best K values for each fold\n")
print(best_k_each_fold)


#the root of the decision tree can be also checked with the graps
ggplot(balanced_africa, aes(systemic_crisis, colour = banking_crisis)) +
  geom_freqpoly(binwidth = 0.25) + labs(title="Banking Crisis Distribution by Systemic Crisis")

ggplot(balanced_africa, aes(sovereign_external_debt_default, colour = banking_crisis)) +
  geom_freqpoly(binwidth = 0.25) + labs(title="Systemic-Crises Distribution by BankingCrisis")





