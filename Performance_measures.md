#Performace measures
You'll learn how to assess the performance of both supervised and unsupervised learning
algorithms. Next, you'll learn why and how you should split your data in a trainging set and 
a test set. Finally, the concepts of bias and variance are explained.

##Measuring model performance or error
###Is our model any good?
####Context of task
Accuracy
Computation time
Interpretability
####3 types of tasks
Classification
Regression 
Clustering

###Classification
Accuracy and error
system is right or wrong
Accuracy goes up when error goes down
Accuracy = correctly classified instances / total amount of classified instances

##The confusion Matrix
Have you ever wondered if you would have survived the Titanic disaster in 1912? Our friends from Kaggle have some historical data on this event. The titanic dataset is already available in your workspace.

In this exercise, a decision tree is learned on this dataset. The tree aims to predict whether a person would have survived the accident based on the variables Age, Sex and Pclass (travel class). The decision the tree makes can be deemed correct or incorrect if we know what the person's true outcome was. That is, if it's a supervised learning problem.

Since the true fate of the passengers, Survived, is also provided in titanic, you can compare it to the prediction made by the tree. As you've seen in the video, the results can be summarized in a confusion matrix. In R, you can use the table() function for this.

In this exercise, you will only focus on assessing the performance of the decision tree. In chapter 3, you will learn how to actually build a decision tree yourself.

Note: As in the previous chapter, there are functions that have a random aspect. The set.seed() function is used to enforce reproducibility. Don't worry about it, just don't remove it!

##Instructions
Have a look at the structure of titanic. Can you infer the number of observations and variables?
Inspect the code that build the decision tree, tree. Don't worry if you do not fully understand it yet.
Use tree to predict() who survived in the titanic dataset. Use tree as the first argument and titanic as the second argument. Make sure to set the type parameter to "class". Assign the result to pred.
Build the confusion matrix with the table() function. This function builds a contingency table. The first argument corresponds to the rows in the matrix and should be the Survived column of titanic: the true labels from the data. The second argument, corresponding to the columns, should be pred: the tree's predicted labels.

```
# The titanic dataset is already loaded into your workspace

# Set random seed. Don't remove this line
set.seed(1)

# Have a look at the structure of titanic
str(titanic)
summary(titanic)

# A decision tree classification model is built on the data
tree <- rpart(Survived ~ ., data = titanic, method = "class")

# Use the predict() method to make predictions, assign to pred
pred = predict(tree, titanic, type = "class")

# Use the table() method to make the confusion matrix
table(titanic$Survived, pred)
```

##Deriving rations from the confusion matrix
The confusion matrix from the last exercise provides you with the raw performance of the decision tree:

The survivors correctly predicted to have survived: true positives (TP)
The deceased who were wrongly predicted to have survived: false positives (FP)
The survivors who were wrongly predicted to have perished: false negatives (FN)
The deceased who were correctly predicted to have perished: true negatives (TN)
  P  N
p TP FN
n FN TP

The confusion matrix is called "conf", try this in the console for its specific values:
```
 conf

      1   0
  1 212  78
  0  53 371
```

In the video, you saw that these values can be used to estimate comprehensive ratios to asses the performance of a classification algorithm. An example is the accuracy, which in this case represents the percentage of correctly predicted fates of the passengers.

Accuracy = (TP+TN) / (TP+FN+FP+TN)

Apart from accuracy, precision and recall are also key metrics to assess the results of a classification algorithm:

Precision = TP / (TP+FN)
Recall = TP / (TP+FN)

The confusion matrix you've calculated in the previous exercise is available in your workspace as conf.

##Instructions
Assign the correct values of the confusion matrix to FP and TN. Fill in the ___.
Calculate the accuracy as acc and print it out.
Finally, also calculate the precision and the recall, as prec and rec. Print out both of them.

```
# The confusion matrix is available in your workspace as conf

# Assign TP, FN, FP and TN using conf
TP <- conf[1, 1] # this will be 212
FN <- conf[1, 2] # this will be 78
FP <- conf[2, 1] # fill in
TN <- conf[2, 2] # fill in

# Calculate and print the accuracy: acc
acc = (TP + TN) / (TP + TN + FN + FP)
print(acc)


# Calculate and print out the precision: prec
prec = TP / ( TP + FP)
print(prec)


# Calculate and print out the recall: rec
rec = TP / (TP + FN)
print(rec)

```
##The quality of a regression

Imagine this: you're working at NASA and your team measured the sound pressure produced by an airplane's wing under different settings. These settings are the frequency of the wind, the angle of the wing, and several more. The results of this experiment are listed in the air dataset (Source: UCIMLR).

Your team wants to build a model that's able to predict the sound pressure based on these settings, instead of having to do those tedious experiments every time.

A colleague has prepared a multivariable linear regression model, fit. It takes as input the predictors: wind frequency (freq), wing's angle (angle), and chord's length (ch_length). The response is the sound pressure (dec). All these variables can be found in air.

Now, your job is to assess the quality of your colleague's model by calculating the RMSE:
file:///Users/Hsia/Kaggle/Camp/Screen%20Shot%202016-10-04%20at%2016.33.34.png
For example: if truth$colwas a column with true values of a variable and pred is the prediction of that variable, the formula could be calculated in R as follows:
```
sqrt((1/nrow(truth)) * sum( (truth$col - pred) ^ 2))
```
##Instructions
Take a look at the structure of air. What does it tell you?
Inspect your colleague's code that builds a multivariable linear regression model based on air. Not familiar with multiple linear regression? No problem! It will become clear in chapter 4. For now, you'll stick to assessing the model's performance.
Use the predict() function to make predictions for the observations in the air dataset. Simply pass fit to predict(); R will know what to do. Assign the result to pred.
Calculate the RMSE using the formula above. yiyi corresponds to the actual sound pressure of observation ii, which is in air$dec. ŷ iy^i corresponds to the predicted value of observation ii, which is in pred. Assign the resulting RMSE to rmse.
Print out rmse.

```
# The air dataset is already loaded into your workspace

# Take a look at the structure of air
str(air)

# Inspect your colleague's code to build the model
fit <- lm(dec ~ freq + angle + ch_length, data = air)


# Use the model to predict for all values: pred
pred = predict(fit, air)

# Use air$dec and pred to calculate the RMSE 
rmse = sqrt((1/nrow(air)) * sum((air$dec - pred) ^ 2))

# Print out rmse
print(rmse)

```




##Adding complexity to increase quality
In the last exercise, your team's model had 3 predictors (input variables), but what if you included more predictors? You have the measurements on free-stream velocity, velocity and suction side displacement thickness, thickness available for use in the air dataset as well!

Adding the new variables will definitely increase the complexity of your model, but will it increase the performance? To find out, we'll take the RMSE from the new, more complex model and compare it to that of the original model.

A colleague took your code from the previous exercise and added code that builds a new extended model, fit2! It's your job to once again assess the performance by calculating the RMSE.

###Instructions
Use the predict() function to make predictions using fit2, for all values in the air dataset. Assign the resulting vector to pred2.
Calculate the RMSE using the formula above. Assign this value to rmse2.
Print rmse2 and compare it with the earlier rmse. What do you conclude?

```
# The air dataset is already loaded into your workspace

# Previous model
fit <- lm(dec ~ freq + angle + ch_length, data = air)
pred <- predict(fit)
rmse <- sqrt(sum( (air$dec - pred) ^ 2) / nrow(air))
rmse

# Your colleague's more complex model
fit2 <- lm(dec ~ freq + angle + ch_length + velocity + thickness, data = air)

# Use the model to predict for all values: pred2
pred2 = predict(fit2, air)

# Calculate rmse2
rmse2 = sqrt(sum((air$dec - pred2) ^ 2) / nrow(air))

# Print out rmse2
rmse2

```

##Let's do some clustering
In the dataset seeds you can find various metrics such as area, perimeter and compactness for 210 seeds. (Source: UCIMLR). However, the seeds' labels were lost. Hence, we don't know which metrics belong to which type of seed. What we do know, is that there were three types of seeds.

The code on the right groups the seeds into three clusters (km_seeds), but is it likely that these three clusters represent our seed types? Let's find out.

There are two initial steps you could take:

Visualize the distribution of cluster assignments among two variables, for example length and compactness.
Verify if the clusters are well separated and compact. To do this, you can calculate the between and within cluster sum of squares respectively.

###Instructions
Take a look at the structure of the seeds dataset.
Extend the plot() command by coloring the observations based on their cluster. Do this by setting the col argument equal to the cluster element of km_seeds.
Print out the ratio of the within sum of squares to the between cluster sum of squares, so WSS/BSSWSS/BSS. These measures can be found in the cluster object km_seeds as tot.withinss and betweenss. Is the within sum of squares substantially lower than the between sum of squares?
https://archive.ics.uci.edu/ml/datasets/seeds

```
# The seeds dataset is already loaded into your workspace

# Set random seed. Don't remove this line
set.seed(1)

# Explore the structure of the dataset
str(seeds)

# Group the seeds in three clusters
km_seeds <- kmeans(seeds, 3)

# Color the points in the plot based on the clusters
plot(length ~ compactness, data = seeds, col = km_seeds$cluster)

# Print out the ratio of the WSS to the BSS
print(km_seeds$tot.withinss / km_seeds$betweenss)
```



##What to do with all these performance measures?
By now, you should have a good understanding of the different techniques and their properties. Can you tell which one of the following statements is true?

A) Defining performance metrics for unsupervised learning (cf. clustering) is easy and straightforward. 
--> Oops! It's not so straightforward to design a performance measure for clustering that will help you choose the optimal amount of clusters. Remember the exercises and try again!
B) You can use the RMSE to determine if a classification was good. 
--> RMSE can only be calculated if your output variable has a numeric value.
CORRECT C)If you want to build a system that can automatically categorize email as spam or not, a confusion matrix can help you assess its quality. 
D) The classification model you learned on a dataset shows an error rate of  0.10.1 . You're sure this model will be useful.
-->Incorrect! Error rate can be a great way to explore the performance of your model. However, always check other ratios like precision and recall as well. Remember the example from the video, where a model classified all patients as healthy when trying to label a very rare disease. This model would have a low error rate as well, but it wouldn't be useful at all.

##Traing set and test set
###Machine learning - statistics
####Predictive power vs. descriptive power
####Supervised learning: model must predict
unseen observations
####Classical statistics:model must fit data
explain or describe data
###Predictive model
####Training
not on complete dataset
trainging set
####Test set to evaluate performance of model
####Sets are disjoint: NO OVERLAP
####Model tested on unseen observations
--> Generalization!
###Split the dataset
N instances(=observations):X
K features:F
Class labels:y
Split the dataset
Use to predict y:^y  <--> real y
###When to use trainging/test set
Supervised learning
Not for unsupervised(clustering)
-->bcoz data not labeled
###Predictive power of model
file:///Users/Hsia/Kaggle/Camp/Screen%20Shot%202016-10-04%20at%2017.11.17.png
###How to split the sets?
Which observations go where?
Training set larger test set
Typically about 3/1
Quite arbitrary
Generally: more data = better model
Test set not too small
###Distribution of the sets
####Classification
classes must have similar distributions
avoid a class not being available in a set
####Classification&regression
shuffle dataset before splitting
###Effect of sampling
Sampling can affect performance measures
Add robustness to these measures: cross validation
Idea:sample multiple times, with different separations
###Cross-validation
E.g.:4-fold cross-validation
file:///Users/Hsia/Kaggle/Camp/Screen%20Shot%202016-10-04%20at%2017.31.16.png
###n-fold cross-validation
Fold test set over dataset n times
Each test set is 1/n size of total dataset.

##Split the sets
Let's return to the titanic dataset for which we set up a decision tree. In exercises 2 and 3 you calculated a confusion matrix to assess the tree's performance. However, the tree was built using the entire set of observations. Therefore, the confusion matrix doesn't assess the predictive power of the tree. The training set and the test set were one and the same thing: this can be improved!

First, you'll want to split the dataset into train and test sets. You'll notice that the titanic dataset is sorted on titanic$Survived , so you'll need to first shuffle the dataset in order to have a fair distribution of the output variable in each set.

For example, you could use the following commands to shuffle a data frame df and divide it into training and test sets with a 60/40 split between the two.
```
n <- nrow(df)
shuffled_df <- df[sample(n), ]
train_indices <- 1:round(0.6 * n)
train <- shuffled_df[train_indices, ]
test_indices <- (round(0.6 * n) + 1):n
test <- shuffled_df[test_indices, ]
```
Watch out, this is an example of how to do a 60/40 split! In the exercise you have to do a 70/30 split. However, you can use the same commands, just change the numbers!
###Instructions
The first part of the exercise is done for you, we shuffled the observations of the titanic dataset and store the result in shuffled.
Split the dataset into a train set, and a test set. Use a 70/30 split. The train set should contain the rows in 1:round(0.7 * n) and the test set in (round(0.7 * n) + 1):n. The example in the exercise description can help you!
Print out the structure of both train and test with str(). Does your result make sense?
```
# The titanic dataset is already loaded into your workspace

# Set random seed. Don't remove this line.
set.seed(1)

# Shuffle the dataset, call the result shuffled
n <- nrow(titanic)
shuffled <- titanic[sample(n),]

# Split the data in train and test

train_indices <- 1:round(0.7 * n)
train <- shuffled[train_indices, ]
test_indices <- (round(0.7 * n) + 1):n
test <- shuffled[test_indices, ]


# Print the structure of train and test
print(str(train))
print(str(test))
```





##First you train, then you test

Time to redo the model training from before. The titanic data frame is again available in your workspace. This time, however, you'll want to build a decision tree on the training set, and next assess its predictive power on a set that has not been used for training: the test set.

On the right, the code that splits titanic up in train and test has already been included. Also, the old code that builds a decision tree on the entire set is included. Up to you to correct it and connect the dots to get a good estimate of the model's predictive ability.
###Instructions
Fill in the ___ in the decision tree model, rpart(...) so that it is learned on the training set.
Use the predict() function with the tree model as the first argument and the correct dataset as the second argument. Set type to "class". Call the predicted vector pred. Remember that you should do the predictions on the test set.
Use the table() function to calculate the confusion matrix. Assign this table to conf. Construct the table with the test set's actual values (test$Survived) as the rows and the test set's model predicted values (pred) as columns.
Finally, print out conf.

```
# The titanic dataset is already loaded into your workspace

# Set random seed. Don't remove this line.
set.seed(1)

# Shuffle the dataset; build train and test
n <- nrow(titanic)
shuffled <- titanic[sample(n),]
train <- shuffled[1:round(0.7 * n),]
test <- shuffled[(round(0.7 * n) + 1):n,]

# Fill in the model that has been learned.
tree <- rpart(Survived ~ ., train , method = "class")

# Predict the outcome on the test set with tree: pred
pred = predict(tree, test, type = "class")


# Calculate the confusion matrix: conf
conf = table(test$Survived, pred)

# Print this confusion matrix
conf

```

##Using cross validation

You already did a great job in assessing the predictive performance, but let's take it a step further: cross validation.

In this exercise, you will fold the dataset 6 times and calculate the accuracy for each fold. The mean of these accuracies forms a more robust estimation of the model's true accuracy of predicting unseen data, because it is less dependent on the choice of training and test sets.

Note: Other performance measures, such as recall or precision, could also be used here.
###Instructions
The code to split the dataset correctly 6 times and build a model each time on the training set is already written for you inside the for loop; try to understand the code. accs is intialized for you as well.
Use the model to predict the values of the test set. Use predict() with three arguments: the decision tree (tree), the test set (test) and don't forget to set type to "class". Assign the result to pred.
Make the confusion matrix using table() and assign it to conf. test$Survived should be on the rows, pred on the columns.
Fill in the ___ in the statement to define accs[i]. The result should be the accuracy: the sum of the diagonal of the confusion matrix divided by the total sum of the confusion matrix.
Finally, print the mean accuracy of the 6 iterations.
```
# The shuffled dataset is already loaded into your workspace

# Set random seed. Don't remove this line.
set.seed(1)

# Initialize the accs vector
accs <- rep(0,6)

for (i in 1:6) {
  # These indices indicate the interval of the test set
  indices <- (((i-1) * round((1/6)*nrow(shuffled))) + 1):((i*round((1/6) * nrow(shuffled))))
  
  # Exclude them from the train set
  train <- shuffled[-indices,]
  
  # Include them in the test set
  test <- shuffled[indices,]
  
  # A model is learned using each training set
  tree <- rpart(Survived ~ ., train, method = "class")
  
  # Make a prediction on the test set using tree
pred = predict(tree, test, type = "class")
  
  # Assign the confusion matrix to conf
  
  conf = table(test$Survived, pred)

  
  # Assign the accuracy of this model to the ith index in accs
  accs[i] <- sum(diag(conf))/sum(conf)
}

# Print out the mean of accs
print(mean(accs))
```

##How many folds?
Let's say you're doing cross validation on a dataset with 22680 observations. This number is stored as n in your workspace. You want the training set to contain 21420 entries, saved as tr. How many folds can you use for your cross validation? In other words, how many iterations with other test sets can you do on the dataset?

Remember you can use the console to the right as a calculator.


##Bias and Variance
What you've learned?
Accuracy and other performance measures
Training and test set
###Knitting it all together
Effect of splitting dataset(train/test) on accuracy
Over and uderfitting
###Introducing
BIAS and VARIANCE
Main goal of supervised learning: prediction
Prediction error - reducible + irreducible error
###Irreducible-reducible error
Irreducible: noise--don't minimize
Reducible:error due to unfit model -- minimize
Reducible error is split into bias and variance
###Bias
Error due to bias:wrong assumptions
Difference predictions and truth
--Using modeld trained by specific learning algorithm
###Example
Quadratice data
Assumption: data is linear--use linear regression
Error due to bias is high: more restrictions on model
###Bias
Complexity of model
More restrictions lead to high bias
###Variance
Error due to variance:error due to the sampling of the trainging set
Model with high variance fits training set closely
###Example
Quadratic data
Few restrictions: fit polynomial perfectly through training set
If you change training set, model will change completely

high variance: generalizes bas to test set
###Bias variance tradeoff
low bias--high variance
low variance--high bias

###Overfitting
Accuract will depend on dataset split(train/test)
High variance will heavily depend on split
Overfitting=model fits training set a lot better than test set
Too specific
Underfitting
Restricting your model too much
High bias
Too general
###Example-spam or not?
##Overfitting the spam!
Do you remember the crude spam filter, spam_classifier(), from chapter 1? It filters spam based on the average sequential use of capital letters (avg_capital_seq) to decide whether an email was spam (1) or not (0).

You may recall that we cheated and it perfectly filtered the spam. However, the set (emails_small) you used to test your classifier was only a small fraction of the entire dataset emails_full (Source: UCIMLR).

Your job is to verify whether the spam_classifier() that was built for you generalizes to the entire set of emails. The accuracy for the set emails_small was equal to 1. Is the accuracy for the entire set emails_full substantially lower?
###Instructions
Apply spam_classifier() on the avg_capital_seq variable in emails_full and save the results in pred_full.
Create a confusion matrix, using table(): conf_full. Put the true labels found in emails_full$spam in the rows.
Use conf_full to calculate the accuracy: acc_full. The functions diag() and sum() will help. Print out the result.
```
# The spam filter that has been 'learned' for you
spam_classifier <- function(x){
  prediction <- rep(NA, length(x)) # initialize prediction vector
  prediction[x > 4] <- 1 
  prediction[x >= 3 & x <= 4] <- 0
  prediction[x >= 2.2 & x < 3] <- 1
  prediction[x >= 1.4 & x < 2.2] <- 0
  prediction[x > 1.25 & x < 1.4] <- 1
  prediction[x <= 1.25] <- 0
  return(factor(prediction, levels = c("1", "0"))) # prediction is either 0 or 1
}

# Apply spam_classifier to emails_full: pred_full
pred_full = spam_classifier(emails_full$avg_capital_seq)

# Build confusion matrix for emails_full: conf_full
conf_full = table(emails_full$spam, pred_full)

# Calculate the accuracy with conf_full: acc_full
acc_full = sum(diag(conf_full)) / sum(conf_full)

# Print acc_full
print(acc_full)
```
##Increasing the bias
It's official now, the spam_classifier() from chapter 1 is bogus. It simply overfits on the emails_small set and, as a result, doesn't generalize to larger datasets such as emails_full.

So let's try something else. On average, emails with a high frequency of sequential capital letters are spam. What if you simply filtered spam based on one threshold for avg_capital_seq?

For example, you could filter all emails with avg_capital_seq > 4 as spam. By doing this, you increase the interpretability of the classifier and restrict its complexity. However, this increases the bias, i.e. the error due to restricting your model.

Your job is to simplify the rules of spam_classifier and calculate the accuracy for the full set emails_full. Next, compare it to that of the small set emails_small, which is coded for you. Does the model generalize now?

###Instructions
Simplify the rules of the spam_classifier. Emails with an avg_capital_seq strictly longer than 4 are spam (labeled with 1), all others are seen as no spam (0).
Inspect the code that defines conf_small and acc_small.
Set up the confusion matrix for the emails_full dataset. Put the true labels found in emails_full$spam in the rows and the predicted spam values in the columns. Assign to conf_full.
Use conf_full to calculate the accuracy. Assign this value to acc_full and print it out. Before, acc_small and acc_full were 100% and 65%, respectively; what do you conclude?
```
# The all-knowing classifier that has been learned for you
# You should change the code of the classifier, simplifying it
spam_classifier <- function(x){
  prediction <- rep(NA, length(x))
  prediction[x > 4] <- 1
  prediction[x <= 4] <- 0
 
  return(factor(prediction, levels = c("1", "0")))
}

# conf_small and acc_small have been calculated for you
conf_small <- table(emails_small$spam, spam_classifier(emails_small$avg_capital_seq))
acc_small <- sum(diag(conf_small)) / sum(conf_small)
acc_small

# Apply spam_classifier to emails_full and calculate the confusion matrix: conf_full
conf_full <- table(emails_full$spam, spam_classifier(emails_full$avg_capital_seq))

# Calculate acc_full
acc_full <- sum(diag(conf_full)) / sum(conf_full)

# Print acc_full
print(acc_full)
```
##Interpretability
Which of the following models do you think would have the highest interpretability? Remember that a high interpretability usually implies a high bias.

It can help to try to describe what the model has to do to classify instances. Usually if you are able to describe what all aspects of a model do and what they mean, the model is quite interpretable.
###Possible answers
A) We've made a regression which fits perfectly through all 15 points in the training set. We used a high-degree polynomial. 
-->Oops! That answer is not correct. Do you think you could explain why the polynomial behaves like it does? Wouldn't it be easier if you just had to explain the slope in a linear model, for example.
Correct B) A model predicts whether a person with certain attributes is male or female. It uses one threshold on the height attribute to make its prediction. 

C) A spam filter uses various rules on 10 attributes to classify whether an email is spam or not. It uses attributes such as specific word count, character count, and so on. 
-->Incorrect! Allowing a learning algorithm to use a lot of attributes will decrease the bias on the model. This will generally increase the complexity.
D) We made a prediction model for car insurance claims. It uses a few attributes and relations between these attributes to predict the amount of claims one will make.
-->Almost. Using few attributes will reduce the complexity, but adding relationships between these attributes will increase the complexity. There is a simpler model in the list, try to find it!


