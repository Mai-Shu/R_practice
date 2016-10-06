#3. Classification
You'll gradually take your first steps to correctly perform classification, one of the most important tasks in machine learning today. By the end of this chapter, you'll be able to learn and build a decision tree and to classify unseen observations with k-Nearest Neighbors.
##Decision trees
###Task of classification
####Automatically assign class to observations with features
####Observation: vector of features, with a class
####Automatically assign class to new observation with features, using previous observations
####Binary classification: two classes
####Multiclass classification: more than two classes

###Example
A dataset consisting of persons
Feature: age, weight and income
Class: 
   binary: happy or not happy
   multiclass: happy,satisfied or not
###Examples of features
Features can be numerical
--> age: 23, 25, 75
--> height: 175.3, 179.5
Features can be categorical
--> travel_class: first class, business class, coach class
--> smokes?: yes, no
###The decision tree
Suppose you're classifying patients as sick or not sick
Intuitive way of classifying: ask questions
###Define the tree
###Questions to ask
###Categorical feature
Can be a feature test on itself
travel_class: coach, business or first
###Classifying with the tree
Observation: patient of 40 years, vaccinated and didn't smoke
###Learn a tree
Using trainging set
Come up with queries(feture tests) at each node
###Learn the tree
Goal:end up with pure leafs--leafs that contain observations of one particular class
In practice:almost never the case--noise
When classifying new instances
end up in leaf
At each node
--Iterate over different feature tests
--Choose the best one
Comes down to two parts
--Make lists of feature tests
--Choose test with best split
###Construct list of tests
Categorical features
-->Parents/grandparents/...didn't use the test yet
Numerical features
-->Choose feature
-->Choose threshold
###Choose best feature test
More complex
Use splitting criteria to decide which test to use
Information gain ~ entropy
###Information gain
Information gained from split based on feature test

Test leads to nicely divided classes
--> high information gain

Test leads to scrambled classes
--> low information gain

Test with highest information gain will be chosen
###Pruning
Number of nodes influences chance on overfit
Restrict size--higher bias
-->Decrease chance on overfit
-->Pruning the tree

##Learn a decision tree
As a big fan of shipwrecks, you decide to go to your local library and look up data about Titanic passengers. You find a data set of 714 passengers, and store it in the titanic data frame (Source: Kaggle). Each passenger has a set of features - Pclass, Sex and Age - and is labeled as survived (1) or perished (0) in the Survived column.

To test your classification skills, you can build a decision tree that uses a person's age, gender, and travel class to predict whether or not they survived the Titanic. The titanic data frame has already been divided into training and test sets (named train and test).

In this exercise, you'll need train to build a decision tree. You can use the rpart() function of the rpart package for this. Behind the scenes, it performs the steps that Vincent explained in the video: coming up with possible feature tests and building a tree with the best of these tests.

Finally, a fancy plot can help you interpret the tree. You will need the rattle, rpart.plot, and RColorBrewer packages to display this.

Note: In problems that have a random aspect, the set.seed() function is used to enforce reproducibility. Don't worry about it, just don't remove it!
###Instructions
Load in all the packages that are mentioned above with library().
Use rpart() to learn a tree model and assign the result to tree. You should use three arguments:

The first one is a formula: Survived ~ .. This represents the function you're trying to learn. We're trying to predict the Survived column, given all other columns (writing . is the same as writing Pclass + Sex + Age in this formula).
The second one is the dataset on which you want to train. Remember, training is done on the train set.
Finally, you'll have to set method to "class" to tell rpart this is a classification problem.
Create a plot of the learned model using fancyRpartPlot(). This function accepts the tree model as an argument.
```
# The train and test set are loaded into your workspace.

# Set random seed. Don't remove this line
set.seed(1)

# Load the rpart, rattle, rpart.plot and RColorBrewer package
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)





# Fill in the ___, build a tree model: tree
tree <- rpart(Survived ~ ., data = train, method = "class")

# Draw the decision tree
fancyRpartPlot(tree)
```
##Understanding the tree plot
In the previous exercise you made a fancy plot of the tree that you've learned on the training set. Have another look at the close-up of a node:


Remember how Vincent told you that a tree is learned by separating the training set step-by-step? In an ideal world, the separations lead to subsets that all have the same class. In reality, however, each division will contain both positive and negative training observations. In this node, 76% of the training instances are positive and 24% are negative. The majority class thus is positive, or 1, which is signaled by the number 1 on top. The 36% bit tells you which percentage of the entire training set passes through this particular node. On each tree level, these percentages thus sum up to 100%. Finally, the Pclass = 1,2 bit specifies the feature test on which this node will be separated next. If the test comes out positive, the left branch is taken; if it's negative, the right branch is taken.

Now that you can interpret the tree, can you tell which of the following statements is correct?
A. The majority class of the root node is positive, denoting survival. 
-->Incorrect, on the contrary. In the leftmost leaf 94% of the training instances are positive, which denotes very high purity!
B. The feature test that follows when the Sex is not female, is based on a categorical variable. 
--> Incorrect. The feature test Age < 3.5 is based on a numerical variable.
C. The tree will predict female passengers in class 3 to not survive, although it's close. 

D. The leftmost leaf is very impure, as the vast majority of the training instances in this leaf are positive.
-->Incorrect, on the contrary. In the leftmost leaf 94% of the training instances are positive, which denotes very high purity!

##Classify with the decision tree
The previous learning step involved proposing different tests on which to split nodes and then to select the best tests using an appropriate splitting criterion. You were spared from all the implementation hassles that come with that: the rpart() function did all of that for you.

###Instructions
Now you are going to classify the instances that are in the test set. As before, the data frames titanic, train and test are available in your workspace. You'll only want to work with the test set, though.
Use tree to predict the labels of the test set with the predict() function; store the resulting prediction in pred.
Create a confusion matrix, conf, of your predictions on the test set. The true values, test$Survived, should be on the rows.
Use the confusion matrix to print out the accuracy. This is the ratio of all correctly classified instances divided by the total number of classified instances, remember?
```
# The train and test set are loaded into your workspace.

# Code from previous exercise
set.seed(1)
library(rpart)
tree <- rpart(Survived ~ ., train, method = "class")

# Predict the values of the test set: pred
pred = predict(tree, test, type = "class")

# Construct the confusion matrix: conf
conf = table(test$Survived, pred)

# Print out the accuracy
print(sum(diag(conf)) / sum(conf))
```
##Pruning the tree
A like-minded shipwreck fanatic is also doing some Titanic predictions. He passes you some code that builds another decision tree model. The resulting model, tree, seems to work well but it's pretty hard to interpret. Generally speaking, the harder it is to interpret the model, the more likely you're overfitting on the training data.

You decide to prune his tree. This implies that you're increasing the bias, as you're restricting the amount of detail your tree can model. Finally, you'll plot this pruned tree to compare it to the previous one.
###Instructions
A model, tree is already coded on the right. Use fancyRpartPlot() to plot it. What do you think?
Use the prune() method to shrink tree to a more compact tree, pruned. Also specify the cp argument to be 0.01. This is a complexity parameter. It basically tells the algorithm to remove node splits that do not sufficiently decrease the impurity.
Take a look at this pruned tree by drawing a fancy plot of the pruned tree. Compare the two plots.
```
# All packages are pre-loaded, as is the data

# Calculation of a complex tree
set.seed(1)
tree <- rpart(Survived ~ ., train, method = "class", control = rpart.control(cp=0.00001))

# Draw the complex tree
fancyRpartPlot(tree)

# Prune the tree: pruned
pruned = prune(tree, cp = 0.01)

# Draw pruned
fancyRpartPlot(pruned)
```
##Interpreting the tree
A 25-year old man was in first class on the Titanic. Using the tree model that's shown on the right, it's your task to predict whether he survived the Titanic accident or not. Remember: 1 corresponds to survived, 0 corresponds to deceased.
###Possible Answers
Correct A) The model predicts the man survived the accident. 
B) You're certain the man survived the accident. 
-->Try again! The model may predict the man will survive, but how certain are you about this prediction? Unless he was your grandfather, it's best you take a look at the distribution of classes in the leaf you end up in.
C) The man perished in the accident. 
--> Alas, wrong answer! You should start at the root of the tree and use the given features to navigate down to a leaf in the tree.
D) The model predicts the man perished in the accident.
--> Oops! You should start at the root of the tree and use the given features to navigate down to a leaf in the tree.
##Splitting criterion
Do you remember the spam filters we built and tested in chapter 1 and 2? Well, it's time to make the filter more serious! We added some relevant data for every email that will help filter the spam, such as word and character frequencies. All of these can be found in the emails dataset, which is loaded in your workspace. Also, a training and test set have already been built from it: train and test.

In this exercise, you'll build two decision trees based on different splitting criteria. In the video you've learned about information gain: the higher the gain when you split, the better. However, the standard splitting criterion of rpart() is the Gini impurity.

It is up to you now to compare the information gain criterion with the Gini impurity criterion: how do the accuracy and resulting trees differ?
###Instructions
Have a look at the code that computes tree_g, pred_g, conf_g and acc_g. Here, the tree was trained with the Gini impurity criterion, which rpart() uses by default.
Change the arguments of the rpart() function in the next block of code so that it will split using the information gain criterion. It is coded as "information". The code that calculates pred_i, conf_i and acc_i is already there.
Draw a fancy plot of tree_g and tree_i using fancyRpartPlot().
Print out the accuracy of both the first and second models.
```
# All packages, emails, train, and test have been pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Train and test tree with gini criterion
tree_g <- rpart(spam ~ ., train, method = "class")
pred_g <- predict(tree_g, test, type = "class")
conf_g <- table(test$spam, pred_g)
acc_g <- sum(diag(conf_g)) / sum(conf_g)

# Change the first line of code to use information gain as splitting criterion
tree_i <- rpart(spam ~ ., train, method = "class", parms = list(split = "information"))
pred_i <- predict(tree_i, test, type = "class")
conf_i <- table(test$spam, pred_i)
acc_i <- sum(diag(conf_i)) / sum(conf_i)

# Draw a fancy plot of both tree_g and tree_i
fancyRpartPlot(tree_g)
fancyRpartPlot(tree_i)


# Print out acc_g and acc_i
print(acc_g)
print(acc_i)
```
##k-Nearest Neighbors
###Instance-based learning
Save training set in memory
No real model like decision tree
Compare unseen instances to training set
Predict using the comparison of unseen data and the training set
###k-Nearest Neighbor
Form of instance-based learning
Simplest form: 1-Nearst Neighbor or Nearest Neighbor
###Nearst Neighbor-example
2 features: X1 and X2
Class:red or blue
Binary classification
Save complete training set
Given: unseen observation with features X=(1.3, -2)
Compare training set with new observation
Find closet observation--nearest neighbor--and assign same class
(just Euclidean distance, nothing fancy)
###k-Nearest Neighbors
k is the amount of neighbors
If k = 5
Use 5 most similar obserbations(neighbors)
Assigned class will be the most represented class within the 5 neighbors
###Distance metric
Important aspect of k-NN
Euclidian distance
Manhattan distance
###Scaling-example
Dataset with
2 features: weight and height
3 observations
###Scaling
Normalize all features
e.g. rescale values between 0 and 1
Gives better measure of real distance
Don't forget to scale new observations
###Categorical features
How to use in distance metric?
Dummmy variables
1 categorical features with N possible outcomes to N binary features(2 outcomes)
###Dummy variables--Example
##Preprocess the data
Let's return to the tragic titanic dataset. This time, you'll classify its observations differently, with k-Nearest Neighbors (k-NN). However, there is one problem you'll have to tackle first: scale.

As you've seen in the video, the scale of your input variables may have a great influence on the outcome of the k-NN algorithm. In your case, the Age is on an entirely different scale than Sex and Pclass, hence it's best to rescale first!

For example, to normalize a vector xx, you could do the following:

x−min(x)max(x)−min(x)
x−min(x)max(x)−min(x)
Head over to the instructions to normalize Age and Pclass for both the training and the test set.

###Instructions
Assign the class label, Survived, of both train and test to separate vectors: train_labels and test_labels.
Copy the train and test set to knn_train and knn_test. You can just use the assignment operator (<-) to do this.
Drop the Survived column from knn_train and knn_test. Tip: dropping a column named column in a data frame named df can be done as follows: df$column <- NULL.
For this instruction, you don't have to write any code. Pclass is an ordinal value between 1 and 3. Have a look at the code that normalizes this variable in both the training and the test set. To define the minimum and maximum, only the training set is used; we can't use information on the test set (like the minimums or maximums) to normalize the data.
In a similar fashion, normalize the Age column of knn_train as well as knn_test. Fill in the ___ in the code. Again, you should only use features from the train set to decide on the normalization! You should use the intermediate variables min_age and max_age.
```
# train and test are pre-loaded

# Store the Survived column of train and test in train_labels and test_labels
train_labels <- train$Survived
test_labels <-  test$Survived


# Copy train and test to knn_train and knn_test
knn_train <- train
knn_test <- test


# Drop Survived column for knn_train and knn_test
knn_train$Survived <- NULL
knn_test$Survived <- NULL


# Normalize Pclass
min_class <- min(knn_train$Pclass)
max_class <- max(knn_train$Pclass)
knn_train$Pclass <- (knn_train$Pclass - min_class) / (max_class - min_class)
knn_test$Pclass <- (knn_test$Pclass - min_class) / (max_class - min_class)

# Normalize Age
min_age <- min(knn_train$Age)
max_age <- max(knn_train$Age)
knn_train$Age <- (knn_train$Age - min_age) / (max_age - min_age)
knn_test$Age <- (knn_test$Age - min_age) / (max_age - min_age)

```


##The knn() function
Now that you have your preprocessed data - available in your workspace as knn_train, knn_test, train_labels and test_labels - you are ready to start with actually classifying some instances with k-Nearest Neighbors.

To do this, you can use the knn() function which is available from the class package.

Let's try it out and see what the result looks like.

###Instructions
Load the class package.
Use knn() to predict the values of the test set based on 5 neighbors. Fill in variables available in your workspace on the ___. The prediction result is assigned to pred. The function takes four arguments:

train: observations in the training set, without the class labels, available in knn_train
test: observations in the test, without the class labels, available in knn_test
cl: factor of true class labels of the training set, available in train_labels
k: number of nearest neighbors you want to consider, 5 in our case
With test_labels and pred, the predicted labels, use table() to build a confusion matrix: conf. Make test_labels the rows in the confusion matrix.

Print out the confusion matrix.
```
# knn_train, knn_test, train_labels and test_labels are pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Load the class package
library(class)

# Fill in the ___, make predictions using knn: pred
pred <- knn(train = knn_train, test = knn_test, cl = train_labels, k = 5)

# Construct the confusion matrix: conf
conf = table(test_labels, pred)

# Print out the confusion matrix
print(conf)
```

##K's choice
A big issue with k-Nearest Neighbors is the choice of a suitable k. How many neighbors should you use to decide on the label of a new observation? Let's have R answer this question for us and assess the performance of k-Nearest Neighbor classification for increasing values of k.

Again, knn_train, knn_test, train_labels and test_labels that you've created before are available in your workspace.
###Instructions
The range, a vector of KK values to try, and an accs vector to store the accuracies for these different values, have already been created. You don't have to write extra code for this step.
Fill in the ___ inside the for loop:

Use knn() to predict the values of the test set like you did in the previous exercise. This time set the k argument to k, the loop index of the for loop. Assign the result to pred.
With test_labels and pred, the predicted labels, use table() to build a confusion matrix.
Derive the accuracy and assign it to the correct index in accs. You can use sum() and diag() like you did before.
The code to create a plot with range on the x-axis and accs on the y-axs is there for you, notice how it changes the xlab argument.

Calculate the best k (giving the highest accuracy) with which.max() and print it out to the console. Tip: you want to find out which index is highest in the accs vector.
```
# knn_train, knn_test, train_labels and test_labels are pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Load the class package, define range and accs
library(class)
range <- 1:round(0.2 * nrow(knn_train))
accs <- rep(0, length(range))

for (k in range) {

  # Fill in the ___, make predictions using knn: pred
  pred <- knn(train = knn_train, test = knn_test, cl = train_labels, k = k)

  # Fill in the ___, construct the confusion matrix: conf
  conf <- table(test_labels, pred)

  # Fill in the ___, calculate the accuracy and store it in accs[k]
  accs[k] <- sum(diag(conf)) / sum(conf)
}

# Plot the accuracies. Title of x-axis is "k".
plot(range, accs, xlab = "k", )

# Calculate the best 
which.max(accs)
```
##Interpreting a Voronoi diagram
A cool way to visualize how 1-Nearest Neighbor works with two-dimensional features is the Voronoi Diagram. It's basically a plot of all the training instances, together with a set of tiles around the points. This tile represents the region of influence of each point. When you want to classify a new observation, it will receive the class of the tile in which the coordinates fall. Pretty cool, right?

In the plot on the right you can see training instances that belong to either the blue or the red class. Each instance has two features: xx and yy. The top left instance, for example, has an xx value of around 0.05 and a yy value of 0.9.

Suppose you are given an unseen observation with features (x,y)=(0.5,0.5)(x,y)=(0.5,0.5). Looking at the Voronoi diagram, which class would you give this observation?
Possible answers:
Correct 1. Blue
2. Red
3. It's impossible to tell!
##The ROC curve
###Introducing
Very powerful performance measure
For binary classification
Reiceiver Operator Characteristic Curve(ROC Curve)
###Probabilities as output
Used decision trees and k-NN to predict class
They can also output probability that instance belongs to class
###Probabilities as output--example
Binary classification
Decide whether patient is sick or not sick
Define probability threshold from which you decide patient to be sick
Decision tree:
New patient: 70% 30%
higher than 50% classify as sick
Avoid sending sick patient home: lower threshold to 30%
More sick patients classified as sick patients
but also
more no sick patients classified as sick patients
###The ROC curve
Confusion matrix
other performance measure for classification
important to construct the ROC curve
###Confusion matrix
Binary classifier: positive or negative(1 or 0)
True positives
Prediction: P
Truth: P
False Negatives
Prediction: N
Truth: P
False Positives
Prediction: P
Truth: N
True Negatives
Prediction: N
Truth: N
###Ratios in the confusion matrix
True positive rate(TPR) = recall
False positive rate(FPR)
TPR
TP/(TP+FN)
FPR
FP/(FP+TN)
###ROC curve
Horizontal axis: FPR
Vertical axis: TPR
How to draw the curve?

###Draw the curve
Need classifier which output probabilities
The decision function

probability --> decide to diagnose --> sick
                                   --> healthy
                                   


probability of sick ---> decide to diagnose ---> sick
                                            ---> healthy
                                            ---> threshold by decision function
>= 50%: sick
< 50%: healthy

probability(all healthy) = 1
probability(all sick) = 1

###Interpreting the curve 
Is it a good curve?
Closer to left upper corner = better
Good classifiers have big area under the curve
AUC(Area under the curve) = 0.905 
>0.9 = very good

##Creating the ROC curve(1)
In this exercise you will work with a medium sized dataset about the income of people given a set of features like education, race, sex, and so on. Each observation is labeled with 1 or 0: 1 means the observation has annual income equal or above $50,000, 0 means the observation has an annual income lower than $50,000 (Source: UCIMLR). This label information is stored in the income variable.

A tree model, tree, is learned for you on a training set and tries to predict income based on all other variables in the dataset.

In previous exercises, you used this tree to make class predictions, by setting the type argument in predict() to "class".

To build an ROC curve, however, you need the probabilities that the observations are positive. In this case, you'll want to to predict the probability of each observation in the test set (already available) having an annual income equal to or above $50,000. Now, you'll have to set the type argument of predict() to "prob".
###Instructions
Predict the probabilities of the test set observations using predict(). It takes three arguments:

The first argument should be the tree model that is built, tree
The second argument should be the test set, on which you want to predict
Finally, don't forget to set type to "prob".
Assign the result to all_probs.

Print out all_probs. Ask yourself the question; what kind of data structure is it?

Select the second column of all_probs, corresponding to the probabilities of the observations belonging to class 1. Assign to probs.
```
# train and test are pre-loaded

# Set random seed. Don't remove this line
set.seed(1)

# Build a tree on the training set: tree
tree <- rpart(income ~ ., train, method = "class")

# Predict probability values using the model: all_probs
all_probs = predict(tree, test, type = "prob")

# Print out all_probs
print(all_probs)

# Select second column of all_probs: probs
probs = all_probs[,2]

```
##Creating the ROC curve(2)
Now that you have the probabilities of every observation in the test set belonging to the positive class (annual income equal or above $50,000), you can build the ROC curve.

You'll use the ROCR package for this. First, you have to build a prediction object with prediction(). Next, you can use performance() with the appropriate arguments to build the actual ROC data and plot it.

probs, which you had to calculate in the previous exercise, is already coded for you.
###Instructions
Load the ROCR package.
Use prediction() with probs and the true labels of the test set (in the income column of test) to get a prediction object. Assign the result to pred.
Use performance() on pred to get the ROC curve. The second and third argument of this function should be "tpr" and "fpr". These stand for true positive rate and false positive rate, respectively. Assign to result to perf.
Plot perf with plot().
```
# train and test are pre-loaded

# Code of previous exercise
set.seed(1)
tree <- rpart(income ~ ., train, method = "class")
probs <- predict(tree, test, type = "prob")[,2]

# Load the ROCR library
library(ROCR)

# Make a prediction object: pred
pred = prediction(probs, test$incom)
# Make a performance object: perf
perf = performance(pred, "tpr", "fpr")

# Plot this curve
plot(perf)

```


##The area under the curve
The same package you used for constructing the ROC curve can be used to quantify the area under the curve, or AUC. The same tree model is loaded into your workspace, and the test set's probabilities have again been calculated for you.

Again using the ROCR package, you can calculate the AUC. The use of prediction() is identical to before. However, the performance() function needs some tweaking.
###Instructions
Load the ROCR package once more, just for kicks!
Use prediction() with the probabilities and true labels of the test set to get a prediction object. Assign to pred.
Use performance() with this prediction object to get the ROC curve. The second argument of this function should be "auc". This stands for area under curve. Assign to perf.
Print out the AUC. This value can be found in perf@y.values[[1]].
```
# test and train are loaded into your workspace

# Build tree and predict probability values for the test set
set.seed(1)
tree <- rpart(income ~ ., train, method = "class")
probs <- predict(tree, test, type = "prob")[,2]

# Load the ROCR library
library(ROCR)

# Make a prediction object: pred
pred = prediction(probs, test$income)

# Make a performance object: perf
perf = performance(pred, "auc")


# Print out the AUC
print(perf@y.values[[1]])

```
##Interpreting the curves
In the given plot you can see two ROC curves. The blue one belongs to a first decision tree model, DT1, and the red one belongs to another decision tree model, DT2.

These curves are formed on the test set used in the previous exercises, i.e. a test set of the income dataset.

Which of the classifiers would have the highest AUC, according to this plot?

###Possible answers
1.The first decision tree model, by far.
-->Almost, the lines are very close together… So the difference in AUC isn't big at al
2.The second decision tree model, by far. 
-->Try again! Which curve has the biggest area beneath it?
Correct3.The first decision tree model, but it's close. 
4. The second decision tree model, but it's close. 
-->Try again! Which curve has the biggest area beneath it?
##Comparing the methods
In this exercise you're going to assess two models: a decision tree model and a k-Nearest Neighbor model. You will compare the ROC curves of these models to draw your conclusions.

You finished the previous chapter by building a spam filter. This time, we have some predictions from two spam filters! These spam filters calculated the probabilities of unseen observations in the test set being spam. The real spam labels of the test set can be found in test$spam.

It is your job to use your knowledge about the ROCR package to plot two ROC curves, one for each classifier. The assigned probabilities for the observations in the test set are loaded into your workspace: probs_t for the decision tree model, probs_k for k-Nearest Neighbors.

The test set is loaded into your workspace as test. It's a subset of the emails dataset.
###Instructions
Load the ROCR package.
probs_t and probs_k are the probabilities of being spam, predicted by the two classifiers. Use prediction() to create prediction objects for probs_t and probs_k. Call them pred_t and pred_k.
Use these prediction objects, pred_t and pred_k, to create performance objects. You can use the performance() function for this. The second and third arguments should be "tpr" and "fpr" for both calls. Call them perf_t and perf_k.
A predefined functions has been defined for you: draw_roc_lines(). It takes two arguments: the first is the performance object of the tree model, perf_t, and the second is the performance object of the k-Nearest Neighbor model, perf_k.
```
# Load the ROCR library
library(ROCR)

# Make the prediction objects for both models: pred_t, pred_k
pred_t = prediction(probs_t, test$spam)
pred_k = prediction(probs_k, test$spam)

# Draw the ROC lines using draw_roc_lines()
draw_roc_lines(perf_t, perf_k)

## Make the performance objects for both models: perf_t, perf_k
perf_t = performance(pred_t, "tpr", "fpr")
perf_k = performance(pred_k, "tpr", "fpr")
# Make the performance objects for both models: perf_t, perf_k
perf_t = performance(pred_t, "tpr", "fpr")
perf_k = performance(pred_k, "tpr", "fpr")

```
