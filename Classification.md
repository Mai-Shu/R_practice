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




##Prining the tree
##Interpreting the tree
##Splitting criterion
##k-Nearest Neighbors
##Preprocess the data
##The knn() function
##K's choice
##Interpreting a Voronoi diagram
##The ROC curve
##Creating the ROC curve(1)
##Creating the ROC curve(2)
##The area under the curve
##Interpreting the curves
##Comparing the methods
