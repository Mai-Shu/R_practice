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



##Understanding the tree plot
##Classify with the decision tree
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
