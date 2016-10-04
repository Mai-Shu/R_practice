setwd("~/Kaggle/Camp")
emails <- read.csv("emails_small.csv")
#[Classification: Filtering spam]
Filtering spam from relevant emails is a typical machine learning task. Information such as word frequency, character frequency and the amount of capital letters can indicate whether an email is spam or not.
In the following exercise you'll work with the dataset emails, which is loaded in your workspace (Source: UCI Machine Learning Repository). Here, several emails have been labeled by humans as spam (1) or not spam (0) and the results are found in the column spam. The considered feature in emails to predict whether it was spam or not is avg_capital_seq. It is the average amount of sequential capital letters found in each email.
In the code, you'll find a crude spam filter we built for you, spam_classifier() that uses avg_capital_seq to predict whether an email is spam or not. In the function definition, it's important to realize that x refers to avg_capital_seq. So where the avg_capital_seq is greater than 4, spam_classifier() predicts the email is spam (1), if avg_capital_seq is inclusively between 3 and 4, it predicts not spam (0), and so on. This classifier's methodology of predicting whether an email is spam or not seems pretty random, but let's see how it does anyways!
Your job is to inspect the emails dataset, apply spam_classifier to it, and compare the predicted labels with the true labels. If you want to play some more with the emails dataset, you can download it here. And if you want to learn more about writing functions, consider taking the Writing Functions in R course taught by Hadley and Charlotte Wickham.
##Instructions
Check the dimensions of this dataset. Use dim().
Inspect the definition of spam_classifier(). It's a simple set of statements that decide between spam and no spam based on a single input vector.
Pass the avg_capital_seq column of emails to spam_classifier() to determine which emails are spam and which aren't. Assign the resulting outcomes to spam_pred.
Compare the vector with your predictions, spam_pred, to the true spam labels in emails$spam with the == operator. Simply print out the result. This can be done in one line of code! How many of the emails were correctly classified?

```
###Show the dimensions of emails
dim(emails)

#Inspect definition of spam_classifier()
spam_classifier <- function(x){
	prediction <- rep(NA, length(x)) #initialize prediction vector
	prediction[x > 4] <- 1
	prediction[x >= 3 & x <= 4] <- 0
	prediction[x >= 2.2 & x < 3] <- 1
	prediction[x >= 1.4 & x < 2.2] <- 0
	prediction[x > 1.25 & x < 1.4] <- 1
	prediction[x <= 1.25] <- 0
	return(prediction) # prediction is either 0 or 1
}

#Apply the classifier to the avg_capital_seq column: spam_pred
spam_pred = spam_classifier(emails$avg_capital_seq)

#Compare spam_pred to emails$spam. Use ==
spam_pred == emails$spam
```

#[Regression: LinkedIn views for the next 3 days]
It's time for you to make another prediction with regression! More precisely, you'll analyze the number of views of your LinkedIn profile. With your growing network and your data science skills improving daily, you wonder if you can predict how often your profile will be visited in the future based on the number of days it's been since you created your LinkedIn account.
The instructions will help you predict the number of profile views for the next 3 days, based on the views for the past 3 weeks. The linkedin vector, which contains this information, is already available in your workspace.
##Instructions 
Create a vector days with the numbers from 1 to 21, which represent the previous 3 weeks of your linkedin views.
You can use the seq() function, or simply :.
days = seq(1:21)
Fit a linear model that explains the LinkedIn views. Use the lm() function such that linkedin ( number of views) is a function of days (number of days since you made your account). As an example, lm(y ~ x) builds a linear model such that y is a function of x, or more colloquially, y is based on x. Assign the resulting linear model to linkedin_lm.
```
linkedin_lm = lm(linkedin ~ days)
Using this linear model, predict the number of views for the next three days (days 22, 23 and 24). Use predict() and the predefined future_days data frame. Assign the result to linkedin_pred.
future_days <- data.frame(days = 22:24)
linkedin_pred <- predict(linkedin_lm, future_days)

#See how the remaining code plots both the historical data and the predictions. Try to interpret the result.
#Plot historical data and predictions
plot(linkedin ~ days, xlim = c(1, 24))
points(22:24, linkedin_pred, col = "green")
```

#[Clustering: separating the iris species]
Last but not least, there's clustering. This technique tries to group your objects. It does this without any prior knowledge of what these groups could or should look like. For clustering, the concepts of prior knowledge and unseen observations are less meaningful than for classification and regression.
In this exercise, you'll group irises in 3 distinct clusters, based on several flower characteristics in the iris dataset. It has already been chopped up in a data frame my_iris and a vector species, as shown in the sample code on the right.
The clustering itself will be done with the kmeans() function. How the algorithm actually works, will be explained in the last chapter. For now, just try it out to gain some intuition!
Note: In problems that have a random aspect (like this problem with kmeans()), the set.seed() function will be used to enforce reproducibility. If you fix the seed, the random numbers that are generated (e.g. in kmeans()) are always the same.

##Instructions
Use the kmeans() function. The first argument is my_iris; the second argument is 3, as you want to find three clusters in my_iris. Assign the result to a new variable, kmeans_iris.
#The actual species of the observations is stored in species. Use table() to compare it to the groups that the clustering came up with. These groups can be found in the cluster attribute of kmeans_iris.
Inspect the code that generates a plot of Petal.Length against Petal.Width and colors by cluster.
```
# Set random seed. Don't remove this line.
set.seed(1)

# Chop up iris in my_iris and species
my_iris <- iris[-5]
species <- iris$Species

# Perform k-means clustering on my_iris: kmeans_iris
kmeans_iris = kmeans(my_iris, 3)

# Compare the actual Species to the clustering using table()
table(kmeans_iris$cluster, species)

# Plot Petal.Width against Petal.Length, coloring by cluster
plot(Petal.Length ~ Petal.Width, data = my_iris, col = kmeans_iris$cluster)
```

#[Supervised vs. Unsupervised]
Machine Learing Tasks
Classification/Regression/Clustering
Classification, Regression --> quite similar

##Supervised Learning
Find: function f which can be used to assign a class or value to unseen observations.
Given: a set of labeled observations

##Unsupervised Learning
Labeling can be tedious, often done by humans
Some techniques don't require labeled data
Unsupervised learning
Clusteing: find groups observation that are similar
Does not require labeled observations

##Performance of the model
[Supervised learning]
Compare real labels with predicted labels
Predictions should be similar to real labels
[Unsupervised learning]
No real labels to compare

##[Semi-Supervised Learning]
A lot of unlabeled observations
A few labeled
Group similar observations using clustering
Use clustering information and classes of labeled observations to assign a class to unlabeled observations
More labeled observations for supervised learning

#Getting practical with supervised learning
Previously, you used kmeans() to perform clustering on the iris dataset. Remember that you created your own copy of the dataset, and dropped the Species attribute? That's right, you removed the labels of the observations.
In this exercise, you will use the same dataset. But instead of dropping the Species labels, you will use them do some supervised learning using recursive partitioning! Don't worry if you don't know what that is yet. Recursive partitioning (a.k.a. decision trees) will be explained in Chapter 3.
##Instructions
Take a look at the iris dataset, using str() and summary().
The code that builds a supervised learning model with the rpart() function from the rpart package is already provided for you. This model trains a decision tree on the iris dataset.
Use the predict() function with the tree model as the first argument. The second argument should be a data frame containing observations of which you want to predict the label. In this case, you can use the predefined unseen data frame. The third argument should be type = "class". Simply print out the result of this prediction step.
```
# Set random seed. Don't remove this line.
set.seed(1)

# Take a look at the iris dataset
str(iris)
summary(iris)


# A decision tree model has been built for you
tree <- rpart(Species ~ Sepal.Length + Sepal.Width + Petal.Length + Petal.Width,
              data = iris, method = "class")

# A dataframe containing unseen observations
unseen <- data.frame(Sepal.Length = c(5.3, 7.2),
                     Sepal.Width = c(2.9, 3.9),
                     Petal.Length = c(1.7, 5.4),
                     Petal.Width = c(0.8, 2.3))

# Predict the label of the unseen observations. Print out the result.

predict(tree, unseen, type = "class")
```

#[How to do unsupervised learning(1)]
In this exercise, you will group cars based on their horsepower and their weight. You can find the types of car and corresponding attributes in the cars data frame, which has been derived from the mtcars dataset. It's available in your To cluster the different observations, you will once again use kmeans().
In short, your job is to cluster the cars in 2 groups, but don't forget to explore the dataset first!

##Instructions
Explore the dataset using str() and summary().
Use kmeans() with two arguments to group the cars into two clusters based on the contents of the cars data frame. Assign the result to km_cars.
Print out the cluster element of km_cars; it shows which cars belong to which clusters.
```
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line.
set.seed(1)

# Explore the cars dataset
str(cars)
summary(cars)

# Group the dataset into two clusters: km_cars
km_cars = kmeans(cars, 2)


# Print out the contents of each cluster
print(km_cars$cluster)
```

#[How to do unsupervised learning(2)]
In the previous exercise, you grouped the cars based on their horsepower and their weight. Now let's have a look at the outcome!
An important part in machine learning is understanding your results. In the case of clustering, visualization is key to interpretation! One way to achieve this is by plotting the features of the cars and coloring the points based on their corresponding cluster.
In this exercise you'll summarize your results in a comprehensive figure. The dataset cars is already available in your workspace; the code to perform the clustering is already available.

##Instructions
Finish the plot() command by coloring the cars based on their cluster. Do this by setting the col argument to the cluster partitioning vector: km_cars$cluster.
Print out the clusters' centroids, which are kind of like the centers of each cluster. They can be found in the centers element of km_cars.
Replace the ___ in points() with the clusters' centroids. This will add the centroids to your earlier plot. To learn about the other parameters that have been defined for you, have a look at the graphical parameters documentation.
https://www.rdocumentation.org/packages/graphics/versions/3.3.1/topics/par?

```
# The cars data frame is pre-loaded

# Set random seed. Don't remove this line
set.seed(1)

# Group the dataset into two clusters: km_cars
km_cars <- kmeans(cars, 2)

# Add code: color the points in the plot based on the clusters
plot(cars, col = km_cars$cluster)

# Print out the cluster centroids
print(km_cars$centers)

# Replace the ___ part: add the centroids to the plot
points(km_cars$centers, pch = 22, bg = c(1, 2), cex = 2)
```

#[Tell the difference]
Wow, you've come a long way in this chapter. You've now acquainted yourself with 3 machine learning techniques. Let's see if you understand the difference between these techniques. Which ones are supervised, and which ones aren't?
From the following list, select the supervised learning problems:
(1) Identify a face on a list of Facebook photos. You can train your system on tagged Facebook pictures. (2) Given some features, predict whether a fruit has gone bad or not. Several supermarkets provided you with their previous observations and results. (3) Group DataCamp students into three groups. Students within the same group should be similar, while those in different groups must be dissimilar.

Possible Answers : only (1) and (2) are supervised.


