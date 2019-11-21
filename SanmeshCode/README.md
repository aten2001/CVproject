# Project title: Movie Revenue Prediction
## Team members: Sanmeshkumar Udhayakumar, Aaron Reich, Tarushree Gandhi, Aastha Agrawal, Prithvi Alva Suresh  
## 7641 Team2

---
<p align="center">
  <img src="https://storage.googleapis.com/kaggle-datasets-images/138/287/229bfb5d3dd1a49cc5ac899c45ca2213/dataset-cover.png"> 
</p>

# 1. Overview of the project and Motivation 
### Motivation: 
The goal of our project is to predict the Box office revenue of a movie based on it's characteristics. 
Our analysis will not only allow Directors/Producers to predict how much money their movie will make, but it will also allow them to justify their movie characteristics, such as movie budget and choice of actors in order to reach a certain revenue. Directors/Producers can also understands what to modify in their selection of actors or investment in the movies to maximize their profit. Such analysis will also allow other interested third parties to predict the success of a film before it is released.                  
In the process of our analysis, we also aim to find the variables most associated with film revenue, and to see how the various revenue prediction models are affected by them.

---
# 2. Dataset and visualization 

### Dataset: The Movies DataBase (TMDB) 5000 Movie Dataset (from Kaggle)
#### Features in the dataset: 10 features in total &nbsp;&nbsp;&nbsp;
<!--
1. budget    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;2. genres
3. keywords &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;4. production companies
5. release date
6. runtime
7. spoken_languages
8. status
9. cast
10. crew  
-->
<table>
  <tr>
    <td>Budget</td>
    <td>Genres</td>
    <td>Keywords</td>
    <td>Production Companies</td>
    <td>Crew</td>
  </tr>
  <tr>
    <td>Release Date</td>
    <td>Runtime</td>
    <td>Spoken Languages</td>
    <td>Status</td>
    <td>Cast</td>
  </tr>
</table>

#### Visualization: 
Preliminary visualization to see the distribution of revenue's of the Movies we are studying.
We sorted movies into appropriate bin size ($100 million) to view the frequency of movies belonging to each bin size.
<p align="center">
  <img src="PrithviCodes/RevenueVSCount.png">
</p>


---
# 3. Data pre-processing
#### Steps followed for Data cleaning & Data pre-processing:
- Removal of data points with missing revenues
- Removing zero REVENUES from the data 
- Adjusting revenue for inflation.
- Separation of Date into Year and day of the year, since we theorized that film revenue will be highly correlated with which season the movie is released in.
- Encoding categorical features: conversion of data into binary format.
  - Different classes in a column (Lists) allotted their own column, and each row will indicate if column existed or not by assigning either a 1 or a 0. 
  - If Stan Lee is in the list of producers for any movie, then now 'Stan Lee' will become a binary feature. If a movie has Stan Lee as the producer the feature will be a 1, otherwise it'll be a 0.
- Data was then divided into Test, Validation, and Training sets (60%, 20% and 20%) for further model training and testing.


---
# 4. Feature Reduction 

Our data has 10955 features, which is huge, especially in relation to the 3376 data points. To reduce the number of features to increase speed of running supervised learning algorithms for revenue prediction of the movies, feature reduction was deemed required. To achieve this, PCA and feature selection were pursued.
 
### (1). PCA 
 
PCA was done in two ways:
1. (PCA_noScale_20Comp) Data wasn't scaled, and number of principal components selected = 20
2. (PCA_Scale_0.995VarRecov) Z-Score normalization was done on the features, and number of principal components = # to recover 99% of the variance. To achieve normalization, remove the mean of the feature and scale to unit variance. The Z-Score of a sample x is calculated as: z = (x - u) / s.

#### PCA_noScale_20Comp DETAILS
Recovered Variance: 99.99999999999851%  
Original # features: 10955  
Reduced # features: 20  
Recovered Variance Plot Below for PCA_noScale_20Comp%    
Note: Huge first principal component is probably due to othe feature of budget, which is much bigger than all other features (average = 40,137,051.54)  
<p align="center">
  <img src="SanmeshCodes/Figures/20CompPCAGraph.png" >
</p>

#### PCA_Scale_99%VarRecov DETAILS
Recovered Variance:  99.00022645866223%  
Original # features: 10955  
Reduced # features: 2965  
Recovered Variance Plot Below for PCA_Scale_99%VarRecov  
<p align="center">
  <img src="SanmeshCodes/Figures/99PercRecovVarPCAGraph.png" >
</p>

### (2). Feature selection 

#### Using XGBRegressor

We used XGBRegressor to check out the correlation of various features to the revenue. 
Once we visualized the graphs we then manually set a threshold and gathered 150 features for testing our models on.

#### Graphs



##### Feature importances of encoded movie data
######  2000 features sorted by feature importance scores of XGBRegressor
<p align="center">
  <img src="PrithviCodes/plots/xgb_2000.png" >
</p>

We plotted the feature importances of the features WRT revenue which was obtained through XGBRegressor.
As shown in the graph out of the 10,000+ features, less than 200 have a meaningful correlation with Revenue. 


###### 150 to 200 features feature importance scores of XGBRegressor

To determine threshold for cutoff for feature selection, we plotted the graph for the lowest contributors out of the features (from feature no. 150 to 200). From this we obtained a threshold of 0.0002. All features with scores above this were considered for further predictive analysis.

<p align="center">
  <img src="PrithviCodes/plots/xgb_150_200.png" >
</p>




##### Top 25 Revenue predictors

<p align="center">
  <img src="PrithviCodes/plots/25_top_XGB.png" >
</p>

We observed the most important features which have high correlation to the box office revenue of a movie, and have plotted the top 25 of these.

# 5. Movie Revenue Prediction 

## Experiments and Model Testing

<p align="center">
  <img src="PrithviCodes/plots/flow_chart.png">
</p>

### Linear Ridge Regression  


First, we tried to predict the exact revenue of the test set of movies using linear ridge regression. Ridge regression was chosen because it would protect against overfitting of the data, especially when there are huge number of features. 
Cross validation was performed to find the optimal alpha or regularization value.
The data sets were the two PCA data sets, and the feature selection dataset mentioned previously. Ridge Regression was trained on 80% of each data set, and then finally tested on the remaining 20% of the data sets.  The results are below.

### (1). PCA No scaling, 20 components
RMSE: 160266397.7589437  
Normalized RMSE: 0.050410109822282445    
R2 score 0.49805732362034183  

### (2). PCA Scaling,99% variance recovery:
RMSE: 225957444.3019453  
Normalized RMSE: 0.07107253761050907  
R2 score 0.00224829444458019  

### (3). Feature Selection:
RMSE: 126001088.6944168  
Normalized RMSE: 0.03963231723948973   
R2 score 0.6897457309459162  

Comparing RMSE and R2 of Ridge Regression on Three Input Data
<p align="center">
  <img src="SanmeshCodes/Figures/RidgeRegressionRMSE.PNG">
</p>
<p align="center">
  <img src="SanmeshCodes/Figures/RidgeRegressionR%5E2.PNG">
</p>

The plots below are the predicted revenue vs actual revenue from Ridge Regression. The data was sorted by the actual y revenue values in order to make it easier to view the results. Alpha was determined through kfold method (Leave-one-out cross validation) and was 0.5 for feature selection.  

Revenue Prediction with PCA_noScale_20Comp data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA20Comp.png">
</p>
  
Revenue Prediction with PCA_Scale_99%VarRecov data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_PCA99PercVarRecov.png">
</p>  
  
Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtest_xgbFeatures.png">
</p>

Closeup of Revenue Prediction with Feature Selection data as input
<p align="center">
  <img src="SanmeshCodes/Figures/ridgeRegressionPlotYPredVsYtestCloseup_xgbFeatures.png">
</p>


------------
### Linear Ridge Regression Results  
The ranking of the input data that gave the highest R^2 scores and lowest RMSE values from best to worst are:
1. Feature Selection 
2. PCA No scaling, 20 components  
3. PCA Scaling, 99% variance recovery

Feature Selection gave us the best performance for ridge regression. Our target R^2 value to indicate a good model is 0.6 to 0.9, according to this literature [1], and this is acheived only through the feature selection data input with ridge regression. Thus we deam ridge regression model with feature selection input as a success in predicting movie revenue.

We can see that for feature selection input, there is bigger error in prediction for bigger test revenues. The predicted revenue plot does have a similar shape to the actual revenue, this showing that the prediction values are trying to follow the actual values. However, the predicted value is not able to keep up with the increase of the actual revenue. This may be because there is a smaller % of actual revenues that are bigger. This may be corrected by having a bigger dataset to train on than only having 3376 samples. 

It isn't clear completely why feature selection performs better than PCA, but one factor may be that there are some features such as a particular actor name, with values only binary 1 or 0 indicating whether the actor is in the movie. So maybe PCA doesn't work as well on binary value features, and there is conflicting opinion on why this is the case in the community. Maybe in the future, we will look into other methods of encoding the feature of actors into numerical data. One potential example is having one feature for all actors, and just encoding the actors into a numerical value from 0 to the # of actors.
  
What is interesting is that the PCA data with normalization performed worse than the PCA without normalization. It is counterintuitive because the goal of PCA is to identify the axis with the most variance, and this goal may be impeded when one feature has much bigger values than other features (in our case, the feature is budget). However, the non-normalized PCA might have performed better because the data captures the budget mainly in the first principal component. We see from our correlation graphs and other literature [1] that budget is one of the leading indicators to predicting movie revenue, so it makes sense that when using PCA data without normalization, it will perform better than pca with normalization.

# 6. Classification Models


### Binning of Y Revenue values 
  
To be able to classify movies into different categories of revenue, we needed to bin the prediction class into various intervals.

We set different values of the bin-size in our following experiments. Given a bin-size B and a bin number x, the intervals formed will be of the form:   
    
[x * B * 1e^6 , (x+1) * B * 1e^6], where x = 0, 1, 2 ...

The below formula is used to categorize the Y values:

y' = y / (B * 1e^6), where y is the actual revenue value and y' is the integer value after binning.

The value of B determines how the y values will get combined. 

e.g:    
Using a bin size of 50, a y value equal to 1e^8 will get mapped to a value equal to 2.
Whereas, Using a bin size of 300, the same value will get mapped to a value equal to 0.

Thus, the value of bin-size is an important variable that can lead to different classification results. We experiment with different values of bin-size in the following sections.

The significance of turning the revenue problem into a classification problem is that if we can get a high accuracy of predicting revenue for a specific bin size, we can tell customers that we can predict revenue at high accuracy within margins of error of +- bin size, or within bin-size intervals of revenue.

  
### SVM
We used Support Vector Machine as a classification model in order to classify the movie data into the correct revenue category. The data is not linearly separable. We were able to get the best performance with the Radial Basis Function kernel. We have experimented with different hyperparameters including gamma, the kernel coefficient, and C the regularization parameter. We found the optimal values for the hyperparameters by using 3-Fold Cross Validation with F1 score as the metric. We used a specific F1 score variation that first calculated the metrics for each class and then found their average weighted by the number of true instances for each class. We needed to use this because of our class imbalance. 

First we kept Gamma constant at its default value of ‘auto’ for the Sklearn Support Vector Classifier call which uses 1/( n features). We then tuned the C parameter to find the optimal value for C. We then fixed C at its default value for the Sklearn Support Vector Classifier call at 1. We tuned the Gamma parameter to find its optimal value. After we obtained the optimal hyperparameters using 3-Fold Cross Validation, we trained the training set with these chose hyperparameters. We then tested our trained model on the test set. 

#### C, Regularization Hyperparameter

C is the regularization hyperparameter. C affects the complexity of the decision rule assigning penalties for the misclassifying of data points. When C is small, the Support Vector Machine classifier is more tolerant of misclassifying data points. The classifier will have high bias, low variance. When C is large, the classifier is very intolerant to the misclassification of data points. It has low bias, high variance.

As seen in our F1-score vs C-hyperparameter plot for classification of data points within a $300 million revenue interval,  the model has a F1 score of 0.8094 when it has a C value of 1. This is thus the optimal value for C from our hyperparameter tuning process. When C is increased higher than 1, the increase in regularization leads to a decrease in model performance.



<p align="center">
  <img src="PrithviCodes/plots/Line_chart_C_f1score.png" >
</p>


#### Gamma, the Kernel Coefficient

Gamma, the kernel coefficient, affects the spread of the kernel and thus affecting the decision boundary. When Gamma is low, the curve of the decision boundary is low. When Gamma is high, the curve of the decision boundary is high. As seen in our F1-score vs Gamma-hyperparameter plot for classification of data points within a $300 million revenue interval,  the model has a F1 score of 0.8443 when it has a Gamma value of 1e-9. As we increase the value of Gamma, the curve of the decision boundary increases. The model begins to decrease in performance.


<p align="center">
  <img src="PrithviCodes/plots/Line_chart_gamma_f1score.png" >
</p> 


#### Error Visualization
  

It can be observed in the Actual Revenue vs Predicted Revenue (SVM) scatter plot below that there is a larger number of data points in the bins with bin size $300 million as compared to the bins with bin size $100 million. Thus we see that the error is higher in the bins with bin size $100 million since the classification problem becomes harder as we increase the number of classes needed to correctly predict. In both bin sizes, the model has a more difficult time predicting higher revenues than it does predicting lower revenues.

<p align="center">
  <img src="PrithviCodes/Scatter_predVSActual.png" >
</p>

<p align="center">
  <img src="PrithviCodes/Scatter_predVSActual100.png" >
</p>

#### Visualizing F1 score & Accuracy
    

In the bar graph below, we have plotted the F1 scores and accuracy versus different bin sizes used for categorizing the revenue class.


We used 3 bin sizes - $50 million, $100 million and $300 million. For each bin size, we used 3-Fold Cross Validation to choose the best values of gamma and C. After using the best values of the 2 hyperparameters to train our model, we tested our model on the test dataset and generated the F1 scores and accuracy scores.


We observed that we get the highest F1 score and accuracy in case of 300 bins. This observation is in accordance with the error visualization graphs presented above that show the least error in case of 300 bins.



<p align="center">
  <img src="PrithviCodes/BinVSF1SVM_2.png" >
</p>


We have plotted our depicted SVM classification results for both bin sizes below:
<p align="center">
  <img src="Figures/SVM_300_Norm_ConfusionMat.png" >
</p> 

<p align="center">
  <img src="Figures/SVM_100_Norm_ConfusionMat.png" >
</p> 

Although we have achieved high accuracy and F1-score, we see from the confusion matrix that majority of our test instances are predicted to be in bin 0 or 1. This can be explained by class imbalance in the training data. There are more number of examples which belong to class 0 and 1 as compared to other categories. To overcome this challenge, we explored Random Forest which performs better with class imbalance in training data.

Another possible avenue to explore for this class imbalance problem with SVM would be to increase the penalty for misclassifying minority classes. This could be done by inversely weighing 'C' with class size. This could be explored in future.


#### Random Forest

<p> We used Random Forest as another classification model in order to classify the movie data into the correct revenue category. We have experimented with different hyperparameters including the number of estimators and the maximum depth of each tree. We found the optimal values for the hyperparameters by using 3-Fold Cross Validation with F1 score as the metric. We used a specific F1 score variation that first calculated the metrics for each class and then found their average weighted by the number of true instances for each class. We needed to use this because of our class imbalance. The criteria used to measure the quality of each split is the Gini Impurity index.</p>
<p> First we kept the number of estimators fixed at its default value of 10 for the Sklearn Random Forest  Classifier call. We then tuned the Max Depth hyperparameter to find the optimal value for Max Depth. We then fixed Max Depth at its default value of “None” for the Sklearn Support Vector Classifier call. We tuned the number of estimators hyperparameter to find its optimal value. After we obtained the optimal hyperparameters using 3-Fold Cross Validation, we trained the model using the training set with these chosen hyperparameters. We then tested our trained model on the test set. </p>
  

#### Number of Estimators

Number of estimators are used to specify the number of trees in the forest. Generally, a higher n=value for the number of trees results in better learning and higher accuracy. But, adding a huge number of trees can make the training and inference process go slow. Hence, we have used a search method to find the best value of n_estimator.

We experimented by fitting Random forest with trees having number of estimators ranging from 1 to 500. Then we plot the F1-score against the Depths.

We observed that as we increase the number of estimators, the F1 score increases to attain its highest value and then it becomes almost constant as we keep increasing the number of estimators further.

We choose the n_estimators parameter corresponding to the highest F1 score value to be the best value.

Below are the graphs for 2 different Bin sizes - 100 and 300.</p>

    

<p align="center">Bin size = 300</p>
<p align="center">
  <img src="Tarushree_RF plots/F1VsEst_300.png" height="500" width="600">
</p>
    

<p align="center">Bin size = 100</p>
<p align="center"> 
  <img src="Tarushree_RF plots/F1VsEst_100.png" height="500" width="600">
</p>



#### Maximum Depth of the Trees

Maximum Depth stands for the depth of each tree in the forest. More the depth of the tree, the more splits it has and more is the tendency to captures fine information about the data. 

We experimented by fitting Random forest with trees having depths ranging from 1 to 250. Then we plot the F1-score against the Depths.

We observed that as we increase the depth of the decision trees, the F1 score increases intitially. But as we keep increasing the depth further, the F1 score falls slightly and then it plateaus to attain a contant value.

We choose the depth corresponding to the highest F1 score value to be the best Depth value.

Below are the graphs for 2 different Bin sizes - 100 and 300.


<p align="center">Bin size = 300</p>
<p align="center">
  <img src="Tarushree_RF plots/F1VsDepth_300.png" height="500" width="600">
</p>

<p align="center">Bin size = 100</p>
<p align="center">
  <img src="Tarushree_RF plots/F1VsDepth_100.png" height="500" width="600">
</p>

#### Error Visualization
  

It can be noticed in the Actual Revenue vs Predicted Revenue, sorted by Actual Revenue scatter plot below that there is a larger number of data points in the bins with bin size $300 million as compared to the bins with bin size $100 million. This is a similar trend to what we observed with SVM. The error is higher in the bins with bin size $100 million since the classification problem becomes harder as we increase the number of classes needed to correctly predict. In both bin sizes, the model has a more difficult time predicting higher revenues than it does predicting lower revenues.

    

<p align="center">Bin size = 300</p>

<p align="center">
  <img src="Tarushree_RF plots/Scatter_RF_300.png" height="500" width="600">
</p>


     

<p align="center">Bin size = 100</p>

<p align="center">
  <img src="Tarushree_RF plots/Scatter_RF_100.png" height="500" width="600">
</p>


#### Visualizing F1 score & Accuracy
    

In the bar graph below, we have plotted the F1 scores and accuracy versus different bin sizes used for categorizing the output label, that is revenue.

We used 3 bin sizes - 50, 100 and 300. For each bin size, we used 3-Fold Cross Validation to choose the best values of number of estimators and maximum depth parameters.
After finding the best values of the 2 parameters, we generated F1 score and accuracy on the test dataset.

We observed that we get the highest F1 score and accuracy in case of 300 bins. This observation is in accordance to the error visualization graphs presented above that show the lest error in case of 300 bins.

<p align="center">
  <img src="Tarushree_RF plots/BinSize_vs_F1_barplot.png" height="500" width="600">
</p>



We have plotted our Random Forest classification results for both bin sizes as confusion matrices below:
<p align="center">
  <img src="Figures/RF_300_Norm_ConfusionMat.png" >
</p>
<p align="center">
  <img src="Figures/RF_100_Norm_ConfusionMat.png" >
</p>
With random forest, we see improved classification as compared to SVM and the overall accuracy also looks good. However, accuracy for each class other than the first class, is not so good. This can be explained by class imbalance in the training data.  


# 7 Final Conclusions  

##### TABLES

##### Classification 
<p align="center">
  <img src="PrithviCodes/plots/Classification_results.png">
</p>

#### Regression
<p align="center">
  <img src="PrithviCodes/plots/Regression_results.png">
</p>

Overall, for dataset dimension reduction, feature selection worked the best across all the supervised learning algorithms.  Ridge regression showed promising results in predicting the revenue, with a R^2 value of 0.69, which is within the range of 0.6 to 0.9, indicating a good model according to [1]. For our classfication models SVM and Random forest, three bin sizes (50 million, 100 million, and 300 million) were experimented with and we achieve higher accuracy with the bigger - 300 bin size. 


Differences in performance using the XGBRegressor features and the PCA principle components can be observed in the table. SVM with a bin size of $300 million has 0.0447 greater of an F1 score with the XGBRegressor features than it does using the principle components from PCA. However SVM with a bin size of $100 million has 0.0032 greater of an F1 score with the principle components from PCA than it does using the XGBRegressor features. 


Random Forest with a bin size of $300 million has 0.0037 greater of an F1 score with the principle components from PCA features than it does using the XGBRegressor features. Random Forest with a bin size of $100 million also has 0.0048 greater of an F1 score with the principle components from PCA features than it does using the XGBRegressor features. However, both SVM and Random Forest are affected by the class imbalance problem in the training data.
  
With random forest, we see improved classification as compared to SVM and overall accuracy also looks better. However, accuracy for each class other than the first class, is not so good. This can be explained by class imbalance in the training data.

<p align="center">
  <img src="PrithviCodes/BinVSF1SVM_2.png" >
</p>

<p align="center">
  <img src="Tarushree_RF plots/BinSize_vs_F1_barplot.png" height="500" width="600">
</p>



# 8. Reference
<ol>
<li>What makes a successful film? Predicting a film’s revenue and user rating with machine learning. (2019). Retrieved 28 September 2019, from https://towardsdatascience.com/what-makes-a-successful-film-predicting-afilms-revenue-and-user-rating-with-machine-learning-e2d1b42365e7</li>
<li>Dan Cocuzzo, Stephen Wu, "Hit or Flop: Box Office Prediction for Feature Films", Stanford University, 2013.</li>
<li>Nahid Quader, Md. Osman Gani, Dipankar Chaki, and Md. Haider Ali, “A Machine Learning Approach to Predict Movie Box-Office Success,”in 20th International Conference of Computer and Information Technology (ICCIT), December 2017.</li>
<li>Mahesh Joshi, Dipanjan Das, Kevin Gimpel, and Noah A. Smith. Movie reviews and revenues: an experiment in text regression. In Human Language Technologies: The 2010 Annual Conference of the North American Chapter of the Association for Computational Linguistics, HLT ’10, 2010.</li>
<li>W. Zhang, S. Skiena Improving movie gross prediction through news analysis International joint conference on web intelligence and intelligent agent technology, IEEE/WIC/ACM (2009)</li>
</ol>

# 9. Contributions

<ol>
<li>Proposal- Sanmesh, Aaron, Tarushree, Aastha, Prithvi</li>
<li>Overview and Project Motivation - Prithvi, Sanmesh</li>
<li>Dataset visualization - Prithvi</li>
<li>Feature Selection - Prithvi</li>
<li>PCA -  Sanmesh</li>
<li>Ridge Regression- Aastha, Sanmesh</li>
<li>SVM - Aaron, Tarushree, Aastha, Prithvi, Sanmesh</li>
<li>Random Forest - Aaron, Tarushree, Aastha, Sanmesh</li>
</ol>