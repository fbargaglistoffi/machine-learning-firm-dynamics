# Supervised Learning for the Prediction of Firm Dynamics

This repository contains the additional material for the Chapter on ["Supervised learning for the prediction of firm dynamics"](https://arxiv.org/abs/2009.06413) by F.J. Bargagli-Stoffi, J. Niederreiter and M. Riccaboni in the book _"Data Science for Economics and Finance: Methodologies and Applications"_ by S. Consoli,  D. Reforgiato Recupero, M. Saisana. 

In the first Section of this repository we introduce a step-by-step guide for the reader that is new to machine learning to guide her/him in designing a supervised learning routine; in the second Section we provide further details on the main algorithms used for prediction tasks at different stages of the company life cycle together with simple examples on their implementation in *R*.

<a href=https://github.com/fbargaglistoffi/supervised-learning-firm-dynamics/blob/master/SL_analysis.Rmd>Here</a> we show how to implement the supervised learning routine to predict firms' bankruptcy on a dataset of Italian firms' financial accounts. The <a href=https://github.com/fbargaglistoffi/supervised-learning-firm-dynamics/blob/master/mock_data.Rdata>dataset</a> is a small, random sample of real firm level data used by F.J. Bargagli-Stoffi, M. Riccaboni and A. Rungi for the main analysis of the paper _"Machine learning for zombie hunting. Firms' failures, financial constraints, and misallocation"_.  For more details on the predictors, we refer the reader to the original paper.

# 1. A Simple Supervised Learning Routine

This simple step-by-step guide should aid the reader in designing a supervised learning (SL) routine to predict outcomes from input data.
    
1. Check that information on the outcome of interest is contained for the observations that are later used to train and test the SL algorithm, i.e. that the data set is labeled. 

2. Prepare the matrix of input attributes to a machine-readable format.
  * In case of missing values in the input attributes, this missingness has to be dealt with as it can impact results (see ["Machine learning for zombie hunting. Firms' failures, financial constraints, and misallocation"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3588410) by Bargagli-Stoffi, Riccaboni and Rungi for more information).
  * In case it is ambiguous how to select the attributes used as input, the user might want to perform a feature selection step first (see [this paper](https://www.sciencedirect.com/science/article/pii/S0925231218302911) for more information).
 
3. Choose how to split your data between training and testing set. Keep in mind that both training and testing set have to stay sufficiently large to train the algorithm or to validate its performance, respectively. Use resampling techniques in case of low data dimensions and stratified sampling whenever labels are highly unbalanced. If the data has a time dimension, make sure that the training set is formed by observations that occured before the ones in the testing set. 

4. Choose the SL algorithm that best suits your need. Possible dimensions to evaluate are prediction performance, simplicity of result interpretation and CPU runtime. Often a horserase between many algorithms is performed and the one with the highest prediction performance is chosen. There are already many algorithms already available "off the shelf" - consult [this page](https://cran.r-project.org/web/views/MachineLearning.html) for a comprehensive review of the main packages for machine learning in *R*.

5. Train the algorithm using the training set only. In case hyper-parameters of the algorithm need to be set, choose them using _crossfold validation_ on the training set, or better keep part of the training set only for hyperparameter tuning - but do not use the testing set until the algorithms are fully specified.

6. Once the algorithm is trained, use it to predict the outcome on the testing set. Compare the predicted outcomes with the true outcomes.

7. Choose the performance measure on which to evaluate the algorithm(s). Popular performance measures are _Accuracy_ and _Area Under the receiver operating Curve_ (AUC). Choose sensitive performance measure in case your data set is unbalanced such as _Balanced Accuracy_ or the _F-score_.

8. Once prediction performance has been assessed, the algorithm can be used to predict outcomes for observations for which the outcome is unknown. Note that valid predictions require that new observations should contain similar features and need to be independent from the outcome of old ones.

# 2. Supervised Learning Algorithms

## 2.1 Decision Trees

### Description
Decision trees commonly consist of a sequence of binary decision rules (nodes) on which the tree splits into branches (edges). At each final branch (leaf node) a decision regarding the outcome is estimated.  The sequence of decision rules and the location of each cut-off point is based on minimizing a measure of node purity (e.g., Gini index, or entropy for classification tasks, mean-squared-error for regression tasks). Decision trees are easy to interpret but sensitive to changes in the feature space, frequently lowering their out of sample performance (see [Breiman 2017](https://www.taylorfrancis.com/books/9781315139470) for a detailed introduction).

### Example usage in R

We focus on the function _rpart_ in the *R* package *Rpart*. The documentation can be found [here](https://www.rdocumentation.org/packages/rpart/versions/4.1-15/topics/rpart).

* <tt>`formula`</tt>: a formula in the format of the formula used to train the decision tree (e.g. outcome ~ predictor1 + predictor2 + ect.);
* <tt>`data`</tt>: specifies the data frame;
* <tt>`method`</tt>: "class" for a classification tree, "anova" for a regression tree;
* <tt>`control`</tt>: optional parameters for controlling tree growth. For example, control=rpart.control(minsplit=30, cp=0.001) requires that the minimum number of observations in a node be 30 before attempting a split and that a split must decrease the overall lack of fit by a factor of 0.001 (cost complexity factor) before being attempted.

```R
# Decision Tree with rpart
install.library("rpart") # if not already installed
library(rpart)

# Grow the tree
dt <- rpart(trainoutcome ~ trainfeatures, method="class", data= train_data, control=rpart.control(minsplit=30, cp=0.001))

    printcp(dt) # display the results
    plotcp(dt) # visualize cross-validation results
    summary(dt) # detailed summary of splits

# Plot tree
plot(dt, uniform=TRUE, main="Classification Tree")
text(dt, use.n=TRUE, all=TRUE, cex=.8)

# Create attractive postscript plot of tree
post(dt, file = "c:/tree.ps",
   title = "Classification Tree")
   
# Get predicted values
dt.pred <- predict(dt, newdata=test, type='class')
   
         # generate table that compares true outcomes of the testing set with predicted outcomes of decisiontree
        dt_tab= table(true=testoutcome, pred= dt)
        # generate ROC object based on predictions in testing set
        dt_roc=roc(testoutcome ~ dt)
        #calculate AUC value of predictions in testing set
        dt_auc=pROC::auc(dt_roc)
```

## 2.2 Random Forest

### Description
Instead of estimating just one _DT_, random forest resamples the training set observations to estimate multiple trees. For each tree at each node a sample of _m_ predictors is chosen randomly from the feature space. To obtain the final prediction the outcomes all trees are averaged or in classification tasks the chosen by majority vote (see also the original contribution of Breiman, 2001)

### Example usage in R
We focus on the function _RandomForest_ in the *R* package *RandomForest*. The documentation can be found [here](https://www.rdocumentation.org/packages/randomForest/versions/4.6-14/topics/randomForest).

Selection of inputs the function takes :

* <tt>`x`</tt>: the feature matrix of the training set (NxP);
* <tt>`y`</tt>: the outcome variable of the training set (Nx1);
* <tt>`xtest`</tt>: (optional) the feature matrix of the testing set (MxP);
* <tt>`ytest`</tt>: (optional) the outcome variable of the testing set (Mx1);
* <tt>`mtry`</tt>: (optional) number of variables randomly sampled as candidates at each split. Note that the default values are different for classification (sqrt(P) where P is number of features and regression (P/3);
* <tt>`ntree`</tt>: (optional) number of trees to grow. This should not be set to too small a number, to ensure that every input row gets predicted at least a few times;
* <tt>`importance`</tt>: (optional) Should importance of predictors be assessed? ;
* <tt>`keep.forest`</tt>: (optional) Should the forest be stored in the object (for later prediction tasks)? ;
* <tt>`seed`</tt>: (optional) set an arbitrary numerical value to make RF estimation result reproducable ;

The _RandomForest_ function returns an object which is a list containing information such as:  the predicted values of the testing set in `$test$predicted`, importance measures in `$importance` and the entire forest `$forest` if `keep.forest==TRUE`.


```R
# Random forest with the randomForest package
install.package("randomForest") # if not already installed
library("randomForest")
library("pROC") 

# Train the Random Forest
obj_rf=randomForest(trainfeatures,y=trainoutcome, xtest=testfeatures,ytest=testoutcome, mtry=8, ntree=500, importance=TRUE, keep.forest=FALSE, seed=34)
      
      #generate table that compares true outcomes of the testing set with predicted outcomes of random forest
        rf_tab= table(true=testoutcome, pred= obj_rf$test$predicted )
      #generate ROC object based on predictions in testing set
        rf_roc=roc(testoutcome~ obj_rf$test$votes[,2])
      #calculate AUC value of predictions in testing set
        rf_auc=pROC::auc(rf_roc)
```


## 2.3 Support Vector Machines

### Description
Support vector machines (SVM) & Support vector machine algorithms estimate a hyperplane over the feature space to classify observations. The vectors that span the hyperplane are called support vectors. They are chosen such that the overall distance (called margin) between the data points and the hyperplane as well as the prediction accuracy is maximized (see also [Steinwart 2008](https://www.springer.com/gp/book/9780387772417)).

### Example usage in R
We focus on the function _svm_ in the *R* package *e1071*. The documentation can be found [here](https://www.rdocumentation.org/packages/e1071/versions/1.7-3/topics/svm)


* <tt>`formula`</tt>:  a formula in the format of the formula used to train the decision tree (e.g. outcome ~ predictor1+predictor2+ect.);
* <tt>`data`</tt>:  an optional data frame containing the variables in the model. By default the variables are taken from the environment which ‘svm’ is called from;
* <tt>`scale`</tt>: a logical vector indicating the variables to be scaled.
* <tt>`type`</tt>: svm can be used as a classification machine, as a regression machine, or for novelty detection;
* <tt>`kernel`</tt>: the kernel used in training and predicting. You might consider changing some of the following parameters, depending on the kernel type.

```R
# Support Vector Machine with the e1071 package
install.package("e1071") # if not already installed
library("e1071")

# Train the Support Vector Machine
obj_svm <- svm(formula, data = train)

# Predicted outcomes Support Vector Machine
svm.pred <- predict(obj_model, newdata = test)
      
      # Generate table that compares true outcomes of the testing set with predicted outcomes of random forest
        svm_tab= table(true=testoutcome, pred= svm.pred)
      # Generate ROC object based on predictions in testing set
        svm_roc=roc(testoutcome ~ svm.pred)
      # Calculate AUC value of predictions in testing set
        svm_auc=pROC::auc(svm_roc)
```

## 2.4 Artificial Neural Network

### Description
(Deep) Artificial Neural Networks (ANN) & Inspired from biological networks, every neural network consists of at least three layers: an input layer containing feature information, at least one hidden layer (deep ANN are ANN with more than one hidden layer), and an output layer returning the predicted values. Each Layer consists of nodes (neurons) who are connected via edges across layers. During the learning process, edges that are more important are reinforced. Neurons may then only send a signal if the signal received is strong enough (see for example [Hassoun 2016](https://mitpress.mit.edu/books/fundamentals-artificial-neural-networks)).

### Example usage in R
We focus on the function _nnet_ in the *R* package *nnet"*. The documentation can be found [here](https://www.rdocumentation.org/packages/nnet/versions/7.3-12/topics/nnet).

* <tt>`formula`</tt>: a formula in the format of the formula used to train the decision tree (e.g. outcome ~ predictor1+predictor2+ect.);
* <tt>`data`</tt>: specifies the data frame;
* <tt>`weights`</tt>: weights for each example -- if missing defaults to 1;
* <tt>`size`</tt>: number of units in the hidden layer. Can be zero if there are skip-layer units;
* <tt>`rang`</tt>: initial random weights on [-rang, rang]. Value about 0.5 unless the inputs are large, in which case it should be chosen so that rang * max(|x|) is about 1;
* <tt>`decay`</tt>: parameter for weight decay (default is 0);
* <tt>`maxit`</tt>: number of iterations (default is 100).

```R
# Neural network with the neural net package
install.packages("neuralnet") # in not already installed
library(neuralnet)

# Train the Neural Network
nnet_fit <- nnet(formula, data = train, size = 2, rang = 0.1, decay = 5e-4, maxit = 200)

test.cl <- function(true, pred) {
    true <- max.col(true)
    cres <- max.col(pred)
    table(true, cres)
}

# Predict fitted values on test
predict.nnet <- predict(nnet_fit, test)
test.cl(test, predict.nnet)
```
