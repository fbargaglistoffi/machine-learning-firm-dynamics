# Supervised Learning for the Prediction of Firm Dynamics

This repository containts the additional material for the Chapter on <a href="https://arxiv.org/abs/1706.09523">"Supervised learning for the prediction of firm dynamics"</a> by F.J. Bargagli-Stoffi, J. Niederreiter and M. Riccaboni in the book "Data Science for Economics and Finance: Methodologies and Applications" by S. Consoli,  D. Reforgiato Recupero, M. Saisana. 

In the first Section of this repository we introduce a step-by-step guide for the reader that is new to machine learning to guide her/him in designing a supervised learning routine; in the second Section we provide further details on the main algorithms used for prediction tasks at different stages of the company life cycle together with simple examples on their implementation in *R*.

<a href=https://github.com/fbargaglistoffi/supervised-learning-firm-dynamics/blob/master/SL_analysis.Rmd>Here</a> we show how to implement the supervised learning routine to predict firms' bankruptcy on a dataset of Italian firms' financial accounts. The <a href=https://github.com/fbargaglistoffi/supervised-learning-firm-dynamics/blob/master/mock_data.Rdata>dataset</a> is a small, random sample of real firm level data used by F.J. Bargagli-Stoffi, M. Riccaboni and A. Rungi for the main analysis of the paper <a href="https://arxiv.org/abs/1706.09523">"Machine learning for zombie hunting. Firms' failures, financial constraints, and misallocation "</a>.  For more details on the predictors, we refer the reader to the original paper.

# 1. A Simple Supervised Learning Routine

This simple step-by-step guide should aid the reader in designing a supervised learning (SL) routine to predict outcomes from input data.
    
1. Check that information on the outcome of interest is contained for the observations that are later used to train and test the SL algorithm, i.e. that the data set is labeled. 

2. Prepare the matrix of input attributes to a machine-readable format.
  * In case of missing values in the input attributes, this missingness has to be dealt with as it can impact results (see xy for more information).
  * In case it is ambiguous how to select the attributes used as input, the user might want to perform a feature selection step first (see XY for more information).
 
3. Choose how to split your data between training and testing set. Keep in mind that both training and testing set have to stay sufficiently large to train the algorithm or to validate its performance, respectively.  Use resampling techniques in case of low data dimensions and stratified sampling whenever labels are highly unbalanced. If the data has a time dimension, make sure that the training set is formed by observations that occured before the ones in the testing set. 

4. Choose the SL algorithm that best suits your need. Possible dimensions to evaluate are prediction performance, simplicity of result interpretation and CPU runtime. Often a horserase between many algorithms is performed and the one with the highest prediction performance is chosen. There are already many algorithms already available "off the shelf" - consult this page for running examples in R.

5. Train the algorithm using the training set only. In case hyperparameters of the algorithm need to be set, choose them using Crossfold validation on the training set, or better keep part of the training set only for hyperparameter tuning - but do not use the testing set until the algorithms are fully specified.

6. Once the algorithm is trained, use it to predict the outcome on the testing set. Compare the predicted outcomes with the true outcomes

7. Choose the performance measure on which to evaluate the algorithm(s). Popular performance measures are Accuracy and Area Under the receiver operating Curve (AUC). Choose sensitive performance measure in case your data set is unbalanced such as balanced Accuracy.

8. Once prediction performance has been assessed, the algorithm can be used to predict outcomes for observations for which the outcome is unknown. Note that valid predictions require that new observations should contain similar features and need to be independent from the outcome of old ones.

# 2. Supervised Learning Algorithms

## 2.1 Decision Trees

### Description
Decision trees commonly consist of a sequence of binary decision rules (nodes) on which the tree splits into branches (edges). At each final branch (leaf node) a decision regarding the outcome is estimated.  The sequence of decision rules and the location of each cut-off point is based on minimizing a measure of node purity (e.g., Gini index, or entropy). Decision trees are easy to interpret but sensitive to changes in the feature space, frequently lowering their out of sample performance (see Breiman 2017 for a detailed introduction).

### Example usage in R



## 2.2 Random Forest

### Description
Instead of estimating just one DT, random forest resamples the training set observations to estimate multiple trees. For each tree at each node a sample of $m$ predictors is chosen randomly from the feature space. To obtain the final prediction the outcomes all trees are averaged or in classification tasks the chosen by majority vote (see also the original contribution of Breiman, 2001)

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
* <tt>`seed`</tt>: (optional) make RF estimation result reproducable? ;

The _RandomForest_ function returns an object which is a list containing information such as:  the predicted values of the testing set in `$test$predicted`, importance measures in `$importance` and the entire forest `$forest` if `keep.forest==TRUE`.




```R
install.package("randomForest") #if not already installed
library("randomForest")
library("pROC") 

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
Support vector machines (SVM) & Support vector machine algorithms estimate a hyperplane over the feature space to classify observations. The vectors that span the hyperplane are called support vectors. They are chosen such that the overall distance (called margin) between the data points and the hyperplane as well as the prediction accuracy is maximized (see also Ssteinwart 2008).

### Example usage in R

## 2.4 Artificial Neural Network

### Description
(Deep) Artificial Neural Networks (ANN) & Inspired from biological networks, every neural network consists of at least three layers: an input layer containing feature information, at least one hidden layer (deep ANN are ANN with more than one hidden layer), and an output layer returning the predicted values. Each Layer consists of nodes (neurons) who are connected via edges across layers. During the learning process, edges that are more important are reinforced. Neurons may then only send a signal if the signal received is strong enough (see for example Hassoun, ).

### Example usage in R

* <tt>`y`</tt>: the outcome variable;
* <tt>`w`</tt>: the reception of the treatment variable (binary);
* <tt>`z`</tt>: the assignment to the treatment variable (binary);
* <tt>`max_depth`</tt>: the maximal depth of the tree generated by the function;
* <tt>`n_burn`</tt>: the number of iterations discarded by the BCF-IV algorithm for the burn-in;
* <tt>`n_sim`</tt>: the number of iterations used by the BCF-IV algorithm  to get the posterior distribution of the estimands;
* <tt>`binary`</tt>: this option should be set to <tt>`TRUE`</tt> when the outcome variable is binary and to <tt>`FALSE`</tt> if the outcome variable is either discrete or continuous.

The _mm_bcf_iv_ function returns the discovered sub-population, the conditional complier average treatment effect (CCACE), the p-value for this effect, the p-value for a weak-instrument test, the proportion of compliers, the conditional intention-to-treat effect (CITT) and the proportion of compliers in the node.

More details on the R code for the BCF-IV function can be found [here](https://github.com/barstoff/BCF-IV/blob/master/Functions/BCF-IV_in_detail.pdf).


```R
source("bcf-iv.R")

bcf_iv(y, w, z, x, max_depth = 2, n_burn= 2000, n_sim= 2000, binary = TRUE)

mm_bcf_iv(y, w, z, x, max_depth = 2, n_burn= 2000, n_sim= 2000, binary = TRUE)
```


