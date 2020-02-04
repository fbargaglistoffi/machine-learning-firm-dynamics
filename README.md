# Supervised Learning for the Prediction of Firm Dynamics

Repository for the chapter on "Supervised learning for the prediction of firm dynamics" by F.J. Bargagli-Stoffi, M. Riccaboni and J. Niederreiter.

## A Simple Supervised Learning Routine

This simple step-by-step guide should aid the reader in designing a supervised learning (SL) routine to predict outcomes from input data.
    
1. Check that information on the outcome of interest is contained for the observations that are later used to train and test the SL algorithm, i.e. that the data set is labeled. 

2. Prepare your the matrix of input attributes to a machine-readable format.
  * In case of missing values in the input attributes, this missingness has to be dealt with as it can impact results (see xy for more information).
  * In case it is ambiguous how to select the attributes used as input, the user might want to perform a feature selection step first (see XY for more information)
 
3. Choose how to split your data between training and testing set. Keep in mind that both training and testing set have to stay sufficiently large to train the algorithm or to validate its performance, respectively.  Use resampling techniques in case of low data dimensions and stratified sampling whenever labels are highly unbalanced. If the data has a time dimension, make sure that the training set is formed by observations that occured before the ones in the testing set. 

4. Choose the SL algorithm that best suits your need. Possible dimensions to evaluate are prediction performance, simplicity of result interpretation and CPU runtime. Often a horserase between many algorithms is performed and the one with the highest prediction performance is chosen. There are already many algorithms already available off the shelf - consult the web appendix for running examples in R.

5. Train the algorithm using the training set only. In case hyperparameters of the algorithm need to be set, choose them using Crossfold validation on the training set.

6. Once the algorithm is trained use it to predict the outcome on the testing set. Compare the predicted outcomes with the true outcomes

7. Choose the performance measure on which to evaluate the algorithm(s). Popular performance measures are Accuracy and Area under the curve. Choose sensitive performance measure in case your data set is unbalanced.

8. Once prediction performance has been assessed, the algorithm can be used to predict outcomes for observations for which the outcome is unknown. Note that valid predictions require that new observations should contain similar features and need to be independent from the outcome of old ones

## Supervised Learning Algorithms
