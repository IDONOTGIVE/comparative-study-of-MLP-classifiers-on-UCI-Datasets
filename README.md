comparative-study-of-MLP-classifiers-on-UCI-Datasets
====================================================

In this project, there are two parts of work to do. First, comparison of different kinds of BP algorithms, after training and testing, CPU time of running the programs is needed. The other part is to implement Auto encoder classifier and minimum distance classifier, and then compare these two classifiers with KNN and RBF. Also Bagging or boosting classifier ensemble should be implementing and compare with other four classifiers. Through obtained result, Bagging MLP classifier is prove to have Accuracy Rate, however CPU time of Bagging is huge.

Howto
-----

1. How to run

1)Add Part2_Bagging MLP folder to path.
2)Input ¡°option_task2¡± in the command windows of matlab, then enter
3)Default load is zoo.mat.If you want to load other dataset, just change the load command in the code, and save it. It will work.


2. Code of Bagging MLP Classifier consists of three parts:

BaggingMLP_train
1)Main function is y =BaggingMLP_train(train_features,test_features,n),
2)Create MLP networks by feedforwardnet().
3)Close the windows using net.trainParam.showWindow = false;
4)Trainging the MLPs using net.train(net,train_features,test_features);
  
BaggingMLP_test.m
1)Use for Loop and size to get features of the trainging

option_task2.m
1)Divided Input into 80% training and 20% test
2)Enter the train data in the BagingMLP_train method
3)Input the result from BaggingMLP_train into BaggingMLP_test method to test the error rate
4)Compare with the target to get error rate
