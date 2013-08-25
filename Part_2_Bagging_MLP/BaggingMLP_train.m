function y =BaggingMLP_train(train_features,test_features,n)

% Classify using a Bagging Ensemble algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	n   - Number of hidden units
%
% Outputs
%   train      - Collection of 80% Input 
%   test       - Collection of 20% Input

% Written by Zhun Shen, 11/12/2011

y = cell(1,n);
Sele = size(train_features,2);  % the number of selection for bagging;

for i = 1:n
    index = ceil(rand(1,Sele)*Sele);
    train_features = train_features(:,index);
    test_features = test_features(:,index);   
	
    net = feedforwardnet(100,'traincgp');       %Initialize the net  
    net.trainParam.showWindow = false;          % close the nntraintool window;
    net = train(net, train_features, test_features);
    y{i} = net;
end