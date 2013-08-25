function result = BaggingMLP_test(train_features, y)

% Classify using a Bagging Ensemble algorithm
% Inputs:
% 	train_features - Train features
%	y              -train reslut from MLP_train
%	
% Outputs
%   result      - Collection of 80% Input 
% Written by Zhun Shen, 11/12/2011

n = size(y, 2);
M = size(train_features,2);
N = y{1}.outputs{2}.size;

temp(N, n) = 0;
result(N,M) = 0;

%loop for caculate the matched features
for i = 1:M
    for j = 1:n
       temp(:,j) = sim(y{1,j}, train_features(:,i));
    end
    result(:,i) = mean(temp,2);
end
