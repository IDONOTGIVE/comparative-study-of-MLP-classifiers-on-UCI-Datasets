clear;

%  This part is used to loading different datasets
load zoo.mat
%load yeast.mat
%load vehicle.mat
%load satimage.mat
%load Glass.mat

t1=cputime;          %record the start time of the program
runningcycles=100;   % Times to train and test the Bagging MLP
s=0;                 %counter to sum the accuarcy of the running cycles

for n=1:runningcycles
%cross validate 80% for training and 20% for testing 
[train test] = crossvalind('HoldOut',size(Input,2),0.2);


%Create a rbf network
net = rbf(16,2,7,'gaussian');    % this fomular is used by zoo.mat
%net = rbf(8,2,10,'gaussian');   % this fomular is used by yeast.mat
%net = rbf(18,2,4,'gaussian');   % this fomular is used by vehicle.mat
%net = rbf(36,2,7,'gaussian');   % this fomular is used by satimage.mat
%net = rbf(9,2,6,'gaussian');    % this fomular is used by Glass.mat

%use rbftrain to trai the network
options(1, :) = foptions;
options(2, :) = foptions;
options(2, 14) = 100;  % 100 iterations of EM
options(2, 5)  = 1;    % Check for covariance collapse in EM
net = rbftrain(net, options, Input(:,train)', Target(:,train)');

%using rnffwd to test the trained network
a = rbffwd(net, Input(:,train)');
result=a - ones(81,7);       % this fomular is used by zoo.mat
%result=a - ones(1188,10);   % this fomular is used by yeast.mat
%result=a - ones(677,4);     % this fomular is used by vehicle.mat
%result=a - ones(5148,7);    % this fomular is used by vehicle.mat
%result =a - ones(172,6);    % this fomular is used by Glass.mat       

%using min() to locate the min distance from 1
[C,I]=min(result.^2,[],2);

%caculate the accuracies of the runningcycles
s=sum(I'==Group(train))/length(train)+s;

end
averagetime=(cputime-t1)/runningcycles;
accuracy=s/runningcycles;
disp(['The accuracy of RBF is: ' num2str(accuracy)]); 
disp(['The CPU time of kNN is: ' num2str(averagetime)]); 

