clear;

%  This part is used to loading different datasets
load zoo.mat
%load yeast.mat
%load vehicle.mat
%load satimage.mat
%load Glass.mat

t1=cputime;         %record the start time of the program
runningcycles=100;     % Times to train and test the Bagging MLP
s=0;                %counter to sum the accuarcy of the running cycles

for n=1:runningcycles
      
	[train test] = crossvalind('HoldOut', size(Input,2), 0.2);  %cross validate 80% for training and 20% for testing 
    y = BaggingMLP_train(Input(:,train), Target(:,train), 5);   %training function and parameter
    result = BaggingMLP_test(Input(:,test), y);                 % test function 
	
	
	%this part is to find out the relevant feature of conlums of target
    target = Target(:,test);
    [C,I]=max(target,[],1);
    target_Group=I;
	
	%row numbers used to be index 
    m = size(result, 2);
    result_index(1,m) = 0;
	
	%find features value of result
for i = 1:m
    temp = abs(result(:,i)');
    result_index(1,i) = find(temp == max(temp));
end

%compute the accuarcy of Bagging MLP
s=length(find(result_index == target_Group))/length(result)+s;
end

%caculate the averages of time and accuracy
averagetime = (cputime-t1)/runningcycles;
accuracy = s/runningcycles;
disp(['The accuracy of Bagging MLP is: ' num2str(accuracy)]); 
disp(['The CPU time of Bagging MLP is: ' num2str(averagetime)]); 