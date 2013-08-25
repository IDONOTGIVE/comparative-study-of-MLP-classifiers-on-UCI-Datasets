clear;

%  This part is used to loading different datasets
load zoo.mat
%load yeast.mat
%load vehicle.mat
%load satimage.mat
%load Glass.mat

t1=cputime;    %record the start time of the program
s=0;            %counter to sum the accuarcy of the running cycles
runningcycles=100;  % Times to train and test the Bagging MLP

for n=1:runningcycles

     %cross validate 80% for training and 20% for testing 
    [train test] = crossvalind('HoldOut',size(Input,2),0.2);  

     %using knnclassify to calssify the training data	
    Class = knnclassify(Input(:,train)',Input(:,test)',Group(test),3);
	
	%sum of accuracies 
    s=sum(Class'==Group(train))/length(train)+s;
end

averagetime=(cputime-t1)/runningcycles;
accuracy=s/runningcycles;
disp(['The accuracy of kNN is: ' num2str(accuracy)]);
disp(['The CPU time of kNN is: ' num2str(averagetime)]); 
