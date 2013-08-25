clear;

%  This part is used to loading different datasets
load zoo.mat
%load yeast.mat
%load vehicle.mat
%load satimage.mat
%load Glass.mat


t1 = cputime;      %record the start time of the program
runningcycles=100; % Times to train and test the Bagging MLP
s=0;               %counter to sum the accuarcy of the running cycles          


for n=1:runningcycles
    %cross validate 80% for training and 20% for testing
    [train test] = crossvalind('HoldOut', size(Input,2), 0.2);
	
	%input praramter for the train
    W =autoencoder_train(Input(:,train), Target(:,train), [1 1]);
	
	%test for the auto encoder
    result = autoencoder_test(W,Input(:,test),1);
    target = Target(:,test);
    m = size(target,2);
	
	%compute the max value of target
    [C,I]=max(target,[],1);
    target_Group=I;
	
	%sum of accuracies 
    s=sum(result==I)/length(result)+s;
end

averagetime=(cputime-t1)/runningcycles;
accuracy=s/runningcycles;
disp(['The accuracy of Auto Encoder is: ' num2str(accuracy)]); 
disp(['The cputime of Auto Encoder is: ' num2str(averagetime)]); 
