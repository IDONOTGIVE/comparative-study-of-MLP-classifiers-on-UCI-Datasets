clear;

load zoo.mat

t1=cputime;       %record the start time of the program
runningcycles=100; % Times to train and test the Bagging MLP 
s=0;                %counter to sum the accuarcy of the running cycles

for n=1:runningcycles
     %cross validate of Input, 80% for training and 20% for testing
    [train test] = crossvalind('HoldOut',size(Input,2),0.2);
	
	%cross validate of Group, 80% for training and 20% for testing
    [gtrain gtest] = crossvalind('HoldOut',size(Group,2),0.2);	
    [prototypes] = mdc_train(Input,Group);%training the data
    
	in =Input(:,test);
    y=mdc_test(in,prototypes);%test the data
    s=sum(y' == Group(:,gtest))/length(Input(:,test))+s;%sum of accuracies 
end

averagetime=(cputime-t1)/runningcycles;
accuracy=s/runningcycles;
disp(['The accuracy of Minmum Distance Classifier is: ' num2str(accuracy)]);
disp(['The cputime of Minmum Distance Classifier is: ' num2str(averagetime)]);
