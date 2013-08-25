% This is the command for "myBackpropagation_SM"
clear;
%  This part is used to loading different datasets

load zoo.mat
%load yeast.mat
%load vehicle.mat
%load satimage.mat
%load Glass.mat

t1=cputime;       %record the start time of the program
runningcycles=100;   % Times to train and test the SM

for n=1:runningcycles
[train,test] = crossvalind('HoldOut',size(Input,2),0.2);
% alpha =0.5
[Wh,Wo] = myBackpropagation_SM(Input(:,train),Target(:,train),[100,7,0.01,0.5,0.015]);
result=Backpropagation_fwd(Wh,Wo,Input(:,train));
end
time=cputime-t1;
disp(['Average time of SM after 100 times training is : ' num2str(time/runningcycles)]);






