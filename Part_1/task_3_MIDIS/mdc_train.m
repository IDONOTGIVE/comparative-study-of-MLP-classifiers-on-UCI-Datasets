function [prototypes ] = mdc_train(xtrain,ytrain)

%Classify using minimum distance algorithm
%Input
%xtrain         the 80%input  
%ytrain         the 20%input
%
%Output
%prototypes     the mean vectors for each of teh classes

%caculate the mean values of xtrain
w1 =mean(xtrain(:,ytrain == 1),2);
w2 =mean(xtrain(:,ytrain == 2),2);
w3 =mean(xtrain(:,ytrain == 3),2);
w4 =mean(xtrain(:,ytrain == 4),2);
w5 =mean(xtrain(:,ytrain == 5),2);
w6 =mean(xtrain(:,ytrain == 6),2);
w7 =mean(xtrain(:,ytrain == 7),2);

%store mean value of xtrain into matrix
prototypes = [w1,w2,w3,w4,w5,w6,w7];

end


    
    
    
    






