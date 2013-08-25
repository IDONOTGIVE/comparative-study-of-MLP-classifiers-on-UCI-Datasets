function y = mdc_test(xtest, prototypes)

%Classify using minimum distance algorithm
%Input
%xtest         the unkown classcenter
%prototypes    the known classcenter
%
%Output
%y             the difference between known points and unknown points

result=[];
for n=1:size(xtest,2);
    for m=1:7
    result(n,m)=xtest(:,n)'*prototypes(:,m)-0.5*prototypes(:,m)'*prototypes(:,m);
    [C,y]=max(result,[],2);% caculate the max value of result to minimun the distance
    end
        
end
end
    
