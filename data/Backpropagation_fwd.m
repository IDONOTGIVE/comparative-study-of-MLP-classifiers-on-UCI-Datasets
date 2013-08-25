
function ypred = Backpropagation_fwd(Whidden, Wout, test_features)

% funtion Backpropagation_fwd is prepared to use for the back-propagation algorithms 
% implemented in the book "Computer Manual in MATLAB to accompany Pattern Classification"
%  written by Bailing Zhang

[Ni, M] = size(test_features); % M is number of samples
nout = length(Wout);


ypred=[];
for m = 1:M
    for n = 1:nout
        eval(['Wo= Wout{' num2str(n) '};'])
        eval(['Wh= Whidden{' num2str(n) '};'])
        Xm = test_features(:,m);
        %tk = test_targets(m);
        
        %Forward propagate the input:
        %First to the hidden units
        gh	= Wh*[Xm; 1];
        y	= activation(gh);
        %Now to the output unit
        go	= Wo*[y; 1];
        zk	= activation(go);
        out(n,1)=zk;
    end
    ypred=[ypred out];
end      
        
        
function f = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
%df	= a*b*sech(b*x).^2; 