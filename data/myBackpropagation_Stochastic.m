function  [Whidden, Wout]= myBackpropagation_Stochastic(train_features, train_targets, params)

% Classify using a backpropagation network with stochastic learning algorithm
% Inputs:
% 	features- Train features
%	targets	- Train targets
%	params   - Number of hidden units, Convergence criterion, Convergence rate
%
% Outputs
%   Whidden      - Collection of Hidden unit weights Wh
%   Wout         - Collection of Output unit weights Wo

% This function was revised from Backpropagation_Stochastic provided in 
% book "Computer Manual in MATLAB to accompany Pattern Classification"
% by Bailing Zhang, Oct 2010

[Nh, No, Theta, eta] = process_params(params);
iter	= 0;

[Ni, M] = size(train_features);
targets = train_targets; 
clear train_targets;

numOut = 0;
while numOut < No,
      numOut = numOut + 1;
      disp(['Training for output node ' num2str(numOut) ':'])
    
      train_targets = targets(numOut,:)';
      means	        = mean(train_features')';
      train_features  = train_features - means*ones(1,M);

     %Initialize the net: In this implementation there is only one output unit, so there
     %will be a weight vector from the hidden units to the output units, and a weight matrix
     %from the input units to the hidden units.
     %The matrices are defined with one more weight so that there will be a bias
      w0		= max(abs(std(train_features')'));
      Wh		= rand(Nh, Ni+1).*w0*2-w0; %Hidden weights
      Wo		= rand(1, Nh+1).*w0*2-w0; %Output weights

      Wo    = Wo/mean(std(Wo'))*(Nh+1)^(-0.5);
      Wh    = Wh/mean(std(Wh'))*(Ni+1)^(-0.5);

      rate	= 10*Theta;
      J       = 1e3;

      while (rate > Theta),
         %Randomally choose an example
         i	= randperm(M);
         m	= i(1);
         Xm = train_features(:,m);
         tk = train_targets(m);
    
         %Forward propagate the input:
         %First to the hidden units
         gh				= Wh*[Xm; 1];
         [y, dfh]		= activation(gh);
         %Now to the output unit
         go				= Wo*[y; 1];
         [zk, dfo]	= activation(go);
    
         %Now, evaluate delta_k at the output: delta_k = (tk-zk)*f'(net)
         delta_k		= (tk - zk).*dfo;
    
         %...and delta_j: delta_j = f'(net)*w_j*delta_k
         delta_j		= dfh'.*Wo(1:end-1).*delta_k;
    
         %w_kj <- w_kj + eta*delta_k*y_j
         Wo			= Wo + eta*delta_k*[y;1]';
    
         %w_ji <- w_ji + eta*delta_j*[Xm;1]
         Wh			= Wh + eta*delta_j'*[Xm;1]';
    
         %Calculate total error
         OldJ = J;
         J    = 0;
         for i = 1:M,
            J = J + (train_targets(i) - activation(Wo*[activation(Wh*[train_features(:,i); 1]); 1])).^2;
         end
         J = J/M; 
         rate  = abs(J - OldJ)/OldJ*100;
    
         iter 			= iter + 1;
         if (iter/100 == floor(iter/100)),
            disp(['Iteration ' num2str(iter) ': Total error is ' num2str(J)])
         end
    end
    %disp(['Backpropagation converged after ' num2str(iter) ' iterations.'])
    eval(['Whidden{' num2str(numOut) '}=Wh;']);
    eval(['Wout{' num2str(numOut) '}=Wo;']);
    
end


function [f, df] = activation(x)

a = 1.716;
b = 2/3;
f	= a*tanh(b*x);
df	= a*b*sech(b*x).^2;