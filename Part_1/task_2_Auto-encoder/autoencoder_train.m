function W = autoencoder_train(train_features, train_targets, params)

% Nh: the number of hidden units;
% beta: the parameter in sigmoidal function;

[Nh, beta] = process_params(params);
[Ni, Mi] = size(train_features);
Nt = size(train_targets, 1);

% There is Nt classes in targets;
for i = 1:Nt
    eval(['w' num2str(i) ' = rand(Ni, Nh);']);
end 

% number of times the training data will be presented to train                                  
notrial = 2; 

% minimum learning rate                         
mu_min=0.2; 

% maximum learning rate  
mu_max=1;  
                      
count=0; trial =0;
while trial < notrial
    trial = trial +1;
    for item = 1:Mi          % total training samples
        count = count +1;
        pr = notrial*Mi/count -1;
        eta = (mu_min+pr*mu_max)/(1+pr); % learning rate
        
        % Reading a training sample and present it to autoencoder here:       
        x  = train_features(:,item); 
        No = find(train_targets(:,item),1);
		
        %['w' num2str(No) ] is used to store weight vectors for each class;
        w  = eval(['w' num2str(No) ]);  
		
        %beta is the parameter for the simoidal function;
        net = beta*w'*x;       
        y = 1./(1+exp(-net)); 
        g = beta*y.*(1-y);    
        m=0;
		
        %M is the number of hidden units;
        while m<Nh
            m  = m+1;  
            er = x-y(m)*w(:,m);              
            dw = er'*w(:,m)*g(m)*x+er*y(m);  
            v  = w(:,m)+ eta*dw; 
            w(:,m) = v;           
        end
        eval(['w' num2str(No) '=w;'])      
        
    end
end

W = cell(Nt,1);
for i = 1:Nt
    W(i) = eval(['{w' num2str(i) '}']);
end 