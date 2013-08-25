function result = autoencoder_test(W, test_features, beta)

CNum = length(W);
err(1,CNum) = 0;
Mt = size(test_features,2);
result(1,Mt) = 0;

for i = 1:Mt
    x  = test_features(:,i);
    for k = 1:CNum
        w  = eval(['W{' num2str(k) '}']);
        %beta is the parameter for the simoidal function;
        net = beta*w'*x;
        y = 1./(1+exp(-net));
        err(1,k) = reconstruction_err(x,w*y);
    end
    result(1,i) = find(err == min(err));    
end

function err = reconstruction_err(before, after)
err = sum((before-after).^2)/length(before);

