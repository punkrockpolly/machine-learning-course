function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % evaluate hypothesis vector for all m examples
    hypothesis = X * theta;

    % calculate delta vector for all hypothesis, y and alpha
    delta1 = (1/m) * (X' * (hypothesis - y));

    % correct answer, but with warning: automatic broadcasting operation applied
    %delta2 = (1/m) * sum((hypothesis - y) .* X);
    %delta2 = delta2';

    % simultaneous update for the vector theta
    theta = theta - (alpha * delta1);


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
