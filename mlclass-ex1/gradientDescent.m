function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
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
    %       of the cost function (computeCost) and gradient here.
    %


    theta_zero = theta(1);
    theta_one = theta(2);

    % use X and theta to evaluate hypothesis on all m examples
    predictions = X * theta;

    % simultaneous update for the vector theta
    theta_zero = theta_zero - alpha * (1/m) * sum(predictions - y);
    theta_one = theta_one - alpha * (1/m) * sum((predictions - y) .* X(:,2));

    % assign new theta values to theta
    theta(1) = theta_zero;
    theta(2) = theta_one;


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

    %plot(iter, J_history);

end

end

%plot(iter, J_history);