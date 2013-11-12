function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% hypothesis using the sigmoid function to predict whether h-theta(x) is >1 or < 0
hypothesis = sigmoid(X * theta);

% regularization term for hypothesis
reg1 = lambda/(2*m) * sum(theta .^2);
reg2 = lambda/(2*m) * (theta(1) .^2);
reg3 = reg1 - reg2;

% cost function for all training data using the hypothesis
J = (-1/m) * sum(y .* log(hypothesis) + (1 - y) .* log(1 - hypothesis)) + reg3;

% regularization term for the gradient descent
reg4 = (lambda/m) .* theta;
reg4(1) = 0;

% gradient decent to minimize theta-J(theta)
grad = (1/m) * (X' * (hypothesis - y)) + reg4;




% =============================================================

end
