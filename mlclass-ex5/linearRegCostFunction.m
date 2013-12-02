function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% use X and theta to evaluate hypothesis on all m examples
hypothesis = X * theta;

% uses predictions and y to determine squared errors
sqrErrors = (hypothesis - y) .^2;

J = 1/(2*m) * sum(sqrErrors);

% regularization term for the gradient descent
reg_term = (lambda/(2*m)) * sum(theta .^2);
reg_term0 = (lambda/(2*m)) * sum(theta(1) .^2);
reg_term = reg_term - reg_term0;
J = J + reg_term;

% gradient decent to minimize theta-J(theta)
grad_reg_term = (lambda/m) * theta;
grad_reg_term(1) = 0;

grad = (1/m) * (X' * (hypothesis - y)) + grad_reg_term;



% =========================================================================

grad = grad(:);

end
