function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% --------------------------------------
% Forward propagation for the hypothesis
% --------------------------------------

% Add ones to the X (A1) data matrix
X = [ones(m, 1) X];

% Hidden Layer
Z2 = X * Theta1';
A2 = sigmoid(Z2);

% Add ones to the A2 data matrix
A2 = [ones(m, 1) A2];

% Output Layer (hypothesis = A3)
Z3 = A2 * Theta2';
A3 = sigmoid(Z3);


% --------------------------------------
% Regularization term for J
% --------------------------------------

reg_1a =  sum(sum(Theta1 .^2));
reg_1b = sum(Theta1(:,1) .^2);
reg_theta1 = reg_1a - reg_1b;

reg_2a = sum(sum(Theta2 .^2));
reg_2b = sum(Theta2(:,1) .^2);
reg_theta2 = reg_2a - reg_2b;

reg_term = lambda/(2*m) * (reg_theta1 + reg_theta2);


% --------------------------------------
% Cost Function
% --------------------------------------

% y(i)k : i-th row of the y column vector, 
% converted to a 10 vector representation of the digit
% if i-th row - y(i,:) = 5: y = [0000100000]

% size(A3)
y_matrix = zeros(size(A3'));

for inum=1:m
	y_matrix(:,inum) = eye(num_labels)(:,y(inum,:))';
end
	
J = sum(sum(-y_matrix .* log(A3') - (1 - y_matrix) .* log(1 - A3'))/m) + reg_term;

% --------------------------------------
% Back Propogation
% --------------------------------------

% for each node j in layer l, compute an “error term” δ(l)j
delta3 = zeros(size(A3));
delta3 = A3' - y_matrix;

delta2 = Theta2'(2:end,:) * delta3 .* sigmoidGradient(Z2');

% size(delta3)
% size(delta2)

% --------------------------------------
% Regularization term for Delta
% --------------------------------------

reg_1 = lambda/m * Theta1;
reg_1(:,1) = 0;

reg_2 = lambda/m * Theta2;
reg_2(:,1) = 0;

Theta1_grad = ((1/m) * delta2 * X) + reg_1;
Theta2_grad = ((1/m) * delta3 * A2) + reg_2;

% size(Theta1_grad)
% size(Theta2_grad)

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
