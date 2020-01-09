function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

% logistic regression prediction is sigmoid of linear regression hypothesis
h = sigmoid(X * theta); % m x n * n x 1
% 

% Cost according to lecture notes
J = (1/m) * ( -y'*log(h) - (1-y)'*log(1-h) );

errs = h - y;  % m x 1 - m x 1 => m x 1
% 1 x m * m x n => 1 x n --> (1xn)' => n x 1
grad = 1/m * (errs' * X)'; % need an n x 1 vector to match theta
% No alpha or lambda??

% =============================================================

end
