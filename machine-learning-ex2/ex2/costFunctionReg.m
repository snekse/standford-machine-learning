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

featureTheta = theta(2:end,:);
calcTheta = [0;featureTheta];

% logistic regression prediction is sigmoid of linear regression hypothesis
h = sigmoid(X * theta); % m x n * n x 1
% 'sigmoid(X * theta)'
% h

% Cost according to lecture notes
J = (1/m) * ( -y'*log(h) - (1-y)'*log(1-h) );
% J;
regFactor = (lambda/(2*m)) * (calcTheta' * calcTheta); % ignore first row
regFactor;
J = J + regFactor;
% J;
errs = h - y;  % m x 1 - m x 1 => m x 1
% 1 x m * m x n => 1 x n --> (1xn)' => n x 1
grad = 1/m * (errs' * X)'; % need an n x 1 vector to match theta
grad = grad + ((lambda/m)*calcTheta);
% No alpha or lambda??

% =============================================================

end
