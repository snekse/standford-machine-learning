function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
learning_weight = alpha * (1/m);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    currentCost = computeCost(X, y, theta);

    h = X * theta; % m x n * n x 1 => m x 1
    errs = h - y;  % m x 1 - m x 1 => m x 1
    % partial deviation adjustment
    % 1 x m * m x n => 1 x n => (1xn)' => n x 1 
    g = learning_weight * (errs' * X)'; % need an n x 1 vector to match theta
    
    % fprintf('Cost is %f\n', currentCost);
    % fprintf('Gradient is %f\n', g);

    theta = theta - g;
 
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
