function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
display(theta);
display('entering for loop');
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    t_theta = zeros(size(theta));
    %display(t_theta);
    diff = (X * theta) - y;
    for i = 1 : size(theta, 1)
        t_theta(i, 1) = theta(i, 1) - (alpha * (1 / m) * sum(diff .* X(:,i)));
    end;
    theta = t_theta;
    %display(size(t_theta));
    %display(t_theta);
    %display(' gradientDescent ================================');
    %display(computeCost(X, y, theta));
    %display("gradientDescent ================================");
    %display(J_history(iter));


    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
