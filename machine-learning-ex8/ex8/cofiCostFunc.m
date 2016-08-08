function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%
J = (1 / 2) * sum(((X * Theta' - Y) .^ 2)(R == 1)) + (lambda / 2) * sum(sum(Theta .^ 2)) + (lambda / 2) * sum(sum(X .^ 2));
for i = 1 : size(X, 1)
	for j = 1 : size(X, 2)
		for k = 1 : size(Theta, 1)
			if R(i,k) == 1
				X_grad(i, j) += (X(i, :) * Theta(k, :)' - Y(i, k)) * Theta(k, j);
			end
		end
		X_grad(i, j) += lambda * X(i, j);
	end
end
for i = 1 : size(Theta, 1)
	for j = 1 : size(Theta, 2)
		for k = 1 : size(X, 1)
			if R(k, i) == 1
				Theta_grad(i, j) += (Theta(i, :) * X(k, :)' - Y(k, i)') * X(k, j);
			end
		end
		Theta_grad(i, j) += lambda * Theta(i, j);	
	end
end
% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
