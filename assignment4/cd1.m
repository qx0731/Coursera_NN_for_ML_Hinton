function ret = cd1(rbm_w, visible_data)
visible_data = sample_bernoulli(visible_data);
% <rbm_w> is a matrix of size <number of hidden units> by <number of visible units>
% <visible_data> is a (possibly but not necessarily binary) matrix of size <number of visible units> by <number of data cases>
% The returned value is the gradient approximation produced by CD-1. It's of the same shape as <rbm_w>.
%    error('not yet implemented');
hidden=sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w,visible_data));
visible2=sample_bernoulli(hidden_state_to_visible_probabilities(rbm_w,hidden));
%hidden2=sample_bernoulli(visible_state_to_hidden_probabilities(rbm_w,visible2));
hidden2=visible_state_to_hidden_probabilities(rbm_w,visible2);
ret=configuration_goodness_gradient(visible_data,hidden);
ret=ret-configuration_goodness_gradient(visible2,hidden2);
end
