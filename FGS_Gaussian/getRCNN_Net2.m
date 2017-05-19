function [net] = getRCNN_Net2(net2)
% [net] = getRCNN_Net2(net2)
% Input: 
%       - net2: The model loaded from load ('xxx.mat') of RCNN;
% Output:
%       - net: Wrapped Dagnn Model.
%Load the network and put it in test mode.
	net = dagnn.DagNN.loadobj(net2);
	net.mode = 'test' ;

	% Mark class and bounding box predictions as `precious` so they are
	% not optimized away during evaluation.
	net.vars(net.getVarIndex('cls_prob')).precious = 1 ;
	net.vars(net.getVarIndex('bbox_pred')).precious = 1 ;
end