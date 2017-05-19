function [boxModel,boxOpt] = EB_Model()
%% [boxModel,boxOpt] = EB_Model()
%% load pre-trained edge detection model and set opts (see edgesDemo.m)
	boxModel=load('models/forest/modelBsds'); 
	boxModel=boxModel.model;
	boxModel.opts.multiscale=0; 
	boxModel.opts.sharpen=2; 
	boxModel.opts.nThreads=4;

	%% set up opts for edgeBoxes (see edgeBoxes.m)
	boxOpt = edgeBoxes();
	boxOpt.alpha = .65;     % step size of sliding window search
	boxOpt.beta  = .75;     % nms threshold for object proposals
	boxOpt.minScore = .02;  % min score of boxes to detect
	boxOpt.maxBoxes = 1000;  % max number of boxes to detect
	boxOpt.gamma = 1.5;
	boxOpt.eta=.9996;
end