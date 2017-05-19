function [net,nnOpt,net2] = getRCNN(class,GPU)
if nargin < 2
		class = {'class'};
		GPU = 1;
end
	nnOpt = getDaggModel(class,GPU);
	% Load the network and put it in test mode.
	net2 = load(nnOpt.modelPath) ;
	net = dagnn.DagNN.loadobj(net2);
	net.mode = 'test' ;

	% Mark class and bounding box predictions as `precious` so they are
	% not optimized away during evaluation.
	net.vars(net.getVarIndex('cls_prob')).precious = 1 ;
	net.vars(net.getVarIndex('bbox_pred')).precious = 1 ;
        if numel(nnOpt.gpu) > 0
            gpuDevice(nnOpt.gpu);
            net.move('gpu');
        end
    

end