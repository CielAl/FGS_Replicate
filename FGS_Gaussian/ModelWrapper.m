function [net,nnOpt,boxModel,boxOpt] = ModelWrapper(class,GPU)
% A wrapper to get Dagnn and EdgeBox model
% [net,nnOpt,boxModel,boxOpt] = getFGS_Model(class,GPU)
% Input:
%       class,GPU
% Output:
%       [net,nnOpt] = getRCNN(class,GPU);
%       [boxModel,boxOpt] = EB_Model();
%       [net,nnOpt,boxModel,boxOpt]
    [net,nnOpt] = getRCNN(class,GPU);
    [boxModel,boxOpt] = EB_Model();
end