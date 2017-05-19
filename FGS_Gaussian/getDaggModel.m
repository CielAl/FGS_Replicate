function [nnOpt] = getDaggModel(class,GPU,paths)
% [nnOpt] = getDaggModel(class,GPU,path)
% class: String, name of target class
% GPU: [] If using CPU, positive Int otherwise
% paths: pretrained RCNN. A cell {...,...,...} that contains all possible paths 
% of models.
    if nargin < 2
		class = {'person'};
		GPU = 1;
    end
    if ~iscell(class)
       class = {class}; 
    end
	nnOpt.modelPath = '' ;
	nnOpt.classes = class ;
	nnOpt.gpu = [GPU] ; %#ok<NBRAK>
	nnOpt.confThreshold = 0.5 ;
	nnOpt.nmsThreshold = 0.3 ;
	nnOpt = vl_argparse(nnOpt, {}) ;
    if nargin< 3
        paths = {nnOpt.modelPath, ...
                 './fast-rcnn-vgg16-dagnn.mat', ...
                 fullfile(vl_rootnn, 'data', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat'), ...
                 fullfile(vl_rootnn, 'data', 'models-import', 'fast-rcnn-vgg16-pascal07-dagnn.mat'),...
                 'fast-rcnn-vgg16-pascal07-dagnn.mat', './fast-rcnn-vgg16-pascal07-dagnn.mat'
                 } ;
    end
         %ok = min(find(cellfun(@(x)exist(x,'file'), paths))) ;
        ok = find(cellfun(@(x)exist(x,'file'), paths), 1 ) ;
	if isempty(ok)
	  fprintf('Downloading the Fast RCNN model ... this may take a while\n') ;
	  nnOpt.modelPath = fullfile(vl_rootnn, 'data', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat') ;
	  mkdir(fileparts(nnOpt.modelPath)) ;
	  urlwrite('http://www.vlfeat.org/matconvnet/models/fast-rcnn-vgg16-pascal07-dagnn.mat', ...
			   nnOpt.modelPath) ;
	else
	  nnOpt.modelPath = paths{ok} ;
	end
end