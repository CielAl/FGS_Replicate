function [cls_dets,net,nnOpt,im_1] = project515_demo(filename,class,GPU)
	 if (nargin<1)
        filename = '000005.jpg';
        class = {'person'};
        GPU = [1];
     elseif nargin<2   
        class = {'person'};
        GPU = [1];
     elseif nargin<3
         GPU = [1];
     end
     
    [net,nnOpt] = getRCNN(class,GPU);
   
  %% Generate edge boxes
     [boxModel,boxOpt] = EB_Model();
  tic;
    imD = imread(filename);
    boxes=edgeBoxesWrapper(imD,boxModel,boxOpt);
  toc;
     tic;
    [cls_dets,labels,gpBox,gpLabel] = predImageSingle(net,boxes,nnOpt,imD,1);   
    toc;
    figure;
    if size(gpBox(gpLabel==1,:),1)<=0
            class_name =' Nothing found';
    else
            class_name = class{1};
    end
   

    im_1 = bbox_draw(imD,gpBox(gpLabel==1,:)); 
        title(sprintf('Detections for class ''%s''', class_name)) ;
    figure;
   
       im_2 = bbox_draw(imD,cls_dets(labels==1,:)); 
            title(sprintf('No GP Detections for class ''%s''', class_name)) ;
end
%{
function [boxes]=  edgeBoxesWrapper(imD,boxModel,boxOpt)
imdb_gtBox{revID}(imdb_gtLabel{revID}==CLS_PERSON,:)
   %% Define the conventional formation of bounding box; 
      boxes=edgeBoxes(imD,boxModel,boxOpt);
      boxes = floor(boxes(:,1:4));
      boxes(:,3) = boxes(:,1)+boxes(:,3);
      boxes(:,4) = boxes(:,2)+boxes(:,4);
 end
%}
%{
function [net,nnOpt,net2] = getRCNN(class,GPU)
	if nargin < 2
		class = 'class';
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
end
%}

%{
function [nnOpt] = getDaggModel(class,GPU)
	if nargin < 2
		class = 'class';
		GPU = 1;
	end
	nnOpt.modelPath = '' ;
	nnOpt.classes = {class} ;
	nnOpt.gpu = [GPU] ;
	nnOpt.confThreshold = 0.5 ;
	nnOpt.nmsThreshold = 0.3 ;
	nnOpt = vl_argparse(nnOpt, {}) ;
	paths = {nnOpt.modelPath, ...
			 './fast-rcnn-vgg16-dagnn.mat', ...
			 fullfile(vl_rootnn, 'data', 'models', 'fast-rcnn-vgg16-pascal07-dagnn.mat'), ...
			 fullfile(vl_rootnn, 'data', 'models-import', 'fast-rcnn-vgg16-pascal07-dagnn.mat')} ;
	ok = min(find(cellfun(@(x)exist(x,'file'), paths))) ;

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
%}
%{
function [boxModel,boxOpt] = EB_Model()

	%% load pre-trained edge detection model and set opts (see edgesDemo.m)
	boxModel=load('models/forest/modelBsds'); 
	boxModel=boxModel.model;
	boxModel.opts.multiscale=0; 
	boxModel.opts.sharpen=2; 
	boxModel.opts.nThreads=4;

	%% set up opts for edgeBoxes (see edgeBoxes.m)
	boxOpt = edgeBoxes();
	boxOpt.alpha = .65;     % step size of sliding window search
	boxOpt.beta  = .90;     % nms threshold for object proposals
	boxOpt.minScore = .02;  % min score of boxes to detect
	boxOpt.maxBoxes = 500;  % max number of boxes to detect
	boxOpt.gamma = 1.5;
	boxOpt.eta=.9996;
end
%}
%{
function [res_Box,imo] = predImageSingle(net,boxes,nnOpt,imD)
% Load a test image and candidate bounding boxes.
	opts = nnOpt;
	if nargin<5
		imD = imread( '000005.jpg');
	end
	%imD = imread(filename);
	im = single(imD) ;
	imo = im; % keep original image
	%boxes = load('000004_boxes.mat') ;%Changed
	boxes = single(boxes') + 1 ;

	boxeso = boxes - 1; % keep original boxes

	% Resize images and boxes to a size compatible with the network.
	imageSize = size(im) ;
	fullImageSize = net.meta.normalization.imageSize(1) ...
		/ net.meta.normalization.cropSize ;
	scale = max(fullImageSize ./ imageSize(1:2)) ;
	im = imresize(im, scale, ...
				  net.meta.normalization.interpolation, ...
				  'antialiasing', false) ;
	boxes = bsxfun(@times, boxes - 1, scale) + 1 ;

	% Remove the average color from the input image.
	imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;

	% Convert boxes into ROIs by prepending the image index. There is only
	% one image in this batch.
	rois = [ones(1,size(boxes,2)) ; boxes] ;

	% Evaluate network either on CPU or GPU.
	if numel(opts.gpu) > 0
	  gpuDevice(opts.gpu) ;
	  imNorm = gpuArray(imNorm) ;
	  rois = gpuArray(rois) ;
	  net.move('gpu') ;
	end

	net.conserveMemory = true ;
	net.eval({'data', imNorm, 'rois', rois});

	% Extract class probabilities and  bounding box refinements
	%squeeze(gather(res(end).x))
	probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
	deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ;

	% Visualize results for one class at a time

	for i = 1:numel(opts.classes)
	  c = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
	  cprobs = probs(c,:) ;
	  cdeltas = deltas(4*(c-1)+(1:4),:)' ;
	  cboxes = bbox_transform_inv(boxeso', cdeltas);
	  cls_dets = [cboxes cprobs'] ;

	  keep = bbox_nms(cls_dets, opts.nmsThreshold) ;
	  cls_dets = cls_dets(keep, :) ;

	  sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
	  if 1
		imo = bbox_draw(imo/255,cls_dets(sel_boxes,:));
	  end
	  title(sprintf('Detections for class ''%s''', opts.classes{i})) ;

	  fprintf('Detections for category ''%s'':\n', opts.classes{i});
	  for j=1:size(sel_boxes,1)
		bbox_id = sel_boxes(j,1);
		fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
				cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
				cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
				cls_dets(bbox_id,end));
	  end
	end
	%res_Box = cls_dets(sel_boxes,:);
     
    %[regBox]=GP_BoxReg(res_Box);
   % Box = [];
    Box = cls_dets(sel_boxes,:);
    net.move('gpu') ;
    net.conserveMemory = true ;
 for jj = 1 :20
        %gpuDevice([]);
        [regBox]=GP_BoxReg(Box,jj);
        if isempty(regBox)
           continue; 
        end
        regBox = single(regBox') + 1 ;
        boxeso = regBox-1';
    	regBox = bsxfun(@times, regBox - 1, scale) + 1 ;
        imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;
    	rois = [ones(1,size(regBox,2)) ; regBox] ;
        if numel(opts.gpu) > 0
          imNorm = gpuArray(imNorm) ;
          rois = gpuArray(rois) ;
        end
        net.eval({'data', imNorm, 'rois', rois}); 
        probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
        deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ; 
        %reset(gpuDevice(opts.gpu));
    	for i = 1:numel(opts.classes)
          c = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
          cprobs = probs(c,:) ;
          cdeltas = deltas(4*(c-1)+(1:4),:)' ;
          cboxes = bbox_transform_inv(boxeso', cdeltas);
          cls_dets = [cboxes cprobs'] ;
          keep = bbox_nms(cls_dets, opts.nmsThreshold) ;
          cls_dets = cls_dets(keep, :) ;
          sel_boxes = find(cls_dets(:,end) >= opts.confThreshold) ;
        end
        
    Box = [Box;cls_dets(sel_boxes,:)];
    
 end
    keep = bbox_nms(Box, opts.nmsThreshold)
    res_Box = Box(keep,:);
end
%}
%{
function min_func = getMin()
    Solver_Timeout = 10;
    minFunc_Method = 'lbfgs';
    minFuncX_OPTS = struct();
    minFuncX_OPTS.timeout = Solver_Timeout;
    minFuncX_OPTS.Display = 'off';
    minFuncX_OPTS.Method  = minFunc_Method;
    min_func = @(func,x0) minFuncX( func,x0, minFuncX_OPTS);
end
%}
%{
function [regBox]=GP_BoxReg(Box,Idx) %%Assume x1y1x2y2p
    if (size(Box,1))<=0
        regBox = [];
        return;
    end
    min_func = getMin();
    %% Convert Box
    newBox = [Box(:,2) Box(:,1) Box(:,4) Box(:,3)];
    fN = Box(:,5);
    %%
    [GPModel22] = InitModel();
    PsiN1 = bbox_ltrb2param( newBox, 'yxhwl').'; %% Must transpose: column as bbox
     latent_obj = @(z) sgp_negloglik( GPmodel(c), z, PsiN1, fN );
       z0 = 0;
       try
            z_hat = min_func( latent_obj, z0);
       catch
          % warning( 'Optimization on z is failed' );
          z_hat = 0;
       end
        expnz = exp(-z_hat);
        if Idx> size(Box,1)
           Idx = size(Box,1); 
        end
           PsiN = PsiN1;
           PsiN = PsiN.*expnz;
         KN = sgp_cov( GPModel22, 0, PsiN ); %% Non-rescaled
         fN_hat = max(fN);
         psiNp1_0 = PsiN1(:,Idx);
         search_obj = @(psiNp1) sgp_neg_acquisition_ei( GPModel22, ...
                    psiNp1, PsiN, fN, fN_hat, KN );
         psiNp1_hat = min_func( search_obj, psiNp1_0 );
         %PsiN = [PsiN psiNp1_hat]; % 
         %PsiN = PsiN/expnz;
         %regBox = psiNp1_hat;Debug
         %regBox = bbox_param2ltrb(PsiN','yxhwl'); %Transpose back in the input
         regBox = bbox_param2ltrb(psiNp1_hat','yxhwl'); %Transpose back in the input
         regBox = [regBox(:,2) regBox(:,1) regBox(:,4) regBox(:,3)];

end
%}
%{
function [GPModel22] = InitModel()
    meanfunc = @meanConst;
    hyp.mean = 0;
    covfunc = @covSEard; 
    ell = 2.0; sf = 1.0; 
    likfunc = @likGauss; hyp.lik = log(0.1);    
    hyp.cov = log([ell ell ell ell sf]);
       % hyp = minimize.....
    GPModel22 = sgp_model_from_general(hyp);
 
end
%}
%{
function [net] = getRCNN_Net2(net2)

	% Load the network and put it in test mode.
	net = dagnn.DagNN.loadobj(net2);
	net.mode = 'test' ;

	% Mark class and bounding box predictions as `precious` so they are
	% not optimized away during evaluation.
	net.vars(net.getVarIndex('cls_prob')).precious = 1 ;
	net.vars(net.getVarIndex('bbox_pred')).precious = 1 ;
end
%}
