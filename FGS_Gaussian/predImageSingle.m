function [baseBox,Labels,gpBox,gpLabel] = predImageSingle(net,boxes,nnOpt,imD,useGP)
%% Intiate argins.
	opts = nnOpt;
    if nargin<5
       useGP = 1; 
    elseif nargin<4
		imD = imread( '000005.jpg');
        useGP = 1; 
    end
    %% Process the image as single-precision
	im = single(imD) ;
	%imo = im; % keep original image for print
	%% Resize images and boxes to a size compatible with the network.
	imageSize = size(im) ;
	fullImageSize = net.meta.normalization.imageSize(1) ...
		/ net.meta.normalization.cropSize ;
	scale = max(fullImageSize ./ imageSize(1:2)) ;
	im = imresize(im, scale, ...
				  net.meta.normalization.interpolation, ...
				  'antialiasing', false) ;
    %% Calculate Features and original boxes (after rcnn)
    [boxeso,probs,deltas] = rcnn_Features(boxes,net,im,scale);
    [box_det,~,Labels] = extractBox(opts,net,boxeso,probs,deltas);

    %% return if no GP. 
    baseBox = cell2mat(box_det);
    Labels = cell2mat(Labels);
         keep_label = bbox_nms(baseBox, opts.nmsThreshold) ;
          baseBox = baseBox(keep_label, :) ;
          Labels = Labels(keep_label,:);
    if ~useGP
       % sel_boxes = cell2mat(sel_boxes);      
        gpBox = [];
        gpLabel = [];
        return; 
    end
    %% Sector of GP

 [refineBoxes_det,rawLabels] = FGSWrapper(box_det,net,scale,opts,im)  ;  
    gpBox = cell2mat(refineBoxes_det);
    gpLabel = cell2mat(rawLabels);
end

%% Sub process
%{
function [cls_dets,sel_boxes,Labels] = extractBox(opts,net,boxeso,probs,deltas,prtFlag,imo)
if nargin <6
   prtFlag = 0;
   imo = [];
end
classLen = numel(opts.classes);
[c, cprobs, cdeltas, cboxes , keep] = deal(cell(classLen,1));
[cls_dets, sel_boxes ,Labels] = deal(cell(classLen,1));
    for i = 1:classLen
          c{i} = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
          cprobs{i} = probs(c{i},:) ;
          cdeltas{i} = deltas(4*(c{i}-1)+(1:4),:)' ;
          cboxes{i} = bbox_transform_inv(boxeso', cdeltas{i});
          cls_dets{i} = [cboxes{i} cprobs{i}'] ;
          keep{i} = bbox_nms(cls_dets{i}, opts.nmsThreshold) ;
          cls_dets{i} = cls_dets{i}(keep{i}, :) ;
          sel_boxes{i} = find(cls_dets{i}(:,end) >= opts.confThreshold) ;
          cls_dets{i} = cls_dets{i}(sel_boxes{i}, :) ;
          Labels{i} = i*ones(size(sel_boxes{i},1),1);
    end
    if prtFlag
        for i = 1:classLen
          printProposal(imo,cls_dets,sel_boxes,opts.classes{i}) 
          printRes(sel_boxes,cls_dets);
        end 
    end
end
%}

%% Utility functions


%% code Log
%boxes = load('000004_boxes.mat') ;%Changed  
%{
% Replace those by [....] = cell(classLen,1);
c = cell(classLen,1);
cprobs = cell(classLen,1);
cdeltas = cell(classLen,1);
cboxes = cell(classLen,1);
cls_dets = cell(classLen,1);
keep = cell(classLen,1);
sel_boxes = cell(classLen,1);
Labels = cell(classLen,1);
%}
    
 %{
% rcnn_Features
       boxes = single(boxes') + 1 ;
	boxeso = boxes - 1; % keep original boxes
	boxes = bsxfun(@times, boxes - 1, scale) + 1 ;

	% Remove the average color from the input image.
	imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;
	% Convert boxes into ROIs by prepending the image index. There is only
	% one image in this batch.
	rois = [ones(1,size(boxes,2)) ; boxes] ;

	% Evaluate network either on CPU or GPU.
	if numel(opts.gpu) > 0
	  %gpuDevice(opts.gpu) ; %Caution: Init in getRCNN
	  imNorm = gpuArray(imNorm) ;
	  rois = gpuArray(rois) ;
	  net.move('gpu') ;
	end

	net.conserveMemory = true ; % Prevent "Insufficient Graphic memory" 
	% Get scores
    net.eval({'data', imNorm, 'rois', rois});

	% Extract class probabilities and  bounding box refinements
	probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
	deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ; 
    %}

%{
 for jj = 1 :20
        [regBox]=GP_BoxReg(Box,jj);
        if isempty(regBox)
           continue; 
        end
         [boxeso,probs,deltas] = rcnn_Features(regBox,net,im,scale);

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
    keep = bbox_nms(Box, opts.nmsThreshold);
    gpBox = Box(keep,:);
%}