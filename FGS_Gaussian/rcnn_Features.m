function [boxeso,probs,deltas] = rcnn_Features(regBox,net,im,scale)
% Adapted from code of vlfeat: MatConvNet.
% [boxeso,probs,deltas] = rcnn_Features(regBox,net,im,scale)
% Calculate the feature of bboxes via RCNN.
% Input:
%     regBox: Proposals. Rows defined as [x1 y1 x2 y2]
%     net: RCNN net obj
%     im: image
%     scale: scaling factor. scale = max(fullImageSize ./ imageSize(1:2)) ;
% Output:
%          boxeso: Original boxes without transformation.
%          probs: Probability score of box for all classes.
%           deltas: bbox_pred. ??? Maybe for boxregression
        regBox = single(regBox') + 1 ;
        boxeso = regBox-1';
    	regBox = bsxfun(@times, regBox - 1, scale) + 1 ;
        imNorm = bsxfun(@minus, im, net.meta.normalization.averageImage) ;
    	rois = [ones(1,size(regBox,2)) ; regBox] ;
        if strcmp(net.device ,'gpu')> 0
          imNorm = gpuArray(imNorm) ;
          rois = gpuArray(rois) ;
        end
        net.eval({'data', imNorm, 'rois', rois}); 
        probs = squeeze(gather(net.vars(net.getVarIndex('cls_prob')).value)) ;
        deltas = squeeze(gather(net.vars(net.getVarIndex('bbox_pred')).value)) ; 
end