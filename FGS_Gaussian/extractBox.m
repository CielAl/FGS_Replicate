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
          sel_boxes{i} = find(cls_dets{i}(:,end) >= opts.confThreshold) ;
          cls_dets{i} = cls_dets{i}(sel_boxes{i}, :) ;
       %   keep{i} = bbox_nms(cls_dets{i}, opts.nmsThreshold) ;
          %cls_dets{i} = cls_dets{i}(keep{i}, :) ;
          Labels{i} = i*ones(size(sel_boxes{i},1),1);
    end
    if prtFlag
        for i = 1:classLen
          printProposal(imo,cls_dets,sel_boxes,opts.classes{i}) 
          printRes(sel_boxes,cls_dets);
        end 
    end
end
function printRes(sel_boxes,cls_dets)
        for j=1:size(sel_boxes,1)
		bbox_id = sel_boxes(j,1);
		fprintf('\t(%.1f,%.1f)\t(%.1f,%.1f)\tprobability=%.6f\n', ...
				cls_dets(bbox_id,1), cls_dets(bbox_id,2), ...
				cls_dets(bbox_id,3), cls_dets(bbox_id,4), ...
				cls_dets(bbox_id,end));
         end
end
function [imo] = printProposal(imo,cls_dets,sel_boxes,class_name)
   figure;
   imo = bbox_draw(imo/255,cls_dets(sel_boxes,:));
   title(sprintf('Detections for class ''%s''', class_name)) ;
   fprintf('Detections for category ''%s'':\n', class_name);
end