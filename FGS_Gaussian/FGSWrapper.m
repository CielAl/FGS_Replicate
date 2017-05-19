function [rawBox,rawLabels] = FGSWrapper(box_det,net,scale,opts,im)   
 rawBox = box_det;
 classLen = numel(opts.classes);
[c, cprobs, cdeltas, cboxes , keep] = deal(cell(classLen,1));
[cls_dets, sel_boxes ,rawLabels,batchBuff] = deal(cell(classLen,1));

iterLimit = 4; %min(4,size(rawBox{1},1));%*size(rawBox{1},1);
    for i = 1:classLen 
        for gpRound = 1:iterLimit
                batchSize = size(rawBox{i},1);
                 batchBuff{i} = [];
             for jj = 1 :batchSize
                 try
                    [regBox]=GP_BoxReg(rawBox{i},jj);
                    batchBuff{i}= [batchBuff{i};regBox];
                 catch                  
                      batchBuff{i}= [batchBuff{i};[]];
                 end
             end
                 [boxeso,probs,deltas] = rcnn_Features(batchBuff{i},net,im,scale);
                 if isempty(boxeso)
                  
                    continue;
                 end
                 
                  c{i} = find(strcmp(opts.classes{i}, net.meta.classes.name)) ;
                  cprobs{i} = probs(c{i},:) ;
                  cdeltas{i} = deltas(4*(c{i}-1)+(1:4),:)' ;
                  cboxes{i} = bbox_transform_inv(boxeso', cdeltas{i});
                  cls_dets{i} = [cboxes{i} cprobs{i}'] ;
                  sel_boxes{i} = find(cls_dets{i}(:,end) >= opts.confThreshold) ;
                  cls_dets{i} = cls_dets{i}(sel_boxes{i}, :) ;
                  rawBox{i} = [rawBox{i};cls_dets{i}];
                  keep{i} = bbox_nms(rawBox{i}, opts.nmsThreshold) ;
                  rawBox{i} = rawBox{i}(keep{i}, :) ;
        end

        rawLabels{i} = i*ones(size(rawBox{i},1),1);
    end
end