function [boxes]=  edgeBoxesWrapper(imD,boxModel,boxOpt)
   %% Define the conventional formation of bounding box; 
      boxes=edgeBoxes(imD,boxModel,boxOpt);
      boxes = floor(boxes(:,1:4));
      boxes(:,3) = boxes(:,1)+boxes(:,3);
      boxes(:,4) = boxes(:,2)+boxes(:,4);
 end