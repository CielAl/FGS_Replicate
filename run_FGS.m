

  %% ii->imageID->imdb.boxes
  Delta=cell(subsetSize,1);
  Res_Non = cell(subsetSize,1);
  Res_GP = cell(subsetSize,1);

  for (ii = 1:subsetSize) %parfor sucks with CUDA
     
    imD =ImageBatch_Tiny{ii};
    boxes=edgeBoxesWrapper(imD,boxModel,boxOpt);
    [cls_dets,~,gpBox,~] = predImageSingle(net,boxes,nnOpt,imD,1); 
    seqID = (imageID(shuffle(ii))) ;
    imName_Temp = [sprintf('%06d', seqID),'.jpg'];
    revID =  find(strcmp(imdb.images.name, [sprintf('%06d', 5120),'.jpg'])==1);
    revID = revID(1);