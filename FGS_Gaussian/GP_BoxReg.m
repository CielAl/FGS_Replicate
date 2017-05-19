function [regBox]=GP_BoxReg(Box,Idx,GPModel,gpml_struct) %%Assume x1y1x2y2p
    %% Check input
    if (size(Box,1))<=0
        regBox = [];
        return;
    elseif Idx> size(Box,1) || Idx<=0
           Idx = size(Box,1); 
    end
    if nargin < 3 
      [GPModel, gpml_struct] = InitGPmodel(); 
    end
    min_func = getMinFunc();
    %% Convert Box from [x1 y1 x2 y2] to [y1 x1 y2 x2]
    newBox = [Box(:,2) Box(:,1) Box(:,4) Box(:,3)];
    fN = Box(:,5);
    %%

    PsiN1 = bbox_ltrb2param( newBox, 'yxhwl').'; %% Must transpose: column as bbox
     latent_obj = @(z) sgp_negloglik( GPModel, z, PsiN1, fN );
     z0 = 0;
     if 1
       try
          z_hat = min_func( latent_obj, z0);
       catch
%     %     warning( 'Optimization on z is failed' );
          z_hat = 0;
        end
        expnz = exp(-z_hat);

           PsiN = PsiN1;
           PsiN = PsiN.*expnz;
     else     
           
           %% modified
           PsiN = PsiN1;
           hyp = gpml_struct.hyp;
           meanfunc = gpml_struct.meanfunc;
           covfunc = gpml_struct.covfunc;
           likfunc = gpml_struct.likfunc;
           hyp2 = minimize(hyp, @gp, -100, @infEP, meanfunc, covfunc, likfunc, PsiN', fN);
           [GPModel, gpml_struct]= InitGPmodel(hyp2);
     end
        %%
           KN = sgp_cov( GPModel, 0, PsiN ); %% Non-rescaled
         fN_hat = max(fN);
         psiNp1_0 = PsiN1(:,Idx);
         search_obj = @(psiNp1) sgp_neg_acquisition_ei( GPModel, ...
                    psiNp1, PsiN, fN, fN_hat, KN );
         psiNp1_hat = min_func( search_obj, psiNp1_0 );
         %PsiN = [PsiN psiNp1_hat]; % 
         %PsiN = PsiN/expnz;
         %regBox = psiNp1_hat;Debug
         %regBox = bbox_param2ltrb(PsiN','yxhwl'); %Transpose back in the input
         regBox = bbox_param2ltrb(psiNp1_hat','yxhwl'); %Transpose back in the input
         regBox = [regBox(:,2) regBox(:,1) regBox(:,4) regBox(:,3)];

end