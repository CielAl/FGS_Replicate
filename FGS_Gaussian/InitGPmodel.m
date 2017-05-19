function [GPModel22,gpml_struct] = InitGPmodel(hyp)
% [GPModel22,gpml_struct] = InitGPmodel()
% Initialize the models for GP.
% GPModel22: The prior info for GP in FGS.
% gpml_struct: A struct that embedded all elements for GP in GPML lib.
    meanfunc = @meanConst;

    covfunc = @covSEard; 
   
    likfunc = @likGauss; 
    if nargin<1
        ell = 2.0; sf = 1.0; 
        hyp.mean = [0];
        hyp.lik = log(0.1);    
        hyp.cov = log([ell ell ell ell sf]);
    end
       % hyp = minimize.....
    GPModel22 = sgp_model_from_general(hyp);
    gpml_struct.hyp = hyp;
    gpml_struct.meanfunc = meanfunc;
    gpml_struct.covfunc = covfunc;
    gpml_struct.likfunc = likfunc;
end