function min_func = getMinFunc()
% min_func = getMinFunc() 
% Return the function handle of minFunc.
    Solver_Timeout = 10;
    minFunc_Method = 'lbfgs';
    minFuncX_OPTS = struct();
    minFuncX_OPTS.timeout = Solver_Timeout;
    minFuncX_OPTS.Display = 'off';
    minFuncX_OPTS.Method  = minFunc_Method;
    min_func = @(func,x0) minFuncX( func,x0, minFuncX_OPTS);
end