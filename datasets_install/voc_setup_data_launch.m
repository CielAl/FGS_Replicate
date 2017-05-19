run(fullfile(fileparts(mfilename('fullpath')), ...
  '..', '..', 'matlab', 'vl_setupnn.m')) ;
addpath(fullfile(vl_rootnn,'examples','fast_rcnn','bbox_functions'));
addpath(fullfile(vl_rootnn,'examples','fast_rcnn','datasets'));

opts.dataDir   = fullfile(vl_rootnn, 'data') ;
opts.sswDir    = fullfile(vl_rootnn, 'data', 'SSW');
opts.expDir    = fullfile(vl_rootnn, 'data', 'fast-rcnn-vgg16-pascal07') ;
opts.imdbPath  = fullfile(opts.expDir, 'imdb.mat');
opts.modelPath = fullfile(opts.dataDir, 'models', ...
  'imagenet-vgg-verydeep-16.mat') ;

opts.piecewise = true;  % piecewise training (+bbox regression)
opts.train.gpus = [] ;
opts.train.batchSize = 2 ;
opts.train.numSubBatches = 1 ;
opts.train.continue = true ;
opts.train.prefetch = false ; % does not help for two images in a batch
opts.train.learningRate = 1e-3 / 64 * [ones(1,6) 0.1*ones(1,6)];
opts.train.weightDecay = 0.0005 ;
opts.train.numEpochs = 12 ;
opts.train.derOutputs = {'losscls', 1, 'lossbbox', 1} ;
opts.lite = false  ;
opts.numFetchThreads = 2 ;

opts = vl_argparse(opts, {}) ;
display(opts);

opts.train.expDir = opts.expDir ;
opts.train.numEpochs = numel(opts.train.learningRate) ;


if exist(opts.imdbPath,'file') == 2
  fprintf('Loading imdb...');
  imdb = load(opts.imdbPath) ;
else
  if ~exist(opts.expDir,'dir')
    mkdir(opts.expDir);
  end
  fprintf('Setting VOC2007 up, this may take a few minutes\n');
  imdb = cnn_setup_data_voc07_ssw(...
    'dataDir', opts.dataDir, ...
    'sswDir', opts.sswDir, ...
    'addFlipped', true, ...
    'useDifficult', true) ;
  save(opts.imdbPath,'-struct', 'imdb','-v7.3');
  fprintf('\n');
end
fprintf('done\n');

%imdb = cnn_setup_data_voc07(...
 % 'dataDir', opts.dataDir, ...
  %'useDifficult', opts.useDifficult, ...
  %'addFlipped', opts.addFlipped) ;