%   This an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.

function param = globalParam()

%% The function generates global parameters for training and testing

%% train/test dataset
param.trainDataset = 'pascal';
param.testDataset = 'pascal';

%% images
param.trainImgFolder = sprintf('../datasets/imgs/%s', param.trainDataset);
param.testImgFolder = sprintf('../datasets/imgs/%s', param.testDataset);

%% salient object data
param.trainSobjFolder = sprintf('../datasets/masks/%s', param.trainDataset);
param.testSobjFolder = sprintf('../datasets/masks/%s', param.testDataset);
if strcmp(param.trainDataset, 'bruce')
  param.trainThresh = 30./255-eps;
else
  param.trainThresh = 0.5;
end
if strcmp(param.testDataset, 'bruce')
  param.testThresh = 30./255-eps;
else
  param.testThresh = 0.5;
end

%% number of images
files = dir(sprintf('%s/*.png', param.trainSobjFolder));
param.numTrainImgs = length(files);
files = dir(sprintf('%s/*.png', param.testSobjFolder));
param.numTestImgs = length(files);

%% fixation prediction algs
% you can modify this line to switch between different fixation models
% including gbvs, aws, aim, sig, dva, sun, itti and humanFix
% we have include pre-computed maps for these methods on all datasets
% we also include gbvs code so you can test a model on a new image
param.fixAlg = {'gbvs'};
% param.fixAlg = {'humanFix'};
param.trainFixAlgFolder =  sprintf('../algmaps/%s/%s', param.trainDataset, param.fixAlg{1});
param.testFixAlgFolder =  sprintf('../algmaps/%s/%s', param.testDataset, param.fixAlg{1});
% smooth the fixation map if necessary (0 to turn this off)
param.fixSigma = 0.01;

%% setup folder for results 
param.resultMaskFolder = sprintf('../algmaps/%s/mcg_%s', param.testDataset, param.fixAlg{1});
if ~exist(param.resultMaskFolder, 'dir')
  mkdir(param.resultMaskFolder);
end

%% object proposal params
% A relative small number of props is good enough for most datasets
param.maxTrainProps = 400;           % number of proposals for training
param.maxTestProps = 400;            % you can use less proposals for testing
param.minArea = 200;                 % minimum area for a proposal
param.topK = 20;                     % top K proposals to average
% MCG results are cached into the following folders
param.trainCacheFolder = sprintf('../tmp/%s', param.trainDataset);
param.testCacheFolder = sprintf('../tmp/%s', param.testDataset);
if ~exist(param.trainCacheFolder, 'dir')
  mkdir(param.trainCacheFolder);
end
if ~exist(param.testCacheFolder, 'dir')
  mkdir(param.testCacheFolder);
end

%% train/test split
% if train and test on the same dataset
if strcmp(param.trainDataset, param.testDataset)
  % we will use 40% of the samples for training and 60% for testing
  % use the same random number generator for producing same results
  rng(9527);                          % comment out this line for random sampling
  param.pTrain = 0.4;
  rndList = randperm(param.numTrainImgs);
  param.numTrainImgs = round(param.pTrain*param.numTrainImgs);
  param.trainList = rndList(1:param.numTrainImgs);
  param.numTestImgs = numel(rndList) - param.numTrainImgs;
  param.testList = rndList(end - param.numTestImgs + 1:end);
else
  param.trainList = 1:param.numTrainImgs;
  param.testList = 1:param.numTestImgs;
end

%% for random forest
param.ntree = 16;     % number of trees
param.mtry = 5;       % number of feature dims per node
param.minChild = 5;   % minimum number of samples per node
param.featDim = 37;   % dims of shape and fixation features, total feature dim is larger
param.nbins = 8;      % discretize the label space
% we will resample the data to balance the categories 
param.numSamples = round((0.8*param.maxTrainProps/param.nbins)*param.numTrainImgs);  

%% benchmark params
param.gtThreshold = param.testThresh;
param.thNum = 50;
param.beta = sqrt(0.3);
param.thList = linspace(0, 1, param.thNum);

%% modelPath
% the naming format: 
% trainingDataset_(percentage of training samples)_fixationMethod_forestModel.mat
if isfield(param, 'pTrain')
  param.modelPath = sprintf('./models/%s_%d_%s_forestModel.mat', param.trainDataset, round(param.pTrain*100), param.fixAlg{1});
else
  param.modelPath = sprintf('./models/%s_%s_forestModel.mat', param.trainDataset, param.fixAlg{1});
end

% fix for the playing session
param.modelPath = sprintf('./models/pascal_100_gbvs_forestModel.mat');

end