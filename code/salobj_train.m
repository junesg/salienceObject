%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.


function forest = salobj_train(param)
%% the main training function
% train our salient object segmentation model
% using object proposals + fixation prediction

%% skip training if possible
if exist(param.modelPath, 'file')
  model = load(param.modelPath);
  forest = model.forest;
  return;
end

%% pack gt object mask into memory
allMasks = loadAllMasks(param, 'train');

%% load structured edge model
% Load pre-trained Structured Forest model
sf_model = loadvar(fullfile(mcg_root, 'datasets', 'models', 'sf_modelFinal.mat'),'model');
pareto_n_cands = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_pareto_point_train2012.mat'),'n_cands');
rf_regressor = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_rand_forest_train2012.mat'),'rf');

%% now get all features
cellFeats = cell(param.numTrainImgs, 1);
cellScores = cell(param.numTrainImgs, 1);
parfor curFile = 1:param.numTrainImgs
  
  fprintf('Training Stage: Processing %d th image\n', curFile)
  % optmized for parfor
  cur_sf_model = sf_model;
  cur_pareto_n_cands = pareto_n_cands;
  cur_rf_regressor = rf_regressor;
  cur_allMasks = allMasks;
  
  % get file names
  t_start = tic;
  imgfile = fullfile(param.trainImgFolder, sprintf('%d.jpg', param.trainList(curFile)));
  fixfile = fullfile(param.trainFixAlgFolder, sprintf('%d.png', param.trainList(curFile)));
  
  % load image, fixation and salient object masks
  img = imread(imgfile);
  fixRes = im2double(imread(fixfile));
  objMask = cur_allMasks{curFile};
  objCC = bwconncomp(objMask, 4);
  objMask = uint16(labelmatrix(objCC));
  
  % img stats
  imgH = size(img, 1); imgW = size(img, 2); imgD = norm([imgH, imgW]);
  
  % we switched to mcg from cpmc, this is faster and slightly better
  % run mcg for object proposals, we also get the features from MCG
  cachefile = fullfile(param.trainCacheFolder, sprintf('%d.mat', param.trainList(curFile)));
  if ~exist(cachefile, 'file')
    %tic
    [candidates_mcg, ~, mcg_feats] = im2mcg_simple(img, 'accurate', cur_sf_model, cur_pareto_n_cands, cur_rf_regressor);
    % remove redudant features from mcg
    mcg_feats =  mcg_feats(:, [1:3, 6:13, 15:16]);
    parsave(cachefile, candidates_mcg, mcg_feats)
  else
    data = load(cachefile);
    mcg_feats = data.mcg_feats;
    candidates_mcg = data.candidates_mcg;
  end
  numProps = size(candidates_mcg.scores, 1);
  %t_prop = toc;
  %fprintf('  MCG takes %0.2f Sec with %d proposals\n', t_prop, numProps);
  
  % sort mcg results
  numProps = min(numProps, param.maxTrainProps);
  masks = false([size(img, 1), size(img, 2), numProps]);
  scores = zeros([1 numProps]);
  [sorted_scores, sorted_idx] = sort(candidates_mcg.scores, 1, 'descend');
  
  % truncate scores and features
  scores(1:numProps) = sorted_scores(1:numProps); sorted_idx = sorted_idx(1:numProps);
  mcg_feats = [mcg_feats(sorted_idx, :), scores'];
  
  % convert mcg results into masks
  % they are then used for training our random forest
  %tic
  props = candidates_mcg.labels(sorted_idx);
  for curProp = 1:numProps
    masks(:,:,curProp) = ismember(candidates_mcg.superpixels, props{curProp});
  end
  
  % filter out small segments
  [masks, validMasks, maskCCs] = filterMasks(masks, param.minArea);
  mcg_feats = mcg_feats(validMasks, :);
  numSegs = size(masks, 3);
  
  % match proposals to gt masks
  % this is our label
  bestScores = matchMasks(masks, objMask);
  %t_match = toc;
  %fprintf('  Filter and Matching takes %0.2f Sec \n', t_match);
  
  % generate saliency map from fixation results
  % add proper upsampling / smoothing if necessary
  %tic
  if size(fixRes, 1) ~= imgH
    % upsampling
    fixRes = imresize(fixRes, [imgH, imgW]);
    
    % smoothing if necessary
    if param.fixSigma > 0
      kSize = imgD.*param.fixSigma;
      curH = fspecial('gaussian', round([kSize, kSize]*5), kSize);
      % rescale the salmap
      salMap = mat2gray(conv2(fixRes, curH, 'same')) .* max(fixRes(:));
    else
      salMap = fixRes;
    end
    
  else
    salMap = fixRes;
  end
  %t_fix = toc;
  %fprintf('  Fixation map takes %0.2f Sec \n', t_fix);
  
  % compute all features
  %tic
  [curFeats, curScores, maskIdx] = computeFeatures(maskCCs, salMap, bestScores, param);
  mcg_feats = single(mcg_feats(maskIdx, :));
  %t_feat = toc;
  %fprintf('  Features takes %0.2f Sec \n', t_feat);
  
  cellFeats{curFile} = [curFeats, mcg_feats];
  cellScores{curFile} = curScores;
  fprintf('All done, %d segments with total time %0.2f Sec \n',  numSegs, toc(t_start));
  
end

%% stack all features and send them to random forest
% stack all features
allFeats = cell2mat(cellFeats);
allLabels = cell2mat(cellScores);
clear cellFeats; clear cellScores;
featfile = fullfile(param.trainCacheFolder, 'feats.mat');
save(featfile, 'allFeats', 'allLabels');

% discretize the output space
allLabels = ceil((allLabels + eps)*param.nbins);
allLabels(allLabels>param.nbins) = param.nbins;
clear cellFeats; clear cellScores;

% sample with replacement (balance samples for training)
trainFeats = zeros([param.numSamples*param.nbins size(allFeats, 2)]);
trainLabels = zeros([param.numSamples*param.nbins 1]);
for curLabelIdx = 1:param.nbins
  curLabels = allLabels(allLabels==curLabelIdx);
  curFeats = allFeats(allLabels==curLabelIdx, :);
  [sampleFeats, sampleIdx] = datasample(curFeats, param.numSamples);
  sampleLabels = curLabels(sampleIdx);
  
  trainFeats((curLabelIdx-1)*param.numSamples+1: curLabelIdx*param.numSamples, :) = sampleFeats;
  trainLabels((curLabelIdx-1)*param.numSamples+1: curLabelIdx*param.numSamples) = sampleLabels;
end

% train the forest
tic
forest = forestTrain( trainFeats, trainLabels, 'M', param.ntree, 'F1', param.mtry, 'minChild', param.minChild);
toc
save(param.modelPath, 'forest');

end

