%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.

function [] = salobj_test(param)
%% the main test function
% test a trained model on the testing set
% this function is used for batch test

%% load trained models
if exist(param.modelPath, 'file')
  model = load(param.modelPath);
  forest = model.forest; 
  clear model;
else
  fprintf('Can not load model at %s\n', param.modelPath);
  return;
end

%% load structured edge model
% Load pre-trained Structured Forest model
sf_model = loadvar(fullfile(mcg_root, 'datasets', 'models', 'sf_modelFinal.mat'),'model');
pareto_n_cands = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_pareto_point_train2012.mat'),'n_cands');
rf_regressor = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_rand_forest_train2012.mat'),'rf');


%% now generating all results
parfor curFile = 1:param.numTestImgs
  
  fprintf('Testing Stage: Processing %d th image\n', curFile)

  % get file names
  t_start = tic;
  imgfile = fullfile(param.testImgFolder, sprintf('%d.jpg', param.testList(curFile)));
  fixfile = fullfile(param.testFixAlgFolder, sprintf('%d.png', param.testList(curFile)));
  
  % optmized for parfor
  cur_sf_model = sf_model;
  cur_pareto_n_cands = pareto_n_cands;
  cur_rf_regressor = rf_regressor;
  
  % load image and fixation results
  img = imread(imgfile);
  fixRes = im2double(imread(fixfile));
  
  % img stats
  imgH = size(img, 1); imgW = size(img, 2); imgD = norm([imgH, imgW]);
  
  % run mcg
  cachefile = fullfile(param.testCacheFolder, sprintf('%d.mat', param.testList(curFile)));
  if ~exist(cachefile, 'file')
    [candidates_mcg, ~, mcg_feats] = im2mcg_simple(img, 'accurate', cur_sf_model, cur_pareto_n_cands, cur_rf_regressor);
    mcg_feats =  mcg_feats(:, [1:3, 6:13, 15:16]);
    parsave(cachefile, candidates_mcg, mcg_feats)
  else
    data = load(cachefile);
    mcg_feats = data.mcg_feats;
    candidates_mcg = data.candidates_mcg;
  end
  numProps = size(candidates_mcg.scores, 1);

  % sort mcg results 
  numProps = min(numProps, param.maxTestProps);
  masks = false([size(img, 1), size(img, 2), numProps]);
  scores = zeros([1 numProps]);
  [sorted_scores, sorted_idx] = sort(candidates_mcg.scores, 1, 'descend');
  
  % truncate scores and features
  scores(1:numProps) = sorted_scores(1:numProps); sorted_idx = sorted_idx(1:numProps);
  mcg_feats = [mcg_feats(sorted_idx, :), scores'];
  
  % convert mcg results into masks
  props = candidates_mcg.labels(sorted_idx);
  for curProp = 1:numProps
    masks(:,:,curProp) = ismember(candidates_mcg.superpixels, props{curProp});
  end
  
  % filter out small segments
  [masks, validMasks, maskCCs] = filterMasks(masks, param.minArea);
  mcg_feats = mcg_feats(validMasks, :);
  numSegs = size(masks, 3);
  
  % generate saliency map from fixation results
  % add proper upsampling / smoothing if necessary
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
  
  % compute all features
  [curFeats,~, maskIdx] = computeFeatures(maskCCs, salMap, [], param);
  masks = masks(:,:,maskIdx); mcg_feats = single(mcg_feats(maskIdx, :));
  allFeats = [curFeats, mcg_feats];
  [labels, probs] = forestApply(allFeats, forest);
  
  % sort the results
  [labels, index] = sort(labels, 1, 'descend');
  index = index(1:param.topK); 
  topMasks = masks(:,:,index); 
  scores = reshape((labels(index)/param.nbins), [1 1 param.topK]);
  
  % find top-K results and average them
  finalMask = topMasks .* repmat(scores, [imgH, imgW, 1]);
  finalMask = mat2gray(sum(finalMask, 3));
  
  resultfile = fullfile(param.resultMaskFolder, sprintf('%d.png', param.testList(curFile)));
  imwrite(finalMask, resultfile);
  fprintf('All done, %d segments with total time %0.2f Sec \n',  numSegs, toc(t_start));
  
end

end

