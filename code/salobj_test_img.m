%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.

function finalMask = salobj_test_img(img, param, forest)
%% test a trained model on a single image

%% try loading trained models
param.fixAlg = {'gbvs'};    % modify this line if you want to try your own fixation model
if nargin < 3
  if exist(param.modelPath, 'file')
    model = load(param.modelPath);
    forest = model.forest;
    clear model;
  else
    fprintf('Can not load model at %s\n', param.modelPath);
    finalMask = [];
    return;
  end
end

%% load structured edge model
% Load pre-trained Structured Forest model for mcg
sf_model = loadvar(fullfile(mcg_root, 'datasets', 'models', 'sf_modelFinal.mat'),'model');
pareto_n_cands = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_pareto_point_train2012.mat'),'n_cands');
rf_regressor = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_rand_forest_train2012.mat'),'rf');

%% run GBVS on input image
t_start = tic;
fprintf('Begin processing \n')
tic
gbvsParam = makeGBVSParams();
gbvsResult = gbvs(img, gbvsParam);
fixRes = gbvsResult.master_map;

% img stats
imgH = size(img, 1); imgW = size(img, 2); imgD = norm([imgH, imgW]);

%% generate saliency map from fixation results
% add proper upsampling / smoothing if necessary
% upsampling
fixRes = imresize(fixRes, [imgH, imgW]);
% smoothing if necessary
if param.fixSigma > 0
  kSize = round(imgD.*param.fixSigma);
  curH = fspecial('gaussian', round([kSize, kSize]*5), kSize);
  % rescale the salmap
  salMap = mat2gray(conv2(fixRes, curH, 'same')) .* max(fixRes(:));
else
  salMap = fixRes;
end
t_gbvs = toc;
fprintf('  GBVS takes %0.2f sec\n', t_gbvs)

%% run mcg
tic
[candidates_mcg, ~, mcg_feats] = im2mcg_simple(img, 'accurate', sf_model, pareto_n_cands, rf_regressor);
mcg_feats =  mcg_feats(:, [1:3, 6:13, 15:16]);
numProps = size(candidates_mcg.scores, 1);
t_mcg = toc;
fprintf('  MCG takes %0.2f sec\n', t_mcg)

%% process mcg results
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

% filter out the masks
[masks, validMasks, maskCCs] = filterMasks(masks, param.minArea);
mcg_feats = mcg_feats(validMasks, :);
numSegs = size(masks, 3);


%% compute all features
tic
[curFeats,~, maskIdx] = computeFeatures(maskCCs, salMap, [], param);

masks = masks(:,:,maskIdx); mcg_feats = single(mcg_feats(maskIdx, :));
allFeats = [curFeats, mcg_feats];
t_feat = toc;
t_mcg = toc;
fprintf('  Features takes %0.2f sec\n', t_mcg)

%% run the forest and generate the final result
[labels, probs] = forestApply(allFeats, forest);

% sort the results
[labels, index] = sort(labels, 1, 'descend');
index = index(1:param.topK);
topMasks = masks(:,:,index);
scores = reshape((labels(index)/param.nbins), [1 1 param.topK]);

% find top-K results and average them
finalMask = topMasks .* repmat(scores, [imgH, imgW, 1]);
finalMask = mat2gray(sum(finalMask, 3));
fprintf('All done, %d segments with total time %0.2f Sec \n',  numSegs, toc(t_start));

end


%% end of file