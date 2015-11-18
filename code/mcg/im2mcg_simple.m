% ------------------------------------------------------------------------
% Modified wrapper for MCG, good for parfor
% ------------------------------------------------------------------------
% This function computes the MCG candidates given an image.
%  INPUT:
%  - image : Input image
%  - mode  : It can be: + 'fast'     (SCG in the paper)
%                       + 'accurate' (MCG in the paper)
%  - thresh : threshold for eliminating overlapping proposals
%
%  OUTPUT:
%  - candidates : Struct containing the following fields
%          + superpixels : Label matrix of the superpixel partition
%          + labels : Cell containing the superpixel labels that form
%                     each of the candidates
%          + scores : Score of each of the ranked candidates
%  - ucm2       : Ultrametric Contour Map from which the candidates are
%                 extracted
%  - feats      : features used to rank the proposals
%
% ------------------------------------------------------------------------
function [candidates, ucm2, feats, times] = im2mcg_simple(image,mode, sf_model, pareto_n_cands, rf_regressor, thresh)
if nargin<2
  mode = 'fast';
end


% Level of overlap to erase duplicates
if nargin<6
  J_th = 0.9;
else
  J_th = thresh;
end

% Load pre-trained Structured Forest model
% sf_model = loadvar(fullfile(mcg_root, 'datasets', 'models', 'sf_modelFinal.mat'),'model');


% Max margin parameter
theta = 0.7;

if strcmp(mode,'fast')
  % Which scales to work on (MCG is [2, 1, 0.5], SCG is just [1])
  scales = 1;
  
  % Get the hierarchies at each scale and the global hierarchy
  [ucm2,~,times] = img2ucms(image, sf_model, scales);
  all_ucms = ucm2;
  
  % Load pre-trained pareto point
  %pareto_n_cands = loadvar(fullfile(mcg_root, 'datasets', 'models', 'scg_pareto_point_train2012.mat'),'n_cands');
  
  % Load pre-trained random forest regresssor for the ranking of candidates
  %rf_regressor = loadvar(fullfile(mcg_root, 'datasets', 'models', 'scg_rand_forest_train2012.mat'),'rf');
  
elseif strcmp(mode,'accurate')
  % Which scales to work on (MCG is [2, 1, 0.5], SCG is just [1])
  scales = [2, 1, 0.5];
  
  % Get the hierarchies at each scale and the global hierarchy
  [ucm2,ucms,times] = img2ucms(image, sf_model, scales);
  all_ucms = cat(3,ucm2,ucms(:,:,3),ucms(:,:,2),ucms(:,:,1)); % Multi, 0.5, 1, 2
  
  % Load pre-trained pareto point
  %pareto_n_cands = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_pareto_point_train2012.mat'),'n_cands');
  
  % Load pre-trained random forest regresssor for the ranking of candidates
  %rf_regressor = loadvar(fullfile(mcg_root, 'datasets', 'models', 'mcg_rand_forest_train2012.mat'),'rf');
else
  error('Unknown mode for MCG: Possibilities are ''fast'' or ''accurate''')
end
% ------------------------------------

% Transform ucms to hierarchies (dendogram) and put them all together
n_hiers = size(all_ucms,3);
lps = [];
ms  = cell(n_hiers,1);
ths = cell(n_hiers,1);
for ii=1:n_hiers
  % Transform the UCM to a hierarchy
  curr_hier = ucm2hier(all_ucms(:,:,ii));
  ths{ii}.start_ths = curr_hier.start_ths';
  ths{ii}.end_ths   = curr_hier.end_ths';
  ms{ii}            = curr_hier.ms_matrix;
  lps = cat(3, lps, curr_hier.leaves_part);
end

% Get full cands, represented on a fused hierarchy
[f_lp,f_ms,cands,start_ths,end_ths] = full_cands_from_hiers(lps,ms,ths,pareto_n_cands);

% Hole filling and complementary candidates
if ~isempty(f_ms)
  [cands_hf, cands_comp] = hole_filling(double(f_lp), double(f_ms), cands); %#ok<NASGU>
else
  cands_hf = cands;
  cands_comp = cands; %#ok<NASGU>
end

% Select which candidates to keep (Uncomment just one line)
cands = cands_hf;                       % Just the candidates with holes filled
% cands = [cands_hf; cands_comp];         % Holes filled and the complementary
% cands = [cands; cands_hf; cands_comp];  % All of them

% Compute base features
b_feats = compute_base_features(f_lp, f_ms, all_ucms);
b_feats.start_ths = start_ths;
b_feats.end_ths   = end_ths;
b_feats.im_size   = size(f_lp);

% Filter by overlap
red_cands = mex_fast_reduction(cands-1,b_feats.areas,b_feats.intersections,J_th);

% Compute full features on reduced cands
feats = compute_full_features(red_cands,b_feats);

% Rank candidates
class_scores = regRF_predict(feats,rf_regressor);
[scores, ids] = sort(class_scores,'descend');
red_cands = red_cands(ids,:);
if isrow(scores)
  scores = scores';
end

% Max margin
new_ids = mex_max_margin(red_cands-1,scores,b_feats.intersections,theta); %#ok<NASGU>
cand_labels = red_cands(new_ids,:);
candidates.scores = scores(new_ids);


% Get the labels of leave regions that form each candidates
candidates.superpixels = f_lp;
if ~isempty(f_ms)
  candidates.labels = cands2labels(cand_labels,f_ms);
else
  candidates.labels = {1};
end

% pick contour feats
