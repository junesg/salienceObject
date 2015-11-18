%   This an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   modified by juneysg@umich.edu for eecs542
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.

function [curFeats, curScores, featIdx] = computeFeatures(maskCCs, salMap, bestScores, param)
%% compute geometry and saliency features for each segment

% get stats
totalEnergy = sum(salMap(:));
numSegs = maskCCs.NumObjects;

% img stats
imgH = maskCCs.ImageSize(1); imgW = maskCCs.ImageSize(2);
imgD = norm([imgH, imgW]); imgArea = imgH*imgW; imgP = 2.*(imgH + imgW);

if isempty(bestScores)
  bestScores = zeros([numSegs, 1], 'single');
end

% memory allocation
curFeats = zeros([numSegs, param.featDim], 'single');
curScores = zeros([numSegs, 1], 'single');
featIdx = false([numSegs, 1]);

% shape features, almost the same as CPMC
segStats = regionprops(maskCCs, 'Area', 'Centroid', ...
  'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',...
  'Orientation', 'EquivDiameter', 'Extent', 'BoundingBox');

% loop over each segment
for curMaskIdx = 1:numSegs
  
  % current seg mask
  segMask = false([imgH, imgW]);
  segMask(maskCCs.PixelIdxList{curMaskIdx}) = 1;
  
  % check the bounding box
  if isempty(segStats(curMaskIdx)) || segStats(curMaskIdx).BoundingBox(3) < 10 || segStats(curMaskIdx).BoundingBox(4) < 10 ...
      || segStats(curMaskIdx).MajorAxisLength < 10 || segStats(curMaskIdx).MinorAxisLength < 10
    continue;
  end
  
  % shape features
  shapeFeats = [segStats(curMaskIdx).Area./imgArea; ...
    segStats(curMaskIdx).Centroid(1)./imgW; segStats(curMaskIdx).Centroid(2)./imgH; ...
    %segStats(curMaskIdx).ConvexArea./imgArea; ...
    segStats(curMaskIdx).Perimeter./imgP; 
    segStats(curMaskIdx).MajorAxisLength/imgD; segStats(curMaskIdx).MinorAxisLength./imgD;...
    segStats(curMaskIdx).Eccentricity; ...
    segStats(curMaskIdx).Orientation/180*pi;...
    segStats(curMaskIdx).EquivDiameter./imgD; ...
    %segStats(curMaskIdx).Solidity; ...
    segStats(curMaskIdx).Extent];
  
  % saliency features
  if sum(salMap(segMask)) >= 0.01
    salSegStats = regionprops(segMask, salMap, 'MaxIntensity', 'MinIntensity', 'WeightedCentroid', 'MeanIntensity');
    salSegCenter =  (salSegStats.WeightedCentroid - segStats(curMaskIdx).Centroid);
    energyRatio  = salSegStats.MeanIntensity*segStats(curMaskIdx).Area ./ totalEnergy;
    
    % histogram of fixation
    segSalMap = salMap; segSalMap(~segMask) = 0;
    segSalMap = imcrop(segSalMap, segStats(curMaskIdx).BoundingBox);
    segSalMask = imcrop(segMask, segStats(curMaskIdx).BoundingBox);
    hos = computeHOS(segSalMap, [3 3]);
    hosRot = computeHOSRot(segSalMap, segSalMask, [3, 4], segStats(curMaskIdx).Orientation);
    
    % kind of hacky features, we will move to CNN
    salFeats = [salSegStats.MeanIntensity; energyRatio; ...
      salSegCenter(1)./segStats(curMaskIdx).BoundingBox(3); ...
      salSegCenter(2)./segStats(curMaskIdx).BoundingBox(4);...
      salSegStats.MaxIntensity; salSegStats.MinIntensity; hos; hosRot];
  else
    salFeats = zeros([27 1]);
  end
  
  % stack features
  curFeats(curMaskIdx, :) = single([shapeFeats', salFeats']);
  curScores(curMaskIdx) = single(bestScores(curMaskIdx));
  featIdx(curMaskIdx) = 1;
  
end

% truncate features
curFeats = curFeats(featIdx, :);
curScores = curScores(featIdx, :);
