function [allMasks] = loadAllMasks(param, stage)

%% load all salient object mask into memory
allMasks = cell(param.numTrainImgs, 1);

if strcmp(stage, 'train')
  thresh = param.trainThresh;
end

if strcmp(stage, 'test')
  thresh = param.testThresh;
end

fprintf(sprintf('Loading all masks...'))
%% sum all masks to global map
for curImg = 1:param.numTrainImgs
    
    % load gt object mask
    maskFile = sprintf('%s/%d.png', param.trainSobjFolder, param.trainList(curImg));
    mask = imread(maskFile);
    mask = im2double(mask);
    mask = mask>thresh;
   
    allMasks{curImg} = mask;
   
end
fprintf(sprintf('Done\n'))

