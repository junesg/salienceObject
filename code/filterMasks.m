function [masks, validMasks, CCs]= filterMasks(masks, thresh)
% filter out segments that is smaller than a threshold
% also convert the masks into cc structure

numSegs = size(masks, 3);
validMasks = false([numSegs, 1]);
CCs = struct('Connectivity', 8, 'ImageSize', [size(masks, 1), size(masks, 2)],...
  'NumObjects', 0, 'PixelIdxList', cell(1));

if nargin<2
  thresh = 0;
end

for curSeg = 1:numSegs
  
  % connected component
  bwcc = bwconncomp(masks(:,:,curSeg), 8);
  
  % delete current masks
  if bwcc.NumObjects ==0 || (bwcc.NumObjects == 1 && numel(bwcc.PixelIdxList{1}) < thresh)
    continue;
  end
  
  % num of components
  if bwcc.NumObjects ==1
    validMasks(curSeg) = true;
    CCs.NumObjects = CCs.NumObjects + 1;
    CCs.PixelIdxList{end+1} = bwcc.PixelIdxList{1}; 
    continue;
  end
  
  % MCG already include hole filling
  % we do not need to anything here
  
end

masks = masks(:,:, validMasks);

