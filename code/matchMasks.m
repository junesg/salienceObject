function [bestScores, bestMatches] = matchMasks(masks, antMask)
% match every mask to every gt mask

% memory allocation
numSegs = size(masks, 3);
bestScores = zeros([1 numSegs]);
bestMatches = zeros([1 numSegs]);

% get all objects
numObj = max(antMask(:));

if numObj < 1
    return;
end

% match object to segs 
oScores = zeros([numObj, numSegs]);
for curSeg = 1:numSegs 
    
    % current object mask
    curSegMask = masks(:,:,curSeg);
    
    for curObj = 1:numObj
        
        % intersection over union
        curObjMask = (antMask == curObj);
        
        oScores(curObj, curSeg) = nnz(curSegMask & curObjMask) ./ nnz(curSegMask  | curObjMask);
        
    end
end

% get the best scores
[bestScores, bestMatches] = max(oScores, [], 1);

    