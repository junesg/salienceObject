function [feat] = computeHOS(salmap, grids)
% spatial histogram of saliency

% height / width
height = size(salmap, 1);
width = size(salmap, 2);

% set up the grid for histogram, ignore regions that are too small
hStep = floor(height/grids(1));
hGrids = 1: hStep :height -hStep +1;
hGrids = hGrids(1:grids(1));

wStep = floor(width/grids(2));
wGrids = 1: wStep :width - wStep +1;
wGrids = wGrids(1:grids(2));


% integral image
integralImage = cumsum(cumsum(double(salmap), 2), 1);
salmap = padarray(integralImage,[1 1],'pre');
% in case you do not have cv toolbox
%salmap = integralImage(salmap);
feat = zeros([prod(grids) 1]);

% for loop
count = 1;
for h = hGrids
    for w = wGrids
        feat(count, 1) = salmap(h,w) + salmap(h+hStep, w+wStep) - salmap(h+hStep, w) - salmap(h, w+wStep);
        count = count+1;
    end
end
feat(feat<0) = 0;

feat = feat./(norm(feat)+eps);
feat = sqrt(feat);