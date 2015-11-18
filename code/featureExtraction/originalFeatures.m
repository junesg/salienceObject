function [shapeFeats, salFeats] = originalFeatures(aMask, img, salMap)
	% input: a mask, which is an array of indices associated with 1 in the mask
	%       an image, the origianl image  eg. img = imread('th.jpg'); this
	%       is RGB image.
	% 		a saliience map for the image, which is image size saliency map, higher numbers refers to better salience
	
  
    totalEnergy = sum(salMap(:));
	% create segmentation mask 
    gimg = rgb2gray(img)
	[imgH, imgW] = size(gimg);
	segMask = false([imgH, imgW]);
    imgArea = imgH*imgW;
    imgP = 2.*(imgH + imgW);
    
	segMask(aMask) = 1;


	% measure the properties of the region
	salSegStats = regionprops(segMask, salMap, 'MaxIntensity', 'MinIntensity', 'WeightedCentroid', 'MeanIntensity');
    segStats = regionprops(segMask, 'Area', 'Centroid', ...
  'Perimeter', 'MajorAxisLength', 'MinorAxisLength', 'Eccentricity',...
  'Orientation', 'EquivDiameter', 'Extent', 'BoundingBox');
        
    salSegCenter =  (salSegStats.WeightedCentroid - segStats.Centroid);
    energyRatio  = salSegStats.MeanIntensity*segStats.Area ./ totalEnergy;
    
    % histogram of fixation
    segSalMap = salMap; segSalMap(~segMask) = 0;
    segSalMap = imcrop(segSalMap, segStats.BoundingBox);
    segSalMask = imcrop(segMask, segStats.BoundingBox);
    hos = computeHOS(segSalMap, [3 3]);
    hosRot = computeHOSRot(segSalMap, segSalMask, [3, 4], segStats.Orientation);
    

     salFeats = [salSegStats.MeanIntensity; energyRatio; ...
      salSegCenter(1)./segStats(curMaskIdx).BoundingBox(3); ...
      salSegCenter(2)./segStats(curMaskIdx).BoundingBox(4);...
      salSegStats.MaxIntensity; salSegStats.MinIntensity; hos; hosRot];


  
 
    shapeFeats = [segStats(curMaskIdx).Area./imgArea; ...
    segStats(curMaskIdx).Centroid(1)./imgW; segStats(curMaskIdx).Centroid(2)./imgH; ...
    segStats(curMaskIdx).Perimeter./imgP; 
    segStats(curMaskIdx).MajorAxisLength/imgD; segStats(curMaskIdx).MinorAxisLength./imgD;...
    segStats(curMaskIdx).Eccentricity; ...
    segStats(curMaskIdx).Orientation/180*pi;...
    segStats(curMaskIdx).EquivDiameter./imgD; ...
    segStats(curMaskIdx).Extent];



end