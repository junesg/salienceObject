function f,d=  statisticsSalFeatures(aMask, img, salMap, weighted)
% this function gets the SIFT feature of the masked area
% aMask: array of pixel index in the image, img: RGB image, salMap: salience Map
% weighted is a boolean to see if this sift source image should be modified by salMap weight

 % F = VL_SIFT(I) computes the SIFT frames [1] (keypoints) F of the
 %    image I. I is a gray-scale image in single precision. Each column
 %    of F is a feature frame and has the format [X;Y;S;TH], where X,Y
 %    is the (fractional) center of the frame, S is the scale and TH is
 %    the orientation (in radians).
   
 %    [F,D] = VL_SIFT(I) computes the SIFT descriptors [1] as well. Each
 %    column of D is the descriptor of the corresponding frame in F. A
 %    descriptor is a 128-dimensional vector of class UINT8.
   

	DEBUG = 0

	% import vlfeat

	if DEBUG
		img = imread('../th.jpg');
		aMask = imread('../firstImageMask.jpg');
		%normalize aMask
		aMask = aMask./ (max(aMask(:)) - min(aMask(:)));
		aMask = find(aMask == 1);
		salMap = imread('../testsalMap.jpg');
	end


	% create segmentation mask 
    gimg = rgb2gray(img)
	[imgH, imgW] = size(gimg);
	segMask = false([imgH, imgW]);
	imgArea = imgH*imgW; 
	segMask(aMask) = 1;


	% get statistics of the saliency map within the boundary


end