% input path to your image files 
img_pth  = '../datasets/imgs/pascal';
% input path to your mask files
mask_pth = '../datasets/masks/pascal';
% if you want to have particular image name, this is a cell of names
% #### this can be left empty ####
img_ids = {};
% set threshold
threshold  = 0.5;

%% now start running the image properties

images = dir(img_pth);
masks = dir(mask_pth);
AllCenterBiases = [];
AllSizeProp  = [];

% loop through each image
for ii = 1:length(images)
    if images(ii).name(1) == '.', continue;end
    if ~isempty(img_ids)
        find_one =  false;
        for kk = 1:length(img_ids)
            if strfind(images(ii).name, img_ids{kk}) == 1
                find_one = true;
            end
        end
        if ~find_one, continue;end
    end
    
    % now search for mask
    comp  =  strsplit(images(ii).name,'.');
    find_two  = false;
    for jj = 1:length(masks)
        
        if strcmp(masks(jj).name, [comp{1},'.png'])==1
            find_two = true;
        end
        
    end
    
    if find_two
        % now process the centers, and proportions of this image:
        [CenterBias, sizeProp] = getImageProperty([img_pth,'/',images(ii).name], [mask_pth,'/',comp{1},'.png'], threshold);
    
        AllCenterBiases = [AllCenterBiases;CenterBias];
        AllSizeProp = [AllSizeProp; sizeProp];
    
    end
end
