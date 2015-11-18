function [CenterBias, sizeProp] = getImageProperty(image_dir, mask_dir, threshold)
       
    %%
    % example 
    %
    %     image_dir = '../datasets/imgs/pascal/1.jpg';
    %     mask_dir = '../datasets/masks/pascal/1.png';
    %     threshold  = 0.5;
    % first change the mask to binary
    mask = imread(mask_dir);
    mask2 = mask > threshold;
%     orig_im = imread(image_dir);
    
    % get the connectivity
    CC = bwconncomp(mask2);
    imsize  = CC.ImageSize;
    
    % calcuate the center bias , size proportion of the mask
    centers = [];
    sizeProp = [];
    
    pixelIdList = CC.PixelIdxList;
    
    % now look at each center
    for ii = 1:CC.NumObjects
        %first get the rows and cols of this segmentation 
        assert(max(pixelIdList{ii}) < imsize(1)*imsize(2));
        rows = [];
        cols = [];
        for jj = 1:length(pixelIdList{ii})
            pixel_pos = pixelIdList{ii}(jj);
            rows = [rows , ceil(pixel_pos/imsize(1))];
            this_col = mod(pixel_pos,imsize(2));
            if this_col == 0 
                this_col =  imsize(2);
            end
            cols = [cols,this_col];
        end
        centers = [centers; [mean(rows)/imsize(1), mean(cols)/imsize(2)]];
        sizeProp = [sizeProp;length(pixelIdList{ii})/(imsize(1)*imsize(2)) ];
        
    end
    
    % establish center bias
    CenterBias = centers;
    % now establish size 

end