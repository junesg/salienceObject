%% main function to report the evaluation
clear
clc

setup_env;

clc
%% select the data set to be evaluated

% select pre-established data set info
uiopen('load');
disp(['Selected data set is: ', data_info.name]);

%% load their param

param = globalParam;

% load the model
load('./models/pascal_100_gbvs_forestModel', 'forest');

%% select the image to play
imgNo = 220075;

img = imread([data_info.dataPath, num2str(imgNo),'.jpg']);

padSize = round(0.3*size(img));

% left padding
img_left = padarray(img,[0, padSize(2)],'replicate','pre');
% imagesc(img_left)

img_left_mask = salobj_test_img(img_left, param, forest);

img_left_mask1 = img_left_mask(:,padSize(2)+1:end);


% right padding
img_right = padarray(img,[0, padSize(2)],'replicate','post');
% imagesc(img_right)

img_right_mask = salobj_test_img(img_right, param, forest);

img_right_mask1 = img_right_mask(:,1:end-padSize(2));


% up padding
img_up = padarray(img,[padSize(1), 0],'replicate','pre');
% imagesc(img_up)

img_up_mask = salobj_test_img(img_up, param, forest);

img_up_mask1 = img_up_mask(padSize(1)+1:end,:);


% down padding
img_down = padarray(img,[padSize(1), 0],'replicate','post');
% imagesc(img_down)

img_down_mask = salobj_test_img(img_down, param, forest);

img_down_mask1 = img_down_mask(1:end-padSize(1),:);


figure, subplot(221)
imagesc(img_left_mask1)

subplot(222)
imagesc(img_right_mask)
axis equal tight
colormap(gray)

subplot(223)
imagesc(img_up_mask1)
axis equal tight
colormap(gray)

subplot(224)
imagesc(img_down_mask1)
axis equal tight
colormap(gray)

img_final_mask = img_left_mask1 + img_right_mask1 + img_up_mask1 + img_down_mask1;
img_final_mask = img_final_mask/max(img_final_mask(:));

figure, imagesc(img_final_mask)














