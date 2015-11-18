%% set up dataset information structure: data_info
clear 
clc

%% set up paths
data_info.name = 'BSDS';

% input image and gt are in the same folder
data_info.dataPath = '/home/xizh/EECS542/Playing Session/BSDS300/images/';

data_info.resultPath = '/home/xizh/EECS542/Playing Session/BSDS300/results/';

%% number of images to test
data_info.numImg = 300;

%% randomly select images from the dataset
% all the images
files = dir([data_info.dataPath,'*.jpg']); % input image is jpg (gt is png)

% 3 columns: 1 for image name, 2 for test flag, 3 for evaluate flag
data_info.testImgTable = zeros(data_info.numImg,3); 

for cur_img = 1:data_info.numImg

    rand_idx = max(1,round(rand()*length(files))); 
    rand_img = str2num(files(rand_idx).name(1:end-4));
    
    while(sum(rand_img == data_info.testImgTable(:,1)) > 0)
        rand_idx = max(1,round(rand()*length(files))); 
        rand_img = str2num(files(rand_idx).name(1:end-4));
    end

    data_info.testImgTable(cur_img,1) = rand_img;
end

%% save the structure as .mat file
if exist(['Playing_',data_info.name,'.mat'],'file')
    warning('The data set information exists! Dont over load!');
else
    save(['Playing_',data_info.name,'.mat'], 'data_info');
end
