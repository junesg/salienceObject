%% main function to test images
clear
clc

setup_env;

clc
%% select the data set to be tested

% select pre-established data set info
uiopen('load');
disp(['Selected data set is: ', data_info.name]);

%% initialize the model

% get the param
param = globalParam;

% load the model
load('./models/pascal_100_gbvs_forestModel', 'forest');

%% start test

for cur_img = 1:data_info.numImg
    
    % check if this image has been tested
    if (data_info.testImgTable(cur_img,2) == 0)
        
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' is under test...']); 
        
        % load image
        img = imread([data_info.dataPath, num2str(data_info.testImgTable(cur_img,1)),'.jpg']);

        % ++ apply the model ++
        finalMask = salobj_test_img(img, param, forest);
        
        % save the result
        save([data_info.resultPath, num2str(data_info.testImgTable(cur_img,1)),'.mat'], 'finalMask');
        
        % mark the flag and update data info
        data_info.testImgTable(cur_img, 2) = 1;
        save(['Playing_',data_info.name,'.mat'], 'data_info');
          
    else
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' has been tested. Skip.']); 
    end    
end