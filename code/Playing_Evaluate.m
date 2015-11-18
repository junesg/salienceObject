%% main function to evaluate the test results
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

%% start to evaluate

for cur_img = 1:data_info.numImg
    
    % check if this image has been tested
    if (data_info.testImgTable(cur_img,2) == 1 && data_info.testImgTable(cur_img,3) == 0)
        
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' is under evaluation...']); 
        
        % load test result: finalMask
        load([data_info.resultPath, num2str(data_info.testImgTable(cur_img,1)),'.mat']);

        
        % load ground truth
        gt =  double(imread([data_info.dataPath, num2str(data_info.testImgTable(cur_img,1)),'.png']));
        gt = gt/max(gt(:));
        
        % precision and recall for the most salient object segmentation       
        [prec, recall] = prCount(gt, finalMask, 0, param);     
        
        % save P & R
        save([data_info.resultPath, 'PR_', num2str(data_info.testImgTable(cur_img,1)),'.mat'], 'prec','recall');
        
        % mark the evaluation flag and update data info
        data_info.testImgTable(cur_img, 3) = 1;
        save(['Playing_',data_info.name,'.mat'], 'data_info');
          
    elseif (data_info.testImgTable(cur_img,3) == 1)
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' has been evaluated. Skip.']); 
        
    elseif (data_info.testImgTable(cur_img,2) == 0)
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' has not been tested yet. Skip.']);
        
    end    
end