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


%% calculate F-measure

beta2 = 0.3;

F_measure = zeros(data_info.numImg, param.thNum);
PR = zeros(data_info.numImg, param.thNum, 2);

for cur_img = 1:data_info.numImg
    
    % check if this image has been evaluted
    if (data_info.testImgTable(cur_img,3) == 1)
        
        % load evaluation result: PR_XXXX.mat which includes "prec" and "recall"
        load([data_info.resultPath, 'PR_',num2str(data_info.testImgTable(cur_img,1)),'.mat']);

        % calculate Fmeasure per image
        F_measure(cur_img, :) = (1+beta2)*prec.*recall./(beta2*prec + recall);
        
        % PR
        PR(cur_img,:,1) = prec;
        PR(cur_img,:,2) = recall;
        
        disp(['++ Index ', num2str(cur_img, '%.4d'), ' is done.']);     
    else
        disp(['Error: Index ', num2str(cur_img, '%.4d'), ' has not been evaluated!!!!!']); 
        
    end    
end

%% show F-Measure

F_measure(isnan(F_measure)) = 0;

% mean
F_measure_mean = mean(F_measure,1);

plot(param.thList, F_measure_mean,'b','L[]ineWidth',2)
hold on
xlabel('Threshold')
ylabel('F-measure')
ylim([0 1])


[~,max_th_idx] = max(F_measure_mean);

% std
% F_measure_std = std(F_measure,1,1);
% plot(param.thList, F_measure_mean + F_measure_std)
% hold on
% 
% plot(param.thList, F_measure_mean - F_measure_std)
% hold on

figure
plot(F_measure(:,max_th_idx),'r','LineWidth',2)
xlabel('Images')
ylabel('F-measure')
ylim([0 1])

figure
scatter(PR(:,max_th_idx,2),PR(:,max_th_idx,1))
xlabel('Recall')
ylabel('Precision')

%% analyze failure images qualitatively

[sorted_F, sorted_index] = sort(F_measure(:,max_th_idx), 'ascend');
folder = '/home/xizh/EECS542/Playing Session/Playing_PostProcess/';

for ii = 280
    close all
    index = sorted_index(ii);
    img = imread([data_info.dataPath, num2str(data_info.testImgTable(index,1)),'.jpg']);
    gt = imread([data_info.dataPath, num2str(data_info.testImgTable(index,1)),'.png']);
    
    load([data_info.resultPath, num2str(data_info.testImgTable(index,1)),'.mat']);
    
    th_mask = finalMask>0.6;
    
    figure('Position',[100 500 2000 1500])
    subplot(221)
    imagesc(img)
    axis equal tight
    
    
    subplot(222)
    imagesc(gt)
    axis equal tight
    colormap(gray)
    
    subplot(223)
    imagesc(finalMask)
    axis equal tight
    colormap(gray)
    
    subplot(224)
    imagesc(th_mask)
    axis equal tight
    colormap(gray)
    
    axis equal tight
    
    w = waitforbuttonpress;
    if w == 0
        disp('Skip')
    else  % keyboard
        copyfile([data_info.dataPath, num2str(data_info.testImgTable(index,1)),'.jpg'],folder);
        copyfile([data_info.dataPath, num2str(data_info.testImgTable(index,1)),'.png'],folder);
        
        imwrite(finalMask,[folder,num2str(data_info.testImgTable(index,1)),'_Failure.jpg']);
        imwrite(th_mask,[folder,num2str(data_info.testImgTable(index,1)),'_th_Failure.jpg']);
    end
    
end
    
%% failure modes analysis quantatively

centerBias = zeros(data_info.numImg,1);
objectSize = zeros(data_info.numImg,1);

for cur_img = 1:data_info.numImg
    
    bw = imread([data_info.dataPath, num2str(data_info.testImgTable(cur_img,1)),'.png']);
    bw = bw/max(bw(:));
    
%     imagesc(bw)
    stat = regionprops(bw, 'centroid','area');
    
    dist2center = norm(stat.Centroid./fliplr(size(bw)) - [0.5 0.5]);
    objSize = stat.Area/(size(bw,1)*size(bw,2));
    
    centerBias(cur_img) = dist2center;
    objectSize(cur_img) = objSize;
    
    disp(num2str(cur_img))

%     [center, objSize] = getImageProperty('~', [data_info.dataPath, num2str(data_info.testImgTable(cur_img,1)),'.png'], 0.5);
%     
%     meanCenterBias = 0;
%     meanSize = 0;
%     
%     for i = 1: size(center,1)
%         meanCenterBias = meanCenterBias + norm(center(i,:)-[0.5 0.5]);  
%         
%         meanSize = meanSize + objSize(i);     
%     end
%     
%     centerBias(cur_img) = meanCenterBias/size(center,1);
%     objectSize(cur_img) = meanSize/size(center,1);
end

%% show stat

figure
scatter(centerBias, F_measure(:,max_th_idx))
xlim([0 0.5])
xlabel('Normalized Distance between Object Center and Image Center')
ylabel('F-measure')

figure
scatter(objectSize, F_measure(:,max_th_idx))
% xlim([0 1])
xlabel('Normalized Object Size')
ylabel('F-measure')

% center bias
th_centerBias = 0.2;

farCenter = F_measure(find(centerBias>=th_centerBias),max_th_idx);
nearCenter = F_measure(find(centerBias<th_centerBias),max_th_idx);

mean(farCenter)
mean(nearCenter)

std(farCenter)
std(nearCenter)

[h, p] = ttest2(farCenter, nearCenter,'Tail','left')

% size
th_size = 0.1;

bigSize = F_measure(find(objectSize>=th_size),max_th_idx);
smallSize = F_measure(find(objectSize<th_size),max_th_idx);

mean(bigSize)
mean(smallSize)

[h, p] = ttest2(bigSize, smallSize,'Tail','right' )