%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.


%% A quick demo for using a pre-trained model
clear;
clc;

% get the param
param = globalParam;
assert(exist(param.modelPath) ==2);

% load the model
load('./models/pascal_100_gbvs_forestModel', 'forest');

% load image
% img = imread('football.jpg');
img = imread('th.jpg');

%%
% run our code
finalMask = salobj_test_img(img, param, forest);

% display results
figure(1), imshow(img)
figure(2), imshow(finalMask)