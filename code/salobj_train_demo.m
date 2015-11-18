%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.


%% A quick demo for our training and testing pipeline

clear;
clc;

% generate global params
param = globalParam;

% train the model
model = salobj_train(param);

% test the model -> salient object masks
salobj_test(param);

% benchmark the results
salobj_benchmark(param);