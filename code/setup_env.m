%% setup the env, include all necessary 3rd-party libs

% add mcg
cd mcg;
install;
cd ..;

% add piotr's toolbox
addpath(genpath('./dollar_toolbox'));
 
% add gbvs
addpath(genpath('./gbvs'), '-begin');
cd gbvs
gbvs_install;
cd ..;

% add benchmark code by Xiaodi
addpath('../benchmark')
