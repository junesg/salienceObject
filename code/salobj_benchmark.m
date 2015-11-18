%   This is an implmentation of the paper
%   The Secrets of Salient Object Segmentation, CVPR 2014
%   Yin Li (yli440@gatech.edu)
%   Please consider cite our paper if you are using the code
%   Email me if you find bugs or have questions.

function [] = salobj_benchmark(param)
%% the main benchmark function
% benchmark the results of a trained model

colorData = load('customColor.mat');
customColor = colorData.customColor;
customStyle = colorData.customStyle;

salObjAlgs = {'sf'; 'gc'; 'pcas'; 'ft'; ...         % salient object methods
  sprintf('mcg_%s', param.fixAlg{1}) };             % our method
lh = [];

for curAlgNum = 1:size(salObjAlgs, 1)
  
  prec = cell(1, param.numTestImgs);
  recall = cell(1, param.numTestImgs);
  resultMaskFolder = fullfile('../algmaps', param.testDataset, salObjAlgs{curAlgNum});
  
  parfor curFile = 1:param.numTestImgs   
    
    gtfile = fullfile(param.testSobjFolder, sprintf('%d.png', param.testList(curFile)));
    curGT = im2double(imread(gtfile));
    
    algfile = fullfile(resultMaskFolder, sprintf('%d.png', param.testList(curFile)));
    curAlgMap = im2double(imread(algfile));
    
    [curPrec, curRecall] = prCore.prCount(curGT, curAlgMap, false, param);
    
    prec{curFile} = curPrec;
    recall{curFile} = curRecall;
    
  end
  prec = mean(cell2mat(prec), 2);
  recall = mean(cell2mat(recall), 2);
  
  %% draw the curve and best f-measure
  figure(1), lh(end+1) = plot(recall, prec, 'LineWidth', 2);
  set(lh(end),'Color', customColor(curAlgNum, :));
  set(lh(end),'LineStyle', customStyle{curAlgNum}); hold on
  
  [curScore, curTh] = max((1+param.beta^2).*prec.*recall./(param.beta^2.*prec+recall));

  figure(1), sh = scatter(recall(curTh), prec(curTh));
  set(sh,'MarkerFaceColor', customColor(curAlgNum, :));
  set(sh,'MarkerEdgeColor', customColor(curAlgNum, :));
  grid on; hold on;
  
  fprintf('Method %s F-Score=%.4f at th=%.2f\n', salObjAlgs{curAlgNum}, ...
    curScore, param.thList(curTh));
  
end

legend(lh, strrep(salObjAlgs, '_', '+'));




