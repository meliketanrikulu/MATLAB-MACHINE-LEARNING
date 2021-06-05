function stats = function1(group,grouphat)
% INPUT
% group = true class labels
% grouphat = predicted class labels
%
% OR INPUT
% stats = confusionmatStats(group);
% group = confusion matrix from matlab function (confusionmat)
%
% OUTPUT
% stats is a structure array
% stats.confusionMat
%               Predicted Classes
%                    p'    n'
%              ___|_____|_____| 
%       Actual  p |     |     |
%      Classes  n |     |     |
%
% stats.accuracy = (TP + TN)/(TP + FP + FN + TN) ; the average accuracy is returned
% stats.precision = TP / (TP + FP)                  % for each class label
% stats.sensitivity = TP / (TP + FN)                % for each class label
% stats.specificity = TN / (FP + TN)                % for each class label
% stats.recall = sensitivity                        % for each class label
% stats.Fscore = 2*TP /(2*TP + FP + FN)            % for each class label
%
% TP: true positive, TN: true negative, 
% FP: false positive, FN: false negative
% 

field1 = 'confusionMat';
if nargin < 2
    value1 = group;
else
    [value1,gorder] = confusionmat(group,grouphat);
end

numOfClasses = size(value1,1);
totalSamples = sum(sum(value1));

[TP,TN,FP,FN,accuracy,sensitivity,specificity,precision,f_score] = deal(zeros(numOfClasses,1));
for class = 1:numOfClasses
   TP(class) = value1(class,class);
   fprintf('class%d_TPR = %d   ',class, TP(class));
   tempMat = value1;
   tempMat(:,class) = []; % remove column
   tempMat(class,:) = []; % remove row
   TN(class) = sum(sum(tempMat));
   FP(class) = sum(value1(:,class))-TP(class);
   fprintf('class%d_FPR = %d   ',class, FP(class));
   FN(class) = sum(value1(class,:))-TP(class);
end

for class = 1:numOfClasses
    accuracy(class) = (TP(class) + TN(class)) / totalSamples;
    sensitivity(class) = TP(class) / (TP(class) + FN(class));
    specificity(class) = TN(class) / (FP(class) + TN(class));
    precision(class) = TP(class) / (TP(class) + FP(class));
    f_score(class) = 2*TP(class)/(2*TP(class) + FP(class) + FN(class));
end

field2 = 'accuracy';  accuracy_value = accuracy;
field3 = 'sensitivity';  sensitivity_value = sensitivity;
field4 = 'specificity';  specificity_value = specificity;
field5 = 'precision';  precision_value = precision;
field6 = 'recall';  sensitivity_value = sensitivity;
field7 = 'Fscore';  f_score_value = f_score;





stats = struct(field1,value1,field2,accuracy_value,field3,sensitivity_value,field4,specificity_value,field5,precision_value,field6,sensitivity_value,field7,f_score_value);

if exist('gorder','var')
    stats = struct(field1,value1,field2,accuracy_value,field3,sensitivity_value,field4,specificity_value,field5,precision_value,field6,sensitivity_value,field7,f_score_value,'groupOrder',gorder);
end
    


