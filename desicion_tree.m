clear all;close all; clc;
load fisheriris
my_new_table=meas(:,3:4);
rand_num=randperm(150);
Train_Data=my_new_table(rand_num(1:120),:);
Test_Data=my_new_table(rand_num(121:end),:);
Labels = species(rand_num(121:150),:);

meas = Train_Data(:,1:2);
species = species(rand_num(1:120));

f = figure;
gscatter(meas(:,1), meas(:,2), species,'rgb','osd');
xlabel('Sepal length');
ylabel('Sepal width');
%N = size(X,1);



t = fitctree(meas(:,1:2), species,'PredictorNames',{'SL' 'SW' });

%[grpname,node] = predict(t,[x y]);
%gscatter(x,y,grpname,'grb','sod')

view(t,'Mode','graph');

dtResubErr = resubLoss(t)

% cvt = crossval(t,'CVPartition',cp);
% dtCVErr = kfoldLoss(cvt)

resubcost = resubLoss(t,'Subtrees','all');
[cost,secost,ntermnodes,bestlevel] = cvloss(t,'Subtrees','all');
plot(ntermnodes,cost,'b-', ntermnodes,resubcost,'r--')
figure(gcf);
xlabel('Number of terminal nodes');
ylabel('Cost (misclassification error)')
legend('Cross-validation','Resubstitution')

[mincost,minloc] = min(cost);
cutoff = mincost + secost(minloc);
hold on
plot([0 20], [cutoff cutoff], 'k:')
plot(ntermnodes(bestlevel+1), cost(bestlevel+1), 'mo')
legend('Cross-validation','Resubstitution','Min + 1 std. err.','Best choice')
hold off
A=(t)
pt = prune(t,'Level',bestlevel);
view(pt,'Mode','graph')

cost(bestlevel+1)

%%%%%%%%%%%%%prediction%%%%%%%%%%%%%%%%%%
Test_Label = predict(t,Test_Data)
%%%%%%%%%%%%%%%%%5functionparameter%%%%%%%%%%%%%%%
Y=categorical(species);
Test_Label=categorical(Test_Label);
Labels=categorical(Labels);
sonuc=function1(Labels,Test_Label)
Tablo=struct2table(sonuc);
disp(Tablo)


%%%%%%%%%%%%%%%%%%%%%%%%%%confusionmatrix%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
X = Train_Data;
Z = species;
figure;
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
predictedY = resubPredict(Mdl);
cm = confusionchart(Labels,Test_Label);