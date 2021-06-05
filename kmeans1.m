clear all; close all;clc;
load fisheriris
my_new_table=meas(:,3:4);
rand_num=randperm(150);
Train_Data=my_new_table(rand_num(1:120),:);
Test_Data=my_new_table(rand_num(121:150),:);


X = Train_Data(:,1:2);
Y = species(rand_num(1:120));

k=3;      
[idx,C] = kmeans(X,k);

figure
gscatter(X(:,1),X(:,2),idx,'bgm')
hold on
plot(C(:,1),C(:,2),'kx')
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid')

[~,idx_test] = pdist2(C,Test_Data,'euclidean','Smallest',1);
gscatter(Test_Data(:,1),Test_Data(:,2),idx_test,'bgm','ooo')
legend('Cluster 1','Cluster 2','Cluster 3','Cluster Centroid', ...
    'Data classified to Cluster 1','Data classified to Cluster 2', ...
    'Data classified to Cluster 3')

Labels = species(rand_num(121:150),:);
tic;
Md1 = fitcecoc(X,Y);
Training_Time=toc;
fprintf('Training Time= %.3f \n',Training_Time)
tic;
Test_Label=predict(Md1,Test_Data);
Test_Time=toc;
fprintf('Test Time= %.3f \n',Test_Time)
Y=categorical(Y);
Labels=categorical(Labels);
Test_Label=categorical(Test_Label);
sonuc=function1(Labels,Test_Label);
 
Tablo=struct2table(sonuc)
disp(Tablo)
