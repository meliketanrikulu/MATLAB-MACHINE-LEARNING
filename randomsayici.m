%my_new_table=readtable('fisheriris.csv');

load fisheriris
my_new_table=meas(:,3:4);
rand_num=randperm(150);
Train_Data=my_new_table(rand_num(1:120),:);
Test_Data=my_new_table(rand_num(121:end),:);
%cell2mat(Train_Data)
%load fisheriris;
% for i=0:1:120
%  A  =Test_Data(i)
% %meas(A,:)
% end

