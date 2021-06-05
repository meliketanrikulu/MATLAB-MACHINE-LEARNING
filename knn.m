clear all;close all;clc;
load fisheriris
my_new_table=meas(:,3:4);
rand_num=randperm(150);
Train_Data=my_new_table(rand_num(1:120),:);
Test_Data=my_new_table(rand_num(121:end),:);

x = Train_Data(:,1:2);
Labels = species(rand_num(121:150),:)
species=species(rand_num(1:120),:)
gscatter(x(:,1),x(:,2),species)
legend('Location','best')
Mdl = KDTreeSearcher(x)

% newpoint2 = [5 1.45;6 2;2.75 .75];

newpoint2 = [Test_Data(:,1) Test_Data(:,2)];
gscatter(x(:,1),x(:,2),species)
legend('location','best')
[n2,d2] = knnsearch(Mdl,newpoint2,'k',10);
line(x(n2,1),x(n2,2),'color',[.5 .5 .5],'marker','o',...
   'linestyle','none','markersize',10)
line(newpoint2(:,1),newpoint2(:,2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')

tb = tabulate(species(n2(17,:)));
%Labels = species(rand_num(121:150),:)
for i=1:30
    tb = tabulate(species(n2(i,:)));
     Test_Label(i)=(tb(1))
end
%sonuc=function1(Labels,Test_Label);
Y=categorical(species)
Test_Label=categorical(Test_Label);
Labels=categorical(Labels);
sonuc=function1(Labels,Test_Label);
Tablo=struct2table(sonuc);
disp(Tablo)
%%%%%%%%%%%%%%%%%%%%%%%%%%confusionmatrix%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
X = Train_Data;
Z = species;
figure;
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
predictedY = resubPredict(Mdl);
cm = confusionchart(Labels,Test_Label);



