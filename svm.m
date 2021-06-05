clear all;close all;clc;
load fisheriris;
my_new_table=meas(:,3:4);
rand_num=randperm(150);
Train_Data=my_new_table(rand_num(1:120),:);
Test_Data=my_new_table(rand_num(121:end),:);
Labels = species(rand_num(121:150),:);

X = Train_Data(:,1:2);
Y = species(rand_num(1:120));

figure
gscatter(X(:,1),X(:,2),Y);
h = gca;
lims = [h.XLim h.YLim]; % Extract the x and y axis limits
title('{\bf Scatter Diagram of Iris Measurements}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend('Location','Northwest');

SVMModels = cell(3,1); %empty matris
classes = unique(Y); %tekrarlananlar silinir 3 veri kalýr
rng(1); % For reproducibility

for j = 1:numel(classes)%numel=number of element in array
    indx = strcmp(Y,classes(j)); % Create binary classes for each classifier KARÞILAÞTIr
    SVMModels{j} = fitcsvm(X,indx,'ClassNames',[false true],'Standardize',true,...
        'KernelFunction','linear','BoxConstraint',1);
end

d = 0.02;
[x1Grid,x2Grid] = meshgrid(min(X(:,1)):d:max(X(:,1)),...
    min(X(:,2)):d:max(X(:,2)));
xGrid = [x1Grid(:),x2Grid(:)];
N = size(xGrid,1);
Scores = zeros(N,numel(classes));

for j = 1:numel(classes)
    [~,score] = predict(SVMModels{j},xGrid);
    Scores(:,j) = score(:,2); % Second column contains positive-class scores
end

[~,maxScore] = max(Scores,[],2);

figure
h(1:3) = gscatter(xGrid(:,1),xGrid(:,2),maxScore,...
    [0.1 0.5 0.5; 0.5 0.1 0.5; 0.5 0.5 0.1]);
hold on
h(4:6) = gscatter(X(:,1),X(:,2),Y);
title('{\bf Iris Classification Regions}');
xlabel('Petal Length (cm)');
ylabel('Petal Width (cm)');
legend(h,{'setosa region','versicolor region','virginica region',...
    'observed setosa','observed versicolor','observed virginica'},...
    'Location','Northwest');
axis tight
hold off
%%%%%%%%%
% XTest = X(1:10,1:2);
% [label,score] = predict(SVMModel,Test_Data);

newpoint2 = [Test_Data(:,1) Test_Data(:,2)];

line(newpoint2(:,1),newpoint2(:,2),'marker','x','color','k',...
   'markersize',10,'linewidth',2,'linestyle','none')
Y=categorical(Y);
Labels=categorical(Labels);
Md1 = fitcecoc(Train_Data,Y);
Test_Label=predict(Md1,Test_Data);
sonuc=function1(Labels,Test_Label);
%sonuz=cell2struct(sonuc);
Tablo=struct2table(sonuc);
disp(Tablo)

%%%%%%%%%%%%%%%%%%%%%%%%%%confusionmatrix%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%55
X = Train_Data;
Z = species;
figure;
Mdl = fitcknn(X,Y,'NumNeighbors',5,'Standardize',1);
predictedY = resubPredict(Mdl);
cm = confusionchart(Labels,Test_Label);

