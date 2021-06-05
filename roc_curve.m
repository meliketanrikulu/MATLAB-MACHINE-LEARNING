
[X,Y,T,AUC] = perfcurve(species(51:end,:),scores,'virginica');
AUC
plot(X,Y)
xlabel('False positive rate') 
ylabel('True positive rate')
title('ROC for Classification by Logistic Regression')