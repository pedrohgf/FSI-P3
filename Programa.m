Dataset = readtable('dataset/wdbc.data.csv');
ID = Dataset{:,1};
Labels = Dataset{:,2};
Features = Dataset{:,3:end};
Partitions = cvpartition(Labels, 'HoldOut', 0.3);

TestFeatures = Features(Partitions.test,:);
TestLabels = Labels(Partitions.test,:);
TrainingFeatures = Features(Partitions.training,:);
TrainingLabels = Labels(Partitions.training,:);


C = logspace(log10(2^-15), log10(2^5), 100);
LinearRightLabels = [];
GaussianRightLabels = [];
LinearRightLabels = [];

for i=1:length(C)
    SVMModelLinear = fitcsvm(TrainingFeatures,TrainingLabels,'KernelFunction','linear','KernelScale','auto','Standardize',true,'BoxConstraint',C(i));   
    LabelsPredicted = predict(SVMModelLinear,TestFeatures);
    LinearRightLabels(i) = sum(strcmp(LabelsPredicted, TestLabels) == 1);
    SVMModelGaussian = fitcsvm(TrainingFeatures,TrainingLabels,'KernelFunction','Gaussian','KernelScale','auto','Standardize',true,'BoxConstraint',C(i));   
    LabelsPredicted = predict(SVMModelGaussian,TestFeatures);
    GaussianRightLabels(i) = sum(strcmp(LabelsPredicted, TestLabels) == 1);
end

SVMModelMultiquadratic = fitcsvm(TrainingFeatures,TrainingLabels,'KernelFunction','Multiquadratic','Standardize',true);
LabelsPredicted = predict(SVMModelGaussian,TestFeatures);
MultiquadraticRightLabels = sum(strcmp(LabelsPredicted, TestLabels) == 1);

figure(1),
semilogx(C, 100*(LinearRightLabels./length(TestLabels)));
title('Box Constraint vs Acurácia (%) - Kernel Linear'),
xlabel('Box Constraint'),
ylabel('Acurácia (%)'),
axis([2^-15 2^5 60 100]),


figure(2),
semilogx(C, 100*(GaussianRightLabels./length(TestLabels)));
title('Box Constraint vs Acurácia (%) - Kernel Gaussiano'),
xlabel('Box Constraint'),
ylabel('Acurácia (%)'),
axis([2^-15 2^5 60 100]),

%% Gerando Modelos Comparáveis de Kernels Linear, Gaussiano e Multiquadrático
BooleanTrainingLabels = zeros(size(TrainingLabels));
BooleanTrainingLabels = strcmp(TrainingLabels, 'B');
% BooleanTrainingLabels(BooleanTrainingLabels == 'M') = '1'; 
SVMModelMultiquadratic = fitcsvm(TrainingFeatures,BooleanTrainingLabels,'KernelFunction','Multiquadratic','Standardize',true);
SVMModelLinear = fitcsvm(TrainingFeatures,BooleanTrainingLabels,'KernelFunction','linear','Standardize',true);
SVMModelGaussian = fitcsvm(TrainingFeatures,BooleanTrainingLabels,'KernelFunction','gaussian','Standardize',true);

SVMModelMultiquadratic = fitPosterior(SVMModelMultiquadratic);
[~,MultiquadraticScore] = resubPredict(SVMModelMultiquadratic);
[Xmulti,Ymulti,Tmulti,AUCmulti] = perfcurve(BooleanTrainingLabels,MultiquadraticScore(:,SVMModelMultiquadratic.ClassNames),'true');

SVMModelLinear = fitPosterior(SVMModelLinear);
[~,LinearScore] = resubPredict(SVMModelLinear);
[Xlin,Ylin,Tlin,AUClin] = perfcurve(BooleanTrainingLabels,LinearScore(:,SVMModelLinear.ClassNames),'true');

SVMModelGaussian = fitPosterior(SVMModelGaussian);
[~,GaussScore] = resubPredict(SVMModelGaussian);
[Xgauss,Ygauss,Tgauss,AUCgauss] = perfcurve(BooleanTrainingLabels,GaussScore(:,SVMModelGaussian.ClassNames),'true');

figure(3),
hold on
plot(Xlin,Ylin),
plot(Xgauss,Ygauss),
plot(Xmulti,Ymulti),
title('Curvas ROC dos Classificadores'),
xlabel('False Positive Rate'),
ylabel('True Positive Rate'),
legend('Kernel Linear', 'Kernel Gaussiano', 'Kernel Multiquadrático'),
hold off


