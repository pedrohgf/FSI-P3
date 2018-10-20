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
