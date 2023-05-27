% Get the current script's directory
scriptDirectory = pwd;

% The directory images are stored (relative path)
srcDir = fullfile(scriptDirectory, '17flowers');

% The directory where new folders are created (relative path)
dstDir = fullfile(scriptDirectory, 'Classes');


% List of new directory names
dirNames = {'Daffodil','Snowdrop','Lilly Valley','Bluebell','Crocus','Iris','Tigerlily','Tulip','Fritillary','Sunflower','Daisy','Colts Foot','Dandelion','Cowslip','Buttercup','Windflower','Pansy'}; 

% Set the train ratio (setting up train(70%) test(20%) and validation(10%) datafolders in randomize order)
trainRatio = 0.7;
valRatio = 0.1;

train_test_and_labelling(srcDir, dstDir, dirNames, trainRatio, valRatio);


target = [256, 256];
resize_images(dstDir, target)


trainDir= fullfile(scriptDirectory, 'Classes/train');
traindata = imageDatastore(trainDir,"IncludeSubfolders",true,"LabelSource","foldernames");

testDir = fullfile(scriptDirectory, 'Classes/test');
testdata = imageDatastore(testDir,"IncludeSubfolders",true,"LabelSource","foldernames");

valDir = fullfile(scriptDirectory, 'Classes/val');
valdata = imageDatastore(valDir,"IncludeSubfolders",true,"LabelSource","foldernames");

inputSize = [256 256 3];
layers=[
    imageInputLayer(inputSize, "Name", "imageinput")
    convolution2dLayer(3, 64, "Name", "conv_1_1", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm_1_1")
    reluLayer("Name", "relu_1_1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_1")

    convolution2dLayer(3, 128, "Name", "conv_2_2", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm_2_2")
    reluLayer("Name", "relu_2_2")
    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_2")

    convolution2dLayer(3, 256, "Name", "conv_3_1", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm_3_1")
    reluLayer("Name", "relu_3_1")
    maxPooling2dLayer(2, "Stride", 2, "Name", "maxpool_3")
    
    convolution2dLayer(3, 512, "Name", "conv_3_2", "Padding", "same")
    batchNormalizationLayer("Name", "batchnorm_3_2")
    reluLayer("Name", "relu_3_2")

    dropoutLayer(0.4,"Name","dropout")
    globalMaxPooling2dLayer("Name","gmpool")
    fullyConnectedLayer(128, "Name", "fc_2")
    reluLayer("Name", "relu_fc_2")
    fullyConnectedLayer(17, "Name", "fc_output") 
    softmaxLayer("Name", "softmax")
    classificationLayer("Name","classoutput")];

options = trainingOptions('adam', ...
    'MaxEpochs',30,...,
    'MiniBatchSize', 32, ...,
    'ValidationData', valdata, ...
    'ValidationFrequency',30, ...
    'InitialLearnRate',1e-4,...
    'Shuffle','every-epoch' ,...
    'Verbose', false, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'gpu');

imageAugmenter = imageDataAugmenter( ...
    'RandRotation',[-40,40], ...
    'RandXTranslation',[-4 4], ...
    'RandYTranslation',[-4 4], ...
    'RandXReflection', true);

imageAugmenterT = imageDataAugmenter();

imageSize = [256 256 3];
augimds = augmentedImageDatastore(imageSize,traindata,'DataAugmentation',imageAugmenter);


net = trainNetwork(augimds, layers, options);

save('classnet.mat','net');

load('classnet.mat', 'net');
YPred = classify(net,testdata);
scores = predict(net, testdata);
YTest = testdata.Labels;
accuracy = sum(YPred == YTest)/numel(YTest);


% Compute the confusion matrix
confMat = confusionmat(YTest, YPred);

% Display the confusion matrix
disp(confMat)

% Optionally, you can display a confusion chart
figure;
confChart = confusionchart(YTest, YPred);

% Get the number of classes
numClasses = size(confMat, 1);
classNames = unique(YTest);

precision = zeros(numClasses, 1);
recall = zeros(numClasses, 1);
F1_score = zeros(numClasses, 1);

% Calculate precision and recall for each class
for i = 1:numClasses
    TP = confMat(i,i);
    FP = sum(confMat(:,i)) - TP;
    FN = sum(confMat(i,:)) - TP;
    
    if (TP + FP) == 0
        precision(i) = NaN;
    else
        precision(i) = TP / (TP + FP);
    end

    if (TP + FN) == 0
        recall(i) = NaN;
    else
        recall(i) = TP / (TP + FN);
    end

    F1_score(i) = 2 * (precision(i) * recall(i)) / (precision(i) + recall(i));
end



results = table(classNames, precision, recall,F1_score, 'VariableNames', {'Class', 'Precision', 'Recall','F1'});

disp(results)

figure
idx = 6;
subplot(1,1,1);
img_path = testdata.Files{idx};
Label = testdata.Labels(idx);
I_org = imread(img_path);
[YPred,scores] = classify(net,I_org);
imshow(I_org)
title([char(YPred),' ' , num2str(max(scores))])

function resize_images(Dir,size)
    % Get a list of all JPEG files in the source directory and its subdirectories
    srcFiles = dir(fullfile(Dir, '**', '*.jpg')); 

    % Looping over the source files
    for i = 1:numel(srcFiles)
        % Read the image
        img = imread(fullfile(srcFiles(i).folder, srcFiles(i).name));
        
        % Resizing the image
        resized_img = imresize(img, size);
        
        % Createingthe destination directory if it doesn't exist
        destFolder = strrep(srcFiles(i).folder, Dir, Dir);
        if ~exist(destFolder, 'dir')
            mkdir(destFolder);
        end
       
        imwrite(resized_img, fullfile(destFolder, srcFiles(i).name));
    end
end


function train_test_and_labelling(srcDir, dstDir, dirNames, trainRatio, valRatio)
    srcFiles = dir(fullfile(srcDir, '*.jpg')); 

    assert(numel(srcFiles) >= 80 * numel(dirNames)); 

    % Calculate the number of train, validation, and test images per class
    numTrainImages = floor(80 * trainRatio);
    numValImages = floor(80 * valRatio);
    numTestImages = 80 - numTrainImages - numValImages;

    % Create train, validation, and test directories
    trainDir = fullfile(dstDir, 'train');
    valDir = fullfile(dstDir, 'val');
    testDir = fullfile(dstDir, 'test');
    if ~exist(trainDir, 'dir')
        mkdir(trainDir);
    end
    if ~exist(valDir, 'dir')
        mkdir(valDir);
    end
    if ~exist(testDir, 'dir')
        mkdir(testDir);
    end

    % Loop over the new directories
    for i = 1:numel(dirNames)

        % Create the new directory for train, validation, and test
        if ~exist(fullfile(trainDir, dirNames{i}), 'dir')
            mkdir(fullfile(trainDir, dirNames{i}));
        end
        if ~exist(fullfile(valDir, dirNames{i}), 'dir')
            mkdir(fullfile(valDir, dirNames{i}));
        end
        if ~exist(fullfile(testDir, dirNames{i}), 'dir')
            mkdir(fullfile(testDir, dirNames{i}));
        end

        % Get a list of indices for the images for this directory
        imageIndices = (i-1)*80 + 1 : i*80;
        % Randomize the order of the images
        randIndices = randperm(80);
        
        % Loop over the images for this directory
        for j = 1:80
            % Calculate the index of the current image in the srcFiles array
            idx = imageIndices(randIndices(j));
            
            % Copy the image to the new train, validation or test directory
            if j <= numTrainImages
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(trainDir, dirNames{i}, srcFiles(idx).name));
            elseif j <= numTrainImages + numValImages
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(valDir, dirNames{i}, srcFiles(idx).name));
            else
                copyfile(fullfile(srcFiles(idx).folder, srcFiles(idx).name), fullfile(testDir, dirNames{i}, srcFiles(idx).name));
            end
        end
    end
end




