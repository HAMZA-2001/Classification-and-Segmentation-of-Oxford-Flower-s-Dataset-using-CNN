% Load and preprocess images
scriptDir = pwd;
createDir(scriptDir, 'data');
dataDir = fullfile(scriptDir, 'data');
imgDir = fullfile(scriptDir, 'daffodilSeg');

splitTrainTest(imgDir,dataDir,0.8)

imageDir = fullfile(scriptDir, 'data/train/ImageRsz256');
labelDir = fullfile(scriptDir, 'data/train/LabelRsz256');
imds = imageDatastore(imageDir);

classNames = ["flower", "background"];
pixelLabelID = [1, 3];
pxds = pixelLabelDatastore(labelDir, classNames, pixelLabelID, 'FileExtensions', '.png', 'ReadFcn', @(x) uint8(replacePixels(imread(x), [0, 2, 4], 3)));

% Load pre-trained DeepLabv3+ with a more powerful backbone (ResNet-50)
net = deeplabv3plusLayers([256, 256, 3], 2, 'resnet50');

% Data augmentation
augmenter = imageDataAugmenter( ...
    'RandXReflection', true, ...
    'RandRotation', [-10, 10], ...
    'RandXTranslation', [-10 10], ... % Add X translation
    'RandYTranslation', [-10 10]); % Add Y translation

impxds = pixelLabelImageDatastore(imds, pxds, 'DataAugmentation', augmenter);


% Set up training options
opts = trainingOptions( ...
    'adam', ...
    'InitialLearnRate', 1e-4, ...
    'MaxEpochs', 30, ...
    'MiniBatchSize', 32, ...
    'Plots', 'training-progress',...
    'Shuffle', 'every-epoch', ...
    'VerboseFrequency', 2, ...
    'ExecutionEnvironment', 'gpu', ...
    'L2Regularization', 0.0005);


% Train the network
net = trainNetwork(impxds, net, opts);

% Save the trained network
save('segmentnet.mat', 'net')

% Load the trained network
load('segmentnet.mat', 'net')

% test images dir
imgtestdir = fullfile(scriptDir, 'data/test/ImageRsz256');
tsds = imageDatastore(imgtestdir);

testLabelDir = fullfile(scriptDir, 'data/test/LabelRsz256');
tsPxds = pixelLabelDatastore(testLabelDir, classNames, pixelLabelID, 'FileExtensions', '.png', 'ReadFcn', @(x) uint8(replacePixels(imread(x), [0, 2, 4], 3)));

% Create a combined datastore for test images and their labels
testData = combine(tsds, tsPxds);

createDir(scriptDir, 'final_resent_50')

% The directory where your images are stored (relative path)
outDir = fullfile(scriptDir, 'final_resent_50');

%Do segmentation, save output images to disk
pxdsResults = semanticseg(tsds,net,"WriteLocation", outDir);

% Evaluate the performance of the trained network on the test data
evalResult = evaluateSemanticSegmentation(pxdsResults, testData);

% Display the evaluation results
disp(evalResult);


index1 = 1;
index2 = 2;

overlayOut = labeloverlay(readimage(tsds,index1),readimage(pxdsResults,index1)); %overlay
figure
imshow(overlayOut);
title('overlayOut')
overlayOut = labeloverlay(readimage(tsds,index2),readimage(pxdsResults,index2)); %overlay
figure
imshow(overlayOut);
title('overlayOut2')


function createDir(rootDir, dname)
    % Check if the directory exists. If yes, delete it.
    if exist(fullfile(rootDir, dname), 'dir')
        rmdir(fullfile(rootDir, dname), 's'); % 's' option allows for non-empty directory removal
    end

    % Create the directory
    mkdir(rootDir,dname);
end

function img = replacePixels(img, oldVals, newVal)
    for i = 1:numel(oldVals)
        img(img == oldVals(i)) = newVal;
    end
end

function splitTrainTest(srcimageDir,dataSetDir,trainRatio)
    srcImg=fullfile(srcimageDir, 'ImagesRsz256');
    srcLabel=fullfile(srcimageDir, 'LabelsRsz256');
    % List all files in the directories
    files1 = dir(fullfile(srcImg, '*.png'));  % Adjust the file type if needed
    files2 = dir(fullfile(srcLabel, '*.png'));  % Adjust the file type if needed
    
    % Randomly permute the file indices
    indices = randperm(length(files1));
    
    % Determine the index at which to split the files into train and test
    splitIndex = round(trainRatio * length(files1));
    
    % Create directories for the training and testing sets
    if ~exist(fullfile(dataSetDir, 'train'), 'dir')
        mkdir(fullfile(dataSetDir, 'train'));
        mkdir(fullfile(dataSetDir, 'train/ImageRsz256'));
        mkdir(fullfile(dataSetDir, 'train/LabelRsz256'));
    end
    if ~exist(fullfile(dataSetDir, 'test'), 'dir')
        mkdir(fullfile(dataSetDir, 'test'));
        mkdir(fullfile(dataSetDir, 'test/ImageRsz256'));
        mkdir(fullfile(dataSetDir, 'test/LabelRsz256'));
    end

    
    % Move the files to the appropriate directories
    for i = 1:length(indices)
        if i <= splitIndex
            % This file is part of the training set

            copyfile(fullfile(files1(indices(i)).folder,files1(indices(i)).name), fullfile(dataSetDir, 'train/ImageRsz256'));
            copyfile(fullfile(files2(indices(i)).folder,files2(indices(i)).name), fullfile(dataSetDir, 'train/LabelRsz256'));
        else
            % This file is part of the testing set
            copyfile(fullfile(files1(indices(i)).folder,files1(indices(i)).name), fullfile(dataSetDir, 'test/ImageRsz256'));
            copyfile(fullfile(files2(indices(i)).folder,files2(indices(i)).name), fullfile(dataSetDir, 'test/LabelRsz256'));
        end
    end
end



