% Normalisation Code to run before testing 

% Image Normalization - Normalize all images to the common scale 0-1

% Recommand normalising the test data first 
% by RLin, CS@UoB

clear all

% Input Dir
stDir = 'Denoising_earthquakes_project/ARG_EQ_2020/unwrapped/cropped images/Coseismic_interferograms_cropped/';


stList = dir([stDir,'*.tif']);

for k = 1:length(stList)
    % read images
    st = imread(strcat(stDir,stList(k).name()));

    stt = st;             % temo copy st to stt, so st will be intact
    stt(isnan(stt)) = 0;  % first replace nan with 0, in order to calculate mean of stt

%     stt = stt - mean(stt(:)); % shift stt to mean

    % Check ST Nan, isnan(st) remembers the nan postion in st; nan in st
    % will be the same as nan in stt;
    stt(isnan(st)) = min(stt(:)); % Replace NaN values with min value (or any other value) in stt

    
    % Normalisation (Min-Max Scaling)
    Img_st = uint8((stt-min(stt(:)))/range(stt(:))*255);
    %save unwrap ST 
    imwrite(Img_st, ['data/Test_png/', stList(k).name(), '.png']);

    %save wrap ST
%     st_wrap = wrapTo2Pi(st)-pi;
%     st_wrap = (st_wrap-min(st_wrap(:)))/range(st_wrap(:));
%     imwrite(st_wrap, ['data/Test_wrap/', stList(k).name(), '.png']);
end
