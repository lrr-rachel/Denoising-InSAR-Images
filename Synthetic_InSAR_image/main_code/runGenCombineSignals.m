% This step involves combining pre-prepared Noise ST with synthetically generated
% Deformation D (generated after running runGenDeformation.m).

% The combination-signal code has been modified to incorporate essential pre-processing
% steps, including zero-centering (shifting to the mean) and normalization.

% For generating the training dataset, it is recommended to perform normalization
% based on the global range across deformation (D) and combined noisy data (DST).

% run runGlobalMinMax.m to log the Min Max values first

clear all

% synthetic dataset root
rootDir = 'synthesised_patches/';

% parameters
samplesPerClass = 500; % must be smaller than or equal to the maximum number of files in the direc
patchDirUnwrap = 'synthesised_patches/Deformation/unwrap/';
patchDirWrap = 'synthesised_patches/Deformation/wrap/';
stDir = 'NoiseST/';

% calculate and log the global MIN and MAX (runGlobalMinMax.m)
globalMIN = -22.74;
globalMAX = 21.58;

% unwrapped output dir
OutputDirDST = 'synthesised_patches/DST_png/';
OutputDirD = 'synthesised_patches/D_png/';
mkdir(outputDirDST);
mkdir(outputDirD);

indName = 0;

for setnum = 1:10 

    % input directories
    deformList = dir([patchDirUnwrap,'*.mat']);

    stList = dir([stDir,'*.tif']);

    % get shuffled index for signal-combination
    indDeform = randperm(length(deformList),samplesPerClass);
    indST = randperm(length(stList),samplesPerClass);


    % output directories
    outputDirWrap = [patchDirWrap,'combine/'];
    outputDirUnwrap = [patchDirUnwrap, 'combine/'];
    mkdir(outputDirWrap);
    mkdir(outputDirUnwrap);

    % marging process
    for k = 1:samplesPerClass

        % get deformation
        load([patchDirUnwrap, deformList(indDeform(k)).name(1:end-3),'mat']);
        % get ST
        st = imread(strcat(stDir,stList(indST(k)).name()));
    
        % Check ST Nan 
        st(isnan(st)) = 0; % Replace NaN values with 0 (or any other value)
        st = imresize(st,[512 512]);

         % Check D Nan
        los_grid(isnan(los_grid)) = 0;
        los_grid = imresize(los_grid,[512 512]);


        if range(los_grid(:))<=15
            los_grid = los_grid*18/range(los_grid(:));
            save([patchDirUnwrap, deformList(indDeform(k)).name(1:end-3),'mat'],'los_grid');
        elseif range(los_grid(:))>=50
            los_grid = los_grid*40/range(los_grid(:));
            save([patchDirUnwrap,  deformList(indDeform(k)).name(1:end-3),'mat'],'los_grid');
        end


        %##########
        % Shift D to mean
        los_grid = los_grid-mean(los_grid(:));


        % Zero-Mean D + Zero-Mean ST
        insarImg = los_grid + st;


        % save Unwrap D
        % (zero-centre D - global MIN of D and DST)/(global MAX - global MIN)
        los_grid_unwrap = uint8((1-((los_grid-globalMIN)/(globalMAX-globalMIN)))*255);
        imwrite(los_grid_unwrap, [OutputDirD, num2str(indName), '.png']);    

%         disp(['synthesised_patches/D_png/', num2str(indName), '.png'])

        % save wrap D
        los_grid_wrap = wrapTo2Pi(los_grid)-pi;
        los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
        % imwrite(los_grid_wrap, ['synthesised_patches/D_wrap/', num2str(indName), '.png']);


        %save unwrap DST
        Img_unwrap = uint8((1-((insarImg-globalMIN)/(globalMAX-globalMIN)))*255);
        imwrite(Img_unwrap, [OutputDirDST, num2str(indName), '.png']);

        mask = imerode(st~=0,strel('disk',3));
        insarWrap = (wrapTo2Pi(insarImg)-pi);
        insarWrap = (insarWrap-min(insarWrap(:)))/range(insarWrap(:)).*mask;

        %save DST wrap 
        % imwrite(insarWrap, [outputDirWrap, num2str(indName), '.png']);

        % save unwrap ST
        % Img_st = uint8((1-((st-min(st(:)))/range(st(:))))*255);
        % imwrite(Img_st, ['synthesised_patches/ST_png/', num2str(indName), '.png']);
        % 
        % save wrap ST
        % st_wrap = wrapTo2Pi(st)-pi;
        % st_wrap = (st_wrap-min(st_wrap(:)))/range(st_wrap(:));
        % imwrite(st_wrap, ['synthesised_patches/ST_wrap/', num2str(indName), '.png']);

        indName = indName+1;
    end
end





