% This calculates the global min and max values. 

clear all

% synthetic dataset root
rootDir = 'synthesised_patches/';

% parameters
samplesPerClass = 500; % must be smaller than or equal to the maximum number of files in the direc
patchDirUnwrap = 'synthesised_patches/Deformation/unwrap/';
patchDirWrap = 'synthesised_patches/Deformation/wrap/';
stDir = 'NoiseST/';

indName = 0;


for setnum = 1:10 

    % input directories
    deformList = dir([patchDirUnwrap,'*.mat']);

    stList = dir([stDir,'*.tif']);

    % get shuffled index for signal-combination
    indDeform = randperm(length(deformList),samplesPerClass);
    indST = randperm(length(stList),samplesPerClass);

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

        %calculate D min max 
        minValuesD(k) = min(los_grid(:));
        maxValuesD(k) = max(los_grid(:));
%         disp(minValuesD(k))
%         disp(maxValuesD(k))


        % Zero-Mean D + Zero-Mean ST
        insarImg = los_grid + st;

         %calculate DST min max 
        minValuesDST(k) = min(insarImg(:));
        maxValuesDST(k) = max(insarImg(:));

        indName = indName+1;
    end
end

 %calculate min max 
GMinValueD = min(minValuesD);
GMinValueDST = min(minValuesDST);
fprintf('Global Min of D values: %.2f\n', GMinValueD); % -20.11
fprintf('Global Min of DST values: %.2f\n', GMinValueDST); % -22.74

GMaxValueD = max(maxValuesD);
GMaxValueDST = max(maxValuesDST);
fprintf('Global Max of D values: %.2f\n', GMaxValueD); % 5.45
fprintf('Global Max of DST values: %.2f\n', GMaxValueDST); % 21.58

% Compare and print the results
fprintf('Global Min Value:\n');
if GMinValueD < GMinValueDST
    fprintf(GMinValueD);
else
    fprintf(GMinValueDST);
end

fprintf('Global Max Value:\n');
if GMaxValueD > GMaxValueDST
    fprintf(GMaxValueD);
else
    fprintf(GMaxValueDST);
end

