% This code is for generating synthetic deformation

clear all

% addpath(genpath('Synthetic_InSAR_image-main\'));
SAVEWRAP = 1;
outputRoot = 'synthesised_patches\';
mkdir([outputRoot, 'set1\unwrap\deform\']);
mkdir([outputRoot, 'set2\unwrap\deform\']);

if SAVEWRAP == 1
    mkdir([outputRoot, 'set1\wrap\deform\']);
    mkdir([outputRoot, 'set2\wrap\deform\']);
end

% size of input of Alexnet = 227x227 pixels
halfcrop = floor(227/2);

% type to generate deformation
Source_Type = 1;

% default
% Source_Type = 1. %Earthquakes
Quake.Strike = 0;              %strike in degrees
Quake.Dip = 80;                 %dip in degrees
Quake.Rake = -90;               %rake in degrees
Quake.Slip = 1;                 %magnitude of slip vector in metres
Quake.Top_depth = 3;           %depth (measured vertically) to top of fault in kilometres
Quake.Bottom_depth = 6;       %depth (measured vertically) to bottom of fault in kilometres
Quake.Length = 2;             %fault length in kilometres

% Source_Type = 2. Dykes
Dyke.Strike = 0;              %strike in degrees [0-180]
Dyke.Dip = 90;                  %dip in degrees (usually 90 or near 90)
Dyke.Opening = 1;               %magnitude of opening (perpendincular to plane) in metres
Dyke.Top_depth = 2 ;            %depth (measured vertically) to top of dyke in kilometres
Dyke.Bottom_depth = 8 ;        %depth (measured vertically) to bottom of dyke in kilometres
Dyke.Length = 10 ;              %dyke length in kilometres

% Source_Type = 3. Rectangular Sills
Sill.Strike = 0;              %strike (orientation of Length dimension) in degrees [no different]
Sill.Dip = 0;                   %Dip in degrees (usually zero or near zero)
Sill.Opening = 10;             %magnitude of opening (perpendincular to plane) in metres
Sill.Depth = 5;              %depth (measured vertically) to top of dyke in kilometres
Sill.Width = 1;                %depth (measured vertically) to bottom of dyke in kilometres
Sill.Length = 1;               %dyke length in kilometres

% Source_Type = 4. Magma Chamber - point pressure
Mogi.Depth  = 5;                %Depth of Mogi Source
Mogi.Volume = 10*1e6;               %Volume in m^3

% Source_Type = 5. Pressurized Penny-shaped Horizontal Crack (Fialko) - Sill
% Note, this is the slowest to calculate of the various sources

Penny.Depth  = 5;                %Depth of crack in km^3
Penny.Pressure = 1*1e6;         %Pressure of crack in Pa
Penny.Radius  = 5;               %Radius of crack in km^3

maxnum = 5000;
x=-25000:100:25000-100;
y=-25000:100:25000-100;

%% Source_Type = 5. Pressurized Penny-shaped Horizontal Crack (Fialko) - Sill
% -------------------------------------------------------------------------
if Source_Type == 5
    count = 1;
    for Incidence = [2 15 25 160 170 177]
        incidenceName = ['incidence',sprintf('%03d',Incidence)];
        for Heading =  0:72:180
            headingName = [incidenceName, '_heading',sprintf('%03d',Heading)];
            for Radius = 4:6
                Penny.Radius = Radius;	%depth (measured vertically) to bottom of fault in kilometres
                RadiusName = [headingName, '_radius',sprintf('%1d',Radius)];
                for Pressure = 6
                    Penny.Pressure = 10^Pressure;    %depth (measured vertically) to top of fault in kilometres
                    PressureName = [RadiusName, '_pressure',sprintf('%1d',Pressure)];
                    for depth = [4 4.5 5]
                        Penny.Depth  = depth;
                        DepthName = [PressureName, '_depth',sprintf('%0.1f',depth)];
                        for rotate = 0:72:360-72
                            allName = ['Type',num2str(Source_Type),'_',DepthName, '_rotate',sprintf('%03d',rotate)];
                            if count < maxnum
                                disp(allName);
                                [~, los_grid] = generateDeformation(Source_Type, x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
                                % scaling
                                los_grid = los_grid/0.028333*2*pi;
                                los_grid = imrotate(los_grid, rotate,'crop');
                                los_grid = los_grid(round(size(los_grid,1)/2) + (-halfcrop:halfcrop),round(size(los_grid,2)/2) + (-halfcrop:halfcrop));
                                if (range(los_grid(:)) > 10)&&(range(los_grid(:)) < 30)
                                    outputDirUnwrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\unwrap\deform\'];
                                    save([outputDirUnwrap, allName, '.mat'], 'los_grid');
                                    if SAVEWRAP == 1
                                        outputDirWrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\wrap\deform\'];
                                        los_grid_wrap = wrapTo2Pi(los_grid)-pi;
                                        los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
                                        imwrite(los_grid_wrap, [outputDirWrap, allName, '.png']);
                                    end
                                    count = count + 1;
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end
%% Source_Type = 4. Magma Chamber - point pressure
% -------------------------------------------------------------------------
if Source_Type == 4
    count = 1;
    for Incidence = 33%[3 8 12 29 45 140 170 210 330 350]
        incidenceName = ['incidence',sprintf('%03d',Incidence)];
        for Heading =  5:40:330
            headingName = [incidenceName, '_heading',sprintf('%03d',Heading)];
            %for Volume = [6 6.5 6.7 7 7.2 7.5]+0.2
            for Volume = [5 5.5 6 6.2 6.5 6.7 6.9]+0.2
                Mogi.Volume = 10^Volume;                    %dip in degrees
                VolumeName = [headingName, '_vol1e',sprintf('%0.1f',Volume)];
                if Volume<=6
                    %depth_range = [1.5 2];
                    depth_range = [1.5 1.8 2];
                elseif Volume<=6.5
                    depth_range = [1.8 2 2.5 2.8 3];
                elseif Volume<=7
                    %depth_range = [2.5 2.8 3 4 5 5.5 6];
                    depth_range = [2.5 2.8 3 3.5 3.8 4 4.5 5 5.5 6];
                else
                    %depth_range = [5 6 7 7.5 8];
                    depth_range = [5 5.5 6 6.5 7 7.5];
                end
                for depth = depth_range
                    Mogi.Depth = depth;    %depth (measured vertically) to top of fault in kilometres
                    allName = ['Type',num2str(Source_Type),VolumeName, '_depth',sprintf('%0.1f',depth)];
                    
                    if count < maxnum
                        disp(allName);
                        [~, los_grid] = generateDeformation(Source_Type, x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
                        % scaling
                        los_grid = los_grid/0.028333*2*pi;
                        los_grid = los_grid(round(size(los_grid,1)/2) + (-halfcrop:halfcrop),round(size(los_grid,2)/2) + (-halfcrop:halfcrop));
                        disp(range(los_grid(:)))
                        %if (range(los_grid(:)) > 10)&&(range(los_grid(:)) < 60)
                        if (range(los_grid(:)) > 8)&&(range(los_grid(:)) < 60)%60)

                            outputDirUnwrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\unwrap\deform\'];
                            %disp(outputDirUnwrap)
                            save([outputDirUnwrap, allName, '.mat'], 'los_grid');
                            
                            if SAVEWRAP == 1
                                outputDirWrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\wrap\deform\'];
                                los_grid_wrap = wrapTo2Pi(los_grid)-pi;
                                los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
                                imwrite(los_grid_wrap, [outputDirWrap, allName, '.png']);
                            end
                            count = count + 1;
                        end
                    end
                end
            end
        end
    end
end

%% Source_Type = 3. Rectangular Sills
% -------------------------------------------------------------------------
if Source_Type == 3
    count = 1;
    for Incidence = 10:72:360-36
        incidenceName = ['incidence',sprintf('%02d',Incidence)];
        for Heading =  0:72:180
            headingName = [incidenceName, '_heading',sprintf('%03d',Heading)];
            for Dip = [0 10]
                Sill.Dip = Dip;                    %dip in degrees
                DipName = [headingName, '_dip',sprintf('%02d',Dip)];
                if Dip==0
                    Strike_range = 0;
                else
                    Strike_range = 20:90:360;
                end
                for Strike = Strike_range
                    Sill.Strike = Strike;              %strike in degrees
                    strikeName = [DipName, '_strike',sprintf('%03d',Strike)];
                    for Opening = [5 10 15]
                        Sill.Opening = Opening;                  %rake in degrees
                        OpeningName = [strikeName, '_opening',sprintf('%03d',Opening)];
                        for Length = [0.75 1 3]
                            Sill.Length = Length;              %fault length in kilometres
                            LengthName = [OpeningName, '_length',sprintf('%0.1f',Length)];
                            for Width = [2 5]
                                Sill.Width = Width;	%depth (measured vertically) to bottom of fault in kilometres
                                WidthName = [LengthName, '_width',sprintf('%02d',Width)];
                                for depth = [4 5 6]
                                    Sill.Depth = depth;    %depth (measured vertically) to top of fault in kilometres
                                    DepthName = [WidthName, '_depth',sprintf('%1d',depth)];
                                    for rotate = 0:72:360-72
                                        allName = ['Type',num2str(Source_Type),'_',DepthName, '_rotate',sprintf('%1d',rotate)];
                                        if count < maxnum
                                            disp(allName);
                                            [~, los_grid] = generateDeformation(Source_Type, x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
                                            % scaling
                                            los_grid = los_grid/0.028333*2*pi;
                                            los_grid = imrotate(los_grid, rotate,'crop');
                                            los_grid = los_grid(round(size(los_grid,1)/2) + (-halfcrop:halfcrop),round(size(los_grid,2)/2) + (-halfcrop:halfcrop));
                                            if (range(los_grid(:)) > 12)&&(range(los_grid(:)) < 50)
                                                outputDirUnwrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\unwrap\deform\'];
                                                save([outputDirUnwrap, allName, '.mat'], 'los_grid');
                                                if SAVEWRAP == 1
                                                    outputDirWrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\wrap\deform\'];
                                                    los_grid_wrap = wrapTo2Pi(los_grid)-pi;
                                                    los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
                                                    imwrite(los_grid_wrap, [outputDirWrap, allName, '.png']);
                                                end
                                                count = count + 1;
                                            end
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%% Source_Type = 2. Dykes
% -------------------------------------------------------------------------
if Source_Type == 2
    
    count = 1;
    for Incidence = [43 50]
        incidenceName = ['incidence',sprintf('%02d',Incidence)];
        for Heading = [0 192]
            headingName = [incidenceName, '_heading',sprintf('%03d',Heading)];
            for Strike = 0:36:360-36
                Dyke.Strike = Strike;              %strike in degrees
                strikeName = [headingName, '_strike',sprintf('%03d',Strike)];
                for Dip = 70:5:90
                    Dyke.Dip = Dip;                    %dip in degrees
                    DipName = [strikeName, '_dip',sprintf('%02d',Dip)];
                    for Opening = [0.5 0.75]
                        Dyke.Opening = Opening;                  %rake in degrees
                        OpeningName = [DipName, '_opening',sprintf('%03d',Opening)];
                        for Length = [2 5 8]
                            Dyke.Length = Length;              %fault length in kilometres
                            LengthName = [OpeningName, '_length',sprintf('%0.1f',Length)];
                            for Bottom_depth = [2 5]
                                Dyke.Bottom_depth = Bottom_depth;	%depth (measured vertically) to bottom of fault in kilometres
                                Bottom_depthName = [LengthName, '_bdepth',sprintf('%02d',Bottom_depth)];
                                if Bottom_depth==2
                                    Top_depth_range = [0.25 0.5 0.75];
                                else
                                    Top_depth_range = [0.5 0.75 1];
                                end
                                for Top_depth = Top_depth_range
                                    Dyke.Top_depth = Top_depth;    %depth (measured vertically) to top of fault in kilometres
                                    allName = ['Type',num2str(Source_Type),'_',Bottom_depthName, '_tdepth',sprintf('%1d',Top_depth)];
                                    if count < maxnum
                                        disp(allName);
                                        [~, los_grid] = generateDeformation(Source_Type, x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
                                        % scaling
                                        los_grid = los_grid/0.028333*2*pi;
                                        los_grid = los_grid(round(size(los_grid,1)/2) + (-halfcrop:halfcrop),round(size(los_grid,2)/2) + (-halfcrop:halfcrop));
                                        if (range(los_grid(:)) > 12)&&(range(los_grid(:)) < 70)
                                            outputDirUnwrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\unwrap\deform\'];
                                            save([outputDirUnwrap, allName, '.mat'], 'los_grid');
                                            if SAVEWRAP == 1
                                                outputDirWrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\wrap\deform\'];
                                                los_grid_wrap = wrapTo2Pi(los_grid)-pi;
                                                los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
                                                imwrite(los_grid_wrap, [outputDirWrap, allName, '.png']);
                                            end
                                            count = count + 1;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end

%% 1 = Rectangular Dislocation (no opening) - Earthquake
%-------------------------------------------------------------------------
if Source_Type == 1
%     Quake.Slip = 1; %magnitude of slip vector in metres
%     SlipName = ['Slip',sprintf('%03d',Quake.Slip)];

    Heading = 346;    % Heading (azimuth) of satellite measured clockwise from North, in degrees
    Incidence = 39;
    count = 1;
    for Slip = [0.3 0.35 0.4]
        Quake.Slip = Slip; 
        SlipName = ['Slip',sprintf('%03d',Quake.Slip)];
        for Strike = 210-90:10:310-90
            Quake.Strike = Strike;              %strike in degrees
            strikeName = ['strike',sprintf('%03d',Strike)];
            for Dip = 45:2.5:70
                Quake.Dip = Dip;                    %dip in degrees
                DipName = [strikeName, '_dip',sprintf('%02d',Dip)];
                for Rake = -90:5:-40 %[-136:6:-91]
                    Quake.Rake = Rake;                  %rake in degrees
                    RakeName = [DipName, '_rake',sprintf('%03d',Rake)];
                    for Length = [3 5 7] %[5 7 9]
                        Quake.Length = Length;              %fault length in kilometres
                        LengthName = [RakeName, '_length',sprintf('%0.1f',Length)];
                        for Bottom_depth = [9 11.5] %[9 11.5 14]
                            Quake.Bottom_depth = Bottom_depth;	%depth (measured vertically) to bottom of fault in kilometres
                            Bottom_depthName = [LengthName, '_bdepth',sprintf('%02d',Bottom_depth)];
                            %Top_depth_range = [2 4 6];
                            if Bottom_depth==7
                                Top_depth_range = [2 4];
                            elseif Bottom_depth==9
                                Top_depth_range = [2 4 6];
%                             else
%                                 Top_depth_range = [4 6 9];
                            end
                            for Top_depth = Top_depth_range
                                Quake.Top_depth = Top_depth;    %depth (measured vertically) to top of fault in kilometres
                                allName = ['Type',num2str(Source_Type),'_',SlipName, '_',Bottom_depthName, '_tdepth',sprintf('%1d',Top_depth)];
                                if count < maxnum
                                    %                         disp(allName);
                                    [~, los_grid] = generateDeformation(Source_Type, x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
                                    % scaling
                                    los_grid = los_grid/0.028333*2*pi;
                                    los_grid = los_grid(round(size(los_grid,1)/2) + (-halfcrop:halfcrop),round(size(los_grid,2)/2) + (-halfcrop:halfcrop));
                                    if (range(los_grid(:)) > 12)&&(range(los_grid(:)) < 80)
                                        outputDirUnwrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\unwrap\deform\'];
                                        save([outputDirUnwrap, allName, '.mat'], 'los_grid');
%                                         disp(min(los_grid))
%                                         disp(max(los_grid))
%                                         disp(los_grid(:));
%                                         los_grid_unwrap = (los_grid-min(los_grid(:)))/range(los_grid(:));
%                                         imwrite(1-los_grid_unwrap, [outputDirUnwrap, allName, '.png']);

                                        disp(allName)
                                        if SAVEWRAP == 1
                                            outputDirWrap = [outputRoot, 'set', num2str(2-rem(count,2)),'\wrap\deform\'];
                                            los_grid_wrap = wrapTo2Pi(los_grid)-pi;
                                            los_grid_wrap = (los_grid_wrap-min(los_grid_wrap(:)))/range(los_grid_wrap(:));
                                            imwrite(los_grid_wrap, [outputDirWrap, allName, '.png']);
                                        end
                                        count = count + 1;
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
end