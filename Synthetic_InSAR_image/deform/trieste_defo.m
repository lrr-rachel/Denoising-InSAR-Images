%%%%% Script to produce simulated interferograms using simple elastic
%%%%% sources for Earthquakes, Dykes, Sills and point pressure changes at Magma
%%%%% Chambers
%
% Tim J Wright, University of Leeds, 12 August 2009
% t.wright@see.leeds.ac.uk
%
% Note, the script produces wrapped and unwrapped simulations on a 50x50 km
% grid, with the fault/volcano located at the grid centre.
%
% Requires the following additional files:
%
% disloc3d4.m
% dc3d4.m
% dc3d5.m
% dc3d6.m
% penny.m
% Q.m
% RtWt.m
% fpkernel.m
% fredholm.m
% intgr.m
% rngchn_mogi.m
%
% This script and the subroutines will also run in GNU Octave, but some
% minor modifications will be required to the lines that plot the data at
% the bottom of the script (e.g. quiver and colorbar are not functions in
% octave).
%
%%%%%%%%%%%%%%

%% Edit the following lines to define the source type and parameters
clear all

%% Define Source Type
% 1 = Rectangular Dislocation (no opening) - Earthquake
% 2 = Rectangular Dislocation (opening only) - Dyke
% 3 = Rectangular Dislocation (opening only) - Sill (i.e. horizontal)
% 4 = Point Pressure Source (mogi) - Magma Chamber
% 5 = Pressurized Penny-shaped Horizontal Crack (Fialko) - Sill

Source_Type = 4;


%% Define Source Parameters
% Source_Type = 1. Earthquakes
Quake.Strike = 25;             %strike in degrees 
Quake.Dip = 80;                 %dip in degrees
Quake.Rake = -90;              %rake in degrees
Quake.Slip = 1;                 %magnitude of slip vector in metres
Quake.Top_depth = 1 ;           %depth (measured vertically) to top of fault in kilometres
Quake.Bottom_depth = 10 ;       %depth (measured vertically) to bottom of fault in kilometres
Quake.Length = 10 ;             %fault length in kilometres

% Source_Type = 2. Dykes
Dyke.Strike = 0;              %strike in degrees [0-180]
Dyke.Dip = 90;                  %dip in degrees (usually 90 or near 90)
Dyke.Opening = 0.5;               %magnitude of opening (perpendincular to plane) in metres
Dyke.Top_depth = 2;            %depth (measured vertically) to top of dyke in kilometres
Dyke.Bottom_depth = 5;        %depth (measured vertically) to bottom of dyke in kilometres
Dyke.Length = 8;              %dyke length in kilometres

% Source_Type = 3. Rectangular Sills
Sill.Strike = 0;              %strike (orientation of Length dimension) in degrees [no different]
Sill.Dip = 10;                   %Dip in degrees (usually zero or near zero)
Sill.Opening = 10;             %magnitude of opening (perpendincular to plane) in metres
Sill.Depth = 5;              %depth (measured vertically) to top of dyke in kilometres
Sill.Width = 0.5;                %depth (measured vertically) to bottom of dyke in kilometres
Sill.Length = 1;               %dyke length in kilometres

for value = 0:60:330
% Source_Type = 4. Magma Chamber - point pressure
Mogi.Depth  = 2.8;%3.5;                %Depth of Mogi Source
Mogi.Volume = 10^6.5;               %Volume in m^3

% Source_Type = 5. Pressurized Penny-shaped Horizontal Crack (Fialko) - Sill
% Note, this is the slowest to calculate of the various sources
Penny.Depth  = 5;                %Depth of crack in km^3
Penny.Pressure = 10^6;         %Pressure of crack in Pa
Penny.Radius  = 5;               %Radius of crack in km^3

%% Set Heading and Incidence Angle for Satellite
Heading = value;    % Heading (azimuth) of satellite measured clockwise from North, in degrees
Incidence = 5;%20;%      % Incidence angle of satellite in degrees

%%
%%%%%%%%%%%%%%%%%%%
% DO NOT EDIT ANY TEXT BELOW HERE
%%%%%%%%%%%%%%%%%%%%%
x=[-25000:100:25000];
y=[-25000:100:25000];

[los_grid_wrap, los_grid] = generateDeformation(Source_Type, ...
   x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence);
disp([value range(los_grid(:)/0.028333*2*pi)])

%% Plot interferograms
figure;
% subplot(2,2,1)
% imagesc(x/1000,y/1000,los_grid)
% hold on
% if (Source_Type == 1 || Source_Type == 2)
%     plot([end1x;end2x]/1000,[end1y;end2y]/1000,'w')
%     plot([end1x]/1000,[end1y]/1000,'wo')
% elseif (Source_Type==3 || Source_Type==4)
%     plot(1/1000,1/1000,'wo')
% elseif (Source_Type==5)
%     hp=model(1)/1000; kp=model(2)/1000; rp=model(4)/1000; N=256;
%     tp=(0:N)*2*pi/N;
%     plot(rp*cos(tp)+hp, rp*sin(tp)+kp ,'w');
% end
% if (Source_Type==3)
%     plot([c1x;c2x;c3x;c4x;c1x]/1000,[c1y;c2y;c3y;c4y;c1y]/1000,'w')
% end
% axis xy
% axis image
% title('Unwrapped simulation')
% xlabel('easting (km)')
% ylabel('northing (km)')
% h=colorbar('vert');
% ylabel(h,'metres')
%
% subplot(2,2,2)
imagesc(los_grid_wrap/0.028333*2*pi-pi);  colormap jet; axis image;
hold on
% if (Source_Type == 1 || Source_Type == 2)
%     plot([end1x;end2x]/1000,[end1y;end2y]/1000,'w')
%     plot([end1x]/1000,[end1y]/1000,'wo')
% elseif (Source_Type==3 || Source_Type==4)
%     plot(1/1000,1/1000,'wo')
% elseif (Source_Type==5)
%     plot(rp*cos(tp)+hp, rp*sin(tp)+kp ,'w');
% end
% if (Source_Type==3)
%     plot([c1x;c2x;c3x;c4x;c1x]/1000,[c1y;c2y;c3y;c4y;c1y]/1000,'w')
% end
axis xy
axis image
title(['Wrapped simulation value ', num2str(value)])
xlabel('easting (km)')
ylabel('northing (km)')
h=colorbar('vert');
ylabel(h,'radians');
end
