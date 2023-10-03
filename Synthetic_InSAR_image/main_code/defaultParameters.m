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

% Set Heading and Incidence Angle for Satellite
Heading = 192.04;    % Heading (azimuth) of satellite measured clockwise from North, in degrees
Incidence = 23;      % Incidence angle of satellite in degrees
