function [los_grid_wrap, los_grid] = generateDeformation(Source_Type, ...
    x, y, Quake, Dyke, Sill, Mogi, Penny, Heading, Incidence)

%% Define Elastic Lame parameters
lambda = 2.3e10; % units = pascals
mu     = 2.3e10;
v = lambda / (2*(lambda + mu)); %  calculate poisson's ration

%% Calculate LOS_vector from Heading and Incidence
DEG2RAD = pi/180;
sat_inc = 90 - Incidence;
sat_az  = 360 - Heading;
%sat_inc=Incidence
%sat_az=Heading;
los_x=-cos(sat_az*DEG2RAD)*cos(sat_inc*DEG2RAD);
los_y=-sin(sat_az*DEG2RAD)*cos(sat_inc*DEG2RAD);
los_z=sin(sat_inc*DEG2RAD);
LOS_vector = [los_x los_y los_z];  %Unit vector in satellite line of site


%% Set up model parameter vector and calculate coordinates for plotting rectangular extents
if (Source_Type == 1)
    model = [1;1;Quake.Strike;Quake.Dip;Quake.Rake;Quake.Slip;Quake.Length*1000;Quake.Top_depth*1000;Quake.Bottom_depth*1000;1];
    
    
elseif (Source_Type == 2)
    model = [1;1;Dyke.Strike;Dyke.Dip;0;Dyke.Opening;Dyke.Length*1000;Dyke.Top_depth*1000;Dyke.Bottom_depth*1000;2];
elseif (Source_Type == 3)
    model = [1;1;Sill.Strike;Sill.Dip;0;Sill.Opening;Sill.Length*1000;Sill.Depth*1000;Sill.Width*1000;3];
elseif (Source_Type ==4 || Source_Type ==5)
    model = [];
else
    disp (['Error: Source_Type must be 1, 2, 3, 4 or 5'])
    return
end

if (Source_Type == 1 || Source_Type == 2)
    end1x = model(1) + sin(model(3)*DEG2RAD).*model(7)/2;
    end2x = model(1) - sin(model(3)*DEG2RAD).*model(7)/2;
    end1y = model(2) + cos(model(3)*DEG2RAD).*model(7)/2;
    end2y = model(2) - cos(model(3)*DEG2RAD).*model(7)/2;

    c1x = end1x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(8);
    c2x = end1x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9);
    c3x = end2x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9);
    c4x = end2x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(8);
    c1y = end1y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(8);
    c2y = end1y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9);
    c3y = end2y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9);
    c4y = end2y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(8);

        

elseif (Source_Type==3)
    end1x = model(1) + sin(model(3)*DEG2RAD).*model(7)/2;
    end2x = model(1) - sin(model(3)*DEG2RAD).*model(7)/2;
    end1y = model(2) + cos(model(3)*DEG2RAD).*model(7)/2;
    end2y = model(2) - cos(model(3)*DEG2RAD).*model(7)/2;
    c1x = end1x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c2x = end1x - sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c3x = end2x - sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c4x = end2x + sin((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c1y = end1y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c2y = end1y - cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c3y = end2y - cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
    c4y = end2y + cos((model(3)+90)*DEG2RAD).*cos(model(4)*DEG2RAD).*model(9)/2;
end

%% set up regular grid for plots
[xx,yy]=meshgrid(x,y);
xx= reshape(xx,numel(xx),1);
yy= reshape(yy,numel(yy),1);
coords = [xx';yy'];

%% Calculate Displacements using Okada or Mogi formulations
if (Source_Type == 1 || Source_Type ==2 || Source_Type ==3)
    [U,flag]=disloc3d4(model,coords,lambda,mu);
    xgrid = reshape(U(1,:),numel(y),numel(x));
    ygrid = reshape(U(2,:),numel(y),numel(x));
    zgrid = reshape(U(3,:),numel(y),numel(x));
    los_grid = xgrid*LOS_vector(1) + ygrid*LOS_vector(2) + zgrid*LOS_vector(3);
elseif (Source_Type == 4)
    xgrid=rngchn_mogi(1/1000,1/1000,Mogi.Depth,-Mogi.Volume/1e9,coords(2,:)'/1000,coords(1,:)'/1000,v,repmat([1 0 0],length(coords),1));
    ygrid=rngchn_mogi(1/1000,1/1000,Mogi.Depth,-Mogi.Volume/1e9,coords(2,:)'/1000,coords(1,:)'/1000,v,repmat([0 1 0],length(coords),1));
    zgrid=rngchn_mogi(1/1000,1/1000,Mogi.Depth,-Mogi.Volume/1e9,coords(2,:)'/1000,coords(1,:)'/1000,v,repmat([0 0 1],length(coords),1));
    los_grid=rngchn_mogi(1/1000,1/1000,Mogi.Depth,-Mogi.Volume/1e9,coords(2,:)'/1000,coords(1,:)'/1000,v,repmat(LOS_vector,length(coords),1));
    los_grid=reshape(los_grid,numel(y),numel(x));
elseif (Source_Type == 5)
    model = [1,1,Penny.Depth*1000,Penny.Radius*1000,Penny.Pressure];
    [xgrid,ygrid,zgrid]=penny(model,coords',mu,v);
    los_grid = xgrid*LOS_vector(1) + ygrid*LOS_vector(2) + zgrid*LOS_vector(3);
    los_grid=reshape(los_grid,numel(y),numel(x));
end
xgrid = reshape(xgrid,numel(y),numel(x));
ygrid = reshape(ygrid,numel(y),numel(x));
zgrid = reshape(zgrid,numel(y),numel(x));

%% Calculate wrapped interferogram
los_grid_wrap = mod(los_grid+10000,0.028333);
