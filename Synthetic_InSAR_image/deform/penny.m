function [Ux,Uy,Uz]=penny(mod,coord,mu,v)
% function [Ux,Uy,Uz]=penny(mod,coord,mu,v)
% mod=[x_c,y_c,z_c,R,P]
% *_c = 3D coordinate of centre of crack (metres)
% R = crack radius (metres)
% P = pressure in crack (Pa)
% coord=[x,y] (columns) (metres)
% mu = shear modulus (Pa)
% v = Poisson's ratio
%
% script to create x y z displacement field for penny shaped crack model of
% Fialko et al., 2001
%
% tjw sept 08

R=mod(4);
P=mod(5);
x=coord(:,1)/R;
y=coord(:,2)/R;
x_c=mod(1)/R;
y_c=mod(2)/R;
z_c=mod(3);

% convert pressure into non-dimensional factor
mu = 3e10; %shear modulus
v = 0.25;  %poisson's ratio
Pf = 2*(1-v)*R*P/mu; %Pf = factor to multiply non-dimensional uplift by

h = z_c/R;  %dimensionless crack depth

% The following parameters need to be user-supplied:   
nis=2;       % number of sub-intervals on [0,1] on which integration is done
             % using a 16-point Gauss quadrature (i.e., total of nis*16 points)
eps=1e-5;    % solution accuracy for Fredholm integral equations (stop 
             % iterations when relative change is less than eps)
             
[fi,psi,t,Wt]=fredholm(h,nis,eps);


r = sqrt((x-x_c).^2 + (y-y_c).^2);
[Uz,Ur]=intgr(r,fi,psi,h,Wt,t);

Uz = -Uz*Pf;
Ur = Ur*Pf;

Nx = (x-x_c)./r;  %unit vector from crack centre to obs
Ny = (y-y_c)./r;  %unit vector from crack centre to obs

Ux = Ur.*Nx;
Uy = Ur.*Ny;





