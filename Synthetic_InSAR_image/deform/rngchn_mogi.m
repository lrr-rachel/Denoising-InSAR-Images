function [del_rng]=rngchn_mogi(n1,e1,depth,del_v,ning,eing,v,plook);
%
%  USEAGE: [del_rng]=rngchn_mogi(n1,e1,depth,del_v,ning,eing,v,plook);
%
%  INPUT: n1 = local north coord of center of Mogi source (km)
%         e1 = local east coord of center of Mogi source (km)
%         depth = depth of Mogi source (km) for points to calculate
%                 range change. This vector is the depth of the 
%                 mogi source plus the elevation of the point
%                 taken from the DEM.
%         del_v = Volume change of Mogi source (km^3)
%         ning = north coord's of points to calculate range change
%         eing = east coord's of points to calculate range change
%         v = Poisson's ration of material
%
%  OUTPUT: del_rng = range change at coordinates given in ning and eing.
%                    If ning and eing are vectors with the same dimensions,
%                    del_rng is a vector. If ning is a row vector and eing
%                    is a column vecor, del_rng is a matrix of deformation
%                    values...compliant with Feigle and Dupre's useage.
%

%----the parameters below will be fed into this script when it
%----becomes a function:
%ning=50:-.09:0;
%ning=ning';
%eing=0:.09:50;
%plook=[0.380717  -0.087895   0.920505];
%del_v=-0.01;
%depth=8.0;
%n1=21.738;
%e1=23.901;

[m,n]=size(ning);
[mm,nn]=size(eing);

%----coef for bulk modulus pressure <--> volume relation is below
%dsp_coef=(1000000*del_v*15)/(pi*16);

%----coef for McTigue's pressure <--> volume relation is below
%dsp_coef=(1000000*del_v*3)/(pi*4);

%----coef for pressure <--> volume relation is below, allowing variable
%    Poisson's ration v (tjw nov 09)
dsp_coef=1000000*del_v*(1-v)/pi;

if(mm == 1 & n == 1)
  disp('Calculating a matrix of rngchg values')
  del_rng=zeros(m,nn);
  del_d=del_rng;
  del_f=del_rng;
  tmp_n=del_rng;
  tmp_e=del_rng;
  for i_loop=1:m
    tmp_e(i_loop,:)=eing;
  end
  for i_loop=1:nn
    tmp_n(:,i_loop)=ning;
  end
  d_mat=sqrt((tmp_n-n1).^2 + (tmp_e-e1).^2);
  tmp_hyp=((d_mat.^2 + depth.^2).^1.5);
  del_d=dsp_coef*d_mat./tmp_hyp;
  del_f=dsp_coef*depth./tmp_hyp;
  azim=atan2((tmp_e-e1),(tmp_n-n1));
  e_disp=sin(azim).*del_d;
  n_disp=cos(azim).*del_d;
  for i_loop=1:nn
    del_rng(:,i_loop)=[e_disp(:,i_loop),n_disp(:,i_loop),...
                       del_f(:,i_loop)]*plook';
  end
elseif ((mm == 1 & m == 1) | (n ==1 & nn == 1))
  if (n ~= nn)
    error('Coord vectors not equal length!')
  end
%  disp('Calculating a vector of rngchng values')
  del_rng=zeros(size(ning));
  del_d=del_rng;
  del_f=del_rng;
  d_mat=sqrt((ning-n1).^2 + (eing-e1).^2);
  tmp_hyp=((d_mat.^2 + depth.^2).^1.5);
  del_d=dsp_coef*d_mat./tmp_hyp;
  del_f=dsp_coef*depth./tmp_hyp;
  azim=atan2((eing-e1),(ning-n1));
  e_disp=sin(azim).*del_d;
  n_disp=cos(azim).*del_d;
%  disp([e_disp, n_disp, del_f])
  del_rng=[e_disp, n_disp, del_f].*plook;
  del_rng=sum(del_rng')';
%  del_rng=[e_disp, n_disp, del_f]*plook';
%  del_rng=-1.0*del_rng;
  del_rng=-1.0*del_rng/1000; %convert from mm to m tjw ???
else
  error('Coord vectors make no sense!')
end

