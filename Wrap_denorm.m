% de-normalisation and warpping code
% by RLin, CS@UoB

% read original Input image
I = imread('data/cropped_20201107_20201201.geo.unw.tif');
tempI = I;             % temp copy I to tempI, so I will be intact
tempI(isnan(tempI)) = 0;  % first replace nan with 0, in order to calculate mean of tempI


% Check Nan, isnan(I) remembers the nan postion in I; nan in I
% will be the same as nan in tempI;
tempI(isnan(I)) = min(tempI(:)); % Replace NaN values with min value (or any other value) in tempI


% read denoised Output image
O = imread('data/output.png');

% convert input to double, range 0-1
ConvI = im2double(tempI);   
ConvO = im2double(O);  

% calculate range and min for original input
minI = min(tempI(:));
rangeI = max(tempI(:))-min(tempI(:));

% de-normalisation
% convert back to initial pixel value (beform norm)
O = ( ConvO*(double(rangeI)) + double(minI) ); 


% wrapping
% wrapTo2Pi wraps to interval [0, 2*pi] 
wrapped = wrapTo2Pi(O)-pi;
imwrite(wrapped, 'wrapped.png');
imshow(wrapped);


