IM=imread('sunderland.jpg');  % Read in a image
whos
figure;
subplot(4,2,1);
imshow(IM);                     % Display image
title('Input image');
FF = fft(IM);  % Take FFT
IF = uint8(((FF)));
subplot(4,2,2);

imshow(IF);
title('Fourier of Input image:complex representation (real part)');

whos
subplot(4,2,3);
IF = uint8((abs(FF)));

imshow(IF);
title('Fourier of Input image:modulus representation');

whos

subplot(4,2,4);

imshow(angle(FF));
title('Fourier of Input image:argument representation');

whos

IFF = ifft((FF));                 % take IFFT
whos
FINAL_IM = uint8((IFF));      % Take real part and convert back to UINT8
whos
subplot(4,2,5);

imshow(FINAL_IM);
title('Inverse Fourier of Input image fourier:whole');

IFF = ifft(abs(FF));
FINAL_IM = uint8(IFF);      % Take real part and convert back to UINT8
whos
subplot(4,2,6);

imshow(FINAL_IM);
title('Inverse Fourier of Input image fourier:modulus');

