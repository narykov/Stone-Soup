function []=read_bplane()

y1=4096;
x1=4096;
ppisize=4096;

%make nice figure size
scrsz = get(0,'ScreenSize');
scn=[50, -30, 1*(scrsz(3)-200), 1*(scrsz(4)-50)];

fn='/home/denbridge/rawrecs/fn2.raw'


fid4=fopen(fn)

bpl=fread(fid4,[y1, x1],'uint8');
 figure
 imagesc(bpl)
axis xy
colorbar

fclose(fid4);

figure
[imC] = Polar2Im(bpl,ppisize,'cubic');
imagesc(imC)
colorbar
xlim([0 ppisize])
ylim([0 ppisize])

end

function [imC] = Polar2Im(imP,W,method)
  %Polar2Im turns a polar image (imP) into a cartesian image (imC) of width W
  %method can be: '*linear', '*cubic', '*spline', or '*nearest'.
  imP(isnan(imP))=0;
  w = round(W/2);
  xy = (1:W-w);
  [M N P]= size(imP);
  [x y] = meshgrid(xy,xy);
  n = round(N/4);
  rr = linspace(1,w,M);
  W1 = w:-1:1;
  PM = [2 1 3;1 2 3;2 1 3;1 2 3];
  W2 = w+1:2*w;
  nn = [1:n; n+1:2*n; 2*n+1:3*n; 3*n+1:N;];
  w1 = [W1;W2;W2;W1];
  w2 = [W2;W2;W1;W1];
  aa = linspace(0,90*pi/180,n);
  r = sqrt(x.^2 + y.^2);
  a = atan2(y,x);
  imC= zeros(W,W,P);
  for i=1:4 %turn each quarter into a cartesian image
    imC(w1(i,:),w2(i,:),:)=permute(interp2(rr,aa,imP(:,nn(i,:))',r,a,method),PM(i,:));
  end
  imC(isnan(imC))=0;

  end

