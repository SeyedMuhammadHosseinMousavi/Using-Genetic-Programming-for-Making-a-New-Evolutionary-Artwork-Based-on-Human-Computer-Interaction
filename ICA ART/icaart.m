clc;clear;
close all;
%% Data reading
bodyc=imread('bodyc.jpg');
bodyd=imread('bodyd.jpg');
facec=imread('facec.jpg');
faced=imread('faced.jpg');
% Human detection
peopleDetector = vision.PeopleDetector;
[bboxes,scores] = peopleDetector(bodyc);
I1 = insertObjectAnnotation(bodyc,'rectangle',bboxes,scores,'LineWidth',7,'FontSize',45,'Color',{'cyan'});
figure, imshow(I1);title('detection scores');
% Hint
disp(['Check if subject is in 2.5 meter distance from sensor']);
disp(['Yes = Stop   ,   No = Please got to the specific distance']);
% Face detection
faceDetector = vision.CascadeObjectDetector;
bboxes = faceDetector(facec);
IFaces = insertObjectAnnotation(facec,'rectangle',bboxes(3,:),'Face','LineWidth',4,'FontSize',30,'Color',{'magenta'});   
figure;imshow(IFaces);title('Detected faces');
% Extracting SURF features
imset = imageSet('tst','recursive'); 
% Create a bag-of-features from the image database
bag = bagOfFeatures(imset,'VocabularySize',10,'PointSelection','Detector');
% Encode the images as new features
surf = encode(bag,imset);
% Labels
sizefinal=size(surf);
sizefinal=sizefinal(1,2);
surf(1:40,sizefinal+1)=1;
surf(41:80,sizefinal+1)=2;
dataknn=surf(:,1:10);
lblknn=surf(:,end);
% KNN classification
Mdl = fitcknn(dataknn,lblknn,'NumNeighbors',5,'Standardize',1)
rng(1); % For reproducibility
knndat = crossval(Mdl);
classError = kfoldLoss(knndat)
disp(['K-NN Classification Accuracy :   ' num2str(100-classError) ]);
% Hint
disp(['Generating Next artwork based on last feedback ...']);

%% ICA Evolutionary Algorithm Problem Definition
CostFunction=@(x) Sphere(x);        % Cost Function
nVar=5;             % Number of Decision Variables
VarSize=[1 nVar];   % Decision Variables Matrix Size
VarMin=-10;         % Lower Bound of Variables
VarMax= 10;         % Upper Bound of Variables
% ICA Parameters
MaxIt=100;          % Maximum Number of Iterations
nPop=50;            % Population Size
nEmp=10;            % Number of Empires/Imperialists
alpha=1;            % Selection Pressure
beta=2;             % Assimilation Coefficient
pRevolution=0.1;    % Revolution Probability
mu=0.05;            % Revolution Rate
zeta=0.1;           % Colonies Mean Cost Coefficient
ShareSettings;
% Initialization
% Initialize Empires
emp=CreateInitialEmpires();
% Array to Hold Best Cost Values
BestCost=zeros(MaxIt,1);
% ICA Main Loop
for it=1:MaxIt    
    % Assimilation
    emp=AssimilateColonies(emp);    
    % Revolution
    emp=DoRevolution(emp);    
    % Intra-Empire Competition
    emp=IntraEmpireCompetition(emp);    
    % Update Total Cost of Empires
    emp=UpdateTotalCost(emp);    
    % Inter-Empire Competition
    emp=InterEmpireCompetition(emp);    
    % Update Best Solution Ever Found
    imp=[emp.Imp];
    [~, BestImpIndex]=min([imp.Cost]);
    BestSol=imp(BestImpIndex);    
    % Update Best Cost
    BestCost(it)=BestSol.Cost;
    % Show Iteration Information
    disp(['Iteration ' num2str(it) ': Best Cost = ' num2str(BestCost(it))]);
    
end
% Results
figure;
semilogy(BestCost,'LineWidth',2);
xlabel('ICA Iteration');
ylabel('ICA Best Cost');

%% Generating artwork based on ICA step
% Reading
im=imread('t1.png');im=rgb2gray(im);
im2=imread('t4.png');im2=rgb2gray(im2);
im3=imread('t5.png');im3=rgb2gray(im3);
im4=imread('t2.png');im4=rgb2gray(im4);
% Fourier transform
FreqColor=fft2(im4); 
FreqColor = fftshift(FreqColor);
% Customizing variables
finsiz=256;
loopend=round(BestCost(1,1));
findiv=1+(BestImpIndex/10)+0.6;
  y=num2str(BestSol.Cost);
  finfuse=str2num(y(1));
stdmed=alpha;
% Median and std filter
if stdmed == 1
im=medfilt2(imadjust(cast(stdfilt(im),'uint8')));imor=im;
im2=medfilt2(imadjust(cast(stdfilt(im2),'uint8')));im2or=im2;
im3=medfilt2(imadjust(cast(stdfilt(im3),'uint8')));im3or=im3;
im4=medfilt2(imadjust(cast(stdfilt(im4),'uint8')));im4or=im4;
end;
% Same sizing
im=imresize(im, [finsiz finsiz]);
im2=imresize(im2, [finsiz finsiz]);
im3=imresize(im3, [finsiz finsiz]);
im4=imresize(im4, [finsiz finsiz]);
% Main process
tempang=0;
for i = 1 : loopend
imr = imrotate(im,tempang+20);
imr2 = imrotate(im2,tempang-40);
imr3 = imrotate(im3,tempang+40);
imr4 = imrotate(im4,tempang-20);
tform = affine2d([1 0 0; .5 1 0; 1 0 1]);
imr = imwarp(imr,tform);
imr2 = imwarp(imr2,tform);
%
imrsize=size(imr);imrsize=imrsize(1,1);
imr=imresize(imr, [finsiz finsiz]);
imr2=imresize(imr2, [finsiz finsiz]);
imr3=imresize(imr3, [finsiz finsiz]);
imr4=imresize(imr4, [finsiz finsiz]);
%
tmpimr=imresize(imr, [finsiz finsiz]);
tmpimr2=imresize(imr2, [finsiz finsiz]);
tmpimr3=imresize(imr3, [finsiz finsiz]);
tmpimr4=imresize(imr4, [finsiz finsiz]);
%
im=imresize(im, [finsiz finsiz]);
im2=imresize(im2, [finsiz finsiz]);
im3=imresize(im3, [finsiz finsiz]);
im4=imresize(im4, [finsiz finsiz]);
% Blending images
im = imfuse(im,imr);
im2 = imfuse(im2,imr2);
im3 = imfuse(im3,imr3);
im4 = imfuse(im4,imr4);
% Next image with reduction
im=im-tmpimr/findiv;
im2=im2-tmpimr2/findiv;
im3=im3-tmpimr3/findiv;
im4=im4-tmpimr4/findiv;
%
imreg{i}=im;
imreg2{i}=im2;
imreg3{i}=im3;
imreg4{i}=im4;
end;
% Plots
f1=figure;
movegui(f1,'northeast');
for i=1:loopend
    imshow(imreg{i});end;
%
f2=figure;
movegui(f2,'northwest');
for i=1:loopend
    imshow(imreg2{i});end;
%
f3=figure;
movegui(f3,'southeast');
for i=1:loopend
    imshow(imreg3{i});end;
%
f4=figure;
movegui(f4,'southwest');
for i=1:loopend
    imshow(imreg4{i});end;
%
figure;
fr=(log(1+abs(FreqColor))); 
fr=imresize(fr, [finsiz finsiz]);
C1 = imfuse(imreg{finfuse},imreg2{finfuse},'blend','Scaling','joint');
C2 = imfuse(C1,imreg3{finfuse},'blend','Scaling','joint');
C3 = imfuse(C2,imreg4{finfuse},'blend','Scaling','joint');
J = imcomplement(C3);
J2=imcomplement(fr);
imshowpair(J,J2);
subplot(1,3,1)
imshowpair(C3,fr);
subplot(1,3,2)
imshowpair(C3,J);
subplot(1,3,3)
imshowpair(J,J2);
%
figure;
subplot(2,5,1)
subimage(imor);
subplot(2,5,2)
subimage(im2or);
subplot(2,5,3)
subimage(im3or);
subplot(2,5,4)
subimage(im4or);
subplot(2,5,5)
subimage(imreg{end});
subplot(2,5,6)
subimage(imreg2{end});
subplot(2,5,7)
subimage(imreg3{end});
subplot(2,5,8)
subimage(imreg4{end});
subplot(2,5,9)
subimage(C3);
subplot(2,5,10)
imshow(log(1+abs(fr)),[]); 
disp(['Rehabilitation Estimation ...']);

% figure;
% montage(imreg);
% figure;
% montage(imreg2);
% figure;
% montage(imreg3);
% figure;
% montage(imreg4);



