clear;
% Add Utility Function to the MATLAB Path
utilpath = fullfile(matlabroot, 'toolbox', 'imaq', 'imaqdemos','html', 'KinectForWindows');
addpath(utilpath);
% separate VIDEOINPUT object needs to be created for each of the color and depth(IR) devices
% The Kinect for Windows Sensor shows up as two separate devices in IMAQHWINFO.
hwInfo = imaqhwinfo('kinect')
hwInfo.DeviceInfo(1)
hwInfo.DeviceInfo(2)
% Create the VIDEOINPUT objects for the two streams
colorVid = videoinput('kinect',1)
depthVid = videoinput('kinect',2)
% Set the triggering mode to 'manual'
triggerconfig([colorVid depthVid],'manual');
%  In this example 100 frames are acquired to give the Kinect for Windows sensor sufficient time to
%  start tracking a skeleton.
%
numberofframe=300;
%
colorVid.FramesPerTrigger = numberofframe;
depthVid.FramesPerTrigger = numberofframe;
% Start the color and depth device. This begins acquisition, but does not
% start logging of acquired data.
start([colorVid depthVid]);
% Trigger the devices to start logging of data.
trigger([colorVid depthVid]);
% Retrieve the acquired data
[colorFrameData,colorTimeData,colorMetaData] = getdata(colorVid);
[depthFrameData,depthTimeData,depthMetaData] = getdata(depthVid);
% Stop the devices
stop([colorVid depthVid]);

%% converting 4-d matrix to 3-d rgb images
rgb4=size(colorFrameData)
for i=1:rgb4(1,4)
    rgb{i}=colorFrameData(:,:,:,i)
end;
%%%%%%%%%%%%%%%%%%%%%%%%%

% converting 4-d matrix to 3-d depth images
depth4=size(depthFrameData)
for i=1:depth4(1,4)
    depth{i}=depthFrameData(:,:,:,i)
end;

%% Saving image (Color)
%first delete previews files from specific folder 
delete('rec\*.jpg');
%then saving new files
for i = 1 : rgb4(1,4)
imwrite(rgb{i},['rec\rgb image' num2str(i) '.jpg']);
color{i}=rgb{i};
    disp(['No of saved RGB image :   ' num2str(i) ]);
end;
%%%%%%%%%%%%%%%%%%%%%%%%%
% Saving image (DEPTH)
%first delete previews files from specific folder 
% delete('rec\*.jpg');
%then saving new files
for i = 1 : depth4(1,4)
imwrite(imadjust(depth{i}),['rec\depth image' num2str(i) '.png']);
depthad{i}=imadjust(depth{i});
    disp(['No of saved DEPTH image :   ' num2str(i) ]);
end;
% Converting 2d image to 3d one
for i = 1:depth4(1,4)   
depth3d{i} = cat(3, depthad{i}, depthad{i}, depthad{i});
end;
% uint 16 to uint 8
for i = 1:depth4(1,4)   
depth3dd{i} = im2uint8(depth3d{i}); 
end;
%
delete('rec\*.png');
%then saving new files
for i = 1 : depth4(1,4)
imwrite(depth3dd{i},['rec\depth image' num2str(i) '.jpg']);
    disp(['Depth Save :   ' num2str(i) ]);
end;