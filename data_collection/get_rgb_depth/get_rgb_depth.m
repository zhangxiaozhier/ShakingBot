vid_rgb = videoinput('kinect', 1, 'BGR_1920x1080');
vid_depth = videoinput('kinect', 2, 'Depth_512x424');

src1 = getselectedsource(vid_rgb);
src2 = getselectedsource(vid_depth);

vid_rgb.FramesPerTrigger = 1;
vid_depth.FramesPerTrigger = 1;

preview(vid_rgb);
% preview(vid_depth);
i = 0;
while (i<100)
    start(vid_rgb);
    start(vid_depth);
    rgb_data = getdata(vid_rgb);
    depth_data = getdata(vid_depth);
    imwrite(rgb_data,['./rgb/rgb_',num2str(i),'.jpg']);
    imwrite(depth_data,['./depth/depth_',num2str(i),'.png']);
    fprintf('Currently saving the %d rgb image\n',i);
    fprintf('Currently saving the %d depth image\n',i);
    i = i + 1;
    pause(0.3)
end
% stoppreview(vid_rgb);
% stoppreview(vid_depth);
disp('end')


