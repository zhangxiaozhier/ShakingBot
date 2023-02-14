Input_path_rgb = './rgb/rgb_';
Input_path_depth = './depth/depth_';
Output_path='./colorizedDepth/CD_';

namelist = dir(strcat(Input_path_rgb,'*.jpg'));
len = length(namelist);

% disp(['len:',num2str(len)]);

for i = 0:len
    depth_data_path = [Input_path_depth,num2str(i),'.png'];
    rgb_data_path = [Input_path_rgb,num2str(i),'.jpg'];
    [colorizedDepth,coord,binaryDepth] = alignImage(depth_data_path,rgb_data_path,'IntrinsicRGB','InvIntrinsicIR','TransformationD-C');
    imwrite(colorizedDepth,[Output_path,num2str(i),'.jpg']);
    fprintf('Currently saving the %d alignment image\n',i);
%     disp([Input_path_rgb,num2str(i),'.png'])
end
disp('Image alignment ends')
