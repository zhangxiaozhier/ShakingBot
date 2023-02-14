function [colorizedDepth,coordinateFrame,binaryDepth] = alignImage(depthImg, rgbImg, intrinsicRGB, intrinsicIR, transformMat)
    
    %read images
    depthImg_raw = imread(depthImg);
    rgbImg_raw = imread(rgbImg);
    intrinsicRGB = load(intrinsicRGB);
    intrinsicIR = load(intrinsicIR);
    transformMat = load(transformMat);
    intrinsicIR = inv(intrinsicIR);
    
    %generate camera parameter object
    RGBcamPar = cameraParameters('IntrinsicMatrix',intrinsicRGB);
    IRcamPar = cameraParameters('IntrinsicMatrix',intrinsicIR);
    
    %undistort images first if there's any skew or lense distortion
    depthImg = undistortImage(depthImg_raw,IRcamPar);
    rgbImg = undistortImage(rgbImg_raw, RGBcamPar);
    
    %get focal length and optical centers out of intrinsic matrices
    fx_d = IRcamPar.FocalLength(1);
    fy_d = IRcamPar.FocalLength(2);
    cx_d = IRcamPar.PrincipalPoint(1);
    cy_d = IRcamPar.PrincipalPoint(2);
    
    fx_c = RGBcamPar.FocalLength(1);
    fy_c = RGBcamPar.FocalLength(2);
    cx_c = RGBcamPar.PrincipalPoint(1);
    cy_c = RGBcamPar.PrincipalPoint(2);
    
    depthHeight = size(depthImg,1);
    depthWidth = size(depthImg,2);
    rgbHeight = size(rgbImg,1);
    rgbWidth = size(rgbImg,2);
    
    %colorizedDepth_6layer image has X,Y,Z,R,G,B values in its layers
   
    colorizedDepth_6layer = zeros(depthHeight,depthWidth,6);
%     colorizedDepth_6layer = repelem(255,depthHeight,depthWidth,6);
    binaryDepth = zeros(depthHeight,depthWidth,3);
    
    for v = 1:(depthHeight)
        for u = 1:(depthWidth)
            %apply intrinsics
            z = single(depthImg(v,u));
            x = single((u - cx_d)*z)/fx_d;
            y = single((v - cy_d)*z)/fy_d;
            %apply extrinsic
            transformed = (transformMat * [x;y;z;1])';
            colorizedDepth_6layer(v,u,1) = transformed(1);
            colorizedDepth_6layer(v,u,2) = transformed(2);
            colorizedDepth_6layer(v,u,3) = transformed(3);
                  
        end
    end
    coordinateFrame = colorizedDepth_6layer(:,:,1:3);
    
    for v = 1 : (depthHeight)
        for u = 1 : (depthWidth)
            % Apply RGB intrinsics
            x = (colorizedDepth_6layer(v,u,1) * fx_c / colorizedDepth_6layer(v,u,3)) + cx_c;
            y = (colorizedDepth_6layer(v,u,2) * fy_c / colorizedDepth_6layer(v,u,3)) + cy_c;
            % "x" and "y" are indices into the RGB frame, but they may contain
            % invalid values (which correspond to the parts of the scene not visible
            % to the RGB camera.
            % Do we have a valid index?
            if (x > rgbWidth || y > rgbHeight ||...
                x < 1 || y < 1 ||...
                isnan(x) || isnan(y))
                continue;
            end
            
            % Need some kind of interpolation. I just did it the lazy way
            x = round(x);
            y = round(y);

            colorizedDepth_6layer(v,u,4) = uint8(rgbImg(y, x, 1));
            colorizedDepth_6layer(v,u,5) = uint8(rgbImg(y, x, 2));
            colorizedDepth_6layer(v,u,6) = uint8(rgbImg(y, x, 3));
        end
    end    
    colorizedDepth = uint8(colorizedDepth_6layer(:,:,4:6));
end
