# License: Apache 2.0. See LICENSE file in root directory.
# Copyright(c) 2015-2017 Intel Corporation. All Rights Reserved.

"""
OpenCV and Numpy Point cloud Software Renderer

This sample is mostly for demonstration and educational purposes.
It really doesn't offer the quality or performance that can be
achieved with hardware acceleration.

Usage:
------
Mouse: 
    Drag with left button to rotate around pivot (thick small axes), 
    with right button to translate and the wheel to zoom.

Keyboard: 
    [p]     Pause
    [r]     Reset View
    [d]     Cycle through decimation values
    [z]     Toggle point scaling
    [c]     Toggle color sourcepipeline
    [s]     Save PNG (./out.png)
    [e]     Export points to ply (./out.ply)
    [q\ESC] Quit
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import settings
from d3_viewer_functions import *
from additional_functions import GetMouseInfo
from menu_functions import *

def init():
    if not settings.camera():
        settings.filename = GetFile(settings.window_name, 'Inputs')
        if not settings.filename:
            settings.filename = ''
            return
        settings.camera()
    settings.filename = ''

    MakeWindow(settings.window_name)
    cv2.setMouseCallback(settings.window_name, mouse_cb)
    view_state = 0
    paused  = False

    while True:
        if settings.pipeline1 == 'NO':
            
            # Grab camera data
            if not paused:
                # Wait for a coherent pair of frames: depth and color
                frames = settings.pipeline.wait_for_frames()

                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()


                depth_frame = settings.decimate.process(depth_frame)
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())

                depth_colormap = np.asanyarray(
                    settings.colorizer.colorize(depth_frame).get_data())

                if settings.state.color:
                    mapped_frame, color_source = color_frame, color_image
                else:
                    mapped_frame, color_source = depth_frame, depth_colormap

                points = settings.pc.calculate(depth_frame)
                settings.pc.map_to(mapped_frame)

                # Pointcloud data to arrays
                v, t = points.get_vertices(), points.get_texture_coordinates()
                verts = np.asanyarray(v).view(np.float32).reshape(-1, 3)  # xyz
                #Scale it up
                verts = verts*10
        
                texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # uv


            if view_state < 2:
                d3_image = Render(settings.depth_intrinsics, verts, texcoords, color_source)
            
            ShowScreens(settings.window_name, 'Live View', view_state, color_image, d3_image)
           
        else:

            # Grab camera data
            if not paused:
                # Wait for a coherent pair of frames: depth and color
                frames2 = settings.pipeline2.wait_for_frames()
                frames1 = settings.pipeline1.wait_for_frames()
                

                depth_frame1 = frames1.get_depth_frame()
                color_frame1 = frames1.get_color_frame()

                
                depth_frame2 = frames2.get_depth_frame()
                color_frame2 = frames2.get_color_frame()


                depth_frame1 = settings.decimate.process(depth_frame1)
                depth_image1 = np.asanyarray(depth_frame1.get_data())
                color_image1 = np.asanyarray(color_frame1.get_data())

                depth_frame2 = settings.decimate.process(depth_frame2)
                depth_image2 = np.asanyarray(depth_frame2.get_data())
                color_image2 = np.asanyarray(color_frame2.get_data())

                depth_colormap1 = np.asanyarray(
                    settings.colorizer.colorize(depth_frame1).get_data())
                depth_colormap2 = np.asanyarray(
                    settings.colorizer.colorize(depth_frame2).get_data())

                if settings.state.color:
                    mapped_frame1, color_source1 = color_frame1, color_image1
                    mapped_frame2, color_source2 = color_frame2, color_image2
                else:
                    mapped_frame1, color_source1 = depth_frame1, depth_colormap1
                    mapped_frame2, color_source2 = depth_frame2, depth_colormap2
                
                #combine here
                h,w = color_image1.shape[:2]
                
                points1 = settings.pc.calculate(depth_frame1)
                settings.pc.map_to(mapped_frame1)
                # Pointcloud data to arrays
                v1, t1 = points1.get_vertices(), points1.get_texture_coordinates()
                verts1 = np.asanyarray(v1).view(np.float32).reshape(-1, 3)  # xyz
                texcoords1 = np.asanyarray(t1).view(np.float32).reshape(-1, 2)  # uv
                verts1 = verts1.reshape((h,w,3))
                texcoords1 = texcoords1.reshape((h,w,2))

                points2 = settings.pc.calculate(depth_frame2)
                settings.pc.map_to(mapped_frame2)
                # Pointcloud data to arrays
                v2, t2 = points2.get_vertices(), points2.get_texture_coordinates()
                verts2 = np.asanyarray(v2).view(np.float32).reshape(-1, 3)  # xyz
                texcoords2 = np.asanyarray(t2).view(np.float32).reshape(-1, 2)  # uv
                verts2 = verts2.reshape((h,w,3))
                texcoords2 = texcoords2.reshape((h,w,2))

                verts = np.vstack((verts2,verts1))
                texcoords = np.vstack((texcoords2,texcoords1))

                verts = verts.reshape((2*h*w,3))
                texcoords = texcoords.reshape((2*h*w,2))
                #Scale it up
                verts = verts*10
        
                color_source = np.vstack((color_source2,color_source1))
                

            if view_state < 2:
                d3_image = Render(settings.depth_intrinsics, verts, texcoords, color_source)
            
            ShowScreens(settings.window_name, 'Live View', view_state, color_source, d3_image)
            
        key = cv2.waitKey(1)

        if key == ord("r"):
            settings.state.reset()

        if key == ord("p"):
            paused ^= True

        if key == ord("v"):
            view_state = (view_state+1)%3

        if key == ord("z"):
            settings.state.scale ^= True

        if key == ord("c"):
            settings.state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', settings.out)

        if key == ord("e"):
            points.export_to_ply('./out.ply', mapped_frame)

        if key in (27, ord("q"), 8):
            #Stop streaming
            cv2.destroyAllWindows()
            settings.pipeline.stop()
            return
        try:
            cv2.getWindowProperty(settings.window_name, cv2.WND_PROP_AUTOSIZE)
        except:
            #Stop streaming
            cv2.destroyAllWindows()
            settings.pipeline.stop()
            return
    
            
        
            
