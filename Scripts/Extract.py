import cv2
import numpy as np
import pyrealsense2 as rs
import settings
from d3_viewer_functions import *
from additional_functions import *
from menu_functions import *
import os
import shutil
def OldSave(path):
    outfile = path+'Raw.npz'
    npzfile = np.load(outfile)
    all_verts = npzfile['all_verts']
    all_texcoords = npzfile['all_texcoords']
    all_color_images = npzfile['all_color_images']
    all_depth_colormaps = npzfile['all_depth_colormaps']
    cam_info = npzfile['cam_info']
    npzfile.close()

    shutil.rmtree(path)
    os.makedirs(path)
    os.makedirs(path+'3D Objects')
    os.makedirs(path+'Vids')
    while True:
        i = ClickMenu(settings.window_name,'What To Extract?',['Just Tire Data', 'Tire and Contact Footprint ', 'Tire and Refined Footprint'],True)
        if i == 0:
            return
        else:
            i+=-1
            break
    percentage = np.float64(TypeMenu(settings.window_name,str(len(all_verts))+' Frames, Percentage of Frames to Keep?', 'Press Any Key To Start, Press Enter When Done'))
    if percentage == 0:
        return
    Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps, cam_info, i, percentage)

def init():
        
    all_verts, all_texcoords, all_color_images, all_depth_colormaps = [[],[],[],[]]

    title = "What Type of Data to Analyze"
    options = ['Live Data','Saved Data']
    new = menu_functions.ClickMenu(settings.window_name, title, options, True)
    if new == 0: 
        return
    else:
        new += -1

    if new:
        path = GetFolder()
        if path:
            OldSave(path)
        return

    if not settings.camera():
        settings.filename = GetFile(settings.window_name, 'Inputs')
        if not settings.filename:
            settings.filename = ''
            return
        settings.camera()
    settings.filename = ''
    path = MakeFolder('Outputs') #new
    if path == 0:
        return
    print(path)
    os.makedirs(path+'Temporary Files')
    os.makedirs(path+'3D Objects')
    os.makedirs(path+'Vids')

    MakeWindow(settings.window_name)
    cv2.setWindowTitle(settings.window_name, 'Extract Data')
    while True:
        verts = []
        # Grab camera data
        if not settings.state.paused:
            # Wait for a coherent pair of frames: depth and color
            frames = settings.pipeline.wait_for_frames()

            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()


            depth_frame = settings.decimate.process(depth_frame)

            # Grab new intrinsics (may be changed by decimation)
            depth_intrinsics = rs.video_stream_profile(
                depth_frame.profile).get_intrinsics()
            w, h = depth_intrinsics.width, depth_intrinsics.height

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
            texcoords = np.asanyarray(t).view(np.float32).reshape(-1, 2)  # 
            verts *= 10
            
                        
            
            #store information in array to do math later, need to be copy or memory issues happen
            now_verts = verts.copy()
            now_texcoords = texcoords.copy()
            now_color_images = color_image.copy()
            now_depth_colormaps = depth_colormap.copy()

            all_verts.append(now_verts)
            all_texcoords.append(now_texcoords)
            all_color_images.append(now_color_images)
            all_depth_colormaps.append(now_depth_colormaps)
            
            #makes a temp file for every frame incase program ends unexpectedly
            with open(path+'Temporary Files/'+str(len(all_verts))+'.npz', 'x') as f:
                pass
            np.savez(path+'Temporary Files/'+str(len(all_verts)), now_verts=now_verts, now_texcoords=now_texcoords, now_color_images=now_color_images, now_depth_colormaps=now_depth_colormaps )
            ShowScreens(settings.window_name, 'Exctracting', 2, color_source, 0)
            
        
        key = cv2.waitKey(1)

        if key == ord("r"):
            settings.state.reset()

        if key == ord("p"):
            settings.state.paused ^= True

        if key == ord("d"):
            settings.state.decimate = (settings.state.decimate + 1) % 3
            settings.decimate.set_option(rs.option.filter_magnitude, 2 ** settings.state.decimate)
            print(settings.state.decimate)

        if key == ord("z"):
            settings.state.scale ^= True

        if key == ord("c"):
            settings.state.color ^= True

        if key == ord("s"):
            cv2.imwrite('./out.png', settings.out)

        if key in (8, 27, ord("q")):
            break
        if not cv2.getWindowProperty(settings.window_name,cv2.WND_PROP_VISIBLE):
            return 
    
    #Have something here asking about deleting frames and do loading screen
    #cv2.destroyAllWindows()
    while True:
        i = ClickMenu(settings.window_name,'What To Extract?',['Just Tire Data', 'Tire and Contact Footprint ', 'Tire and Refined Contact Footprint'],True)
        if i == 0:
            j = ClickMenu(settings.window_name, 'Delete Data?',['Yes', 'No'], True)
            if j< 2:
                shutil.rmtree(path)
                return
        else:
            i+=-1
            break
    percentage = np.float64(TypeMenu(settings.window_name,str(len(all_verts))+' Frames, Percentage of Frames to Keep?', 'Press Any Key To Start, Press Enter When Done'))
    if percentage == 0:
        return
    cam_info = [settings.w,settings.h,settings.ppy, settings.frame_rate]
    Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps, cam_info, i,percentage)
    

    # Stop streaming
    settings.pipeline.stop()
