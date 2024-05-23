import time
import cv2
import numpy as np
import pyrealsense2 as rs
import os
import settings
from d3_viewer_functions import *
from additional_functions import *
from menu_functions import *



def init():
    while True:
        settings.camera()
        cv2.setWindowTitle(settings.window_name, 'View Stored Data')
        outfile = GetFile('View Stored Data','Outputs')

        if not outfile:
            #cv2.destroyAllWindows()
            return
        
        #selected temp files
        if outfile[-16:] == '/Temporary Files':
            outfile+='/'
            i = 1
            all_verts = []
            all_texcoords = []
            all_color_images = []
            all_depth_colormaps = []
            #os.listdir(outfile)[0]
            #Go through each temp file, combining them into one
            while os.path.isfile(outfile+str(i)+'.npz'):
                npzfile = np.load(outfile+str(i)+'.npz')
                all_verts.append(npzfile['now_verts'])
                all_texcoords.append(npzfile['now_texcoords'])
                all_color_images.append(npzfile['now_color_images'])
                all_depth_colormaps.append(npzfile['now_depth_colormaps'])
                i+=1
                npzfile.close()

            path = outfile[:-16]

            Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps)
            break

        #If not a temp file, load all the frames
        npzfile = np.load(outfile)
        all_verts = npzfile['all_verts']
        all_texcoords = npzfile['all_texcoords']
        all_color_images = npzfile['all_color_images']
        all_depth_colormaps = npzfile['all_depth_colormaps']
        all_footprints = npzfile['all_footprints']
        try:
            cam_info = np.float64(npzfile['cam_info'])
        except:
            settings.filename = 'Inputs/tire5.bag'
            settings.camera()
            cam_info = [settings.w,settings.h,settings.ppy, settings.frame_rate]

        npzfile.close()


        frame_numb = 0
        frameR = np.float32(settings.frame_rate)
        verts = []
        h,w = all_color_images[0].shape[:2]
        last_slash = len(outfile) - 1 - outfile[::-1].index('/')
        path = outfile[0:last_slash]
        file_name = outfile[last_slash+1:-4]
        view_state = 0
        paused = False
        key = 0
        refine = False

        MakeWindow(settings.window_name)
        cv2.setMouseCallback(settings.window_name, mouse_cb)
        cv2.setWindowTitle(settings.window_name, 'View Stored Data')
     
        obj_path = path+'/3D Objects/'
        vid_path = path+'/Vids/'
        #display the stuff
        while True:

            if not paused or key==ord("n"):
                if view_state == 2 and key != ord("n"):
                    time.sleep(1/frameR)
                frame_numb +=1
                if frame_numb >= len(all_verts):
                    frame_numb = 0
            

                depth_colormap = all_depth_colormaps[frame_numb]
                color_image = all_color_images[frame_numb]
                footprint = all_footprints[frame_numb].copy()


                verts = all_verts[frame_numb]  # xyz
                
                texcoords = all_texcoords[frame_numb]
                if settings.state.color:
                    color_source = color_image
                else:
                    if refine:
                        nextfoot = all_footprints[(frame_numb+1)%len(all_footprints)].copy()//255
                        lastfoot = all_footprints[(frame_numb-1)%len(all_footprints)].copy()//255
                        footprint = footprint*nextfoot*lastfoot
                    color_source = footprint

            if view_state < 2:
                d3_image = Render(cam_info, verts, texcoords, color_source)
        
            color_area = ShowScreens(settings.window_name, 'View Stored Data', view_state, color_source, d3_image)
            key = cv2.waitKey(1)
            if key == 8:
                cv2.destroyAllWindows()
                break

            if key == ord("p"):  
                if paused:
                    cv2.setMouseCallback(settings.window_name, mouse_cb)
                    try: cv2.destroyWindow('Data')
                    except: pass
                elif view_state >= 2:
                    cv2.setMouseCallback(settings.window_name, GetMouseInfo, [color_source, verts, color_area])   
                    GetMouseInfo(0,0,0,0,[color_source, verts,color_area])
                paused ^= True

            if key == ord("v"):
                view_state = (view_state+1)%3

            if key == ord("c"):
                settings.state.color ^= True
            if key == ord("r"):
                refine ^= True

            if key == ord("e"):   

                settings.camera()
                frame_count = len(all_verts)

                menu_functions.MakeWindow('Loading')
                text1 = "Using All "+str(mp.cpu_count())+" cores"
                text2 = str(frame_count)+' frames '+str(frame_count//mp.cpu_count()+1)+' Groups'
                menu_functions.ShowText('Loading', 'Loading', [text1,text2],True,0)
                cv2.waitKey(1)

                startTime = timeit.default_timer()
                #Bundling input for multiproccessing
                inputs = []
                for f in range(len(all_verts)):
                    inputs.append([all_verts[f].copy(),[h,w],obj_path,file_name,f,cam_info[2]])
                pool = mp.Pool(mp.cpu_count())
                pool.starmap(MakeObj, inputs)
                pool.close() 

                endTime = timeit.default_timer()
                print("Done",startTime-endTime)
                text1 = 'Finished! in '+str(endTime-startTime)
                menu_functions.ShowText('Loading', 'Loading', [text1],True,0)
                cv2.waitKey(1000)
                cv2.destroyWindow('Loading')
                
            if key == ord('m'):
                # Define the codec and create VideoWriter object.The output is stored in 'outpy.avi' file.
                name = vid_path+file_name+'.avi'
                out = cv2.VideoWriter(name,cv2.VideoWriter_fourcc('M','J','P','G'), frameR, (w,h))
  
                for i in range(len(all_verts)):
                    # Write the frame into the file 'output.avi'
                    out.write(all_color_images[i])
                out.release()
            if key == ord('w'):
                frameR += 1

            if key == ord('s'):
                if frameR> 1:
                    frameR += -1

            if key == ord('z'):
                if settings.zoom[2]:
                    cv2.setMouseCallback('Color', lambda *args : None)
                    cv2.destroyWindow('Zoom')
                    settings.zoom[2] = False
                else:
                    settings.zoom[2] = True
            if key == ord('x'):
                settings.zoom[1] = not settings.zoom[1]

            if settings.state.color:
                Zoom([color_image, verts])
            else:
                Zoom([footprint, verts]) 

            if key in (27, ord("q")):
                return
            if not cv2.getWindowProperty(settings.window_name,cv2.WND_PROP_VISIBLE):
                cv2.destroyAllWindows()
                return  False
