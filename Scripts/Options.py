import cv2
import numpy as np
import os
import time
from additional_functions import *
from menu_functions import *

def mouse(event, x, y, flag, param):
   
    color, truth_floor = param
    
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = np.ones((242,266,3),dtype=np.uint8)*255

    #display color info
    this_color = color[y][x]
    color_text = str(this_color)

    [c1, c2, c3] = this_color
    cv2.putText(img, "BRG", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.rectangle(img, (200, 5), (250, 25), (int(c1), int(c2), int(c3)), -1)

    cv2.putText(img, 'Is it a Tire pix '+str(not sum(truth_floor[y][x])), (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.imshow('Data',img)

def ColorSeperator():
    def nothing(x):
        pass
    old_color_compare = settings.color_compare.copy()
    live = ClickMenu(settings.window_name,'What Data To Parse',['Stored Data','Live Data'],True)
    if not live: return  
    live += -1

    if not live:
        path = menu_functions.GetFolder()
        if not path:
            return
        npzfile = np.load(path+'Raw.npz')
        all_verts = npzfile['all_verts']
        all_colors = npzfile['all_color_images']
        frame_rate = 5
        frame_numb = 0
        frameR = 6
    else:
        if not settings.camera():
            settings.filename = GetFile(settings.window_name, 'Inputs')
            if not settings.filename:
                settings.filename = ''
                return
            settings.camera()
        settings.filename = ''
    cv2.namedWindow("Seperate", cv2.WINDOW_AUTOSIZE)
    # create trackbars for color change
    cv2.createTrackbar('R','Seperate',0,255,nothing)
    cv2.createTrackbar('G','Seperate',0,255,nothing)
    cv2.createTrackbar('B','Seperate',0,255,nothing)
    
    # create switch for ON/OFF functionality
    switch = '0 : Min \n1 : Max'
    cv2.createTrackbar(switch, 'Seperate',0,1,nothing)
    last_s = 1
    paused = False
    
    while True:

        if not paused:
            if not live:
                time.sleep(1/frame_rate)

                frame_numb +=1
                if frame_numb >= len(all_verts):
                    frame_numb = 0

                color = all_colors[frame_numb]
                
            else:
                frames = settings.pipeline.wait_for_frames()

                color_frame = frames.get_color_frame()

                color = np.asanyarray(color_frame.get_data())
            

        # get current positions of four trackbars
        r = cv2.getTrackbarPos('B','Seperate')
        g = cv2.getTrackbarPos('G','Seperate')
        b = cv2.getTrackbarPos('R','Seperate')
        s = cv2.getTrackbarPos(switch,'Seperate')
        if last_s != s:
            cv2.setTrackbarPos('B','Seperate',settings.color_compare[s][0])
            cv2.setTrackbarPos('G','Seperate',settings.color_compare[s][1])
            cv2.setTrackbarPos('R','Seperate',settings.color_compare[s][2])
        last_s = s
        settings.color_compare[s] = [r,g,b]

        h,w = color.shape[:2]
        verts = np.zeros((h*w,3))
        truth_floor  = Arrays(verts, color)[2]

        cv2.imshow('Orig',color)
        new_color = np.zeros_like(color) + 255
        new_color *= truth_floor
        cv2.imshow('Seperate',new_color)
        

        key = cv2.waitKey(1)
        if key == 8:
            cv2.destroyAllWindows()
            
            i = ClickMenu('Save?', 'Do you want to save this color range',['Yes','No'],True)
            if i == 2:
                settings.color_compare = old_color_compare
            cv2.destroyAllWindows()
            break
        
        if key == ord("r"):
            settings.state.reset()

        if key == ord("p"):
            if paused:
                cv2.setMouseCallback('Orig', lambda *args : None)
                cv2.setMouseCallback('Seperate', lambda *args : None)
                cv2.destroyWindow('Data')
            else:
                cv2.setMouseCallback('Orig', mouse, [color, truth_floor])  
                cv2.setMouseCallback('Seperate', mouse, [color, truth_floor]) 
                mouse(0,0,0,0,[color, truth_floor])
            paused ^= True

        if key == ord('w'):
            frameR += 1

        if key == ord('s'):
            if frameR> 1:
                frameR += -1
        if key in (27, ord("q")):
            cv2.destroyAllWindows()
            settings.color_compare = old_color_compare
            return
        if not cv2.getWindowProperty('Orig',cv2.WND_PROP_VISIBLE):
            settings.color_compare = old_color_compare
            cv2.destroyAllWindows()
            return  False




def init():
    while True:
        npzfile = np.load('Settings.npz')

        dict_settings = {}
        for i in npzfile.files:
            dict_settings[i.replace('_',' ')] = npzfile[i]
        npzfile.close()


        del dict_settings['color compare']
        del dict_settings['menu info']

        option1 = np.array(['Color Seperator'])
        options = []
        for i in dict_settings.keys():
            options.append(i+': '+str(dict_settings[i]))
        
        options = np.concatenate((option1,np.array(options)))
    
        i = ClickMenu(settings.window_name, 'Options', options, False)
        match i:
            case False: return
            case 1: 
                ColorSeperator()
                SaveSettings('color_compare',settings.color_compare)
                return
            case _:
                title = options[i-1][:options[i-1].rfind(':')]
                new_val = TypeMenu(settings.window_name,title,str(dict_settings[title]))
                if new_val:
                    SaveSettings(title.replace(' ','_'),new_val)
