import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp
from menu_functions import *

import settings
import Extract
import View
import LiveView
import Options
import arduino

def StartUp():
    def CreateFolder(folder_name):
        try: os.makedirs(folder_name)
        except: pass
    CreateFolder('Outputs')
    CreateFolder('Pressure Outputs')
    CreateFolder('Images')
    CreateFolder('Inputs')
    try:
        with open('Settings.npz', 'x') as f: pass
  
        color_compare = [[0,0,0],[50,50,50]]
        menu_info = [50,50,1000,800]
  
        Wire_Diameter = .1
        Camera_Floor_Distance = 15
        Gap_Length = 30
        Frame_Rate = 30

        np.savez('Settings', color_compare=color_compare, menu_info=menu_info,Frame_Rate=Frame_Rate,
                 Wire_Diameter=Wire_Diameter, Camera_Floor_Distance=Camera_Floor_Distance,Gap_Length=Gap_Length)
    except:
        pass

    npzfile = np.load('Settings.npz')
    dict_settings = {}
    for i in npzfile.files:
        dict_settings[i] = npzfile[i]
    npzfile.close()
    settings.init(dict_settings)

if __name__ == '__main__':
    mp.freeze_support()
    StartUp()
    while True:
        cv2.namedWindow(settings.window_name, cv2.WINDOW_FREERATIO)
        cv2.setWindowTitle(settings.window_name, 'Main Menu')
        i = ClickMenu(settings.window_name, 'Main Menu', ["Extract Data", "View Stored Data", "Live Viewer", 'Pressure Map Reader',"Options"], True)
        match i:
            case False: break
            case 1: Extract.init()
            case 2: View.init()
            case 3: LiveView.init()
            case 4: arduino.init()
            case 5: Options.init()
