"""
Arduino must out put its number as the first character for every line
when its new arr should output 999 after its first character
"""

import serial  
import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial.tools.list_ports
import menu_functions
import settings
import os
import shutil
import time




def Show(arr,name):

    arr = np.int32(arr)
    arr = np.repeat(arr, 50, axis=0)
    arr = np.repeat(arr, 50, axis=1)
    h, w = arr.shape
    color = 2*np.ones((h,w,3), dtype=np.uint8)*arr.reshape(h,w,1)
    color = np.uint8(color)
    cv2.namedWindow(name, cv2.WINDOW_FREERATIO)
    cv2.imshow(name, color)

    return

def Save(array, name, count, path):
    count[name] += 1
    with open(path+'Temporary Files/'+name+'-'+str(count[name])+'.npz', 'x') as f:
            pass
    np.savez(path+'Temporary Files/'+name+'-'+str(count[name]), array=array)

def ReadNew():         
    ports = [comport.device for comport in serial.tools.list_ports.comports()]
    if not len(ports):
        menu_functions.MakeWindow('No Arduino')
        menu_functions.ShowText('No Arduino', 'No Arduino Dectected', ['Press Any Key To Return Home'], True,0)
        cv2.waitKey()
        cv2.destroyWindow('No Arduino')
        return
    path = menu_functions.MakeFolder('Pressure Outputs')
    if not path:
         return
    os.makedirs(path+'Temporary Files')

    arrays = {}
    count = {}
    sers = []
    paused = False
    for port in ports:
        sers.append(serial.Serial(port, 115200))
    while True:
        if not paused:
            #gets a row from all the boards
            one_row_from_all_boards = []
            for ser in sers:
                try:
                    one_row_from_all_boards.append(ser.readline().decode('utf-8').strip())
                except:
                    print('Decoding err from port',port)
                
            #for each row add it to the appropriate array
            for row in one_row_from_all_boards:
                name = row[row.rfind(" ")+1:]
                if name not in arrays:
                    arrays[name] = []
                    count[name] = 0
                        
                #if this is first row then make new arr
                if row[:3] == '999':
                    if len(arrays[name]) == 16:
                        #display arrays
                        Show(arrays[name],name)
                        Save(arrays[name],name,count,path)
                    arrays[name] = []
                    row = row[4:]

                row = row[:-len(name)-1]

                try:
                    arrays[name].append(row.split('  '))
                except:
                    pass
        img = np.full((100,200,3), [255,255,255], dtype=np.uint8)
        cv2.namedWindow("Close To Return", cv2.WINDOW_AUTOSIZE)
        cv2.putText(img, "Close To Return", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
        cv2.imshow('Close To Return', img)
        key = cv2.waitKey(1)
        if key == ord('p'):
            paused = True
        if key in (8, 27, ord("q")):
            cv2.destroyAllWindows()
            return
        if not cv2.getWindowProperty("Close To Return",cv2.WND_PROP_VISIBLE):
            cv2.destroyAllWindows()
            return

def GetPressureInfo(event, x, y, flag, param):
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = np.full((300,200,3), [255,255,255], dtype=np.uint8)
    name, array, frame_numb = param
    i = int(x/50)
    j = int(y/50)

    cv2.putText(img, "Map: "+name, (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, 'Frame: '+str(frame_numb), (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, 'Positon: '+str(i)+','+str(j), (10, 80), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, 'Value: '+str(array[j][i]), (10, 110), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.imshow('Data', img)



def ReadOld():
    while True:
        outfile = menu_functions.GetFile('View Stored Data', 'Pressure Outputs')
        if not outfile:
            break
        if outfile[-16:] == '/Temporary Files':
            all_arrays = {}
            positions = {}
        
            #Go through each temp file, combining them into one
            for file in os.listdir(outfile):
                npzfile = np.load(outfile+'/'+file)
                array = npzfile['array']
                name = file[:file.rfind("-")]
                frame_numb = file[file.rfind("-")+1:-4]

                if name in all_arrays:
                    all_arrays[name][frame_numb] = array
                else:
                    all_arrays[name] = {frame_numb: array}
                npzfile.close()
            
            #now ask for the x,y and wether the map is to the left or right of this point
            for name in all_arrays.keys():
                pos = menu_functions.TypeMenu(settings.window_name, 'Please Tpye The x,y,O Coordinate of the map '+name,
                                              'O = 0 for Left and O = 1 for Right')
                if not pos:
                    break
                positions[name] = np.float32(pos.split(','))
            if not pos:
                continue

            path = outfile[:-16]
            with open(path+'/'+'Arrays.npz', 'x') as f:
                pass
            all_arrays = [all_arrays]
            np.savez(path+'/'+'Arrays', all_arrays=all_arrays)
            with open(path+'/positions.txt', 'w') as f:
                text = ''
                for name in positions:
                    text += '\n_'+str(name)+'_'+str(positions[name])[1:-1]
                text = text[2:]
                print(text, file=f)

            shutil.rmtree(path+'/Temporary Files')
            continue


        npzfile = np.load(outfile, allow_pickle=True)

        all_arrays = npzfile['all_arrays'][0].copy()
        with open(outfile[:outfile.rfind('/')]+'/positions.txt', 'r') as f:
            text = f.read()[:-1]
            print(text)
            table = str.maketrans("'[]\n'", '     ')
            text = text.translate(table)
            text = text.replace(' ','')
            arr = text.split('_')

            print(arr)
            keys = []
            vals = []
            for i in range(len(arr)):
                if i%2 == 0:
                    keys.append(arr[i])
                else:
                    vals.append(arr[i])

            positions = dict(zip(keys, zip(*vals)))

        tot_frames = len(list(all_arrays.values())[0].values())
        npzfile.close()

        paused  = False
        frameR = 6
        frame_numb = 1
        while True:

            if not paused:
                time.sleep(1/frameR)
                frame_numb +=1
                
                if frame_numb >= tot_frames:
                    frame_numb = 0
            
                for name in all_arrays.keys():
                    #print(name)
                    #print(all_arrays[name][str(frame_numb)])
                    Show(all_arrays[name][str(frame_numb)], name)

            # Render
            key = cv2.waitKey(5)
           
            if key == 8:
                cv2.destroyAllWindows()
                break

            if key == ord("p"):
                if paused:
                    for name in all_arrays.keys():
                        cv2.setMouseCallback(name, lambda *args : None)
                    cv2.destroyWindow('Data')
                else:
                    for name in all_arrays.keys():
                        cv2.setMouseCallback(name, GetPressureInfo, [name, all_arrays[name][frame_numb], frame_numb])   

                paused ^= True

                
            if key == ord('v'):
                #its just which map? do I stitch them together if so, how?
                pass

            if key == ord('w'):
                frameR += 1

            if key == ord('s'):
                if frameR> 1:
                    frameR += -1

            if key in (27, ord("q")):
                return


    
    
    
def init():
    title = "Pressure Map"
    options = ['Read From Maps','Read From Saved Data']
    index = menu_functions.ClickMenu(settings.window_name, title, options, True)
    if index == 0: 
        return
    else:
        index += -1
    if index == 0:
        ReadNew()
    if index == 1:
        ReadOld()
