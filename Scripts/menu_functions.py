import cv2
import os
import settings
import numpy as np
from additional_functions import SaveSettings

def MakeWindow(window_name):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, settings.menu_info[2], settings.menu_info[3]) 
    cv2.moveWindow(window_name, settings.menu_info[0], settings.menu_info[1])
    cv2.namedWindow(window_name, cv2.WINDOW_FREERATIO)
    SaveSettings('menu_info',settings.menu_info)


def MakeFolder(directory):
    first = True
    while True:

        #if folder doesnt exist make it and return the path
        if first:
            msg1 = "Please Type Destination Folder"
            msg2 = "Hit Enter When Finished"
        else:
            msg1 = "That Folder Already Exist"
            msg2 = "Press Any Key To Try Again"
        folder_name = TypeMenu(settings.window_name,msg1,msg2)
 
        if not folder_name:
            cv2.destroyWindow(settings.window_name)
            return False
        
        first = False

        try:
            os.makedirs(directory+'/'+folder_name)
            path = directory+'/'+folder_name+'/'
            cv2.destroyWindow(settings.window_name)
            return path
        #if folder exists inform user and try again
        except:
            pass

def GetFile(window_name, directory):
    #get list of folders in outputs, put most recent at top
    allfolders = [f for f in os.listdir(directory) if not os.path.isfile(os.path.join(directory, f))]
    folder_dict = {}
    for i in allfolders:
        folder_dict[i] = os.path.getmtime(os.path.join(directory, i))
    res = {key: val for key, val in sorted(folder_dict.items(), reverse=True, key = lambda ele: ele[1])}
    allfolders = list(res.keys())

    while True:
        if directory != 'Inputs':
            cv2.setWindowTitle(settings.window_name, window_name)
            #Get what folder user clicked on, if they backspaced at folder level return false
            i = ClickMenu(settings.window_name, 'Select Folder', allfolders, False)
            if i == 0:
                return False
            else:
                folder = allfolders[i-1]  
        else:
            folder = ""

        
        #Get what file they clicked on, if back space cont in loop, else get file and return path
        allfiles = [f for f in os.listdir(directory+'/'+folder) if os.path.isfile(os.path.join(directory+'/'+folder, f))]
        if 'Temporary Files' in os.listdir(directory+'/'+folder):
            allfiles.append('Temporary Files')

        allfiles = np.array(allfiles)
        for i in range(len(allfiles)):
            if allfiles[i][-3:] == 'txt': allfiles[i] = 'DELETE'
        allfiles = allfiles[allfiles != 'DELETE']
        i = ClickMenu(settings.window_name, folder+': Please Select File', allfiles, False)
        if not i and directory == 'Inputs':
            return False
        if i !=0 :
            file = allfiles[i-1]
            path = directory+'/'+folder+'/'+file
            cv2.destroyAllWindows()
            return path

def GetFolder():
    #get list of folders in outputs, put most recent at top
    allfolders = [f for f in os.listdir('Outputs') if not os.path.isfile(os.path.join('Outputs', f))]
    folder_dict = {}
    for i in allfolders:
        folder_dict[i] = os.path.getmtime(os.path.join('Outputs', i))
    res = {key: val for key, val in sorted(folder_dict.items(), reverse=True, key = lambda ele: ele[1])}
    allfolders = list(res.keys())


    #Get what folder user clicked on, if they backspaced at folder level return false
    i = ClickMenu(settings.window_name, 'Select Folder', allfolders, False)
    if i == 0:
        return False
    else:
        folder = allfolders[i-1]  
        path = 'Outputs/'+folder+'/'
        cv2.destroyAllWindows()
        return path

def ClickMenu(window_name, title, options, center):
    #Initializing
    options_dist = 20
    option_i = [-1]
    final = ['NOTHING']
    scroll = [0]
    rect_x = [0]
    rect_y = [0]
    MakeWindow(window_name)
    #Runs when click or pressed enter
    def Select():
        #If mouse is on a clickable spot, Call that function and destroy menu window
        if option_i[0] < len(options) and option_i[0] >= 0:
            final[0] = option_i[0]+1

    #on mouse anything
    def on_mouse_move(event, x, y, p1, p2):

        #event = 10 when 2 fingers on mouse and moves, up when top half down when bottom,
        if event == 10:
            if y<height/2:
                scroll[0] += options_height[0]/5
            else:
                scroll[0] += -options_height[0]/5
            scroll[0] = int(scroll[0])

            #stops from scrolling past tex reg position and from leaving all text of screen
            if scroll[0] > 0: scroll[0] = 0
            if scroll[0] < -tot_height - offset_to_center + options_height[0]: scroll[0] = -tot_height - offset_to_center + options_height[0]

        #Gets postion of mouse relative to the options
        #option_i[0] = int((y-scroll[0]-title_height-offset_to_center-options_height[0]) / (options_height[1]+options_dist) +.5)
        text_y = 0
        i = -1
        option_i[0] = -1
        while y >= text_y: 
            text_y = (i+1)*(options_height[0]+options_dist)+title_height+offset_to_center + scroll[0]
            i += 1
        if text_y - y > options_dist:
            option_i[0] = i-1
        option_i[0] = i-1
        
        #If click
        if event == 4:
            Select()
    #Sets up mouse event, calls it to set up the text
    cv2.setMouseCallback(window_name, on_mouse_move)  
    old_scroll = scroll[0]+1
    while True:
        width, height = cv2.getWindowImageRect(window_name)[2:]
        if width != settings.menu_info[2] or height != settings.menu_info[3] or old_scroll != scroll[0]:
            options_height, options_width, add_on, width,title_height, offset_to_center, tot_height, height, img = ShowText(window_name, title, options, center, scroll[0])
        old_scroll = scroll[0]
        #If mouse is on one of the options make a grey rectangle around that option
        final_img = img.copy()
        if option_i[0] < len(options) and option_i[0] >= 0:
            rect_y[0] = (option_i[0])*(options_height[0]+options_dist)+add_on
            if center:
                rect_x[0] = int(width/2 - options_width[option_i[0]]/2)
            else:
                rect_x[0] = 0
            rect_img = img.copy()
            cv2.rectangle(rect_img, (rect_x[0], rect_y[0]), (rect_x[0]+options_width[option_i[0]], rect_y[0]+options_height[option_i[0]]), (100,100,100), -1)
            alpha = .5
            final_img = cv2.addWeighted(img, alpha, rect_img, 1 - alpha, 0)
  
        cv2.imshow(window_name, final_img)
        key = cv2.waitKey(1)
        #If Enter register select
        if key == 13: Select()
        #If esc q or backspace, end program
        if key in (27, ord("q"), 8):
            #cv2.destroyWindow(window_name)
            return False
        if not cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE):
            return  False
        if final[0] != 'NOTHING':
            return final[0]

def TypeMenu(window_name, title, msg):
    MakeWindow(window_name)

    ShowText(window_name, title, [msg], True,0)
    name = ''
    while True:
        key = cv2.waitKey(1)

        #update screen if size changed
        width, height = cv2.getWindowImageRect(window_name)[2:]
        if width != settings.menu_info[2] or height != settings.menu_info[3]:
            if name !=  '':
                ShowText(window_name, title, [name], True,0)
            else:
                ShowText(window_name, title, [msg], True,0)
        
        #If no key was press skip all other stuff
        if key == -1: continue
        #If backspace was pressed remove last charcter
        if key == 8:
            if len(name) == 0:
                return False
            name = name[:-1]
        #else add char
        else:
            name += chr(key)

        #update screen
        if name !=  '':
            ShowText(window_name, title, [name], True,0)
        else:
            ShowText(window_name, title, [msg], True,0)
        
        #pressed enter button
        if key == 13 :
            #due to key press the name has the enter char appended, remove it
            name = name[:-1]
            return name
            #display img val to screen
        if not cv2.getWindowProperty(window_name,cv2.WND_PROP_VISIBLE):
            return  False

def ShowScreens(window_name, title, view_state, color_image, d3_image):
    margin = 5
    img = ShowText(window_name, '', [], False, 0)[-1]
    width = settings.menu_info[2]
    height = settings.menu_info[3]
    new_h = int((3*height-2*margin)/4)
    if not width or not height:
        return color_image
    top = height//4 - 2*margin

    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 2*(width/1000) , 1)[0]
    cv2.putText(img, title, ((width-text_size[0])//2, int(top-2*margin)),cv2.FONT_HERSHEY_SIMPLEX, 2*(width/1000),(0,0,0),1,cv2.LINE_AA)

    match view_state:
        case 0: 
            #Both
            new_w = int((width - margin*3)/2)
            
            color_image = cv2.resize(color_image,(new_w,new_h))    
            d3_image = cv2.resize(d3_image,(new_w,new_h))

            img[top:top+new_h, margin:margin+new_w, :] = color_image
            img[top:top+new_h, 2*margin+new_w:2*margin+new_w+new_w, :] = d3_image

            color_area = [top,top+new_h, margin,margin+new_w]
        case 1:
            #just d3
            new_w = int(width-2*margin)
            d3_image = cv2.resize(d3_image,(new_w,new_h))
            img[top:top+new_h, margin:margin+new_w, :] = d3_image

            color_area = False
        case 2:
            #just camera
            new_w = int(width-2*margin)
            color_image = cv2.resize(color_image,(new_w,new_h))
            img[top:top+new_h, margin:margin+new_w, :] = color_image
            color_area = [top,top+new_h, margin,margin+new_w]

    cv2.imshow(window_name, img)
    return color_area

def ShowText(window_name, title, options, center, scroll):
    #Initializing
    options_dist = 20
    rect_x = [0]


    
    width, height = cv2.getWindowImageRect(window_name)[2:]
    settings.menu_info = list(cv2.getWindowImageRect(window_name))
    settings.menu_info[0] += -8
    settings.menu_info[1] += -31


    if not width*height:
        width = 1000
        height = 500
    
    title_scale = 4 *(1/.9) *(width/1000) 
    options_scale = (width*height/100000)/5

    options_width = []
    options_height = []
    #Get all the widths and heights of Text that will be on screen
    title_width = 10**9
    while title_width > width:
        title_scale *= .9
        (title_width, title_height) = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, title_scale, 1)[0]
    for i in range(len(options)):
        options_width.append(cv2.getTextSize(options[i], cv2.FONT_HERSHEY_SIMPLEX, options_scale, 1)[0][0])
        options_height.append(cv2.getTextSize(options[i], cv2.FONT_HERSHEY_SIMPLEX, options_scale, 1)[0][1])
        
    #Get the total height to vertically align all the text
    tot_height = title_height + sum(options_height) + len(options)*options_dist
    title_x = int(width/2-title_width/2)
    if tot_height > height:
        offset_to_center = 200
    else:
        offset_to_center = int((height-tot_height)/2)
    img = np.full((height,width,3), [255,255,255], dtype=np.uint8)
    
    title_x = int(width/2-title_width/2)

    try:
        ud = cv2.imread('Images/UD.png')
        spaceg = cv2.imread('Images/SpaceGrant.png')
        
        ud_scale_w = img.shape[1]/(ud.shape[1]*3.5)
        ud_scale_h = img.shape[0]/(ud.shape[0]*5)
        ud_scale = min([ud_scale_w,ud_scale_h])
        ud = cv2.resize(ud,(int(ud.shape[1]*ud_scale),int(ud.shape[0]*ud_scale)))
        
        spaceg_scale_w = img.shape[1]/(spaceg.shape[1]*5)
        spaceg_scale_h = img.shape[0]/(spaceg.shape[0]*5)
        spaceg_scale = min([spaceg_scale_w,spaceg_scale_h])
        spaceg = cv2.resize(spaceg,(int(spaceg.shape[1]*spaceg_scale),int(spaceg.shape[0]*spaceg_scale)))

        if height > spaceg.shape[1] and width > spaceg.shape[0]:
            img[0:spaceg.shape[0], 0:spaceg.shape[1] , :] = spaceg
            img[:ud.shape[0], img.shape[1]-ud.shape[1]: , :] = ud
    except:
        pass
    cv2.putText(img, title, (title_x, title_height+offset_to_center), cv2.FONT_HERSHEY_SIMPLEX , title_scale, (0, 0, 0), 1, cv2.LINE_AA) 

    #Write the text for Each Option, this is after the box so it can be ontop
    for i in range(len(options)):    
        if center:
            rect_x[0] = int(width/2 - options_width[i]/2)
        else:
            rect_x[0] = 0
        text_y = (i+1)*(options_height[0]+options_dist)+title_height+offset_to_center + scroll
        cv2.putText(img, options[i], (rect_x[0], text_y), cv2.FONT_HERSHEY_SIMPLEX , options_scale, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow(window_name, img)
    add_on = title_height+offset_to_center + scroll + options_dist
    return [options_height, options_width, add_on, width,title_height, offset_to_center, tot_height, height, img]
