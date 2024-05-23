import cv2
import settings
import os
import numpy as np
import shutil
import math
import multiprocessing as mp
import timeit
from scipy.spatial import Voronoi, voronoi_plot_2d
from stl import mesh
#from menu_functions import *
import menu_functions
def voronoi_finite_polygons_2d(vor, radius=None):
    """
    Reconstruct infinite voronoi regions in a 2D diagram to finite
    regions.
    Parameters
    ----------
    vor : Voronoi
        Input diagram
    radius : float, optional
        Distance to 'points at infinity'.
    Returns
    -------
    regions : list of tuples
        Indices of vertices in each revised Voronoi regions.
    vertices : list of tuples
        Coordinates for revised Voronoi vertices. Same as coordinates
        of input vertices, with 'points at infinity' appended to the
        end.
    """

    if vor.points.shape[1] != 2:
        raise ValueError("Requires 2D input")

    new_regions = []
    new_vertices = vor.vertices.tolist()

    center = vor.points.mean(axis=0)
    if radius is None:
        radius = vor.points.ptp().max()*2

    # Construct a map containing all ridges for a given point
    all_ridges = {}
    for (p1, p2), (v1, v2) in zip(vor.ridge_points, vor.ridge_vertices):
        all_ridges.setdefault(p1, []).append((p2, v1, v2))
        all_ridges.setdefault(p2, []).append((p1, v1, v2))

    # Reconstruct infinite regions
    for p1, region in enumerate(vor.point_region):
        vertices = vor.regions[region]

        if all(v >= 0 for v in vertices):
            # finite region
            new_regions.append(vertices)
            continue

        # reconstruct a non-finite region
        ridges = all_ridges[p1]
        new_region = [v for v in vertices if v >= 0]

        for p2, v1, v2 in ridges:
            if v2 < 0:
                v1, v2 = v2, v1
            if v1 >= 0:
                # finite ridge: already in the region
                continue

            # Compute the missing endpoint of an infinite ridge

            t = vor.points[p2] - vor.points[p1] # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[[p1, p2]].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            far_point = vor.vertices[v2] + direction * radius

            new_region.append(len(new_vertices))
            new_vertices.append(far_point.tolist())

        # sort region counterclockwise
        vs = np.asarray([new_vertices[v] for v in new_region])
        c = vs.mean(axis=0)
        angles = np.arctan2(vs[:,1] - c[1], vs[:,0] - c[0])
        new_region = np.array(new_region)[np.argsort(angles)]

        # finish
        new_regions.append(new_region.tolist())

    return new_regions, np.asarray(new_vertices)

def Tire(pix):
    if sum(pix > settings.color_compare[0]) == 3 and sum(pix < settings.color_compare[1]) == 3:
        return True
    return False

def Arrays(verts, color):
    
    h,w = color.shape[:2]
    #color = cv2.cvtColor(color, cv2.COLOR_BGR2HSV)
    color = color.reshape((len(verts),3))
    truth_tire = np.ones_like(verts)
    truth_floor = np.ones_like(verts)
    #Falsing the 0's
    truth = np.sign(np.sign(sum(np.transpose(verts != [0,0,0]))-1)+1).reshape(len(verts),1)
    truth_tire  *= truth
    truth_floor *= truth

    #Falsing the z>2's
    max_z = int(settings.camera_floor_distance)/10
    #truth = np.sign(np.sign(sum(np.transpose(verts <= [-np.inf,-np.inf,max_z]))-1)+1).reshape(len(verts),1)
    truth = (verts < max_z)[:,2:]

    truth_tire *= truth
    #truth_floor *= truth

    #Falsing not the not Tire color
    truth1 = color > settings.color_compare[0]
    truth2 = color < settings.color_compare[1]
    truth = sum(np.transpose(truth1*truth2)) == 3
    truth = np.array(truth, dtype=int).reshape(len(truth),1)
    truth_tire *= truth

    #inverting that for floor
    truth_floor *= np.logical_not(truth)
    truth_color_floor = np.ones_like(verts)*np.logical_not(truth)

    #putting back to correct shape
    truth_tire = np.array(truth_tire.reshape(h,w,3), dtype=bool)
    truth_floor = np.array(truth_floor.reshape(h,w,3), dtype=bool)
    truth_color_floor = np.array(truth_color_floor.reshape(h,w,3), dtype=bool)

    return [truth_tire, truth_floor, truth_color_floor]


def DetailedFindGaps(verts, floor_color_arr, shape, step):
    h, w = shape
    tire_gaps = []
    for j in range(0,h,step):
        tire_gaps.append([])
        for i in range(0,w,step):
            tire_gaps[-1].append(True)
            #Square
            this_square_list = []
            for jj in range(j,j+step,1):
                for ii in range(i,i+step,1):
                    try:
                        if not floor_color_arr[jj][ii][0]:
                            tire_gaps[-1][-1] = False
                            break
                        else:
                            #if verts[jj*w+ii][2] < 2:
                                this_square_list.append(verts[jj*w+ii][2])
                    except:
                        pass
  
            if tire_gaps[-1][-1]:
                #adds up and left max to this list
                try:
                    if tire_gaps[j-1][i]:
                        this_square_list.append(tire_gaps[j-1][i])
                    if tire_gaps[j][i-1]:
                        this_square_list.append(tire_gaps[j][i-1])
                except:
                    pass
                if len(this_square_list) > 0:
                    if max(this_square_list) > 0:
                        tire_gaps[-1][-1] = max(this_square_list)
                    else:
                        tire_gaps[-1][-1] = -100

                
    #adds right and down max to this list
    for j in reversed(range(len(tire_gaps))):
        for i in reversed(range(len(tire_gaps[j]))):
            if not tire_gaps[j][i]: continue
            this_square_list = [tire_gaps[j][i]]
            try:
                if tire_gaps[j+1][i]:
                    this_square_list.append(tire_gaps[j+1][i])
                if tire_gaps[j][i+1]:
                    this_square_list.append(tire_gaps[j][i+1])
            except:
                pass
            tire_gaps[j][i] = max(this_square_list)
    
    for j in range(len(tire_gaps)):
        for i in range(len(tire_gaps[j])):
            if not tire_gaps[j][i]: continue
            this_square_list = [tire_gaps[j][i]]
            try:
                if tire_gaps[j-1][i]:
                    this_square_list.append(tire_gaps[j-1][i])
                if tire_gaps[j][i-1]:
                    this_square_list.append(tire_gaps[j][i-1])
            except:
                pass
            tire_gaps[j][i] = max(this_square_list)
              
    return tire_gaps

def FastFindGaps(verts, floor_color_arr, shape, step):
    h, w = shape
    tire_gaps = []
    for j in range(0,h,step):
        tire_gaps.append([])
        for i in range(0,w,step):
            tire_gaps[-1].append(True)
            #Square
            this_square_list = []
            for jj in range(j,j+step,1):
                for ii in range(i,i+step,1):
                    try:
                        if not floor_color_arr[jj][ii][0]:
                            tire_gaps[-1][-1] = False
                            break
                        else:
                            #if verts[jj*w+ii][2] < 2:
                                this_square_list.append(verts[jj*w+ii][2])
                    except:
                        pass
  
            if tire_gaps[-1][-1]:
                #adds up and left max to this list
                try:
                    if tire_gaps[j-1][i]:
                        this_square_list.append(tire_gaps[j-1][i])
                    if tire_gaps[j][i-1]:
                        this_square_list.append(tire_gaps[j][i-1])
                except:
                    pass
                if len(this_square_list) > 0:
                    if max(this_square_list) > 0:
                        tire_gaps[-1][-1] = max(this_square_list)
                    else:
                        tire_gaps[-1][-1] = -100
    return tire_gaps

def NewNewFindGaps(verts,  floor_color_arr, shape, step, color):
    h, w = shape
    long_floor = floor_color_arr.reshape(h*w*3)

def SeperateTire(verts, color,colormap):
    h, w = color.shape[:2]
    truth_tire, truth_floor, truth_color_floor = Arrays(verts, color)
    vert_shaped = verts.reshape(h,w,3)

    tire_color = truth_tire*color
    tire_map = truth_tire*colormap
    tire_verts = (truth_tire*vert_shaped).reshape(len(verts),3)

    floor_color = truth_floor*color
    floor_map = truth_floor*colormap
    floor_verts = (truth_floor*vert_shaped).reshape(len(verts),3)
    return [[tire_verts, tire_color,tire_map],[floor_verts, floor_color,floor_map], [truth_tire, truth_floor, truth_color_floor]]
def GetContact(truth_tire, truth_floor, truth_color_floor, color, step, verts, data_type):
    h, w = color.shape[:2]
    if data_type == 1 :
        tire_gaps = FastFindGaps(verts, truth_color_floor, color.shape[:2], step)
    if data_type == 2:
        tire_gaps = DetailedFindGaps(verts, truth_floor, color.shape[:2], step)

    gap_centers = []
    for j in range(len(tire_gaps)):
        for i in range(len(tire_gaps[j])):
            if tire_gaps[j][i]:
                gap_centers.append([(i+.5)*step,(j+.5)*step])
    gap_centers = np.array(gap_centers)



    #input of gap_length is on cm but we use it as pix
    r = int(settings.gap_length)
    circ = color.copy()*0
    img = color.copy()*0
    vor = Voronoi(gap_centers)
    regions, vertices = voronoi_finite_polygons_2d(vor)

    for region_index in range(len(regions)):
        polygon = [vertices[i] for i in regions[region_index]]

        pts = np.int32([polygon])
        x,y = np.int32(gap_centers[region_index]/step)
        
        depth = tire_gaps[y][x]
        depth = str(int(depth*100000))
        c1 = int(depth[:2])
        c2 = int(depth[2:4])
        c3 = int(depth[4:])
    
        cv2.fillPoly(img, pts, (c1,c2,c3))
        cv2.circle(circ,np.int32(gap_centers[region_index]),r,(100,0,0),-1)
    
    img = img + circ
    colors = img.reshape(h*w,3)
    depths = sum(np.transpose(colors*np.array([10000,100,1])))
    depths = np.float64(depths - 1000000)
    is_close = depths > 0
    depths *=  is_close/100000

    diff_arr = np.abs(depths-verts[:,2])

    truth_contact  = diff_arr < np.float64(settings.wire_diameter)
    truth_contact *= diff_arr > np.float64(settings.wire_diameter)*.3
    truth_contact  = truth_contact.reshape(h*w,1)*truth_tire.reshape(h*w,3)

    contact_color = color.reshape(h*w,3)*np.logical_not(truth_contact) + np.full((h*w,3),[0,255,0])*(truth_contact)

    contact_color = np.uint8(contact_color.reshape(h,w,3))
    truth_contact = np.uint8(truth_contact.reshape(h,w,3))*255
    return [contact_color, truth_contact]

def DoMath(verts, color, colormap, f, data_type):
    npzfile = np.load('Settings.npz')
    dict_settings = {}
    for i in npzfile.files:
        dict_settings[i] = npzfile[i]
    npzfile.close()
    settings.init(dict_settings)

    step = 5
    
    [[tire_verts, tire_color,tire_map],[floor_verts, floor_color,floor_map], [truth_tire, truth_floor, truth_color_floor]] = SeperateTire(verts,color,colormap)

    
    contact_color = 0
    truth_contact = 0
    if data_type > 0:
        """if not np.any(truth_tire):
            return False"""
        contact_color, truth_contact = GetContact(truth_tire, truth_floor, truth_color_floor, color, step, verts, data_type)
        #diff_arr = np.full_like(color, [1,1,1])*diff_arr
        #pressure_color, pressure_verts = GetPressure(truth_contact, verts)


    if f%(mp.cpu_count())==0: 
        text = 'Group'+str((f//mp.cpu_count())+1)+' done'
        print(text)
        menu_functions.MakeWindow('Loading')
        menu_functions.ShowText('Loading', 'Loading', [text], True,0)
        cv2.waitKey(1)
        
        

    return [[verts, contact_color, colormap],[tire_verts, tire_color,tire_map],[floor_verts, floor_color,floor_map], truth_contact ]  

def SaveSettings(setting_name,val):
    #Save settings to file
    npzfile = np.load('Settings.npz')
    p_settings = dict(npzfile)
    p_settings[setting_name] = val
    np.savez('Settings.npz', **p_settings)
    #Save settings to this instace of the program
    dict_settings = {}
    for i in npzfile.files:
        dict_settings[i] = npzfile[i]
    npzfile.close()
    settings.init(dict_settings)

def Save(path, all_verts, all_texcoords, all_color_images, all_depth_colormaps, cam_info, data_type, percentage):
    
    def MakeFile(name):
        try:
            with open(path+name+'.npz', 'x') as f: pass
        except:
            os.remove(path+name+'.npz')
            with open(path+name+'.npz', 'x') as f: pass
    
    MakeFile('Raw')
    all_footprints = all_color_images.copy()
    np.savez(path+'Raw', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, 
             all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints, cam_info=cam_info)
    
    del_index = np.linspace(0,len(all_verts)-1,int((len(all_verts))*(1-(percentage/100))),dtype=np.uint64)
    all_verts = np.delete(all_verts, del_index, axis=0)
    all_texcoords = np.delete(all_texcoords, del_index, axis=0)
    all_color_images = np.delete(all_color_images, del_index, axis=0)
    all_depth_colormaps = np.delete(all_depth_colormaps, del_index, axis=0)
 


    #Bundling input for multiproccessing
    inputs = []
    frame_count = len(all_verts)

    for f in range(frame_count):
        inputs.append([all_verts[f].copy(),all_color_images[f].copy(),all_depth_colormaps[f].copy(),f,data_type])

    #Uses all cores on the computer
    text1 = "Using All "+str(mp.cpu_count())+" cores"
    #menu_functions.MakeWindow('Loading')
    text2 = str(frame_count)+' frames '+str(frame_count//mp.cpu_count()+1)+' Groups'
    cv2.setWindowTitle(settings.window_name, 'Loading')
    menu_functions.ShowText(settings.window_name, 'Loading', [text1,text2],True,0)
    
    cv2.waitKey(1)
    

  
    startTime = timeit.default_timer()
    pool = mp.Pool(mp.cpu_count())
    ########################Uncomment below and delete f=0 and the other results thing#######################
    results = pool.starmap(DoMath, inputs)
    pool.close() 
    #f = 0
    #results = DoMath(all_verts[f].copy(),all_color_images[f].copy(),all_depth_colormaps[f].copy(),f,data_type)
    ###########################################################################################################
    endTime = timeit.default_timer()

    text1 = "Finished! in "+str(int((endTime-startTime)*10)/10)+' seconds'
    text2 = "Please Wait"
    text3 = "Results Are Being Packaged"
    menu_functions.ShowText(settings.window_name, 'Loading', [text1,text2,text3], True,0)
    cv2.waitKey(1)
   

    #print('SHAPE',np.array(results).shape)
    #Unbundling result for use
    contact, tire, floor, all_footprints = [[],[],[],[]]
    for i in range(3):
        contact.append([])
        tire.append([])
        floor.append([])
    
    #dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd
    for f in range(len(all_verts)):
        #ddddddddddddddddddddddddddddddddddddddddddd
        contact_f, tire_f, floor_f, footprint_f = results[f]
        all_footprints.append(footprint_f)

        for i in range(len(contact_f)):
            contact[i].append(contact_f[i])
            tire[i].append(tire_f[i])
            floor[i].append(floor_f[i])
    """h,w = all_color_images[0].shape[:2]
    results = np.array(results)
    print(results.shape)
    contact = results[:,0]
    tire = results[:,1]
    floor = results[:,2]
    
    def unbound(Arr):
        out = []
        for i in range(Arr.shape[1]):
            out.append(Arr[:,i])
        return out"""
            
    
    if data_type>0:
        MakeFile('Contact')
        all_verts, all_color_images, all_depth_colormaps = contact
        np.savez(path+'Contact', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, 
                all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints, cam_info=cam_info)
        
        """MakeFile('Pressure')
        all_verts, all_color_images, all_depth_colormaps = pressure
        np.savez(path+'Pressure', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, 
                all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints, cam_info=cam_info)"""
    
    MakeFile('Tire')
    all_verts, all_color_images, all_depth_colormaps = tire
    np.savez(path+'Tire', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, 
             all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints,cam_info=cam_info )
    
    MakeFile('Floor')
    all_verts, all_color_images, all_depth_colormaps = floor
    np.savez(path+'Floor', all_verts=all_verts, all_texcoords=all_texcoords, all_color_images=all_color_images, 
             all_depth_colormaps=all_depth_colormaps, all_footprints=all_footprints,cam_info=cam_info )
    try:
        f = open(path+'Recorded Settings.txt','w')
    except:
        os.remove(path+'Recorded Settings.txt')
        f = open(path+'Recorded Settings.txt','w')
    
    npzfile = np.load('Settings.npz')
    string = 'These Were Settings Used For This Run\n'
    for i in npzfile.files:
        if i == "menu_info": continue
        string += i.replace('_',' ')+': '+str(npzfile[i])+'\n'
        
    npzfile.close()
    f.write(string)
    f.close()

    try:
        shutil.rmtree(path+'Temporary Files')
    except:
        pass
    #cv2.destroyWindow('Loading')

def Zoom(images):
    if settings.zoom[2]:
        height, width, c = images[0].shape

        def PosToCrop():
            yshift, xshift = [height/2, width/2]
            yclick = settings.zoom[3][0]*settings.zoom[0]
            xclick = settings.zoom[3][1]*settings.zoom[0]
            y1,y2,x1,x2  = [yclick-yshift, yclick+yshift,xclick-xshift, xclick+xshift]

            if y1 < 0:
                y1,y2 = [0,height]
            if y2 > height*settings.zoom[0]:
                y1, y2 = [height*settings.zoom[0]-height, height*settings.zoom[0]]
            
            if x1 < 0:
                x1,x2 = [0,width]
            if x2 > width*settings.zoom[0]:
                x1, x2 = [width*settings.zoom[0]-width, width*settings.zoom[0]]
            
            return [y1,y2,x1,x2]
    
        def ZoomView(event, x, y, p1, p2):
            if event == 10:
                if not settings.zoom[1]:
                    settings.zoom[0] *= .9
                else:
                    settings.zoom[0] *= 1.1
            if settings.zoom[0] < 1: settings.zoom[0] = 1

            settings.zoom[3] = [y,x]

            img = CropAndCursor(images[0].copy(), x, y)

            cv2.imshow('Zoom', img)
            GetMouseInfo(0,x,y,0,images)

        def CropAndCursor(image, x, y):
            
            overlay = image.copy()
            cv2.rectangle(overlay, (x, y-5), (x+1, y+5), (0,0,255), -1)
            cv2.rectangle(overlay, (x-5, y), (x+5, y+1), (0,0,255), -1)
            alpha = 1*(1/settings.zoom[0])  
            img = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)


            img = cv2.resize(img, None, fx=settings.zoom[0], fy=settings.zoom[0])
            y1, y2, x1, x2 = PosToCrop()
            img = img[int(y1):int(y2), int(x1):int(x2)]
            return img

        cv2.setMouseCallback('Color', ZoomView)
        img = CropAndCursor(images[0].copy(), settings.zoom[3][1], settings.zoom[3][0])
        cv2.imshow('Zoom', img)

def GetMouseInfo(event, x, y, flag, param):
    if not param: return
    #images 0 is color source
    #images 1 is vert
    images = param[:2]
    area = param[2]
    color_h, color_w = images[0].shape[:2]
    w = area[3]-area[2]
    h = area[1]-area[0]
    
    x = int((x-area[2])*color_w/w)
    y = int((y-area[0])*color_h/h)
    
    if x >= color_w or y >= color_h:
        return
    cv2.namedWindow("Data", cv2.WINDOW_KEEPRATIO)
    img = np.ones((242,266,3),dtype=np.uint8)*255

    #display color info
    color = images[0][y][x]
    color_text = str(color)

    [c1, c2, c3] = images[0][y][x]
    cv2.putText(img, "[Blue,Green,Red]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, color_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.rectangle(img, (200, 5), (250, 25), (int(c1), int(c2), int(c3)), -1)


    coords_text = str([int(item*10000)/10000 for item in images[1][y*images[0].shape[1] + x]])
    cv2.putText(img, "Coords", (10, 75), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, coords_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.putText(img, "It Is Tire:", (10, 125), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.putText(img, str(Tire(color)), (10, 150), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 

    cv2.putText(img, str(x)+','+str(y), (10, 175), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    cv2.imshow('Data', img) 

def SeperatePixs(color_image):
    color_image = color_image.reshape(color_image.size//3, 3)
    truth1 = color_image > [0,0,0]
    truth2 = color_image < [110,110,110]
    truth = sum(np.transpose(truth1*truth2)) == 3
    settings.tire_pos = truth

    print(settings.tire_pos)
    print(sum(settings.tire_pos), len(settings.tire_pos))

def GetColorInfo(event, x, y, flag, color_image):
    #key = cv2.waitKey(1)
    
    cv2.namedWindow("Data", cv2.WINDOW_AUTOSIZE)
    img = np.ones((242,266,3),dtype=np.uint8)*255
    hsv_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    #display color info
    color = np.array(color_image[y][x],dtype=int)
    if settings.mode == 'Floor':
        print(settings.mode,color)
        settings.floor_colors.append(color)
    if settings.mode == 'Tire':
        print(settings.mode,color)
        settings.tire_colors.append(color)
    cv2.putText(img, settings.mode+str(color), (10, 75), cv2.FONT_HERSHEY_SIMPLEX , .5, (0, 0, 0), 1, cv2.LINE_AA) 
    color_text = str(hsv_image[y][x])+str(color)

    cv2.imshow('Data', img) 
                      


def NewFile(path, points, faces):

    text1 = '\nv '.join(' '.join('%0.3f' %x for x in y) for y in points)
    text1 = 'v '+ text1

    faces += 1
    text2 = '\nf '.join(' '.join('%i' %x for x in y) for y in faces)
    text2 = 'f ' +text2
    txt = text1 +'\n'+ text2

    with open(path, 'x') as f:
        f.write(txt)

def FindColorMath(color_image):
    color_image  = color_image.reshape(color_image.size//3, 3)
    tire_colors  = color_image[settings.tire_pos]
    floor_colors = color_image[np.logical_not(settings.tire_pos)]

    tire_colors = np.transpose(tire_colors)
    lil = np.amin(tire_colors,1)+1
    big = np.amax(tire_colors,1)-1
    print('Above',lil,'Below',big)
    
    truth1 = floor_colors > lil
    truth2 = floor_colors < big
    truth = sum(np.transpose(truth1*truth2)) == 3
    if np.any(truth):
        floor_as_tires = np.count_nonzero(truth)
        total_floor = len(truth)
        print('Oh NO',floor_as_tires,'/',total_floor,floor_as_tires/total_floor,'floor pixels count as tires')
    else:
        settings.color_compare = [lil, big]



def MakeObj(verts, shape, path, file_name, frame, ppy):

    npzfile = np.load('Settings.npz')
    dict_settings = {}
    for i in npzfile.files:
        dict_settings[i] = npzfile[i]
    npzfile.close()
    settings.init(dict_settings)

    startTime = timeit.default_timer()
    if frame%mp.cpu_count() == 0:
        text = 'Exporting Frames'+ str(frame)+':'+str(frame+mp.cpu_count())
        menu_functions.MakeWindow('Loading')
        menu_functions.ShowText('Loading', 'Loading', [text],True,0)
        cv2.waitKey(1)
    

    #deleting all the [0,0,0]
    truth = np.array(np.sign(sum(np.transpose(verts != [0,0,0]))-1)+1, dtype=bool)
    verts = verts[truth]

    #deleting anything with a z>2
    truth = np.array(sum(np.transpose(verts <= [-np.inf,-np.inf,2])), dtype=bool)
    verts = verts[truth]


    h, w = shape
    #consty multiplied by depth gives he height and since square width of the pix
    fovy = ppy*90/math.pi
    consty = 2*math.tan(fovy)/h
    
    arr_1d_0_0_py = (verts*np.full_like(verts, [0,0,consty])).reshape(len(verts)*3)
    arr_1d_0_py_0 = np.append(np.delete(arr_1d_0_0_py, 0), 0)
    add_py_to_y = arr_1d_0_py_0.reshape(len(verts),3)

    arr_1d_py_0_0 = np.append(np.delete(arr_1d_0_py_0, 0), 0)
    add_py_to_x = arr_1d_py_0_0.reshape(len(verts),3)

    points1 = verts.copy()
    points2 = verts + add_py_to_x
    points3 = verts + add_py_to_y
    points4 = points3 + add_py_to_x

    points = np.concatenate((points1,points2,points3,points4))

    #the way points are set up face must 0,v,2v,3v and then 1,v+1,2v+1,3v+1 and so on till v-1, 2v-1, 3v-1, 4v-1
    #for triangles it should be 0,v,2v and v,2v,3v
    face1 = np.arange(3)*len(verts)
    faces1 = np.full((len(verts),3), face1) + np.arange(len(verts)).reshape(len(verts),1)

    face2 =  np.arange(1,4)*len(verts)
    faces2 = np.full((len(verts),3), face2) + np.arange(len(verts)).reshape(len(verts),1)

    faces = np.concatenate((faces1,faces2))

    MathTime = timeit.default_timer()
    obj_path = path+file_name+str(frame+1)+'.stl'
    #NewFile(obj_path,points,faces)
    # Create the mesh
    cube = mesh.Mesh(np.zeros(faces.shape[0], dtype=mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            cube.vectors[i][j] = points[f[j],:]

    # Write the mesh to file
    cube.save(obj_path)
    FileTime = timeit.default_timer()
    if frame%mp.cpu_count() == 0: print("Done", MathTime-startTime, FileTime-MathTime)
