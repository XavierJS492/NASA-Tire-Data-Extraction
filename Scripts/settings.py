# settings.py
import math
import cv2
import numpy as np
import pyrealsense2 as rs
import multiprocessing as mp

def init(p_settings):
    global zoom
    #[zoom factor, zoom in (T) or out (F), show them window (T or F), [y,x] of mouse]
    zoom = [1, True, False, [0,0]]
    global window_name
    window_name = "Winow Name"

    global color_compare
    color_compare = p_settings['color_compare']

    global wire_diameter
    wire_diameter = p_settings['Wire_Diameter']

    global camera_floor_distance
    camera_floor_distance = p_settings['Camera_Floor_Distance']

    global gap_length
    gap_length = p_settings['Gap_Length']

    global frame_rate
    frame_rate = p_settings['Frame_Rate']

    #window x,y,w,h
    global menu_info
    menu_info = p_settings['menu_info']

    global filename
    filename = ""

    
def camera():
    global pipeline, pc, decimate, colorizer, w, h, out, state, depth_intrinsics, ppy
    global pipeline1, pipeline2
    pipeline1 = 'NO'
    class AppState:

        def __init__(self, *args, **kwargs):
            self.WIN_NAME = 'RealSense'
            self.pitch, self.yaw = math.radians(-10), math.radians(-15)
            self.translation = np.array([0, 0, -1], dtype=np.float32)
            self.distance = 2
            self.prev_mouse = 0, 0
            self.mouse_btns = [False, False, False]
            self.paused = False
            self.decimate = 0
            self.scale = True
            self.color = True

        def reset(self):
            self.pitch, self.yaw, self.distance = 0, 0, 2
            self.translation[:] = 0, 0, -1

        @property
        def rotation(self):
            Rx, _ = cv2.Rodrigues((self.pitch, 0, 0))
            Ry, _ = cv2.Rodrigues((0, self.yaw, 0))
            return np.dot(Ry, Rx).astype(np.float32)

        @property
        def pivot(self):
            return self.translation + np.array((0, 0, self.distance), dtype=np.float32)
    state = AppState()

    # Configure depth and color streams
    ctx = rs.context()
    if len(ctx.devices) > 1:
        pipeline1, pipeline2 = TwoCams()
        pipeline = pipeline1
    else:
        pipeline = rs.pipeline()
    config = rs.config()

    pipeline_wrapper = rs.pipeline_wrapper(pipeline)
    
    try:
        pipeline_profile = config.resolve(pipeline_wrapper)
        device = pipeline_profile.get_device()

        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, rs.format.bgr8, 30)
        #if it broke change ^ to config.enable_stream(rs.stream.color, rs.format.rgb8, 30)
    except:
        if filename == "":
            return False
        rs.config.enable_device_from_file(config, filename)
        config.enable_stream(rs.stream.depth, rs.format.z16, 30)
        config.enable_stream(rs.stream.color)

    # Start streaming
    if len(ctx.devices) <= 1:
        pipeline.start(config)

    # Get stream profile and camera intrinsics
    profile = pipeline.get_active_profile()
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    depth_intrinsics = depth_profile.get_intrinsics()
    w, h = depth_intrinsics.width, depth_intrinsics.height
    ppy = depth_intrinsics.ppy

    #print('depth_scale',profile.get_device().first_depth_sensor().get_depth_scale())
    
    # Processing blocks
    pc = rs.pointcloud()
    decimate = rs.decimation_filter()
    decimate.set_option(rs.option.filter_magnitude, 2 ** state.decimate)
    colorizer = rs.colorizer()

    out = np.empty((h, w, 3), dtype=np.uint8)
    return True

def TwoCams():
    print('settings two cam')
    # Configure depth and color streams...
    # ...from Camera 1
    pipeline1 = rs.pipeline()
    config1 = rs.config()

    pipeline_wrapper1 = rs.pipeline_wrapper(pipeline1)


    pipeline_profile1 = config1.resolve(pipeline_wrapper1)
    device1 = pipeline_profile1.get_device()
    config1.enable_device('126122270250')

    config1.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config1.enable_stream(rs.stream.color, rs.format.bgr8, 30)

    # ...from Camera 2
    pipeline2 = rs.pipeline()
    config2 = rs.config()

    pipeline_wrapper2 = rs.pipeline_wrapper(pipeline2)


    pipeline_profile2 = config2.resolve(pipeline_wrapper2)
    device2 = pipeline_profile2.get_device()

    config1.enable_device('218622276349')
    config2.enable_stream(rs.stream.depth, rs.format.z16, 30)
    config2.enable_stream(rs.stream.color, rs.format.bgr8, 30)


    # Start streaming from both cameras
    pipeline1.start(config1)
    pipeline2.start(config2)
    return pipeline1, pipeline2
