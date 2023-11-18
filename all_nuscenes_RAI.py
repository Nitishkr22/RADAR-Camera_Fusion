
import os
import os.path as osp
import sys
import math
import time
import numpy as np
# from nuscenes.utils.data_classes import RadarPointCloud, Box
from PIL import Image
import cv2
from data_classes1 import *
# from radar import *
# import csv

def enrich_radar_data(radar_data):
    """
    This function adds additional data to the given radar data
    
    :param radar_data: The source data which are used to calculate additional metadata
        Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0

    :returns enriched_radar_data:
            [0]: x
            [1]: y
            [2]: z
            [3]: dyn_prop
            [4]: id
            [5]: rcs
            [6]: vx
            [7]: vy
            [8]: vx_comp
            [9]: vy_comp
            [10]: is_quality_valid
            [11]: ambig_state
            [12]: x_rms
            [13]: y_rms
            [14]: invalid_state
            [15]: pdh0
            [16]: vx_rms
            [17]: vy_rms
            [18]: distance
            [19]: azimuth
            [20]: vrad_comp
    """
    assert radar_data.shape[0] == 18, "Channel count mismatch."

    # Adding distance
    # Calculate distance
    dist = np.sqrt(radar_data[0,:]**2 + radar_data[1,:]**2)
    dist = np.expand_dims(dist, axis=0) #appending distance to existing data

    # calculate the azimuth values
    azimuth = np.arctan2(radar_data[1,:], radar_data[0,:]) 
    azimuth = np.expand_dims(azimuth, axis=0) #appending azimuth to existing data

    # Calculate vrad comp
    radial = np.array([radar_data[0,:], radar_data[1,:]]) # Calculate the distance vector
    # print("normal",radial)
    radial = radial / np.linalg.norm(radial, axis=0, keepdims=True)# Normalize these vectors => x/sqrt(x1^2+y1^2) and y/sqrt(x1^2+y1^2)
    # print("normalized",radial)
    v = np.array([radar_data[8,:], radar_data[9,:]]) # Create the speed vector
    vrad_comp = np.sum(v*radial, axis=0, keepdims=True) # Project the speed component onto this vector i.e. (xn*vx_comp + yn*vy_comp)
    # print(vrad_comp)
    data_collections = [
        radar_data,
        dist,
        azimuth,
        vrad_comp
    ]
    data_collection2 = [
        dist,
        azimuth,
        vrad_comp
    ]
    # print(type(data_collection2))

    enriched_radar_data = np.concatenate(data_collections, axis=0)

    return data_collection2

def get_sensor_sample_data(nusc, sample, sensor_channel, dtype=np.float32, size=None):
    """
    This function takes the token of a sample and a sensor sensor_channel and returns the according data
    :param sample: the nuscenes sample dict
    :param sensor_channel: the target sensor channel of the given sample to load the data from
    :param dtype: the target numpy type
    :param size: for resizing the image

    Radar Format:
        - Shape: 19 x n
        - Semantics: 
            [0]: x (1)
            [1]: y (2)
            [2]: z (3)
            [3]: dyn_prop (4)
            [4]: id (5)
            [5]: rcs (6)
            [6]: vx (7)
            [7]: vy (8)
            [8]: vx_comp (9)
            [9]: vy_comp (10)
            [10]: is_quality_valid (11)
            [11]: ambig_state (12)
            [12]: x_rms (13)
            [13]: y_rms (14)
            [14]: invalid_state (15)
            [15]: pdh0 (16)
            [16]: vx_rms (17)
            [17]: vy_rms (18)
            [18]: distance (19)

    Image Format:
        - Shape: h x w x 3
        - Channels: RGB
        - size:
            - [int] size to limit image size
            - [tuple[int]] size to limit image size
    """

    # Get filepath
    sd_rec = nusc.get('sample_data', sample['data'][sensor_channel])
    file_name = osp.join(nusc.dataroot, sd_rec['filename']) # stores file dirctory of the data in file_name
    # print(file_name)

    # Check conditions
    if not osp.exists(file_name):
        raise FileNotFoundError(
            "nuscenes data must be located in %s" % file_name)

    # Read the data
    if "RADAR" in sensor_channel:
        pc = RadarPointCloud.from_file(file_name)  # Load radar points
        data = pc.points.astype(dtype)
        # print(data)

        # with open("output.csv", "w", newline="") as csvfile:
        #     writer = csv.writer(csvfile)
        #     for point in data:
        #         writer.writerow(point)

        #     csvfile.close()
        # print(data.shape)

        data1 = enrich_radar_data(data) # enrich the radar data an bring them into proper format
        # print(data)
    elif "CAM" in sensor_channel:
        i = Image.open(file_name)
        # print(file_name)
        # image = cv2.imread(file_name)  # Replace "path/to/your/image.jpg" with the actual path to your image
        # print("oooooooooo", image)
        # Display the image
        # cv2.imshow("Image", image)
        # # Wait for a key press to close the window
        # cv2.waitKey(0)
        # # Clean up and close the window
        # cv2.destroyAllWindows()

        # resize if size is given
        if size is not None:
            try:
                _ = iter(size)
            except TypeError:
                # not iterable
                # limit both dimension to size, but keep aspect ration
                size = (size, size)
                i.thumbnail(size=size)
            else:
                size = size[::-1]  # revert dimensions
                i = i.resize(size=size)

        data = np.array(i, dtype=dtype) # contans image pixel just in opposite fashion as cv2 image
        data1 = None
        # print("iiiiiii",data)
        # cv2.imshow("Image", data)
        # # Wait for a key press to close the window
        # cv2.waitKey(0)
        # # Clean up and close the window
        # cv2.destroyAllWindows()
        

        if np.issubdtype(dtype, np.floating):
            data = data / 255 # floating images usually are on [0,1] interval

        # cv2.imshow("Image", data)
        # # Wait for a key press to close the window
        # cv2.waitKey(0)
        # # Clean up and close the window
        # cv2.destroyAllWindows()

    else:
        raise Exception("\"%s\" is not supported" % sensor_channel)

    return data, data1

def _resize_image(image_data, target_shape):
    """
    Perfomrs resizing of the image and calculates a matrix to adapt the intrinsic camera matrix
    :param image_data: [np.array] with shape (height x width x 3)
    :param target_shape: [tuple] with (width, height)

    :return resized image: [np.array] with shape (height x width x 3)
    :return resize matrix: [numpy array (3 x 3)]
    """
    # print('resized', type(image_data))
    stupid_confusing_cv2_size_because_width_and_height_are_in_wrong_order = (target_shape[1], target_shape[0])
    resized_image = cv2.resize(image_data, stupid_confusing_cv2_size_because_width_and_height_are_in_wrong_order)
    resize_matrix = np.eye(3, dtype=resized_image.dtype)
    # print(target_shape)
    # print(image_data.shape)
    resize_matrix[1, 1] = target_shape[0]/image_data.shape[0]
    resize_matrix[0, 0] = target_shape[1]/image_data.shape[1]
    return resized_image, resize_matrix

def _radar_transformation(radar_data, height=None):
    """
    Transforms the given radar data with height z = 0 and another height as input using extrinsic radar matrix to vehicle's co-sy

    This function appends the distance to the radar point.

    Parameters:
    :param radar_data: [numpy array] with radar parameter (e.g. velocity) in rows and radar points for one timestep in columns
        Semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 distance
    :param radar_extrinsic: [numpy array (3x4)] that consists of the extrinsic parameters of the given radar sensor
    :param height: [tuple] (min height, max height) that defines the (unknown) height of the radar points

    Returns:
    :returns radar_data: [numpy array (m x no of points)] that consists of the transformed radar points with z = 0
    :returns radar_xyz_endpoint: [numpy array (3 x no of points)] that consits of the transformed radar points z = height  
    """

    # Field of view (global)
    ELEVATION_FOV_SR = 20
    ELEVATION_FOV_FR = 14  

    # initialization
    num_points = radar_data.shape[1]

    # Radar points for the endpoint
    radar_xyz_endpoint = radar_data[0:18,:].copy()

    # variant 1: constant height substracted by RADAR_HEIGHT
    RADAR_HEIGHT = 0.5
    if height:
        radar_data[2, :] = np.ones((num_points,)) * (height[0] - RADAR_HEIGHT) # lower points
        radar_xyz_endpoint[2, :] = np.ones((num_points,)) * (height[1] - RADAR_HEIGHT) # upper points
    
    # variant 2: field of view
    else:
        dist = radar_data[-1,:]
        count = 0
        for d in dist:
            # short range mode
            if d <= 70: 
                radar_xyz_endpoint[2, count] = -d * np.tan(ELEVATION_FOV_SR/2)
                
            # long range mode
            else:
                radar_xyz_endpoint[2, count] = -d * np.tan(ELEVATION_FOV_FR/2)

            count += 1

    return radar_data, radar_xyz_endpoint

def view_points(points: np.ndarray, view: np.ndarray, normalize: bool):
    """
    This function is a modification of nuscenes.geometry_utils.view_points function

    This is a helper class that maps 3d points to a 2d plane. It can be used to implement both perspective and
    orthographic projections. It first applies the dot product between the points and the view. By convention,
    the view should be such that the data is projected onto the first 2 axis. It then optionally applies a
    normalization along the third dimension.

    For a perspective projection the view should be a 3x3 camera matrix, and normalize=True
    For an orthographic projection with translation the view is a 3x4 matrix and normalize=False
    For an orthographic projection without translation the view is a 3x3 matrix (optionally 3x4 with last columns
     all zeros) and normalize=False

    :param points: <np.float32: 3, n> Matrix of points, where each point (x, y, z) is along each column.
    :param view: <np.float32: n, n>. Defines an arbitrary projection (n <= 4).
        The projection should be such that the corners are projected onto the first 2 axis.
    :param normalize: Whether to normalize the remaining coordinate (along the third axis).
    :return: <np.float32: 3, n>. Mapped point. If normalize=False, the third coordinate is the height.
    """

    output = points
    # print(view.shape)
    # print(points.shape)
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] >= 3
    points = output[0:3,:]

    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view # makeing homogeneous intrensic matrix
    # print(viewpad)

    nbr_points = points.shape[1]
    # print(points)
    # Do operation in homogenous coordinates
    points = np.concatenate((points, np.ones((1, nbr_points))))
    # print(points)
    points = np.dot(viewpad, points)
    points = points[:3, :]
    # print(points)

    # normalizing z and making z as 1 by dividing it to x and y
    if normalize:
        points = points / points[2:3, :].repeat(3, 0).reshape(3, nbr_points)

    # print(points)

    output[0:3,:] = points
    # print(output)
    return output

def map_pointcloud_to_image(nusc, radar_points, pointsensor_token, camera_token, target_resolution=(None,None)):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load point-cloud and map it to the image
    plane.
    :param radar_pints: [list] list of radar points
    :param pointsensor_token: [str] Lidar/radar sample_data token.
    :param camera_token: [str] Camera sample_data token.
    :param target_resolution: [tuple of int] determining the output size for the radar_image. None for no change

    :return (points <np.float: 2, n)
    """

    # Initialize the database
    cam = nusc.get('sample_data', camera_token)
    # print("cccaamera",cam)
    pointsensor = nusc.get('sample_data', radar_token) # displays details of radar_token from sample_data.jason
    # print("rrrrrraaadddaaarr",pointsensor)

    pc = RadarPointCloud(radar_points)
    # print(radar_points.shape)
    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the point-cloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token']) # cs_record stores data from calibrated_sensor.jaso of pointsesnor's calibrated_sensor_token
    # print("ccccc",cs_record)
    rot = pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    # print("rot1",Quaternion(cs_record['rotation']).rotation_matrix)
    # print("111111rrrrrrrrr",rot)
    trans =    pc.translate(np.array(cs_record['translation'])) # detections in vehicle frame of reference
    # print("trans1",np.array(cs_record['translation']))
    # print("11111111tttttt",trans)

    # Second step: transform to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # print("gggggggg",poserecord)
    # print(Quaternion(poserecord['rotation']).rotation_matrix)
    # print(np.array(poserecord['translation']))
    rot2 = pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    # print("22222rrrr",rot2)
    trans2 = pc.translate(np.array(poserecord['translation']))
    # print("222222tttt",trans2)

    # Third step: transform into the ego vehicle frame for the timestamp of the image.
    poserecord1 = nusc.get('ego_pose', cam['ego_pose_token'])
    # print(poserecord1)
    rot3 = pc.translate(-np.array(poserecord1['translation']))
    # print(np.array(poserecord1['translation']))
    # print("33333333rrr",rot3)
    trans3 = pc.rotate(Quaternion(poserecord1['rotation']).rotation_matrix.T) #same matrix as in first step
    # print(Quaternion(poserecord1['rotation']).rotation_matrix)
    # print("33333ttt",trans3)

    # Fourth step: transform into the camera.
    cs_record2 = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # print(cs_record2)
    # print(np.array(cs_record2['translation']))
    trans4 = pc.translate(-np.array(cs_record2['translation']))
    # print("tttttttttttttttttttttttttttttttttt",trans4)
    rot4 = pc.rotate(Quaternion(cs_record2['rotation']).rotation_matrix.T)
    # print(Quaternion(cs_record2['rotation']).rotation_matrix)
    # print(rot4)


    # Fifth step: actually take a "picture" of the point cloud.
    # Grab the depths (camera frame z axis points away from the camera).

    # intrinsic_resized = np.matmul(camera_resize, np.array(cs_record['camera_intrinsic']))
    view = np.array(cs_record2['camera_intrinsic'])
    # print(view)
    # print(pc.points)
    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    points = view_points(pc.points, view, normalize=True) #resize here
    # print(points)

    # Resizing to target resolution
    if target_resolution[1]: # resizing width
        points[0,:] *= (target_resolution[1]/cam['width'])

    if target_resolution[0]: # resizing height
        points[1,:] *= (target_resolution[0]/cam['height'])

    # actual_resolution = (cam['height'], cam['width'])
    # for i in range(len(target_resolution)):
    #     if target_resolution[i]:
    #         points[i,:] *= (target_resolution[i]/actual_resolution[i])

    return points

def _create_vertical_line(P1, P2, img):
    """
    Produces and array that consists of the coordinates and intensities of each pixel in a line between two points

    :param P1: [numpy array] that consists of the coordinate of the first point (x,y)
    :param P2: [numpy array] that consists of the coordinate of the second point (x,y)
    :param img: [numpy array] the image being processed

    :return itbuffer: [numpy array] that consists of the coordinates and intensities of each pixel in the radii (shape: [numPixels, 3], row = [x,y])     
    """
    # define local variables for readability
    imageH = img.shape[0]
    imageW = img.shape[1]
    # print(P1)

    # difference and absolute difference between points
    # used to calculate slope and relative location between points
    P1_y = int(P1[1])
    P2_y = int(P2[1])
    dX = 0
    dY = P2_y - P1_y
    # print(dY)
    if dY == 0:
        dY = 1
    dXa = np.abs(dX)
    dYa = np.abs(dY)

    # predefine numpy array for output based on distance between points
    itbuffer = np.empty(
        shape=(np.maximum(int(dYa), int(dXa)), 2), dtype=np.float32)
    # print(itbuffer.shape)
    itbuffer.fill(np.nan)
    # print(itbuffer)


    # vertical line segment
    itbuffer[:, 0] = int(P1[0])
    # print(P1_y)
    if P1_y > P2_y:
        # Obtain coordinates along the line using a form of Bresenham's algorithm
        itbuffer[:, 1] = np.arange(P1_y - 1, P1_y - dYa - 1, -1)
        # print("pz>p2",itbuffer)
    else:
        itbuffer[:, 1] = np.arange(P1_y+1, P1_y+dYa+1)
        # print("p1<p2",itbuffer)

    # Remove points outside of image
    colX = itbuffer[:, 0].astype(int)
    # print(colX)
    colY = itbuffer[:, 1].astype(int)
    itbuffer = itbuffer[(colX >= 0) & (colY >= 0) &
                        (colX < imageW) & (colY < imageH)]

    return itbuffer

def _radar2camera(image_data, radar_data, enrich1, radar_xyz_endpoints, clear_radar=False):
    """
    
    Calculates a line of two radar points and puts the radar_meta data as additonal layers to the image -> image_plus


    :param image_data: [numpy array (900 x 1600 x 3)] of image data
    :param radar_data: [numpy array (xyz+meta x no of points)] that consists of the transformed radar points with z = 0
        default semantics: x y z dyn_prop id rcs vx vy vx_comp vy_comp is_quality_valid ambig_state x_rms y_rms invalid_state pdh0 vx_rms vy_rms distance
    :param radar_xyz_endpoints: [numpy array (3 x no of points)] that consits of the transformed radar points z = height
    :param clear_radar: [boolean] True if radar data should be all zero

    :return image_plus: a numpy array (900 x 1600 x (3 + number of radar_meta (e.g. velocity)))
    """
    # data_collection2 = [
    #     radar_data,
    #     enrich1
    # ]
    # print(type(enrich1[0]))
    
    dist = enrich1[0]
    azimuth = enrich1[1]
    vrad_comp = enrich1[2]
    data_collection2 = [
        radar_data,
        dist,
        azimuth,
        vrad_comp
    ]

    radar_data = np.concatenate(data_collection2, axis=0)
    # print(radar_data)
    # print(image_data)
    radar_meta_count = radar_data.shape[0]-3
    # print(radar_meta_count)
    radar_extension = np.zeros(
        (image_data.shape[0], image_data.shape[1], radar_meta_count), dtype=np.float32)
    # print(radar_extension.shape)
    no_of_points = radar_data.shape[1]

    if clear_radar:
        pass # we just don't add it to the image
    else:
        for radar_point in range(0, no_of_points): #no_of_points
            projection_line = _create_vertical_line(
                radar_data[0:2, radar_point], radar_xyz_endpoints[0:2, radar_point], image_data)
            # print(projection_line.shape)
            for pixel_point in range(0, projection_line.shape[0]):
                y = projection_line[pixel_point, 1].astype(int)
                x = projection_line[pixel_point, 0].astype(int)
                # print(y,x)

                # Check if pixel is already filled with radar data and overwrite if distance is less than the existing
                # print(radar_extension[y, x])
                # print(radar_data[3:, radar_point])
                if not np.any(radar_extension[y, x]) or radar_data[-1, radar_point] < radar_extension[y, x, -1]:

                    radar_extension[y, x] = radar_data[3:, radar_point]
                    # print(radar_extension)


    image_plus = np.concatenate((image_data, radar_extension), axis=2)
    # print(image_plus)

    return image_plus

def imageplus_creation(nusc, image_data, radar_data, enrich1, pointsensor_token, camera_token, height=(0,3),  \
        image_target_shape=(900, 1600), clear_radar=False, clear_image=False):
    """
    Superordinate function that creates image_plus data of raw camera and radar data

    :param nusc: nuScenes initialization
    :param image_data: [numpy array] (900 x 1600 x 3)
    :param radar_data: [numpy array](SHAPE?) with radar parameter (e.g. velocity) in rows and radar points for one timestep in columns
        Semantics:
            [0]: x (1)
            [1]: y (2)
            [2]: z (3)
            [3]: dyn_prop (4)
            [4]: id (5)
            [5]: rcs (6)
            [6]: vx (7)
            [7]: vy (8)
            [8]: vx_comp (9)
            [9]: vy_comp (10)
            [10]: is_quality_valid (11)
            [11]: ambig_state (12)
            [12]: x_rms (13)
            [13]: y_rms (14)
            [14]: invalid_state (15)
            [15]: pdh0 (16)
            [16]: vx_rms (17)
            [17]: vy_rms (18)
            [18]: distance (19)

    :param pointsensor_token: [str] token of the pointsensor that should be used, most likely radar
    :param camera_token: [str] token of the camera sensor
    :param height: 2 options for 2 different modi
            a.) [tuple] (e.g. height=(0,3)) to define lower and upper boundary
            b.) [str] height = 'FOV' for calculating the heights after the field of view of the radar
    :param image_target_shape: [tuple] with (height, width), default is (900, 1600)
    :param clear_radar: [boolean] True if radar data should be all zero
    :param clear_image: [boolean] True if image data should be all zero

    :returns: [tuple] image_plus, image
        -image_plus: [numpy array] (900 x 1600 x (3 + number of radar_meta (e.g. velocity)))
           Semantics:
            [0]: R (1)
            [1]: G (2)
            [2]: B (3)
            [3]: dyn_prop (4)
            [4]: id (5)
            [5]: rcs (6)
            [6]: vx (7)
            [7]: vy (8)
            [8]: vx_comp (9)
            [9]: vy_comp (10)
            [10]: is_quality_valid (11)
            [11]: ambig_state (12)
            [12]: x_rms (13)
            [13]: y_rms (14)
            [14]: invalid_state (15)
            [15]: pdh0 (16)
            [16]: vx_rms (17)
            [17]: vy_rms (18)
            [18]: distance (19)

        -cur_image: [numpy array] the original, resized image
    """

    ###############################
    ##### Preprocess the data #####
    ###############################
    # enable barcode method
    barcode = False
    if height[1] > 20:
        height = (0,1)
        barcode = True

    # Resize the image due to a target shape
    cur_img, camera_resize = _resize_image(image_data, image_target_shape)

    # Get radar points with the desired height and radar meta data
    radar_points, radar_xyz_endpoint = _radar_transformation(radar_data, height)
    # print("ssssssssssS",radar_xyz_endpoint)

    #######################
    ##### Filter Data #####
    #######################
    # Clear the image if clear_image is True
    if clear_image: 
        cur_img.fill(0)
    
    #####################################
    ##### Perform the actual Fusion #####
    #####################################
    # Map the radar points into the image
    radar_points = map_pointcloud_to_image(nusc, radar_points, pointsensor_token=pointsensor_token, camera_token=camera_token, target_resolution=image_target_shape)
    radar_xyz_endpoint = map_pointcloud_to_image(nusc, radar_xyz_endpoint, pointsensor_token=pointsensor_token, camera_token=camera_token, target_resolution=image_target_shape)
    # print(radar_points)
    #above returned points are the calibrated points after multiplying to intrensic parameter of camera


    if barcode:
        radar_points[1,:] = image_data.shape[0]
        radar_xyz_endpoint[1,:] = 0

    # Create image plus by creating projection lines and store them as additional channels in the image
    # print(cur_img.shape)
    image_plus = _radar2camera(cur_img, radar_points,enrich1, radar_xyz_endpoint, clear_radar=clear_radar)
    # print(image_plus)
    #########################
    ##### Quality Check #####
    #########################
    # Check if clear_image worked
    # if clear_image and np.count_nonzero(image_plus[0:3]):
    #     print("Clearing image did not work")

    #image_plus contains the floating point image formate with lines coordinate as an extra channel in the image
    # print(image_plus.shape)
    
    return image_plus



def create_imagep_visualization(image_plus_data, color_channel="distance", \
        draw_circles=False, cfg=None, radar_lines_opacity=1.0):
    """
    Visualization of image plus data

    Parameters:
        :image_plus_data: a numpy array (900 x 1600 x (3 + number of radar_meta (e.g. velocity)))
        :image_data: a numpy array (900 x 1600 x 3)
        :color_channel: <str> Image plus channel for colorizing the radar lines. according to radar.channel_map.
        :draw_circles: Draws circles at the bottom of the radar lines
    Returns:
        :image_data: a numpy array (900 x 1600 x 3)
    """
    # read dimensions
    image_plus_height = image_plus_data.shape[0]
    image_plus_width = image_plus_data.shape[1]
    n_channels = image_plus_data.shape[2]

    ##### Extract the image Channels #####
    if cfg is None:
        image_channels = [0,1,2]
    else:
        image_channels = [i_ch for i_ch in cfg.channels if i_ch in [0,1,2]]
    image_data = np.ones(shape=(*image_plus_data.shape[:2],3))
    # print(image_plus_data.shape[:2],3)
    if len(image_channels) > 0:
        image_data[:,:,image_channels] = image_plus_data[:,:,image_channels].copy() # copy so we dont change the old image
    # print(image_data)
    # Draw the Horizon
    image_data = np.array(image_data*255).astype(np.uint8)
    image_data = cv2.cvtColor(image_data, cv2.COLOR_RGB2BGR)
    # cv2.imshow('image', image_data)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # print(n_channels)
    # print(image_plus_data.shape[:-1])
    ##### Paint every augmented pixel on the image #####
    if n_channels > 3:
        # transfer it to the currently selected channels
        if cfg is None:
            print("Warning, no cfg provided. Thus, its not possible to find out \
                which channel shall be used for colorization")
            radar_img = np.zeros(image_plus_data.shape[:-1]) # we expect the channel index to be the last axis 450x450
            # print(radar_img)
        else:
            available_channels = {radar.channel_map[ch]:ch_idx for ch_idx, ch in enumerate(cfg.channels) if ch > 2}
            ch_idx = available_channels[color_channel]
            # Normalize the radar
            if cfg.normalize_radar: # normalization happens from -127 to 127
                radar_img = image_plus_data[...,ch_idx] + 127.5
            else:
                radar_img = radar.normalize(color_channel, image_plus_data[..., ch_idx],
                                            normalization_interval=[0, 255], sigma_factor=2)

            radar_img = np.clip(radar_img,0,255)

        radar_colormap = np.array(cv2.applyColorMap(radar_img.astype(np.uint8), cv2.COLORMAP_AUTUMN))
        # print(radar_colormap)
        # cv2.imshow('image', radar_colormap)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        # print(image_plus_data[200, 0, :])

        for x in range(0, image_plus_width):
            for y in range(0, image_plus_height):
                radar_channels = image_plus_data[y, x, 3:]
                # print(image_plus_data[y, x, 3:])
                pixel_contains_radar = np.count_nonzero(radar_channels)
                
                if not pixel_contains_radar:
                    continue                
                radar_color = radar_colormap[y,x]
                for pixel in [(y,x)]: #[(y,x-1),(y,x),(y,x+1)]:
                    # print(pixel)
                    # print(image_data.shape)
                    if image_data.shape > pixel:
                        # print("aa")
                        # Calculate the color
                        pixel_color = np.array(image_data[pixel][0:3], dtype=np.uint8)
                        pixel_color = np.squeeze(cv2.addWeighted(pixel_color, 1-radar_lines_opacity, radar_color, radar_lines_opacity, 0))

                        # Draw on image
                        image_data[pixel] = pixel_color
                    
                # only if some radar information is there
                if draw_circles:
                    if image_plus_data.shape[0] > y+1 and not np.any(image_plus_data[y+1, x,3:]):
                        cv2.circle(image_data, (x,y), 3, (0,255,0) , thickness=1) # color=radar_colormap[(y,x)].astype(np.float32)


    return image_data


if __name__ == '__main__':
    from nuscenes.nuscenes import NuScenes
    from pyquaternion import Quaternion
    import json
    # from ..tools_nuscenes.nuscenes_helper import get_sensor_sample_data

    # execute functions
    # Initialize the database
    home =  os.path.expanduser("~")
    nusc = NuScenes(version='v1.0-mini', dataroot=home +'/data/nuscenes/', verbose=True)




    # Specify sensors to use
    radar_channel = 'RADAR_FRONT'
    camera_channel = 'CAM_FRONT'

    # Get all scene tokens in a list
    scene_tokens = [s['token'] for s in nusc.scene]  #token id is stored in scene_tokens from scene.jason
    
    image_target_shape = (360, 640)
    height = (0,3)
    # Choose a scene token (between 0 and 100)
    scene_token = scene_tokens[1]
    # print("ssdsds ",scene_tokens)
    file_path = '/home/nitish/data/nuscenes/v1.0-mini/sample.json'

    with open(file_path, 'r') as json_file:
                sample_t = json.load(json_file)
    sample_tok = []
    scene_cnt = 0
    
    for token in scene_tokens:
        scene_cnt = scene_cnt+1
        sample_cnt = 0
        for dict1 in sample_t:
            if(token==dict1['scene_token']):
                sample_cnt = sample_cnt+1
                sample_tok.append(dict1['token'])
                sample = nusc.get('sample', dict1['token'])
                sample_token = dict1['token']
                sample_record = nusc.get('sample', sample_token)
                radar_token = sample_record['data'][radar_channel]
                camera_token = sample_record['data'][camera_channel]  
                radar_data, enrich1 = get_sensor_sample_data(nusc, sample, radar_channel)  
                filename_csv = "radar_data/scene_"+str(scene_cnt)+"_sample_cnt_"+str(sample_cnt)+".csv"
                np.savetxt(filename_csv, radar_data, delimiter=',')
                image_data, enrich = get_sensor_sample_data(nusc, sample, camera_channel) 
                image_plus_data = imageplus_creation(nusc,
                    image_data, radar_data, enrich1, radar_token, camera_token, height, image_target_shape, clear_radar=False, clear_image=False)
                filename = "image_plus_data/scene_"+str(scene_cnt)+"_sample_cnt_"+str(sample_cnt)+".npy"
                np.save(filename, image_plus_data)
                
                filename_img = "projected_image/scene_"+str(scene_cnt)+"_sample_cnt_"+str(sample_cnt)+".jpg"
                imgp_viz = create_imagep_visualization(image_plus_data)
                cv2.imwrite(filename_img, imgp_viz)

                print("True")
    # print(len(sample_tok))


    # Get the first sample of this scene to be demonstrated
    # scene_rec = nusc.get('scene', scene_token)  #   scene_rec contains scene_token's data from scene.jason
    # print(scene_rec)



    # sample = nusc.get('sample', scene_rec['first_sample_token']) # same as next 2nd step
    # sample = nusc.get('sample', 'ce94ef7a0522468e81c0e2b3a2f1e12d') # same as next 2nd step

    # print(sample)

    # Get the sample token and the records for sensor tokens
    # sample_token = scene_rec['first_sample_token']
    # sample_token = '0a0d6b8c2e884134a3b48df43d54c36a'

    # print(sample_token)

    # sample_record = nusc.get('sample', sample_token) # comtains all the data of first sample token from different jason files
    
    # print(sample_record)

    # Grab the front camera and the radar sensor.
    # radar_token = sample_record['data'][radar_channel]
    # camera_token = sample_record['data'][camera_channel]
    # print(radar_token, camera_token)


    # Get radar and image data
    # radar_data, enrich1 = get_sensor_sample_data(nusc, sample, radar_channel)
    # print(enrich1)

    # print(radar_data.shape)
    # image_data, enrich = get_sensor_sample_data(nusc, sample, camera_channel) # image_data is of floating type [0,1]
    # print(image_data)


    # ## Define parameters for image_plus_creation

    # # Desired image plus shape (resizing)
    # image_target_shape = (480, 640)

    # # Desired height for projection lines (2 options)
    # # ... a.) Tuple (e.g. height=(0,3)) to define lower and upper boundary
    # # ... b.) height = 'FOV' for calculating the heights after the field of view of the radar
    # height = (0.5,0.5) #'FOV'

################# imageplus_creation function starts #############
    # cur_img, camera_resize = _resize_image(image_data, image_target_shape) # resized image#########33

    # cv2.imshow("Image", cur_img)
    # # Wait for a key press to close the window
    # cv2.waitKey(0)
    # # Clean up and close the window
    # cv2.destroyAllWindows()
    # print(camera_resize)

    # radar_points, radar_xyz_endpoint = _radar_transformation(radar_data, height) ###############
    # print(radar_points)
    # print(radar_xyz_endpoint)

    ####################  map_pointcloud_to_image starts inside mageplus_creation function ###################

    # cam = nusc.get('sample_data', camera_token)
    # print(cam)
    # pointsensor = nusc.get('sample_data', radar_token) # displays details of radar_token from sample_data.jason
    # print(pointsensor)

    # pc = RadarPointCloud(radar_points)
    # # print(radar_points.shape)
    # cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token']) # cs_record stores data from calibrated_sensor.jaso of pointsesnor's calibrated_sensor_token
    # print("ccccc",cs_record)
    # rot = pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    # # print("111111rrrrrrrrr",rot)
    # trans =    pc.translate(np.array(cs_record['translation'])) # detections in vehicle frame of reference
    # # print("11111111tttttt",trans)

    # # Second step: transform to the global frame.
    # poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    # # print("gggggggg",poserecord)
    # # print(Quaternion(poserecord['rotation']).rotation_matrix)
    # # print(np.array(poserecord['translation']))
    # rot2 = pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    # # print("22222rrrr",rot2)
    # trans2 = pc.translate(np.array(poserecord['translation']))
    # # print("222222tttt",trans2)

    # # Third step: transform into the ego vehicle frame for the timestamp of the image.
    # poserecord1 = nusc.get('ego_pose', cam['ego_pose_token'])
    # # print(poserecord1)
    # rot3 = pc.translate(-np.array(poserecord1['translation']))
    # # print("33333333rrr",rot3)
    # trans3 = pc.rotate(Quaternion(poserecord1['rotation']).rotation_matrix.T) #same matrix as in first step
    # # print("33333ttt",trans3)

    # # Fourth step: transform into the camera.
    # cs_record2 = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    # # print(cs_record2)
    # # print(-np.array(cs_record2['translation']))
    # pc.translate(-np.array(cs_record2['translation']))
    # pc.rotate(Quaternion(cs_record2['rotation']).rotation_matrix.T)
    # # print(Quaternion(cs_record2['rotation']).rotation_matrix.T)


    # # Fifth step: actually take a "picture" of the point cloud.
    # # Grab the depths (camera frame z axis points away from the camera).

    # # intrinsic_resized = np.matmul(camera_resize, np.array(cs_record['camera_intrinsic']))
    # view = np.array(cs_record2['camera_intrinsic'])
    # # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
    # points = view_points(pc.points, view, normalize=True) #resize here
    # # print(points)

    # # Resizing to target resolution
    # if target_resolution[1]: # resizing width
    #     points[0,:] *= (target_resolution[1]/cam['width'])

    # if target_resolution[0]: # resizing height
    #     points[1,:] *= (target_resolution[0]/cam['height'])


    # image_plus_data = imageplus_creation(nusc,
    #     image_data, radar_data, enrich1, radar_token, camera_token, height, image_target_shape, clear_radar=False, clear_image=False)
    
    # print("aaaaaaaa: ",image_plus_data.shape)
    

    #image_plus_data contains floating image with projection lins at an additional channel in the image_plus_data
    # print(image_plus_data)
    # Visualize the result
    imgp_viz = create_imagep_visualization(image_plus_data)
    # print(imgp_viz)

    cv2.imshow('image', imgp_viz)
    cv2.waitKey(0)
    cv2.destroyAllWindows()