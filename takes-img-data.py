import json
import numpy as np
import cv2
import os

"""
Pipeline:
1/Read image + extract coordinates of roi
2/Create the mask having same dimension with image
3/Create image of roi padded with black having same dimension with orginal
Return roi image and roi coordinate , image height and width 
"""
def extract_image_roi(image_path, json_data):
    roi  = json_data['shapes'][0]['points']
    
    #read the orginal image
    image = cv2.imread(image_path)
   
    image_height, image_width, image_channels = image.shape
    #create the mask have the same dim with orginal
    mask = np.zeros_like(image)
    #take the coordinates of rois
    roi_polygon = np.array([roi],dtype= np.int32)
    #This is a function in OpenCV that fills a convex polygon 
    #on an image with a specified color
    test = cv2.fillConvexPoly(mask,roi_polygon,(255,255,255))
    roi_image = cv2.bitwise_and(image,mask) #type : numpy array
    roi_height, roi_width, roi_channels = roi_image.shape
    
    # Compare the dimensions of the input image and ROI image
    if image_height != roi_height or image_width != roi_width:
        print('Warning: ROI dimensions do not match input image dimensions.')
    return roi_image, roi_polygon , image_height , image_width
   
#Find the x and y coor of the orginal size of the image
def find_corner(x,y,w,h):
    #Top left corner
    tl = (x, y+h)
    # Top right corner
    tr = (x + w, y+h)
    # Bottom right corner
    br = (x + w , y)
    # Bottom left corner
    bl = (x, y)
    x_coor = np.array([[tl[0], tr[0], br[0], bl[0]]])
    y_coor = np.array([[tl[1], tr[1], br[1], bl[1]]])
    return x_coor , y_coor

def mid_point(x1,y1,x2,y2):
    mid_x = (x1+x2)/2
    mid_y = (y1+y2)/2
    return (mid_x, mid_y)

#Extracting the coordinates of the roi 
def extract_coordinates(roi_polygon):
    x_coordinates = roi_polygon[:,:,0]
    y_coordinates = roi_polygon[:,:,1]
    return x_coordinates , y_coordinates

def find_equation_line(point1, point2):
    x1, y1 = point1
    x2, y2 = point2
    if x2 - x1 == 0:
        # Vertical line
        slope = float('inf')
        intercept = x1
    else:
        # Non-vertical line
        slope = (y2 - y1) / (x2 - x1)
        intercept = y1 - slope * x1
    return slope, intercept

def find_intersection(line1, line2):
    (m1, b1) = line1
    (m2, b2) = line2
    if m1 == float('inf'):
        # line1 is vertical
        x = b1
        y = m2 * x + b2
    elif m2 == float('inf'):
        # line2 is vertical
        x = b2
        y = m1 * x + b1
    elif m2 - m1 == 0:
        # lines are parallel
        return None
    else:
        # lines are not parallel and nor vertical
        x = (b2 - b1) / (m1 - m2)
        y = m1 * x + b1
    return x, y


def find_intersections(line,lines):
    intersections_points = []
    for other_line in lines:
        if other_line == line:
            continue
        else:
            intersections_point = find_intersection(line,other_line)
            if intersections_point is not None:
                intersections_points.append(intersections_point)
    return intersections_points

"""
extaract points of 4 corners of roi and the middle point of orginal size
"""
def extract_points(coordinates, img_height, img_width):
    x_coor_roi , y_coor_roi = extract_coordinates(coordinates) #numpy.array . shape = (1,4)
    x_coor_roi = np.ceil(x_coor_roi)
    y_coor_roi = np.ceil(y_coor_roi)
    x_1_roi , y_1_roi = x_coor_roi[0,3] , y_coor_roi[0,3]
    x_2_roi , y_2_roi = x_coor_roi[0,2] , y_coor_roi[0,2]
    x_3_roi , y_3_roi = x_coor_roi[0,1] , y_coor_roi[0,1]
    x_4_roi , y_4_roi = x_coor_roi[0,0] , y_coor_roi[0,0]
    point1 = (x_1_roi , y_1_roi)
    point2 = (x_2_roi , y_2_roi)
    point3 = (x_3_roi , y_3_roi)
    point4 = (x_4_roi , y_4_roi)
    """x_mid_1_roi, y_mid_1_roi = mid_point(x_coor_roi[0,0],y_coor_roi[0,0],x_coor_roi[0,1],y_coor_roi[0,1]) 
    x_mid_2_roi, y_mid_2_roi = mid_point(x_coor_roi[0,2],y_coor_roi[0,2],x_coor_roi[0,3],y_coor_roi[0,3])
"""
    #find midpoints of picture
    x_coor_original , y_coor_orginal = find_corner(0,0,img_width,img_height)
    (x_mid_1_original , y_mid_1_orginal) = mid_point(x_coor_original[0,0],y_coor_orginal[0,0],x_coor_original[0,1],y_coor_orginal[0,1])
    (x_mid_2_original , y_mid_2_orginal) = mid_point(x_coor_original[0,2],y_coor_orginal[0,2],x_coor_original[0,3],y_coor_orginal[0,3])

    x1_origin = (x_mid_1_original , y_mid_1_orginal)
    x2_origin = (x_mid_2_original , y_mid_2_orginal)
    print(x1_origin,x2_origin)
    return point1, point2, point3, point4, x1_origin, x2_origin

#half left image
def create_half_augment_json(old_data,intersections_points,path_left,path_right):
    orig_point1 = old_data['shapes'][0]['points'][1].copy()
    orig_point2 = old_data['shapes'][0]['points'][2].copy()
   
    old_data['shapes'][0]['points'][1] = list(intersections_points[1])
    old_data['shapes'][0]['points'][2] =  list(intersections_points[0])
    old_data["imageWidth"] = 77
    old_data['imagePath'] = path_left
    
    augment_folder_left = os.path.join(os.getcwd(), 'json-halfcut-left')
    if not os.path.exists(augment_folder_left):
        os.mkdir(augment_folder_left)
    base_name_left = os.path.splitext(os.path.basename(old_data["imagePath"]))[0]
    new_file_name_left = f"{base_name_left}_halfcut_left.json"
    old_data['imagePath'] = path_left
    new_file_path_left = os.path.join(augment_folder_left, new_file_name_left)
    

    with open(new_file_path_left, 'w') as f:
        json.dump(old_data, f)
    
    old_data['shapes'][0]['points'][1] = orig_point1
    old_data['shapes'][0]['points'][2] = orig_point2

    old_data['shapes'][0]['points'][0] = list(intersections_points[1])
    old_data['shapes'][0]['points'][3] = list(intersections_points[0])
    old_data['imagePath'] = path_right

    augment_folder_right = os.path.join(os.getcwd(),'json-halfcut-right')
    if not os.path.exists(augment_folder_right):
        os.mkdir(augment_folder_right)
    
    base_name_right = os.path.splitext(os.path.basename(old_data['imagePath']))[0]
    new_file_name_right = f'{base_name_right}_halfcut_right.json'
    new_file_path_right = os.path.join(augment_folder_right,new_file_name_right)
    
    with open(new_file_path_right,'w') as f:
        json.dump(old_data,f)
    
    
def create_half_augment_image(image,index):
    if image is None:
        raise Exception(f"Failed to read image at {image_path}")

    half_width = int(image.shape[1] / 2)
    left_half = image[:, :half_width]

    path_name_left = os.path.join(os.getcwd(),'halfcut_left')
    if not os.path.exists(path_name_left):
        os.mkdir(path_name_left)
    
    file_name_left = f'image_{index}.jpg'
    file_path_left = os.path.join(path_name_left,file_name_left)
    cv2.imwrite(file_path_left,left_half)

    right_half = image[:, half_width:]
    path_name_right = os.path.join(os.getcwd(),'halfcut_right')
    if not os.path.exists(path_name_right):
        os.mkdir(path_name_right)
    
    file_name_right = f'image_{index}.jpg'
    file_path_right = os.path.join(path_name_right,file_name_right)
    cv2.imwrite(file_path_right,right_half)
    return file_path_left , file_path_right
    

if __name__ =="__main__":
    folder_image = os.path.join(os.getcwd(),'folder-img')
    folder_json = os.path.join(os.getcwd(),'folder-img-data')
    for index, filename in enumerate(os.listdir(folder_image)):
        image_path = os.path.join(folder_image,filename)
        image = cv2.imread(image_path)
        json_path = os.path.join(folder_json,filename.replace('.jpg','.json'))
        with open(json_path) as json_file:
            json_data = json.load(json_file)
        roi_images, coordinates , img_height , img_width = extract_image_roi(image_path= image_path,json_data=json_data)
        point1, point2, point3, point4, x1_origin, x2_origin = extract_points(coordinates, img_height, img_width)
        lines = []
        line_origin = find_equation_line(x1_origin,x2_origin)
        line1_roi = find_equation_line(point1,point2)
        line2_roi = find_equation_line(point3,point4)
        print(line_origin)
        print(line1_roi)
        print(line2_roi)
        """ print(line_origin)
        print(line1_roi)
        print(line2_roi)
        """
        lines.append(line1_roi)
        lines.append(line2_roi)
        intersection_points = find_intersections(line_origin,lines)
        print(intersection_points)
        #print(intersection_points)
        
        path_left , path_right = create_half_augment_image(image,index)

        output_path = create_half_augment_json(json_data,intersection_points,path_left,path_right)
       