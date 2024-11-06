from re import template
from telnetlib import BINARY
import cv2
import numpy as np
import math
import os

def morph_shape(val):
    if val == 0:
        return cv2.MORPH_RECT
    elif val == 1:
        return cv2.MORPH_CROSS
    elif val == 2:
        return cv2.MORPH_ELLIPSE
    else:
        return 0

# def remove_staff_lines(image):
#     """Removes staff lines from the given image using Hough Transform and morphological operations.

#     Args:
#         image: The input image.

#     Returns:
#         The image with staff lines removed.
#     """

#     # Convert to grayscale
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     # Detect staff lines using Hough Transform
#     _, img_canny = cv2.threshold(cv2.bitwise_not(gray), 50, 255, cv2.THRESH_BINARY)
#     linesP = cv2.HoughLinesP(img_canny, 1, np.pi / 180, 150, None, 100, 1)

#     # Create a mask for staff lines
#     mask = np.zeros_like(gray)
#     for line in linesP:
#         x1, y1, x2, y2 = line[0]
#         cv2.line(mask, (x1, y1), (x2, y2), (255,255,255), 1)

#     # Dilate the mask to cover the entire staff area
#     kernel = np.ones((2,1),np.uint8)  # Adjust kernel_size
#     mask = cv2.dilate(mask, kernel, iterations=1)

#     cv2.imshow("MASK", mask)
#     # Invert the mask
#     mask_inverted = cv2.bitwise_not(mask)

#     # Apply the mask to the original image
#     no_staff = cv2.bitwise_or(image, np.stack([mask,mask,mask], axis=-1))
#     no_staff_inv = cv2.bitwise_not(no_staff)
#     image_no_staff_lines = cv2.erode(cv2.dilate(no_staff_inv, kernel, anchor=(0,0), iterations=4), kernel, anchor=(0,0), iterations=1)
#     print(image.shape)
#     return cv2.bitwise_not(image_no_staff_lines)

def remove_staff_lines(image):
    thresh = cv2.adaptiveThreshold(~image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2 )
    vertical_size = 3
    vertical_structure = cv2.getStructuringElement( cv2.MORPH_RECT, (1,vertical_size))
    bw = cv2.erode(thresh, vertical_structure, anchor = (-1,-1), iterations=1)
    bw = cv2.dilate(bw, vertical_structure, anchor= (-1,-1), iterations=1)
    bw = cv2.bitwise_not(bw)
    # cv2.imshow("BW",bw)
    # edges = cv2.adaptiveThreshold( bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 3, -2)
    # cv2.imshow("EDGES", edges)
    # kernel =  cv2.getStructuringElement( cv2.MORPH_RECT, (2,2))
    # smooth = cv2.dilate(edges,kernel=kernel, iterations=1)
    # ret, smooth = cv2.threshold(smooth, 240, 255, cv2.THRESH_TOZERO)
    # smooth = cv2.blur(smooth, (2,2))
    # smooth = cv2.bitwise_not(smooth)
    return bw

def get_notes(image):
    image = cv2.bitwise_not(image)
    params = cv2.SimpleBlobDetector.Params()

    # blob detection
    params.filterByColor = True
    params.filterByCircularity = False
    params.filterByConvexity = False
    params.filterByInertia = False
    params.blobColor = 255
    params.minArea = 10
    params.maxArea = 100
    params.minCircularity =.01
    params.maxCircularity = 1
    params.minConvexity = .1
    params.maxConvexity = 1
    params.minInertiaRatio = 0.1
    params.maxInertiaRatio = 1
    # params.minDistBetweenBlobs = 0.01
    params.minRepeatability = 1


    detector = cv2.SimpleBlobDetector().create(params)
    keypoints = detector.detect(image)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    i = 0
    for kp in keypoints:
        # print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
        i+=1
        cv2.rectangle(im_with_keypoints,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),2)
    return im_with_keypoints


class BoundingBox(object):
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        overlap_y = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        return BoundingBox(x, y, w, h)

    def draw(self, img, color, thickness):
        pos = ((int)(self.x), (int)(self.y))
        size = ((int)(self.x + self.w), (int)(self.y + self.h))
        cv2.rectangle(img, pos, size, color, thickness)

    def getCorner(self):
        return self.x, self.y

    def getWidth(self):
        return self.w

    def getHeight(self):
        return self.h

    def getCenter(self):
        return self.middle
    
def match(img, templates, start_percent, stop_percent, threshold):
    img_width, img_height = img.shape[::-1]
    best_location_count = -1
    best_locations = []
    best_scale = 1

    x = []
    y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 3)]:
        locations = []
        location_count = 0

        for template in templates:
            if (scale*template.shape[0] > img.shape[0] or scale*template.shape[1] > img.shape[1]):
                continue

            template = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            result = np.where(result >= threshold)
            location_count += len(result[0])
            locations += [result]

        # print("scale: {0}, hits: {1}".format(scale, location_count))
        x.append(location_count)
        y.append(scale)
        # plt.plot(y, x)
        # plt.pause(0.00001)
        if (location_count > best_location_count):
            best_location_count = location_count
            best_locations = locations
            best_scale = scale
            # plt.axis([0, 2, 0, best_location_count])
        elif (location_count < best_location_count):
            pass
    # plt.close()

    return best_locations, best_scale

def locate_templates(img, templates, start, stop, threshold):
    locations, scale = match(img, templates, start, stop, threshold)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w *= scale
        h *= scale
        img_locations.append([BoundingBox(pt[0], pt[1], w, h) for pt in zip(*locations[i][::-1])])
    print("DONE")
    return img_locations

def merge_boxes(boxes, threshold):
    filtered_boxes = []
    while len(boxes) > 0:
        print(len(boxes))
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    r = r.merge(boxes.pop(i))
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes

def locate_symbols(img, templates):
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    boxes = locate_templates(gray,templates, 20,99, 0.7)    # Change
    boxes = merge_boxes([j for i in boxes for j in i], 0.5)

    if (len(boxes) > 0):
        print("[INFO] Boxes Found.")
    return boxes

def draw_all_boxes( img, boxes ):
    for box in boxes:
        box.draw(img, (0,0,255), 2)
        x = int(box.getCorner()[0] + (box.getWidth() // 2))
        y = int(box.getCorner()[1] + box.getHeight() + 10)
        # img = cv2.putText(img, "{} clef".format("&"), (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.9, (0,0,255))

def main():   
    img = cv2.imread("Piano.jpg", cv2.IMREAD_COLOR)
    base_template_path =  os.path.join( os.getcwd(), "Template")
    
    # Find Clefs
    clef_templates = []
    clef_template_base_path = os.path.join( base_template_path, "clef" )
    clef_types = os.listdir( clef_template_base_path )

    for clef_type in clef_types:
        clef_template_paths = os.listdir( os.path.join(clef_template_base_path, clef_type) )
        
        for clef_path in clef_template_paths :
            template_img = cv2.imread( os.path.join(base_template_path, "clef", clef_type , clef_path), cv2.IMREAD_GRAYSCALE)
            clef_templates.append( template_img )
            cv2.imshow( clef_path, template_img)

        clef_boxes = locate_symbols(img, clef_templates) 
        draw_all_boxes( img, clef_boxes )
    
    # img_canny = cv2.Cann
    # cdst = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    # # cdstP = np.copy(cdst)

    # lines = cv2.HoughLines(img_canny, 1, np.pi/ 180,  400, None, 0, 0)
    # if lines is not None:
    #     for i in range(0, len(lines)):
    #         rho = lines[i][0][0]
    #         theta = lines[i][0][1]
    #         a = math.cos(theta)
    #         b = math.sin(theta)
    #         x0 = a * rho
    #         y0 = b * rho
    #         pt1 = (int(x0 + 1000*(-b)), int(y0 + 1000*(a)))
    #         pt2 = (int(x0 - 1000*(-b)), int(y0 - 1000*(a)))
    #         cv2.line(cdst, pt1, pt2, (0,0,255), 1, cv2.LINE_AA)
    
    
    
    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(cdstP, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, cv2.LINE_AA)
    
    # print("P:", len(linesP))
    # print("L:", len(lines))

    ## cv2.namedWindow(title_dilation_window)
    ## cv2.createTrackbar(title_trackbar_element_shape, title_dilation_window, 0, max_elem, dilatation)
    ## cv2.createTrackbar(title_trackbar_kernel_size, title_dilation_window, 0, max_kernel_size, dilatation)
    ## erosion(0)
    ## dilatation(0)
    
    cv2.imshow("Source", img)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines", get_notes(remove_staff_lines(img)))
    # cv2.imshow("Canny Image", im_with_keypoints)
    
    # # print(img.shape)
    # cv2.imshow("Music", img)
    cv2.waitKey(0)

if( __name__ == "__main__"):
    main()