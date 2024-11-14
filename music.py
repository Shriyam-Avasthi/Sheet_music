from re import template
import sys
from telnetlib import BINARY
import cv2
import numpy as np
import math
import os
from enum import Enum

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
    vertical_size = 5
    vertical_structure = cv2.getStructuringElement( cv2.MORPH_RECT, (1,vertical_size))
    horizontal_structure = cv2.getStructuringElement( cv2.MORPH_RECT, (4,1))

    bw = cv2.erode(thresh, vertical_structure, anchor = (-1,-1), iterations=1)
    bw = cv2.erode(bw, horizontal_structure, anchor = (-1,-1), iterations=1)
    bw = cv2.dilate(bw, vertical_structure, anchor= (-1,-1), iterations=1)
    bw = cv2.dilate(bw, horizontal_structure, anchor= (-1,-1), iterations=1)
    # cv2.imshow("BW",bw)
    # bw = cv2.bitwise_not(bw)
    # edges = cv2.adaptiveThreshold( bw, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 3, -2)
    # cv2.imshow("EDGES", edges)
    kernel =  cv2.getStructuringElement( cv2.MORPH_RECT, (2,2))
    # smooth = cv2.dilate(edges,kernel=kernel, iterations=1)
    # ret, smooth = cv2.threshold(smooth, 240, 255, cv2.THRESH_TOZERO)
    # smooth = cv2.blur(smooth, (2,2))
    # smooth = cv2.bitwise_not(bw)
    return bw

def get_notes(image):
    # image = cv2.bitwise_not(image)
    params = cv2.SimpleBlobDetector.Params()

    # blob detection
    params.filterByColor = True
    params.filterByCircularity = True
    params.filterByConvexity = False
    params.filterByInertia = False
    params.blobColor = 0
    params.minArea = 10
    params.maxArea = 100
    params.minCircularity =0.001
    params.maxCircularity = 1
    params.minConvexity = 0.001
    params.maxConvexity = 1
    params.minInertiaRatio = 0.001
    params.maxInertiaRatio = 1
    params.minDistBetweenBlobs = 0.01
    params.minRepeatability = 1

    detector = cv2.SimpleBlobDetector().create(params)
    keypoints = detector.detect(image)
    # print(keypoints)
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    i = 0
    for kp in keypoints:
        print("(%f,%f)"%(kp.pt[0],kp.pt[1]))
        i+=1
        cv2.rectangle(im_with_keypoints,(int(kp.pt[0]),int(kp.pt[1])),(int(kp.pt[0])+1,int(kp.pt[1])+1),(0,255,0),1)
    return im_with_keypoints

class SymbolType(Enum):
    BASS_CLEF = 0
    TREBLE_CLEF = 1
    QUARTER_NOTE = 2
    WHOLE_NOTE = 3
    HALF_NOTE = 4
    EIGHTH_REST = 5
    QUARTER_REST = 6
    HALF_REST = 7
    EIGHTH_FLAG = 8

class Note():
    def __init__(self, pitch, position, type):
        self.pitch = [pitch]
        self.position = position
        self.type = type

    def get_pitch(self):
        return self.pitch

    def get_position(self):
        return self.position

class Staff():
    def __init__(self, delta = 2):
        self.lines = []
        self.notes = []
        self.delta = delta
        self.line_gap = 0
        self.clef_type = None
        self.clef_position = (0,0)

    def Add_Line(self, line_y):
        self.lines.append(line_y)
        if( len(self.lines) > 1 and self.line_gap == 0 ):
            self.line_gap = self.lines[1] - self.lines[0]

    def add_clef_info(self, clef_type, clef_position ):
        self.clef_type = clef_type
        self.clef_position = clef_position

    def Belongs_To(self, line_y):
        if( len(self.lines) > 1 ):
            dist = self.lines[1] - self.lines[0]
            d = line_y - self.lines[-1]
            if( d > dist - self.delta and d < dist + self.delta):
                return True
            else:
                return False
        else:
            return True

    def position_within_staff(self, position):
        # print("0",self.lines[0])
        # print("-1", self.lines[-1])
        y = position[1]
        if( (y >= (self.get_upper_boundary())) and (y <= (self.get_lower_boundary()) )):
            return True
        else:
            return False

    def get_line_pitch(self, line_y):
        start_pitch = "A"
        line_space = self.lines[1] - self.lines[0]
        factor = round((line_y - self.lines[0]) / line_space)
        factor = factor % 4
        return ( chr(ord(start_pitch) + factor * 2) )

    def get_note_pitch(self, position):
        dist = sys.maxsize
        pitch = ""
        line_y = 0
        for s in self.lines:
            if( dist > abs(position[1] - s) ):
                dist = abs(position[1] - s)
                pitch = self.get_line_pitch(s)
                line_y = s
        if( dist >= self.line_gap/4 ):
            if( position[1] < line_y ):
                pitch = chr(ord(pitch) - 1)
            elif( position[1] > line_y ):
                pitch = chr(ord(pitch) + 1) 

        return pitch

    def get_upper_boundary(self):
        return ( self.lines[0] - 4.5 * self.line_gap )
    
    def get_lower_boundary(self):
        return ( self.lines[-1] + 4.5 * self.line_gap )

    def add_note(self, note):
        ret, index = self.get_nearest_note_pos(note)
        if(not ret):
            if(note.position[0] > self.clef_position[0] + 40 ):      # USE SOME RELATIVE VALUE HERE AS WELL
                self.notes.insert(index+1, note) 
        elif( ret ):
            flag = True
            for p in self.notes[index].pitch:
                if(p == note.pitch[0]):
                    flag = False
                    break
            if( flag ):
                self.notes[index].pitch.append(note.pitch[0])

    def get_nearest_note_pos(self, note):
        index = 0
        pos = (0,0)
        prev_distance = sys.maxsize
        for i,n in enumerate(self.notes):
            if( math.isclose(n.position[0], note.position[0], abs_tol=10) ):  # CHANGE THE ABS_TOL PARAM TO SOMETHING MORE RELATIVE
                return (True, i)
            else:
                dist = abs(n.position[0] - note.position[0])
                if( prev_distance > dist ):
                    prev_distance = dist
                    pos = n.position
                    index = i
        if( note.position[0] < pos[0] ):
            return (False, index - 1)
        else :
            return (False, index)

    def print_notes(self):
        print()
        print( self.clef_type, end = " " )
        for n in self.notes:
            # if( n.type == SymbolType.BASS_CLEF ):
            #     print("\n#", end = " ")
            # elif( n.type == SymbolType.TREBLE_CLEF ):
            #     print("\n&", end = " ")
            # elif( n.type == SymbolType.QUARTER_NOTE ):
            for p in n.pitch:
                print(p, end = "-")
            if(n.type == SymbolType.QUARTER_NOTE):
                print("4", end = " ")
            elif( n.type == SymbolType.HALF_NOTE ):
                print("2", end = " ")
            

class Music():
    def __init__(self, staff_lines, img):
        self.staff_lines = staff_lines
        self.img = img 
        self.notes = []

    def add_note( self, note ):
        staff_index = self.get_staff(note.position)
        if( staff_index != -1 ):
            self.staff_lines[staff_index].add_note(note)
    
    # def add_note_with_position( self, position, type ):
    #     for staff in self.staff_lines:
    #         if( staff.position_within_staff(position)):
    #             pitch = staff.get_note_pitch( position )
    #             note = Note(pitch, position, type)
    #             staff.add_note(note)

    def print_notes(self):
        for staff in self.staff_lines:
            staff.print_notes()
    
    def get_staff(self, position):
        for i,staff in enumerate(self.staff_lines):
            if( staff.position_within_staff(position) ):
                return i
        return -1
    
    def add_clef_info( self, clef_type, clef_position ):
        staff_index = self.get_staff(clef_position)
        if( staff_index != -1 ):
            self.staff_lines[staff_index].add_clef_info(clef_type, clef_position)
    
    def draw_marks(self):
        for staff in self.staff_lines:
            p = 0
            for l in staff.lines:
                pos = (0,l)
                cv2.putText(self.img, staff.get_line_pitch(l), (pos[0] + p, pos[1]), cv2.FONT_HERSHEY_DUPLEX, 0.25, (255,0,0))
                p += 6
            for n in staff.notes:
                # print(n.position)
                pos = (int(n.position[0]), int(n.position[1]))
                cv2.circle(self.img, pos, radius=1, color=(0, 0, 255), thickness=1)
                pitch = "|"
                for p in n.pitch:
                    if(n.type == SymbolType.HALF_NOTE or n.type == SymbolType.WHOLE_NOTE or n.type == SymbolType.QUARTER_NOTE):
                        pitch += p
                pitch += "|"
                if(n.type == SymbolType.QUARTER_NOTE or n.type == SymbolType.QUARTER_REST):
                    pitch += "4"
                elif( n.type == SymbolType.HALF_NOTE or n.type == SymbolType.HALF_REST):
                    pitch += "2"
                elif( n.type == SymbolType.WHOLE_NOTE):
                    pitch += "1"
            
                cv2.putText(self.img, pitch, (pos[0] + 4, pos[1]+4), cv2.FONT_HERSHEY_DUPLEX, 0.4, (0,0,255))
        cv2.imshow("NOTES", self.img)
    
    def get_note_pitch( self, position ):
        staff_index = self.get_staff(position)
        if( staff_index != -1 ):
            return self.staff_lines[staff_index].get_note_pitch(position)

class BoundingBox(object):
    def __init__(self, x, y, w, h, text = ""):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.middle = self.x + self.w/2, self.y + self.h/2
        self.area = self.w * self.h
        self.text = text

    def overlap(self, other):
        overlap_x = max(0, min(self.x + self.w, other.x + other.w) - max(self.x, other.x))
        overlap_y = max(0, min(self.y + self.h, other.y + other.h) - max(self.y, other.y))
        overlap_area = overlap_x * overlap_y
        return overlap_area / self.area

    def distance(self, other):
        dx = self.middle[0] - other.middle[0]
        dy = self.middle[1] - other.middle[1]
        return math.sqrt(dx*dx + dy*dy)

    def merge(self, other, new_text = None):
        x = min(self.x, other.x)
        y = min(self.y, other.y)
        w = max(self.x + self.w, other.x + other.w) - x
        h = max(self.y + self.h, other.y + other.h) - y
        if( new_text == None ):
            return BoundingBox(x, y, w, h, self.text)
        else:
            return BoundingBox(x, y, w, h, new_text)

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
    
def match(img, templates, start_percent, stop_percent, threshold, overlap = 0.7, mask = None, match_criterion = cv2.TM_CCOEFF_NORMED):
    img_width, img_height = img.shape[::-1]
    best_location_count = -1
    best_locations = []
    best_scale = 1

    x = []
    y = []
    for scale in [i/100.0 for i in range(start_percent, stop_percent + 1, 1)]:
        locations = []
        identified_boxes = []
        location_count = 0

        for template in templates:
            if (scale*template.shape[0] > img.shape[0] or scale*template.shape[1] > img.shape[1]):
                print(scale)
                continue

            temp = cv2.resize(template, None,
                fx = scale, fy = scale, interpolation = cv2.INTER_CUBIC)
            result = cv2.matchTemplate(img, temp, match_criterion, mask = mask)
            result = np.where(result >= threshold)
            w, h = temp.shape[::-1]

            for pt in zip(*result[::-1]):
                new_box = BoundingBox(pt[0], pt[1], w, h)
                
                # Check for overlap with existing boxes
                if not any(new_box.overlap(box) > overlap for box in identified_boxes):
                    identified_boxes.append(new_box)
            
            # Count non-overlapping boxes at this scale
            location_count += len(identified_boxes)

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

def locate_templates(img, templates, start, stop, threshold, text, overlap = 0.7, match_criterion = cv2.TM_CCOEFF_NORMED):
    locations, scale = match(img, templates, start, stop, threshold, overlap, match_criterion=match_criterion)
    print(scale)
    img_locations = []
    for i in range(len(templates)):
        w, h = templates[i].shape[::-1]
        w = w*scale
        h = h*scale
        img_locations.append([BoundingBox(pt[0], pt[1], w, h, text) for pt in zip(*locations[i][::-1])])
    print("DONE")
    return img_locations

def merge_boxes(boxes, threshold, music):
    preference_order = ["&", "#", "q", "h", "w", "r_8", "r_4", "r_16", "r_2", "r_1", "f_8"]
    filtered_boxes = []
    while len(boxes) > 0:
        # print(len(boxes))
        r = boxes.pop(0)
        boxes.sort(key=lambda box: box.distance(r))
        merged = True
        while (merged):
            merged = False
            i = 0
            for _ in range(len(boxes)):
                if r.overlap(boxes[i]) > threshold or boxes[i].overlap(r) > threshold:
                    b = boxes.pop(i)
                    i1 = preference_order.index(r.text)
                    i2 = preference_order.index(b.text)
                    idx = min(i1, i2)
                    r = r.merge(b, preference_order[idx])
                    merged = True
                elif boxes[i].distance(r) > r.w / 2 + boxes[i].w / 2:
                    break
                else:
                    i += 1
        filtered_boxes.append(r)
    return filtered_boxes

def draw_all_boxes( img, boxes, color = (0,0,255) ):
    for box in boxes:
        box.draw(img, color, 1)
        x = int(box.getCorner()[0] + (box.getWidth() // 2))
        y = int(box.getCorner()[1] + box.getHeight() + 10)
        img = cv2.putText(img, box.text, (x, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, color)

def main(): 
    SCALING_FACTOR = 1
    img = cv2.imread("Sheet.jpg", cv2.IMREAD_COLOR)
    img = cv2.fastNlMeansDenoising(img, None, 10, 7, 21)
    base_template_path =  os.path.join( os.getcwd(), "Template")
    gray = cv2.cvtColor( img, cv2.COLOR_BGR2GRAY )
    _, gray =cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    # gray = cv2.bitwise_not(gray)
    # gray = remove_staff_lines(gray)

    # cdst = cv2.cvtColor(img_canny, cv2.COLOR_GRAY2BGR)
    # # cdstP = np.copy(cdst)
    
    lines = cv2.HoughLines(cv2.bitwise_not(gray), 1, np.pi/ 180,  250, None, 0, 0, 1.5, 1.6)
    line_ys = []
    staffs = []
    STAFF_DELTA = 20 * SCALING_FACTOR

    if lines is not None:
        DELTA = 5 * SCALING_FACTOR
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + 1000*SCALING_FACTOR*(-b)), int(y0 + 1000*SCALING_FACTOR*(a)))
            pt2 = (int(x0 - 1000*SCALING_FACTOR*(-b)), int(y0 - 1000*SCALING_FACTOR*(a)))
            delta_y = abs(pt1[1] - pt2[1]) // 2
            y = pt2[1]
            line_ys.append(y)

    line_ys = sorted(line_ys)

    for y in line_ys:
        print("LINES:", y)
        if( len(staffs)> 0 and staffs[-1].Belongs_To(y)):
            staffs[-1].Add_Line(y)
            cv2.line(img, (-1000, y), (1000, y), (20*len(staffs),0,0), 1, cv2.LINE_AA)   
        else:
            staff = Staff(STAFF_DELTA)
            staff.Add_Line(y)
            staffs.append(staff) 
            cv2.line(img, (-1000, y), (1000, y), (20*len(staffs),0,0), 1, cv2.LINE_AA)   

    print("L:", len(staffs))
    # line_ys = sorted(line_ys)

    music = Music(staffs, img.copy())

    # Find Clefs
    clef_templates = []
    clef_template_base_path = os.path.join( base_template_path, "clef" )
    clef_boxes = []

    # Find bass clef
    clef_template_paths = os.listdir( os.path.join(clef_template_base_path, "Bass") )
    for clef_path in clef_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "clef", "Bass" , clef_path), cv2.IMREAD_GRAYSCALE )
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        # template_img = cv2.bitwise_not(template_img)
        # template_img = remove_staff_lines(template_img)
        clef_templates.append( template_img )
        # cv2.imshow( clef_path, template_img)

    bass_clef_boxes = locate_templates(gray, clef_templates, int(8*SCALING_FACTOR), int(30*SCALING_FACTOR), 0.55, "#") 
    clef_boxes.extend(bass_clef_boxes)
    
    # Find Treble clef
    clef_templates = []
    clef_template_paths = os.listdir( os.path.join(clef_template_base_path, "Treble") )
    for clef_path in clef_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "clef", "Treble" , clef_path), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        clef_templates.append( template_img )
        # cv2.imshow( clef_path, template_img)

    treble_clef_boxes = locate_templates(gray, clef_templates, int(8 * SCALING_FACTOR), int(100 * SCALING_FACTOR), 0.55, "&")
    clef_boxes.extend(treble_clef_boxes)

    #Finalize the clef identification
    boxes = merge_boxes([j for i in clef_boxes for j in i], 0.5, music)
    boxes.sort(key=lambda box: box.getCenter()[1])
    staff_index = 0
    for box in boxes:
        
        type = None
        if(box.text == "#"):
            type = SymbolType.BASS_CLEF
        elif(box.text == "&"):
            type = SymbolType.TREBLE_CLEF

        music.add_clef_info(type, box.getCenter())
    draw_all_boxes( img, boxes)

    # Find Quarter Notes
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "note", "quarter") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "note", "quarter", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)
    
    note_boxes = locate_templates(gray, note_templates, int(25*SCALING_FACTOR), int(60*SCALING_FACTOR), 0.65, "q", 0.85)

    #Finalize Quarter Note Identification
    # boxes = merge_boxes([j for i in note_boxes for j in i], 0.8, music)
    # # print(staffs[staff_index].position_within_staff(218.7))
    # for box in boxes:
    #     pitch = music.get_note_pitch( box.getCenter() )
    #     if( pitch is not None ):
    #         type = SymbolType.QUARTER_NOTE

    #         note = Note(pitch, box.getCenter(), type )
    #         music.add_note(note)
    # # print(len(boxes))
    # draw_all_boxes( img, boxes)
    
    # draw_all_boxes(gray_noteheads_only, [j for i in note_boxes for j in i])

    # Find Half Notes
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "note", "half") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "note", "half", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)
    
    note_boxes.extend( locate_templates(gray, note_templates, int(30*SCALING_FACTOR), int(60*SCALING_FACTOR), 0.55, "h", 0.7) )

    #Finalize Half Note Identification
    boxes = merge_boxes([j for i in note_boxes for j in i], 0.8, music)
    # print(staffs[staff_index].position_within_staff(218.7))
    for box in boxes:
        pitch = music.get_note_pitch( box.getCenter() )
        
        if( pitch is not None ):
            if(box.text == "h"):
                type = SymbolType.HALF_NOTE
            elif(box.text == "q"):
                type = SymbolType.QUARTER_NOTE

            note = Note(pitch, box.getCenter(), type )
            music.add_note(note)
    # print(len(boxes))
    draw_all_boxes( img, boxes)

    # Find Whole Notes
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "note", "whole") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "note", "whole", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)
    
    # note_boxes.extend( locate_templates(gray, note_templates, int(30*SCALING_FACTOR), int(60*SCALING_FACTOR), 0.55, "h", 0.7) )
    note_boxes.extend(locate_templates(gray, note_templates, int(25*SCALING_FACTOR), int(60*SCALING_FACTOR), 0.6, "w", 0.8) )

    #Finalize Half Note Identification
    boxes = merge_boxes([j for i in note_boxes for j in i], 0.8, music)
    # print(staffs[staff_index].position_within_staff(218.7))
    for box in boxes:
        pitch = music.get_note_pitch( box.getCenter() )
        
        if( pitch is not None ):
            if(box.text == "h"):
                type = SymbolType.HALF_NOTE
            elif(box.text == "q"):
                type = SymbolType.QUARTER_NOTE
            elif( box.text == "w"):
                type = SymbolType.WHOLE_NOTE

            note = Note(pitch, box.getCenter(), type )
            music.add_note(note)
    # print(len(boxes))
    draw_all_boxes( img, boxes)
    


    #Find Eighth Rests
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "rest", "Eighth_rest") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "rest", "Eighth_rest", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)

    rest_boxes = locate_templates(gray, note_templates, int(50*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.6, "r_8", 0.85)

    #Find Quarter Rests
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "rest", "Quarter_rest") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "rest", "Quarter_rest", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)

    rest_boxes.extend(locate_templates(gray, note_templates, int(25*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.55, "r_4", 0.85) )

    #Find Half Rests
    note_templates = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "rest", "Half_rest") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "rest", "Half_rest", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)

    rest_boxes.extend(locate_templates(gray, note_templates, int(25*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.75, "r_2", 0.85) )

    # rest_boxes = locate_templates(gray, note_templates, int(20*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.75, "r_2", 0.85) 

    #Finalize Rest Identification
    boxes = merge_boxes([j for i in rest_boxes for j in i], 0.6, music)
    # print(staffs[staff_index].position_within_staff(218.7))
    for box in boxes:
        pitch = music.get_note_pitch( box.getCenter() )
        
        if( pitch is not None ):
            if(box.text == "h"):
                type = SymbolType.HALF_NOTE
            elif(box.text == "q"):
                type = SymbolType.QUARTER_NOTE
            elif( box.text == "r_8"):
                type = SymbolType.EIGHTH_REST
            elif(box.text == "r_4"):
                type = SymbolType.QUARTER_REST
            elif(box.text == "r_2"):
                type = SymbolType.HALF_REST

            note = Note(pitch, box.getCenter(), type )
            music.add_note(note)

    ############################################################# FINDING FLAGS #################################################################

    #Eighth Flag

    note_templates = []
    flag_boxes = []
    # gray_noteheads_only = cv2.bitwise_and(remove_staff_lines(gray), cv2.bitwise_not(gray))
    note_template_paths = os.listdir( os.path.join(base_template_path, "flag", "eighth_flag") )
    for note_path in note_template_paths :
        template_img = cv2.imread( os.path.join(base_template_path, "flag", "eighth_flag", note_path ), cv2.IMREAD_GRAYSCALE)
        _, template_img = cv2.threshold(template_img, 200, 255, cv2.THRESH_BINARY)
        note_templates.append( template_img )
        # cv2.imshow( note_path, template_img)

    flag_boxes.extend(locate_templates(gray, note_templates, int(25*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.85, "f_8", 0.85) )

    # flag_boxes = locate_templates(gray, note_templates, int(20*SCALING_FACTOR), int(80*SCALING_FACTOR), 0.75, "r_2", 0.85) 

    #Finalize Rest Identification
    boxes = merge_boxes([j for i in flag_boxes for j in i], 0.6, music)
    # print(staffs[staff_index].position_within_staff(218.7))
    for box in boxes:
        pitch = music.get_note_pitch( box.getCenter() )
        
        if( pitch is not None ):
            if(box.text == "h"):
                type = SymbolType.HALF_NOTE
            elif(box.text == "q"):
                type = SymbolType.QUARTER_NOTE
            elif( box.text == "r_8"):
                type = SymbolType.EIGHTH_REST
            elif(box.text == "r_4"):
                type = SymbolType.QUARTER_REST
            elif(box.text == "r_2"):
                type = SymbolType.HALF_REST
            elif(box.text == "f_8"):
                type = SymbolType.EIGHTH_FLAG

            note = Note(pitch, box.getCenter(), type )
            music.add_note(note)
            

    # print(len(boxes))
    draw_all_boxes( img, boxes)
    music.print_notes()
    print()
    music.draw_marks()

    cv2.imshow("Source", img)
    cv2.imshow("GRAY", gray)
    # cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    # cv2.imshow("Detected Lines", get_notes(remove_staff_lines(img)))
    # cv2.imshow("Canny Image", im_with_keypoints)
    
    # # print(img.shape)
    # cv2.imshow("Music", img)
    cv2.waitKey(0)

if( __name__ == "__main__"):
    main()