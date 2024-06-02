import sys
import xml.etree.ElementTree as ET
import cv2
import csv
import numpy
import numpy as np
# import supervision as sv
from collections import defaultdict
from shapely.geometry import Polygon, LineString, MultiPolygon, MultiLineString, MultiPoint
from shapely.ops import unary_union
from datetime import timedelta
from statistics import mode
import math
from scipy.interpolate import UnivariateSpline
import os
import torch




def save_violation_snapshot(frame, frame_count, output_dir, filename, verbose=False):
    ''' Save The SnapShot for the Detected Lane Occurarnce '''

    # Save the frame as an image
    filename_short = "{}_{}.png".format(filename, frame_count)
    filename = os.path.join(output_dir, filename_short)
    cv2.imwrite(filename, frame)
    if verbose:
        print(f"Snapshot saved as {filename_short}")


def reduce_bounding_box_width(bbox, reduction_factor):
    # Unpack the bounding box coordinates
    bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = bbox

    # Calculate the current height and center of the bounding box
    bbox_height = bbox_y_max - bbox_y_min
    bbox_center_y = (bbox_y_min + bbox_y_max) / 2

    # Calculate the current width and center of the bounding box
    bbox_width = bbox_x_max - bbox_x_min
    bbox_center_x = (bbox_x_min + bbox_x_max) / 2

    # Calculate the new width based on the reduction factor
    new_bbox_width = bbox_width * reduction_factor
    new_bbox_height = bbox_height * reduction_factor

    # Calculate the new x-coordinates of the bounding box
    new_bbox_x_min = bbox_center_x - (new_bbox_width / 2)
    new_bbox_x_max = bbox_center_x + (new_bbox_width / 2)

    new_bbox_y_min = bbox_center_y - (new_bbox_height / 2)
    new_bbox_y_max = bbox_center_y + (new_bbox_height / 2)

    # Return the updated bounding box coordinates
    return new_bbox_x_min, new_bbox_y_min, new_bbox_x_max, new_bbox_y_max


def middle_x_y(xyxy, shift = 1):
    """Grans the middle of the bounding box

    Args:
        xyxy (_type_): _description_

    Returns:
        _type_: _description_
    """

    x1, y1, x2, y2 = xyxy

    # Calculate the center of the box
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2
    if shift:
        center_y = np.mean((y1, center_y)) #todo alter
    return center_x, center_y


class ViewTransformer:
    def __init__(self, target: np.ndarray, source: np.ndarray) -> None:
        source = np.array(source.astype(np.float32))
        target = np.array(target.astype(np.float32))
        # target = np.array(target.astype(np.float32))
        self.m = cv2.getPerspectiveTransform(source, target)

    def transform_points(self, points):
        # if points.size == 0:
        #    return points

        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(
            reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


class Transform:
    def __init__(self, world, image, im_width, im_height, scipy_optim=False):
        self.H, self.mask_test = cv2.findHomography(image, world)
        self.H_inv, _ = cv2.findHomography(world, image)
        self.world = world
        self.image = image
        self.mask_image = numpy.zeros((im_height, im_width), dtype=numpy.uint8)
        self.mask_image = cv2.fillConvexPoly(self.mask_image, self.image.astype(int), 1)

        self.c_x = 0
        self.c_y = 0
        self.points = 0
        for i in range(im_width):
            for j in range(im_height):
                if (self.mask_image[j, i] > 0):
                    self.c_x += i
                    self.c_y += j
                    self.points += 1
        self.c_x /= self.points
        self.c_y /= self.points

    def better_distance(self, xyxy):
        # H, _ = cv2.findHomography(image_points1, world_points)
        H = self.H
        # Use the homography matrix to transform the image points from the second frame
        # to their estimated real-world positions
        real_world_estimates = cv2.perspectiveTransform(np.array([xyxy]), H)[0]

        # Calculate the distance the car has moved in the real world
        # by taking one point (for simplicity) or by averaging the movement of all points
        movement_vector = real_world_estimates[0] - self.world[0]
        distance_moved = np.linalg.norm(movement_vector)

        print(f"The estimated distance the car moved is: {distance_moved} units in the real-world scale")

    def project(self, x, y, inverse=False):
        pp = numpy.asarray([[x], [y], [1]], dtype=numpy.float32);
        if (not inverse):
            p_t = numpy.matmul(self.H, pp)
        else:
            p_t = numpy.matmul(self.H_inv, pp)

        if (p_t[2] == 0):
            p_t[2] = 0.01

        p_ret = [p_t[0] / p_t[2], p_t[1] / p_t[2]]
        return p_ret[0], p_ret[1]
    
    def projection_middle(self, xyxy):
        print(1)
        H = self.H
        # Assuming you have the following points (ix, iy) -> (wx, wy)
        image_points = xyxy  # (ix, iy)
        world_points = cv2.perspectiveTransform(np.array([xyxy]), H)[0]  # (wx, wy)

        # Calculate Homography
        h, status = cv2.findHomography(image_points, world_points)

        # Example vehicle bounding box in image coordinates (xmin, ymin, xmax, ymax)
        bbox_image = xyxy  # corners of the bounding box

        # Convert corners to homogeneous coordinates
        ones = np.ones(shape=(len(bbox_image), 1))
        points_homogeneous = np.hstack([bbox_image, ones])

        # Transform to world coordinates
        world_points = h @ points_homogeneous.T

        # Convert from homogeneous to 2D
        world_points /= world_points[2, :]
        world_points = world_points[:2, :].T

        # Compute center bottom in world coordinates
        center_bottom_image = np.array([np.mean(bbox_image[:, 0]), bbox_image[1, 1], 1]).reshape(3, 1)
        center_bottom_world = (h @ center_bottom_image).ravel()
        center_bottom_world /= center_bottom_world[2]

        print("World Coordinates of the Bounding Box Corners:", world_points)
        print("World Coordinates of the Center Bottom of the Bounding Box:", center_bottom_world[:2])

    def is_within(self, x, y):
        return self.mask_image[y, x] > 0

    def geometric_error(self, params):
        """
        Calculate the geometric error as the sum of squared differences
        between the projected points and the corresponding points in
        the image.
        """
        # Reshape params into H matrix
        H = np.reshape(params, (3, 3))
        H[-1, -1] = 1  # Ensure the homography is normalized

        total_error = 0
        for i in range(len(self.world)):
            world_point = np.append(self.world[i], 1)  # Convert to homogeneous coordinates
            image_point = np.append(self.image[i], 1)

            # Project the world point using the current homography
            projected_point = np.dot(H, world_point)
            projected_point /= projected_point[2]  # Normalize to convert back from homogeneous coordinates

            # Calculate the squared error
            error = np.sum((projected_point - image_point) ** 2)
            total_error += error

        return total_error

    def calculate_reprojection_error(self, H):
        """
        Calculate the reprojection error for a given homography matrix.
        """
        error = 0
        for i in range(len(self.world)):
            world_point = np.append(self.world[i], 1)  # Convert to homogeneous coordinates
            image_point_expected = np.array(self.image[i])
            image_point_projected = np.dot(H, world_point)
            image_point_projected /= image_point_projected[2]  # Convert from homogeneous coordinates
            error += np.linalg.norm(image_point_expected - image_point_projected[:2])
        return error

    def distance_from_centroid(self, x, y, euclid=True):
        if euclid:
            return np.sqrt((x - self.c_x) ** 2 + (y - self.c_y) ** 2)
        else:
            return (abs(x - self.c_x) + abs(y - self.c_y))

    def top_left_im(self):
        return numpy.min(self.image, axis=0)

    def bottom_right_im(self):
        return numpy.max(self.image, axis=0)

    def top_left_world(self):
        return numpy.min(self.world, axis=0)

    def bottom_right_world(self):
        return numpy.max(self.world, axis=0)


class CameraView:
    def __init__(self, xml_file, lane_points=[]):
        self.cs = None
        self.left_divider = None
        self.right_divider = None
        self.im_width = None
        self.im_height = None
        self.worlds = None
        self.combined_coords = None
        self.component_coords = None
        self.transforms = []
        self.view_transformer = []
        self.straight_divider = []  # for drawing lines
        self.actual_divider = lane_points
        self.load_xml(xml_file)

    def get_straight_divider(self):
        # to do extend out to image frame
        image_width = self.im_width
        image_height = self.im_height

        # Calculate the slope and intercept of the line
        x_values = [coord[0] for coord in self.straight_divider]
        y_values = [coord[1] for coord in self.straight_divider]
        slope, intercept = np.polyfit(x_values, y_values, 1)

        # Calculate the y-coordinates where the line intersects with the top and bottom borders
        y_top = 0
        x_top = int((y_top - intercept) / slope)
        y_bottom = image_height
        x_bottom = int((y_bottom - intercept) / slope)

        # Extend the line by adding the new points to the existing coordinates
        extended_line = [(x_top, y_top), (x_bottom, y_bottom)]
        self.straight_divider.extend(extended_line)

        ## now Left
        x_values = [coord[0] for coord in self.left_divider]
        y_values = [coord[1] for coord in self.left_divider]
        slope, intercept = np.polyfit(x_values, y_values, 1)

        # Calculate the y-coordinates where the line intersects with the top and bottom borders
        y_top = 0
        x_top = int((y_top - intercept) / slope)
        y_bottom = image_height
        x_bottom = int((y_bottom - intercept) / slope)

        # Extend the line by adding the new points to the existing coordinates
        extended_line = [(x_top, y_top), (x_bottom, y_bottom)]
        self.left_divider.extend(extended_line)

        # now right

        x_values = [coord[0] for coord in self.right_divider]
        y_values = [coord[1] for coord in self.right_divider]
        slope, intercept = np.polyfit(x_values, y_values, 1)

        # Calculate the y-coordinates where the line intersects with the top and bottom borders
        y_top = 0
        x_top = int((y_top - intercept) / slope)
        y_bottom = image_height
        x_bottom = int((y_bottom - intercept) / slope)

        # Extend the line by adding the new points to the existing coordinates
        extended_line = [(x_top, y_top), (x_bottom, y_bottom)]
        self.right_divider.extend(extended_line)

        return self.straight_divider, self.left_divider, self.right_divider

    def load_xml(self, xml_file):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        worlds = []
        im_ww = []
        im_width = int(root.get('width'))
        im_height = int(root.get('height'))
        self.im_width = im_width  # TODO GET MOVEMENT HALFED
        self.im_height = im_height
        for region in root.findall('region'):
            world = []
            image = []
            for correspondance in region.findall('correspondance'):
                world.append([float(correspondance.get('wx')), float(correspondance.get('wy'))])
                image.append([float(correspondance.get('ix')), float(correspondance.get('iy'))])
                if not [float(correspondance.get('wx')), float(correspondance.get('wy'))] in worlds:
                    worlds.append([float(correspondance.get('wx')), float(correspondance.get('wy'))])
            transform = Transform(numpy.array(world), numpy.array(image), im_width, im_height)
            # view_transformer = ViewTransformer(source=np.array([image]), target=np.array([world]))
            im_ww.append(image)

            self.transforms.append(transform)
            #self.view_transformer.append(view_transformer)

        worlds.sort()
        self.define_midpoint(im_ww)
        self.worlds = np.array(worlds)

    def define_midpoint(self, image_boxes: list):
        ultra_left = []
        ultra_list = []
        ultra_right = []
        # Sort the corners based on their y-coordinates
        for image_box in image_boxes:
            avg_x = int(sum([corner[0] for corner in image_box]) / len(image_box))
            min_x = int(min([corner[0] for corner in image_box]))
            max_x = int(max([corner[0] for corner in image_box]))

            # Calculate the average y-coordinate
            avg_y = int(sum([corner[1] for corner in image_box]) / len(image_box))
            min_y = int(min([corner[1] for corner in image_box]))
            max_y = int(max([corner[1] for corner in image_box]))
            # Create the middle point
            ultra_left.append((min_x, avg_y))
            ultra_list.append((avg_x, avg_y))
            ultra_right.append((max_x, avg_y))
        self.left_divider = ultra_left
        self.right_divider = ultra_right
        self.straight_divider = ultra_list

    def find_best_transform(self, x, y):
        dist = [t.distance_from_centroid(x, y) for t in self.transforms]
        return numpy.argmin(dist)

    def project(self, x, y):
        h = self.find_best_transform(x, y)
        #hh = self.view_transformer[h].transform_points(np.array((x, y)))[0]
        x1, y1 = self.transforms[self.find_best_transform(x, y)].project(x, y)
        h = np.array((x1[0], y1[0]))
        return h

    def movement_vector(self, xyxy, x, y):
        n = self.find_best_transform(x, y)
        H = self.transforms[n].H
        center_x = (xyxy[0] + xyxy[2]) / 2
        center_y = (xyxy[1] + xyxy[3]) / 2

        # Define bounding box corners
        bbox_corners = np.array([
            [xyxy[0], xyxy[1]],  # Top-left
            [xyxy[2], xyxy[1]],  # Top-right
            [xyxy[2], xyxy[3]],  # Bottom-right
            [xyxy[0], xyxy[3]],  # Bottom-left
        ], dtype='float32').reshape(-1, 1, 2)

        # Add the center point to the array of points to be transformed
        # It's important to reshape it to (1, 1, 2) to match the expected input format
        center_point = np.array([[center_x, center_y]], dtype='float32').reshape(1, 1, 2)

        # Combine the corner points and the center point into one array for transformation
        points_to_transform = np.concatenate((bbox_corners, center_point), axis=0)

        # Use the homography matrix to transform the points to their estimated real-world positions
        real_world_estimates = cv2.perspectiveTransform(points_to_transform, H)

        # Extract the transformed center point from the resulting array
        transformed_center = real_world_estimates[-1, 0]  # The last point is the center

        # Return the entire movement vector (which includes corners and center)
        # Along with the real-world estimate of the middle of the box
        return real_world_estimates, transformed_center
        # return movement_vector
        # distance_moved = np.linalg.norm(movement_vector)

        # print(f"The estimated distance the car moved is: {distance_moved} units in the real-world scale")

    def is_inside(self, xcam, ycam):
        is_inside = cv2.pointPolygonTest(self.combined_coords.reshape(-1, 2), (xcam, ycam), False)
        if is_inside in [0, 1]:
            return True
        else:
            return False

    def is_outside_x_is_inside_y(self, xcam, ycam):
        original_polygon = self.combined_coords.reshape(-1, 2)
        min_x = np.min(original_polygon[:, 0])
        max_x = np.max(original_polygon[:, 0])

        # Set the minimum x-coordinate to 0 and the maximum x-coordinate to the screen width
        screen_width = self.im_width  # Replace with the actual screen width
        extended_polygon = original_polygon.copy()
        for i, val in enumerate(extended_polygon[:, 0]):
            if val < self.im_width / 2:
                extended_polygon[i, 0] = 0
            else:
                extended_polygon[i, 0] = screen_width

        is_inside = cv2.pointPolygonTest(self.combined_coords.reshape(-1, 2), (xcam, ycam), False)
        if not is_inside:
            is_inside_y = cv2.pointPolygonTest(extended_polygon, (xcam, ycam), False)
            if is_inside_y:
                return True

        return False

    def get_poly_shape(self):
        return self.combined_polygon

    def split_frame(self, xy, divider_points=None):
        # Assuming 'frame' is a rectangular frame represented as a list of coordinates
        if divider_points is None:
            divider_points = self.actual_divider
        # 'xy' = x1, y1

        # Split the frame into left and right regions
        left_region = []
        right_region = []

        # Iterate through each point in the frame

        x = xy[0]
        y = xy[1]

        # Determine the region based on the position relative to the line defined by divider points
        region = 'left' if y < self.get_line_y(x, divider_points) else 'right'

        # Add the point to the corresponding region
        if region == 'left':

            left_region.append(xy)
        else:

            right_region.append(xy)
        return region

    def get_line_y(self, x, line_points):
        #  'line_points' is a list of (x, y) coordinates defining a line
        # 'line_points' = [(x1, y1), (x2, y2), (x3, y3), ...]
        if self.cs is None:
            x_sorted, y_sorted = zip(*sorted(line_points, key=lambda tup: tup[0]))
            smoothness_factor = 3
            coefficients = np.polyfit(x_sorted, y_sorted, 3)
            cs = np.poly1d(coefficients)
            self.cs = cs
        else:
            cs = self.cs
        # cs = UnivariateSpline(x_sorted, y_sorted, k=smoothness_factor)
        # cs = CubicSpline(x_sorted, y_sorted, bc_type='natural')

        # Evaluate the spline at a higher resolution
        t_new = np.linspace(0, self.im_width, self.im_width)
        y_new = cs(x)
        return int(y_new)

    def poly_point(self, image, show=False):

        # off for now
        if show:
            for t in self.transforms:
                colour = (numpy.random.randint(256), numpy.random.randint(256), numpy.random.randint(256))
                image = cv2.polylines(image, [t.image.astype(int)], True, colour, 2)

        polygons = []

        if not hasattr(self, 'combined_polygon'):
            for t in self.transforms:
                # Convert the polyline points to a Shapely Polygon
                # The 't.image' is assumed to contain the x,y coordinates for the polyline
                polygon = Polygon(t.image.astype(np.int32).reshape(-1, 2))
                polygons.append(polygon)

            # Use unary_union to combine the polygons
            combined_polygon = unary_union(polygons)
            self.combined_polygon = combined_polygon
            combined_coords = np.array(combined_polygon.exterior.coords).astype(int).reshape((-1, 1, 2))
            self.combined_coords = combined_coords
            if not show:
                return

        else:
            combined_polygon = self.combined_polygon
            if not show:
                return





        return image

    def is_outside(self, x, y):
        x1, x2, y1, y2 = self.component_coords
        return x1 <= x <= x2 and y1 <= y <= y2

    def IOU(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        # compute the area of intersection rectangle
        interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
        # compute the area of both the prediction and ground-truth
        # rectangles
        boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
        boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
        # compute the intersection over union by taking the intersection
        # area and dividing it by the sum of prediction + ground-truth
        # areas - the interesection area
        iou = interArea / float(boxAArea + boxBArea - interArea)
        # return the intersection over union value
        return iou

    def detect_center_line(self, image):
        # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Apply Hough Line Transform
        lines = cv2.HoughLinesP(edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

        # Calculate the average position of detected lines
        x_sum = 0
        line_count = 0

        for line in lines:
            x1, y1, x2, y2 = line[0]
            x_sum += (x1 + x2) / 2
            line_count += 1

        if line_count > 0:
            center_line_x = int(x_sum / line_count)
        else:
            # If no lines are detected, assume the center line is in the middle of the image
            center_line_x = image.shape[1] // 2

        # Draw the center line on the image
        cv2.line(image, (center_line_x, 0), (center_line_x, image.shape[0]), (0, 0, 255), 2)

        return image, center_line_x

    def sanity_check(self, image):

        for t in self.transforms:
            colour = (255, 0, 255)
            image = cv2.polylines(image, [t.image.astype(int)], True, colour, 2)

            tl = t.top_left_world()
            br = t.bottom_right_world()

            for x in numpy.linspace(tl[0], br[0], 5):
                for y in numpy.linspace(tl[1], br[1], 5):
                    ix, iy = t.project(x, y, inverse=True)
                    image = cv2.circle(image, (int(ix), int(iy)), 2, colour, -1)

        return image

    def sanity_check_alt(self, image):

        for t in self.transforms:
            colour = (numpy.random.randint(256), numpy.random.randint(256), numpy.random.randint(256))
            image = cv2.polylines(image, [t.image.astype(int)], True, colour, 2)

            tl = t.top_left_world()
            br = t.bottom_right_world()

            # Draw the rectangle using the top-left and bottom-right points
            image = cv2.rectangle(image, tuple(tl.astype(int)), tuple(br.astype(int)), colour, 2)

            # Store the previous point coordinates
            prev_ix, prev_iy = None, None

            for x in numpy.linspace(tl[0], br[0], 5):
                for y in numpy.linspace(tl[1], br[1], 5):
                    ix, iy = t.project(x, y, inverse=True)
                    image = cv2.circle(image, (int(ix), int(iy)), 2, colour, -1)

                    # If we have a previous point, draw a line and calculate the distance
                    if prev_ix is not None and prev_iy is not None:
                        # Draw a line between the previous and current point
                        image = cv2.line(image, (int(prev_ix), int(prev_iy)), (int(ix), int(iy)), colour, 2)

                        # Calculate the distance
                        distance = numpy.sqrt((ix - prev_ix) ** 2 + (iy - prev_iy) ** 2)[0]

                        # Define the midpoint where the text will be displayed
                        midpoint = ((int(prev_ix) + int(ix)) // 2, (int(prev_iy) + int(iy)) // 2)

                        # Write the distance on the image
                        image = cv2.putText(image, f"{distance:.2f}", midpoint, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour,
                                            2)

                    # Update the previous point coordinates
                    prev_ix, prev_iy = ix, iy

        return image


class LaneDetector:
    def __init__(self, combined_polygon):
        # Initialize your polygon here (example points, replace with your own)
        self.combined_polygon = np.array(combined_polygon.exterior.coords).astype(np.int32).reshape((-1, 2))

    def process_frame(self, image):
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian blur
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Canny edge detection
        edges = cv2.Canny(blur, 50, 150)

        # Define the region of interest based on the combined_polygon
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, [self.combined_polygon], 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        # Hough line detection
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)

        # Draw the lines on the image
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Check if the line endpoints are inside the polygon
                if cv2.pointPolygonTest(self.combined_polygon, (int(x1), int(y1)), False) >= 0 and cv2.pointPolygonTest(
                        self.combined_polygon, (int(x2), int(y2)), False) >= 0:
                    cv2.line(image, (x1, y1), (x2, y2), (0, 255, 0), 5)

        return image


class TrajectoryProfile:
    def __init__(self, direction_forward_value=0, frame_width=640, frame_height=360, straight_line_divide=[],
                 left_divide=[], right_divide=[], entry_point=['Top', 'Bottom', 'Left', 'Right'], jerk_threshold = 70) -> None:
        self.jerk_threshold = jerk_threshold
        self._max_speed = 120  # max hm/hr
        self.entry_point = entry_point
        self.object_tracks = {}
        self.forward_points = []
        self.backward_points = []

        self.line_forward_function = None
        self.line_backwards_function = None
        self.angles = []
        self.direction_value_forward = direction_forward_value
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.straight_line_divide = straight_line_divide
        self.left_divide = left_divide
        self.right_divide = right_divide

    def define_average_lines(self):
        x_sorted, y_sorted = zip(*sorted(self.forward_points, key=lambda tup: tup[0]))
        x_sorted, y_sorted = self.average_similar_x(x_sorted, y_sorted, 1)
        coefficients = np.polyfit(x_sorted, y_sorted, 3)
        x_sorted_b, y_sorted_b = zip(*sorted(self.backward_points, key=lambda tup: tup[0]))
        x_sorted_b, y_sorted_b = self.average_similar_x(x_sorted_b, y_sorted_b, 1)
        coefficients_b = np.polyfit(x_sorted_b, y_sorted_b, 3)

        self.line_forward_function = np.poly1d(coefficients)
        self.line_backwards_function = np.poly1d(coefficients_b)
    def detect_unsafe_driving(self, object_id, speed_threshold=30, swerving_threshold=40):
        ''' This function is used to detect unsafe driving occurances, return true if deemed as unsafe or false if
        safe'''
        # Extract tracked positions and speeds
        positions_time_space = self.object_tracks[object_id]['positions']
        if len(positions_time_space) <= 1:
            return False, False
        positions = [pos[1] for pos in positions_time_space]

        timestamp = [pos[0] for pos in positions_time_space]
        # Detect swerving behavior
        swerving_detected = False
        for i in range(1, len(self.object_tracks[object_id]['direction'])):

            direction_change = abs(
                self.object_tracks[object_id]['direction'][i - 1] - self.object_tracks[object_id]['direction'][i])

            # Check if the change in direction exceeds the threshold
            if direction_change > swerving_threshold:
                swerving_detected = True
                break

        jerks = self.calculate_jerk_profile(positions, timestamp)
        self.object_tracks[object_id]['jerks'] = jerks
        jerk_safe = self.is_safe_jerk_profile(jerks, self.jerk_threshold)

        # Return unsafe driving detection results
        return swerving_detected, not jerk_safe

    def is_safe_jerk_profile(self, jerk_profile, threshold):
        """
        Determine if the jerk profile is safe based on a specified jerk threshold.
        :param jerk_profile: List of jerk values (each a tuple of x and y components).
        :param threshold: Maximum acceptable jerk magnitude.
        :return: True if safe, False otherwise.
        """
        for jerk in jerk_profile:
            # Calculate the magnitude of the jerk vector
            magnitude = abs(jerk)
            if magnitude > threshold:
                return False  # Unsafe if any jerk magnitude exceeds the threshold
        return True  # Safe if all jerks are within the threshold

    def calculate_velocity(self, positions, timestamps):
        velocities = []
        delta_time_a = []
        for i in range(1, len(positions)):
            delta_pos = np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
            delta_time = timestamps[i] - timestamps[i - 1]
            delta_time_a.append(timestamps[i])
            if delta_time.total_seconds() <= 0:
                velocities.append(0)
            else:
                velocities.append(delta_pos / delta_time.total_seconds())
        return velocities, delta_time_a  # m/s

    def calculate_acceleration(self, velocities, timestamps):
        accelerations = []
        for i in range(1, len(velocities)):
            delta_vel = velocities[i] - velocities[i - 1]
            delta_time = timestamps[i] - timestamps[i - 1]
            dtcheck = delta_time.total_seconds()
            if dtcheck <= 0:
                accelerations.append(0)
            else:
                accelerations.append(delta_vel / delta_time.total_seconds())
        return accelerations

    def is_forward_threshold(self):
        return self.direction_value_forward

    def calculate_jerk(self, accelerations, timestamps):

        jerks = []

        for i in range(1, len(accelerations)):
            delta_acc = accelerations[i] - accelerations[i - 1]
            delta_time = timestamps[i] - timestamps[i - 1]

            dtcheck = delta_time.total_seconds()
            if dtcheck <= 0:
                jerks.append(0)
            else:
                jerks.append(delta_acc / delta_time.total_seconds())
        return jerks

    # Wrapper function that uses the above functions to calculate velocity, acceleration, and jerk.
    def calculate_jerk_profile(self, positions, timestamps):
        ACCEL_CAP = 20
        velocities, delta_t = self.calculate_velocity(positions, timestamps)
        accelerations = self.calculate_acceleration(velocities, delta_t)
        accelerations_alt = [a for a in accelerations if a < ACCEL_CAP]
        delta_t = [t for t, a in zip(delta_t, accelerations) if a < ACCEL_CAP]
        jerks = self.calculate_jerk(accelerations_alt, delta_t)

        return jerks

    def average_similar_x(self, x, y, threshold):
        # Sort x and y by x
        sorted_indices = np.argsort(x)
        x_sorted = x[sorted_indices]
        y_sorted = y[sorted_indices]

        # Group x's that are closer than the threshold
        groups = np.split(x_sorted, np.where(np.diff(x_sorted) > threshold)[0] + 1)
        y_groups = np.split(y_sorted, np.where(np.diff(x_sorted) > threshold)[0] + 1)

        # Average y's in each group
        x_avg = np.array([np.mean(group) for group in groups])
        y_avg = np.array([np.mean(group) for group in y_groups])

        return x_avg, y_avg

    def _draw_line(self, frame, buffer=10, dontShow=False):
        # Convert coordinates to numpy arrays

        x = []
        y = []
        # for a in self.straight_line_divide:

        # cv2.polylines(frame, np.array([self.straight_line_divide]), False, (0, 255, 0),thickness=2 )
        # cv2.polylines(frame, np.array([self.left_divide]), False, (0, 255, 0), thickness=2)
        # cv2.polylines(frame, np.array([self.right_divide]), False, (0, 255, 0), thickness=2)
        for i in self.forward_points:
            x.append((i[0] + i[2]) / 2)
            y.append((i[1] + i[3]) / 2)
            if not dontShow:
                cv2.circle(frame, (int(x[-1]), int(y[-1])), 2, (0, 255, 255), -1)

        if len(x) < 2:
            return frame



        x = []
        y = []
        for i in self.backward_points:
            x.append((i[0] + i[2]) / 2)
            y.append((i[1] + i[3]) / 2)
            if not dontShow:
                cv2.circle(frame, (int(x[-1]), int(y[-1])), 2, (255, 255, 255), -1)
        if len(x) < 2:
            return frame



        return frame

    def save(self, file='ss'):

        converted_dict = {}
        for key, value in self.object_tracks.items():
            converted_key = str(key)  # Convert the key to str

            converted_value = {}
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, timedelta):
                    converted_value[sub_key] = sub_value.isoformat()
                else:
                    converted_value[sub_key] = sub_value

            converted_dict[converted_key] = converted_value

        columns = ['positions', 'speeds', 'accelerations', 'predicted_positions', 'class',
                   'direction', 'accelerations', 'jerks', 'average_speeds']

        # Open the CSV file in write mode
        with open(f'{file}/object_tracks.csv', 'w', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=columns)

            # Write the column names as the header
            writer.writeheader()

            # Write the data for each object id
            for object_id, values in converted_dict.items():
                row = {
                    'positions': values['positions'],
                    'speeds': values['speeds'],
                    'accelerations': values['accelerations'],
                    'predicted_positions': values['predicted_positions'],
                    'direction': values['direction'],
                    'class': values['class'],
                    'jerks': values['jerks'], #TODO I ADDED THIS FROM DOWN BELOW
                    'accelerations': values['accelerations'],
                    'average_speeds': values['average_speeds']

                }
                writer.writerow(row)

        print("CSV data has been saved successfully.")

    def is_predicted_close_to_actual(self, object_id, threshold=1):
        ''' Determines if Predicted Is CLose to Actual, Implying Safety'''
        if len(self.object_tracks[object_id]['predicted_positions']) < 2:
            return True
        predicted_position = self.object_tracks[object_id]['predicted_positions'][-1]
        actual_position = self.object_tracks[object_id]['positions'][-1][1]

        distance = np.sqrt(
            (predicted_position[0] - actual_position[0]) ** 2 + (predicted_position[1] - actual_position[1]) ** 2)

        if distance <= threshold:
            return True
        else:
            return False

    def get_direction(self, object_id):
        """ Get The Direction Of The Object"""
        if self.object_tracks[object_id]['direction'] == []:
            return None
        if self.object_tracks[object_id]['direction'][-1] > self.is_forward_threshold():
            return True
        else:
            return False

    def is_in_trajectory(self, bbox, forward=True):
        if forward is None:
            return True  # Not Enough Info Skip

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = reduce_bounding_box_width(bbox, 1)  # TODo make this bigger

        # Iterate through each point and check if it falls within the bounding box
        if forward:
            if self.line_forward_function is not None:
                point_x = (bbox_x_min + bbox_x_max) / 2
                point_y = self.line_forward_function(point_x)

                if bbox_x_min <= point_x <= bbox_x_max and bbox_y_min <= point_y <= bbox_y_max:
                    return True  # Bounding box intersects a point
        else:
            if self.line_backwards_function is not None:
                point_x = (bbox_x_min + bbox_x_max) / 2
                point_y = self.line_backwards_function(point_x)


                if bbox_x_min <= point_x <= bbox_x_max and bbox_y_min <= point_y <= bbox_y_max:
                    return True  # Bounding box intersects a point
        return False

    def is_in_path(self, bbox, forward=True):
        if forward is None:
            return True  # Not Enough Info Skip

        bbox_x_min, bbox_y_min, bbox_x_max, bbox_y_max = reduce_bounding_box_width(bbox, 0.5)

        # Iterate through each point and check if it falls within the bounding box
        if forward:
            for point in self.forward_points:

                point_x, point_y = middle_x_y(point)

                if bbox_x_min <= point_x <= bbox_x_max and bbox_y_min <= point_y <= bbox_y_max:
                    return True  # Bounding box intersects a point
        else:
            for point in self.backward_points:
                point_x, point_y = middle_x_y(point)

                if bbox_x_min <= point_x <= bbox_x_max and bbox_y_min <= point_y <= bbox_y_max:
                    return True  # Bounding box intersects a point

        return False  # Bounding box does not intersect any point
    def get_region(self, object_id):
        return self.object_tracks[object_id]['region']
    def get_jerk_value(self, object_id):
        if len(self.object_tracks[object_id]['jerks']) == 0:
            return 0
        else:
            return self.object_tracks[object_id]['jerks'][-1]
    def get_roadside(self, object_id):  # TODO get area of entries

        entry_point = ['Top', 'Bottom', 'Left', 'Right']
        entry_point = self.entry_point
        new_entry = ['left', 'right']
        if new_entry[0] in mode(self.object_tracks[object_id]['region']):
            return True
        elif new_entry[1] in mode(self.object_tracks[object_id]['region']):
            return False

        if entry_point[0] in self.object_tracks[object_id]['roadside']:
            return True
        if entry_point[1] in self.object_tracks[object_id]['roadside']:
            return False

        if entry_point[2] in self.object_tracks[object_id]['roadside']:
            return True
        elif entry_point[3] in self.object_tracks[object_id]['roadside']:
            return False
        else:
            print('dont know what side')
            return None

    def predict_roadside(self, bound_box, frame_width, frame_height):
        x_min, y_min, x_max, y_max = bound_box

        box_center_x = (x_min + x_max) / 2
        box_center_y = (y_min + y_max) / 2

        if box_center_x < frame_width / 3:
            horizontal_position = "Left"
        elif box_center_x > (2 * frame_width) / 3:
            horizontal_position = "Right"
        else:
            horizontal_position = "Center"

        if box_center_y < (3 / 2 * frame_height) / 2:
            vertical_position = "Top"
        elif box_center_y > (3 / 2 * frame_height) / 2:
            vertical_position = "Bottom"
        else:
            vertical_position = "Middle"

        position_text = f"{horizontal_position}, {vertical_position}"
        return position_text

    def update_tracks(self, object_id, current_timestamp, world_coordinates, cls, bound_box, region):
        # Initialize the tracking for a new object
        if object_id not in self.object_tracks:
            self.object_tracks[object_id] = {
                'positions': [],  # (tuple) (time, space)
                'speed': [],
                'speeds': [],
                'average_speeds': [],
                'jerks': [],
                'accelerations': [],
                'predicted_positions': [],
                'class': [],
                'direction': [],
                'roadside': [],
                'region': [],
                'world_coordinates': [],
                'current_timestamp': []
            }
        self.object_tracks[object_id]['region'].append(region)
        self.object_tracks[object_id]['world_coordinates'].append(world_coordinates)
        self.object_tracks[object_id]['current_timestamp'].append(current_timestamp)
        # Update the positions
        self.object_tracks[object_id]['positions'].append((current_timestamp, world_coordinates))
        # Update the class
        self.object_tracks[object_id]['class'].append(cls)

        # Initialize average_speed and acceleration for return statement
        average_speed = 0
        acceleration = 0
        speed = 0
        # Calculate speed if we have at least 2 positions
        if len(self.object_tracks[object_id]['positions']) > 1:
            # Use the last few positions to calculate speed
            recent_positions = self.object_tracks[object_id]['positions'][-3:]
            total_distance = 0
            total_time = 0
            for i in range(len(recent_positions) - 1):
                distance = np.linalg.norm(np.array(recent_positions[i + 1][1]) - np.array(recent_positions[i][1]))
                time_elapsed = (recent_positions[i + 1][0] - recent_positions[i][0]).total_seconds()
                if time_elapsed == 0:
                    if distance != 0:
                        print('how')
                    speed = 0
                else:
                    speed = 3.6 * distance / time_elapsed
                if speed <= self._max_speed:
                    total_distance += distance
                    total_time += time_elapsed
            average_speed = 3.6 * total_distance / total_time if total_time > 0 else 0
            self.object_tracks[object_id]['speeds'].append(average_speed)
            self.object_tracks[object_id]['speed'].append(speed)

            # Calculate acceleration if we have at least 2 speeds
            if len(self.object_tracks[object_id]['speeds']) > 1:
                recent_speeds = self.object_tracks[object_id]['speeds'][-2:]
                speed_change = recent_speeds[1] - recent_speeds[0]
                # Assuming uniform time intervals between speed measurements
                if (recent_positions[2][0] - recent_positions[1][0]).total_seconds() == 0:
                    acceleration = 0
                    print("speed change check: " + str(speed_change))
                else:
                    acceleration = speed_change / (recent_positions[2][0] - recent_positions[1][0]).total_seconds()
                self.object_tracks[object_id]['accelerations'].append(acceleration)

        # Predict trajectory using a suitable method
        predicted_trajectory = self.predict_trajectory(self.object_tracks[object_id]['positions'], bound_box=bound_box,
                                                       object_id=object_id)
        angle_proj = self.predict_trajectory(self.object_tracks[object_id]['positions'], angle=True,
                                             bound_box=bound_box, object_id=object_id)
        if predicted_trajectory is not None:
            self.object_tracks[object_id]['predicted_positions'].append(predicted_trajectory)
            if angle_proj is not None:
                self.object_tracks[object_id]['direction'].append(angle_proj)
        return average_speed, acceleration, predicted_trajectory

    def calculate_time_to_collision(self, leading_vehicle_pos, trailing_vehicle_pos, leading_vehicle_speed,
                                    trailing_vehicle_speed):
        # Calculate the distance between the leading and trailing vehicles
        distance = math.dist(trailing_vehicle_pos,leading_vehicle_pos)

        # Calculate the relative velocity of the trailing vehicle
        relative_velocity = trailing_vehicle_speed - leading_vehicle_speed

        # Calculate the estimated time to collision
        if relative_velocity != 0:
            time_to_collision = distance / relative_velocity
        else:
            time_to_collision = float('inf')

        return time_to_collision

    def get_box(self, detection):
        try:
            boxes = detection.xywh.cpu().numpy()[0]
        except AttributeError:
            return False
        return boxes
    def get_x_y_position(self, detection):
        xyxy = detection.xyxy.cpu().numpy().flatten()
        x, y = middle_x_y(xyxy, shift=1)
        return x, y, xyxy

    def filter_contained_boxes(self, boxes):
        filtered_boxes = []
        for i, box in enumerate(boxes):
            is_contained = False
            for j, other_box in enumerate(boxes):
                if i != j and self.is_box_contained(box, other_box):
                    is_contained = True
                    break
            if not is_contained:
                filtered_boxes.append(box)
        return filtered_boxes

    def is_box_contained(self, box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        if x1 >= x2 and y1 >= y2 and x1 + w1 <= x2 + w2 and y1 + h1 <= y2 + h2:
            return True
        return False



    def rear_end_detections(self, current_id, detections_id):
        curr_id = current_id.id
        time_to_collisions = []
        if curr_id.cpu().numpy()[0] not in self.object_tracks:
            return []
        for detection in detections_id:


            other_id = detection.id

            if other_id.cpu().numpy()[0] not in self.object_tracks:
                continue

            # Compare current_id with other_id
            if curr_id < other_id:
                box_curr = self.get_box(current_id)
                if len(box_curr) <= 1:
                    continue
                box_other = self.get_box(detection)
                if len(box_other) <= 1:
                    continue
                if self.is_box_contained(box_curr, box_other):
                    continue
                trailing_vehicle_pos = self.get_x_y_position(current_id)[:2]
                leading_vehicle_pos = self.get_x_y_position(detection)[:2]
                # Perform calculations for time to collision using leading_vehicle_pos,
                # trailing_vehicle_pos, leading_vehicle_speed, trailing_vehicle_speed
                # ...
                if self.get_roadside(other_id.cpu().numpy()[0]) != self.get_roadside(curr_id.cpu().numpy()[0]):
                    continue

                # Return the time to collision if it meets certain criteria
                if self.rear_end_collision_detected(leading_vehicle_pos, trailing_vehicle_pos):

                    time_to_collision = self.calculate_time_to_collision(leading_vehicle_pos, trailing_vehicle_pos,
                                                                         self.get_speed(other_id.cpu().numpy()[0])/3.6,
                                                                         self.get_speed(curr_id.cpu().numpy()[0])/3.6)
                    time_to_collisions.append(time_to_collision)
                else:
                    time_to_collisions.append(1000)
            # If no collision is detected or no suitable time to collision is found, return None
        return time_to_collisions


    def rear_end_collision_detected(self, leading_vehicle_pos, trailing_vehicle_pos):
        # Set a threshold distance for rear-end collision detection
        rear_end_threshold = 5  # Adjust as needed

        # Determine if a rear-end collision is detected
        #TODO might nead to get x y linalg coordinates hear
        if math.dist(trailing_vehicle_pos ,leading_vehicle_pos) <= rear_end_threshold:
            return True
        else:
            return False





    def predict_trajectory(self, positions, angle=False, bound_box=None, object_id=None):
        # Implement your trajectory prediction algorithm here
        angles = []
        if len(positions) >= 3:
            for i in range(1, len(positions) - 1):
                position_change = np.array(positions[i][1]) - np.array(positions[i - 1][1])
                predicted_position = np.array(positions[i][1]) + position_change

                angle_of_projection = np.array(np.arctan2(position_change[1], position_change[0]) * 180 / np.pi)
                angles.append(angle_of_projection)
            actual_position = np.array(positions[-1][1])
            if angle:

                if self.get_roadside(object_id):  # forward
                    if bound_box is not None:
                        self.forward_points.append(bound_box)

                elif not self.get_roadside(object_id):  # backward

                    if bound_box is not None:
                        self.backward_points.append(bound_box)

                    #print('I assume none')
                    # self.backward_points.append(positions[-1][1])
                self.angles.append(angle_of_projection)
                return angle_of_projection.tolist()
            return predicted_position.tolist()
        else:
            if angle:
                return None
            return positions[-1][1] if positions else None

    def get_speed(self, object_id):
        if object_id not in self.object_tracks:
            return 0

        elif len(self.object_tracks[object_id]['speed']) == 0:
            return 0
        else:
            return self.object_tracks[object_id]['speed'][-1]

    def get_average_speed(self, object_id):
        if len(self.object_tracks[object_id]['speeds']) == 0:
            return 0
        else:
            return self.object_tracks[object_id]['speeds'][-1]

    def get_acceleration(self, object_id):

        if len(self.object_tracks[object_id]['accelerations']) == 0:
            return 0
        else:
            return self.object_tracks[object_id]['accelerations'][-1]

    def get_direction_value(self, object_id):
        #TODO does this work
        if len(self.object_tracks[object_id]['direction']) == 0:
            return 0
        else:
            return self.object_tracks[object_id]['direction'][-1]



class VideoFootageProfiler:
    def __init__(self):
        self.detections = []
        self.TTC = []
        self.motorcycle_detections = []
        self.object_types = set()

    def process_frame(self, timestamp, detections):
        # Add all detections with their timestamp to the profiler's detection list
        for detection in detections:
            detection['timestamp'] = timestamp
            self.add_detection_type(detection['class'])
            self.detections.append(detection)

            if detection['class'] == 'motorcycle':
                self.motorcycle_detections.append((timestamp, detection))


    def record_TTC(self, TTC, frame_number):
        self.TTC.append([TTC, frame_number])

    def save_TTC(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(
                ['TTC', 'FRAME'])
            # Write the detection data
            for detection in self.TTC:
                writer.writerow([detection[0], detection[1]])


    def add_detection_type(self, object_type):
        self.object_types.add(object_type)

    def save_detections_to_csv(self, file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write the header
            writer.writerow(
                ['timestamp', 'bounding_box', 'class', 'speed', 'id', 'lane_violation', 'jerk_violation', 'swerving_violation',
                 'roadside', 'world_x', 'world_y', 'time_increase', 'speed_current'])
            # Write the detection data
            for detection in self.detections:
                writer.writerow([detection['timestamp'], detection['bounding_box'], detection['class'], detection.get('speed', 'N/A'),
                                 detection['object_id'], detection['lane_violation'], detection['jerk_violation'],
                                 detection['swerving_violation'], detection['roadside'], detection['world_x'],
                                 detection['world_y'], detection['time_increase'],  detection['speed_current']])

    def get_detections_by_type(self, object_type):
        # Filter detections by object type
        return [d for d in self.detections if d['class'] == object_type]

    def get_total_detections(self):
        return len(self.detections)

    def get_all_detections_info(self):
        # Return all detections information
        return self.detections

    def get_motorcycle_detections(self):
        # Return all motorcycle detections
        return self.motorcycle_detections

    def get_motorcycle_frequency_stats(self):
        # Calculate frequency of motorcycle detections per hour
        frequency_stats = defaultdict(int)
        for timestamp, _ in self.motorcycle_detections:
            # Assuming timestamp is a datetime object, adjust accordingly if it's not
            try:
                hour = timestamp.hour
                frequency_stats[hour] += 1
            except Exception as e:
                print(e)
        return frequency_stats

    def get_summary_statistics(self):
        summary_stats = {}
        for object_type in self.object_types:
            object_detections = self.get_detections_by_type(object_type)
            total_detections = len(object_detections)
            total_speed = 0
            valid_speed_count = 0

            for detection in object_detections:
                # Handle the case where speed is None
                if detection['speed'] is not None:
                    total_speed += detection['speed']
                    valid_speed_count += 1

            # Calculate average speed; handle division by zero if no valid speeds are present
            average_speed = (total_speed / valid_speed_count) if valid_speed_count else 0

            # Store the summary statistics for the object type
            summary_stats[object_type] = {
                'total_detections': total_detections,
                'average_speed': average_speed
            }

        return summary_stats
