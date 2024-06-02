import faulthandler
import sys
import time

from gapcreek.camera_utils import *
import os
import pandas as pd
import argparse
import cv2
import glob
import subprocess
import re
from ultralytics import YOLO
import numpy as np
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta
import pytesseract
import torch


pytesseract.pytesseract.tesseract_cmd = r'C:\Users\n9471103\AppData\Local\Programs\Tesseract-OCR\tesseract.exe'



def list_directories(path):
    return [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]


def list_files(path, max_files=1000, min_size=15000):
    files = []
    count = 0

    with os.scandir(path) as entries:
        for entry in entries:
            if entry.is_file():
                file_path = os.path.join(path, entry.name)

                modification_time = entry.stat().st_mtime
                modified_datetime = datetime.fromtimestamp(modification_time)

                # Check if the modification time is within the specified time range (between 10 PM and 4 AM)
                if modified_datetime.year != 2024:
                    if modified_datetime.month != 12:
                        continue

                if os.path.getsize(file_path) >= min_size:

                    files.append(entry.name)
                    count += 1


                if count >= max_files:
                    break

    return files[:max_files] if max_files is not None else files


def select_entry(entries, index, entry_type):
    try:
        return entries[index]
    except IndexError:
        print(f"Invalid {entry_type} index. Please enter a number between 1 and {len(entries)}.")
        return None


def take_snapshot(video_path: str, snapshot_filename: str, script_dir: str):
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if ret:
        # If a frame was read successfully, write the frame to an image file
        snapshot_path = os.path.join(script_dir, snapshot_filename)
        cv2.imwrite(snapshot_path, frame)
    # Release the video capture object
    cap.release()


def is_same_object(prev_box, curr_box, distance_threshold=1, verbose=False, frame=None):
    """
    Check if the current bounding box is the same object as the previous one.
    This function calculates the Euclidean distance between the centers of two bounding boxes.
    If the distance is less than a threshold, it considers the boxes as the same object.

    Parameters:
    prev_box (tuple): The bounding box of the previous frame. It should be a tuple of (x, y, w, h).
    curr_box (tuple): The bounding box of the current frame. It should be a tuple of (x, y, w, h).
    distance_threshold (int): The threshold for considering two boxes as the same object.

    Returns:
    bool: True if the boxes are the same object, False otherwise.o
    """

    # teting purposes
    if frame is not None:
        cv2.rectangle(frame, (int(prev_box[0]), int(prev_box[1])), (int(prev_box[2]), int(prev_box[3])), (255, 0, 0), 2)
        cv2.rectangle(frame, (int(curr_box[0]), int(curr_box[1])), (int(curr_box[2]), int(curr_box[3])), (255, 0, 255),
                      2)

    # Calculate the center points of the boxes
    prev_center = (prev_box[0] + prev_box[2] / 2, prev_box[1] + prev_box[3] / 2)
    curr_center = (curr_box[0] + curr_box[2] / 2, curr_box[1] + curr_box[3] / 2)

    # Calculate the Euclidean distance between the center points
    distance = np.sqrt((curr_center[0] - prev_center[0]) ** 2 + (curr_center[1] - prev_center[1]) ** 2)

    # Check if the distance is less than the threshold
    if verbose:
        print(f'distance is {distance}')
    if distance < distance_threshold:
        return False
    else:
        return True


def write_world_coords_to_image(image, world_coords, xyxy, text_id, color=(255, 255, 255), font_scale=0.5, thickness=2):
    """
    Writes the world coordinates on the image at the position of the detection.

    Parameters:
    - image: The image on which to write the text.
    - world_coords: The world coordinates to write.
    - xyxy: The bounding box coordinates or point in the image (x, y).
    - color: The color of the text (B, G, R).
    - font_scale: The scale of the font.
    - thickness: The thickness of the font.

    Returns:
    - image: The image with the text written on it.
    """
    # Format the world coordinates as strings to a certain precision
    text = "ID: {:.2f}, X: {:.2f}, Y: {:.2f}, Z: {:.2f}".format(text_id, *world_coords)

    # Determine the position for annotation (bottom-left corner of the bounding box)
    text_position = (int(xyxy[0]), int(xyxy[3]))

    # Write the text on the image
    cv2.putText(image, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

    return image


def convert_bbox_to_z(bbox):
    """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = bbox[0] + w / 2.
    y = bbox[1] + h / 2.
    s = w * h  # scale is just area
    r = w / float(h)
    return np.array([x, y, s, r]).reshape((4, 1))


def extract_timestamp(frame, helper_time=None) -> datetime:
    # Define the region of interest (top-left corner) where the timestamp is located
    roi = frame[10:60, 10:200]  # Adjust the coordinates based on your video's timestamp position
    #cv2.imshow('ff',roi)

    # Convert the region of interest to grayscale
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Apply thresholding to extract the digits of the timestamp
    _, thresholded_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)

    # Perform OCR (Optical Character Recognition) to extract the timestamp text
    timestamp_text = pytesseract.image_to_string(thresholded_roi)
    try:
        timestamp_text_datetime = check_timestamp_text(timestamp_text.strip(), helper_time)
        return timestamp_text_datetime
    except Exception as e:
        return helper_time


def check_timestamp_text(a, real):
    formatted_string = real.strftime("%m-%d-%Y %a %H:%M:%S")
    t_el = 14
    actual_data = formatted_string[:t_el]
    if formatted_string != a[:t_el]:
        a = actual_data + a[t_el:]
    try:
        parsed_datetime = datetime.strptime(a, "%m-%d-%Y %a %H:%M:%S")
        if parsed_datetime.strftime(formatted_string) != a:
            mismatch_details = []
            real_components = real.timetuple()
            parsed_components = parsed_datetime.timetuple()
            if real_components.tm_mon != parsed_components.tm_mon:
                mismatch_details.append("Month")
            if real_components.tm_mday != parsed_components.tm_mday:
                mismatch_details.append("Day")
            if real_components.tm_hour != parsed_components.tm_hour:
                mismatch_details.append("Hour")
            if real_components.tm_min != parsed_components.tm_min:
                mismatch_details.append("Minute")
            if real_components.tm_sec != parsed_components.tm_sec:
                mismatch_details.append("Second")

            mismatch_details_str = ", ".join(mismatch_details)
            print(f"The string '{a}' does not match the format '{formatted_string}'.")
            print(f"The following components have mismatches: {mismatch_details_str}.")
            print("Parsed datetime:", parsed_datetime)
            return parsed_datetime
        else:
            print(f"The string '{a}' matches the format '{formatted_string}'.")
            print("Parsed datetime:", parsed_datetime)
    except ValueError:
        print(f"The string '{a}' does not match the format '{formatted_string}'.")
        return formatted_string


def time_repair_and_check(a, helper_time):
    if a.year != helper_time.year:
        print('year mismatch')
        a = a.replace(year=helper_time.year)
    if a.month != helper_time.month:
        print('month mismatch')
        a = a.replace(month=helper_time.month)
    if a.day != helper_time.day:
        print('day mismatch')
        a = a.replace(day=helper_time.day)

    if a.hour != helper_time.hour:
        print('hour mismatch')
    if a.minute != helper_time.minute:
        print('minute mismatch')
    if a.second != helper_time.second:
        print('second mismatch')

    difference_in_time = (a - helper_time)
    error = difference_in_time.total_seconds()
    return error, a


def extract_datetime(file_name):
    # Split by underscore and take the second and third parts for date and time respectively
    parts = file_name.split('_')
    date_time_str = f"{parts[1]}_{parts[2]}"
    # Create datetime object
    date_time_obj = datetime.strptime(date_time_str, '%Y-%m-%d_%H-%M-%S')
    return date_time_obj


def clean_filename(filename):
    # Remove leading period and any characters after .mkv
    return re.sub(r"^\.*([\w_-]+\.mkv).*", r"\1", filename)


def run_ffmpeg_command(command):
    try:
        command1 = ' '.join(map(str, command))
        # subprocess.run(command1, check=True)
        return_code = subprocess.call(command1, shell=True)
        if return_code:
            print("FFmpeg command ran successfully. Returing..")
            return True
        else:
            print('Failed.. Try Next.')
    except subprocess.CalledProcessError:
        print("FFmpeg command failed.. Try next")
        return False


def reencode_video(input_file, output_file, desired_width, desired_height, target_framerate):
    # Initial re-encode attempt with error detection
    base_command = [
        'ffmpeg',
        '', '-n',
        '-i', input_file,
        '-vf', f'scale={desired_width}:{desired_height}',
        '-vsync cfr -r', f'{target_framerate}',
        '-c:v', 'libx264',
        '-preset', 'slow',
        '-crf', '22',
        '-c:a', 'aac',
        '-b:a', '128k'
    ]
    command = base_command + [output_file]
    if run_ffmpeg_command(command):
        return

    # If the first attempt fails, try re-encoding with a different codec
    command[9] = 'libx265'
    command[11] = '28'  # Adjust CRF for x265
    if run_ffmpeg_command(command):
        return

    # If that still fails, try copying the streams without re-encoding
    print('failed. try copying the streams without re-encoding')
    command[7:12] = ['-c:v', 'copy', '-c:a', 'copy', output_file]
    if run_ffmpeg_command(command):
        return

    # If copying the streams fails, attempt to re-encode video only
    print('Failed. Attempt to re-encode video only.')
    command[7:12] = ['-c:v', 'libx264', '-crf', '22', '-c:a', 'copy', output_file]
    if run_ffmpeg_command(command):
        return

    # If everything fails, the file might be unrecoverable
    print("All attempts to re-encode the video have failed. The file may be corrupt.")
    print('Tying one more time.')
    command = f'ffmpeg -y -i {input_file} -vf "scale={desired_width}:{desired_height}" -vsync cfr -r 30 -c:v libx264 -preset fast -crf 22 -c:a copy {output_file}'
    subprocess.call(command, shell=True)


def main(args):


    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(device)
    if device == 'cpu':
        sys.exit(0)



    logging.getLogger().setLevel(logging.ERROR)
    model = YOLO("model/yolov8x.pt").to(device)
    print(1)

    if args.TEST_HPC:
        return sys.exit(0)



    FAILED: int = 0
    SPEED_THRESHOLD_MIN: int = 0.5
    SPEED_THRESHOLD_MAX: int = 150
    SPEED_THRESHOLD_BIKE: int = 50
    dir_index = args.dir - 1  # Convert to 0-based index
    file_index = args.file - 1  # Convert to 0-based index
    file_path = None
    date_time_object = current_timestamp = None
    if os.name == 'nt':
        HPC_CODE = 0
        base_path = '//hpc-fs/work/SAIVT/gap_creek_road/raw/'
        base_path_alt = '//hpc-fs/work/SAIVT/gap_creek_road/raw/'

    else:
        HPC_CODE = 1
        print(os.name)
        base_path = '//mnt/hpccs01/work/SAIVT/gap_creek_road/raw/'
        base_path_alt = '//mnt/hpccs01/work/SAIVT/gap_creek_road/raw/'

    # Try to get the list of directories from the base path
    directories = list_directories(base_path)
    # If the list is empty, probably due to an exception, try the alternative path
    if not directories:
        directories = list_directories(base_path_alt)

    # Select a directory
    selected_dir = select_entry(directories, dir_index, 'directory')
    print(f"the selected Directory is: {selected_dir}")
    if selected_dir == 'DLBQL':  # TODO assume drawn line
        print('we are looking at camera 1A')
        path_to_create = None
        print('to do check if i can get actual camer matrix, or something close')

        print('camera calibrated....')
        image_test = cv2.imread('gapcreek/1A_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/1Aclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/1a.xml', image_lanes)

        entry_point = ['Top', "Bottom", 'Left', 'Right']

    elif selected_dir == 'YMDPJ':
        print('we are looking at camera 1B')
        print('to do check if i can get actual camer matrix, or something close')
        path_to_create = None
        image_test = cv2.imread('gapcreek/1B_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/1Bclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/1b.xml', image_lanes)
        print('camera calibrated....')
        entry_point = ['Top', "Bottom", 'Left', 'Right']  # defined order for lanes

    elif selected_dir == 'DINUI':
        print(f'we are looking at camera {selected_dir}')
        # camera = CameraView('gapcreek/3a.xml')
        if os.path.exists('gapcreek/3A_input_test.jpg'):
            print('already exist')
            path_to_create = None
        else:
            path_to_create = 'gapcreek/3A_input_test.jpg'

        entry_point = ['Top', "Bottom", 'Left', 'Right']

        image_test = cv2.imread('gapcreek/3A_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/3Aclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/3a.xml', image_lanes)
    elif selected_dir == 'MPKVA':
        print('we are looking at camera...')

        if os.path.exists('gapcreek/3B_input_test.jpg'):

            path_to_create = None
        else:
            path_to_create = 'gapcreek/3B_input_test.jpg'
        entry_point = ['Center', "Bottom", 'Left', 'Right']
        print('check entry point')
        image_test = cv2.imread('gapcreek/3B_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/3Bclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/3b.xml', image_lanes)
    elif selected_dir == 'QLFAZ':
        print('we are looking at camera')
        camera = CameraView('gapcreek/2b.xml')
        entry_point = ['Top', "Bottom", 'Ignore', 'Ignore']
        print('check entry point')
        if os.path.exists('gapcreek/2B_input_test.jpg'):

            path_to_create = None
        else:
            path_to_create = 'gapcreek/2B_input_test.jpg'
        image_test = cv2.imread('gapcreek/2B_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/2Bclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/2b.xml', image_lanes)
    elif selected_dir == 'TPCAY':
        print('we are looking at camera')
        camera = CameraView('gapcreek/2a.xml')
        entry_point = ['Top', "Bottom", 'Left', 'Right']
        print('check entry point')
        if os.path.exists('gapcreek/2A_input_test.jpg'):

            path_to_create = None
        else:
            path_to_create = 'gapcreek/2A_input_test.jpg'
        image_test = cv2.imread('gapcreek/2A_input_test.jpg')
        image_lanes = pd.read_csv('gapcreek/2Aclicked_points.csv')
        image_lanes = list(zip(image_lanes['x'], image_lanes['y']))
        camera = CameraView('gapcreek/2a.xml', image_lanes)
    else:
        raise Exception('not yet implemented')
    straight_line_divide, left_divide, right_divide = camera.get_straight_divider()
    print('lets select the directory')
    if selected_dir:
        # Construct the path to the selected directory
        dir_path = os.path.join(base_path if directories else base_path_alt, selected_dir)
        # List files in the selected directory
        files = list_files(dir_path, max_files=args.MFC)
        files = sorted(files, key=extract_datetime, reverse=True)
        # Select a file
        selected_file = select_entry(files, file_index, 'file')
        selected_file = clean_filename(selected_file)

        # Construct the path to the selected file

        file_path = os.path.join(dir_path, selected_file)

        print(f"Selected file: {selected_file}")
        print(f"Full path to the selected file: {file_path}")

        ##NOW TRY AND GET THE TIME
        date_time_str = selected_file[2:25]  # '2024-01-29_09-52-22'

        # Replace the underscores with colons in the time part to match the format
        date_time_str = date_time_str.replace('_', ' ')  # '2024-01-29 09:52:22'

        # Now parse the string into a datetime object
        date_time_obj = current_timestamp = datetime.strptime(date_time_str, '%Y-%m-%d %H-%M-%S %f')

    output_dir = f"results/output_dir_{args.dir}_line_{args.line}"
    if not os.path.exists(output_dir):
        print('creating directory...')
        os.makedirs(output_dir)

        output_file_path = os.path.join(output_dir, date_time_str + '.txt')

        # Save the output to the text file
        with open(output_file_path, 'w') as output_file:
            output_file.write(file_path)

        print('Output saved to:', output_file_path)
    elif args.CV == False:
        output_file_path = os.path.join(output_dir, date_time_str + '.txt')

        # Save the output to the text file
        with open(output_file_path, 'w') as output_file:
            output_file.write(file_path)

        print('Output saved to:', output_file_path)

    else:
        try:
            file_path = glob.glob(os.path.join(output_dir, '*.mp4'))[0]
            date_time_path = glob.glob(os.path.join(output_dir, '*.txt'))[0]
            date_time_str = os.path.basename(date_time_path)[:-4]
            date_time_obj = current_timestamp = datetime.strptime(date_time_str, '%Y-%m-%d %H-%M-%S %f')
            print(1)
            args.CV = False
            args.OCV = False
        except FileNotFoundError:
            print('continue to proceed')

    video_path = file_path

    if not args.CV:
        print('Do not convert video')
        print('the video path is {}'.format(video_path))
        cap = cv2.VideoCapture(video_path)
    else:
        print('Attempting to convert video...')
        output_file = f"results/output_dir_{vars(args)['dir']}_line_{vars(args)['line']}/output.mp4"
        target_frame_rate = 6  # Set the desired frame rate here
        # ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
        # Run FFmpeg command to convert VFR to CFR

        desired_width = image_test.shape[1]  # replace with your desired width
        desired_height = image_test.shape[0]  # replace with your desired height, or use -1 to maintain aspect ratio

        # Execute the command
        if args.CV:

            if not os.path.exists(output_file):
                print('this shou; try to rencode')
                reencode_video(video_path, output_file, desired_width, desired_height, target_frame_rate)
        print('grab the capture')
        cap = cv2.VideoCapture(output_file)

    if args.OCV:
        sys.exit(0)

    # end main
    speed = None

    # Initialize the dictionary to store the previous world coordinates and timestamps for each object ID
    prev_world_coords = {}
    prev_timestamps = {}
    prev_bounding_box = {}

    # helpers for visualisation
    # Initialize a simple counter for assigning IDs to each new object
    object_id_counter = 0

    # Get the resolution of the video source
    frame_width = int(cap.get(3))

    print(f" The frame width is: {frame_width} ")
    frame_height = int(cap.get(4))

    print(f" The frame height is: {frame_height} ")
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    if source_fps == 1000:
        print('source is variable, not likley, grab the capture')
        source_fps = 6
        print('assume convert convert options..')
        # raise Exception('variable framerate. code will not work.')
    print(f"fps: {source_fps}")
    total_frames = cap.get(7)
    print(f"total frames: {total_frames}")

    frame_count = 0
    previous_frame_time = 0
    frame_number = 0

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

    # Assuming a 640x480 webcam resolution, adjust if necessary
    output_video_path = os.path.join(output_dir, 'output.avi')
    #output_video_path_jerk = os.path.join(output_dir, 'output_jerk.avi')
    #output_video_path_swerve = os.path.join(output_dir, 'output_swerve.avi')
    out = cv2.VideoWriter(output_video_path, fourcc, source_fps, (frame_width, frame_height))
    #out_jerk = cv2.VideoWriter(output_video_path_jerk, fourcc, source_fps, (frame_width, frame_height))
    #out_swerve = cv2.VideoWriter(output_video_path_swerve, fourcc, source_fps, (frame_width, frame_height))

    # Set the duration of the footage (in seconds)
    duration = timedelta(days=1, seconds=1000)  # Duration in seconds
    # Calculate the timestamp when recording should end
    record_end = None
    profiler = VideoFootageProfiler()
    lane_violation = False

    swerving_vio, jerk_safe_vio = False, False
    TP = TrajectoryProfile(0, frame_width, frame_height, straight_line_divide, left_divide, right_divide, entry_point)

    if args.HM:
        from ultralytics.solutions import heatmap
        heatmap_obj = heatmap.Heatmap()
        heatmap_obj.set_args(colormap=cv2.COLORMAP_PARULA,
                             imw=frame_width,
                             imh=frame_height,
                             shape="circle", decay_factor=.35)

    frame_capture = 0
    FROZEN = 0
    max_attempts = 3
    print(f'current time stamp is {date_time_obj}')
    error_counter = 0
    while cap.isOpened():

        ret, frame = cap.read()
        if ret:
            frame_count += 1
            if path_to_create is not None:
                cv2.imwrite(path_to_create, frame)
                print('does this exit the code')
                return
            if frame_count <= 2:
                camera.poly_point(frame, show=False)
            #preprocess_image_predict(frame)

            if frame_count >= total_frames:
                print('total frames reached')
                break
            if args.ET:
                current_timestamp = current_timestamp + timedelta(seconds=1 / source_fps)
            else:
                if args.FR_BASED:
                    current_timestamp = date_time_obj + timedelta(
                        seconds=cap.get(cv2.CAP_PROP_POS_FRAMES) / source_fps)  # TODO get video time
                else:
                    try:
                        timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
                        timestamp_sec = timestamp_ms / 1000
                        current_timestamp = date_time_obj + timedelta(
                        seconds=timestamp_sec)
                    except Exception as e:
                        print(e)
                        break


                # extracted_time_stamp_difference = extract_timestamp(frame, current_timestamp)
                # print(extracted_time_stamp_difference)
            # print(f" {cap.get(cv2.CAP_PROP_POS_MSEC)/1000.0} seconds")

            # Get the current frame number from the video capture
            captured_frame_no = cap.get(cv2.CAP_PROP_POS_FRAMES)

            # Calculate the expected frame number based on the elapsed time and source FPS
            # expected_frame_no = source_fps * (date_time_obj + current_timestamp).total_seconds()
            offset = cap.get(cv2.CAP_PROP_POS_MSEC)

           # print("CAP_PROP_POS_MSEC: ", offset)
            # Check if the expected frame number is less than the captured frame number
            # Get the current frame timestamp in seconds
            if args.FR_BASED:
                current_frame_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / source_fps
            else:

                current_frame_time = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0


            # Calculate the expected time difference between frames
            expected_delta = 1.0 / source_fps

            previous_frame_time = current_frame_time
            previous_offset = offset
            frame_number += 1
            wanted_classes = [1, 2, 3, 5, 7]
            # Perform inference
            results = model.track(frame, persist=True, verbose=False, classes=wanted_classes, iou = 0.7,
                                  agnostic_nms = True,nms = True, retina_masks = True,
                                  device=device)

            if args.HM:
                frame = heatmap_obj.generate_heatmap(frame, results)

            # Extract results
            detections = results[0].boxes  # Assuming 'boxes' is a list of detected boxes
            profiler_detections = []  # List
            if detections.id is not None:
                #detections = detections[camera.is_inside(x, y)] #TODO implement

                for detection in detections:
                    # Extract individual detection
                    #TP.get_x_y_position(detection)
                   # xyxy = detection.xyxy.cpu().numpy().flatten()
                    #x, y = middle_x_y(xyxy, shift=1)
                    x, y, xyxy = TP.get_x_y_position(detection)
                    if not camera.is_inside(x, y):  # TODO i want to see if this profiles

                        if camera.is_outside_x_is_inside_y(x, y):
                            if args.TEST_OFF:
                                if args.SV:
                                    filename = "potentialviolation_{}.png".format(frame_count)
                                    cv2.imwrite(filename, frame)
                                    print(f"Snapshot saved as {filename}")
                        #continue
                    cv2.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (255, 0, 0), 2)
                    conf = detection.conf
                    cls = detection.cls

                    if record_end is None:
                        record_end = current_timestamp + duration
                        print(f" recording will be finished at  {record_end}")

                    # Assign an ID to each detection (for simplicity, we will assign aqq new ID to each detection)
                    object_id = detection.id.cpu().numpy()[0]
                    if object_id > object_id_counter:
                        object_id_counter = object_id

                    region = camera.split_frame((x, y), divider_points=None)
                    # world_coordinates = image_to_world(xyxy, camera_matrix, dist_coeffs, rvec, tvec)

                    # Project the image coordinates to world coordinates
                    world_x, world_y = camera.project(x, y)
                    #print('to do check below')
                    camera.projection_middle(xyxy) #TODO add in this code
                    print('hold')
                    # movement_cam, rr = camera.movement_vector(xyxy_convert, x, y)
                    world_coordinates = np.array((world_x, world_y, 1))

                    if object_id in prev_world_coords:

                        TP.update_tracks(object_id, current_timestamp, (world_x, world_y), model.names[int(cls)], xyxy,
                                         region=region)
                        try:
                            swerving_vio, jerk_vio = TP.detect_unsafe_driving(object_id, swerving_threshold=vars(args).get(
                            'jerk_threshold', 50))
                        except Exception as e:
                            print(e)
                            swerving_vio, jerk_vio = 0, 0
                        if swerving_vio:
                            if args.SV:
                                save_violation_snapshot(frame, frame_count, output_dir, 'swerving',
                                                        vars(args).get("Verbose", False)
                                                        )
                                frame_capture = 0
                        if jerk_vio:
                            if args.SV:
                                save_violation_snapshot(frame, frame_count, output_dir, 'jerking',
                                                        vars(args).get("Verbose", False))
                                frame_capture = 0

                        if not TP.is_predicted_close_to_actual(object_id, threshold=5):
                            if args.SV:
                                save_violation_snapshot(frame, frame_count, output_dir, 'not_close_enough',
                                                        vars(args).get("Verbose", False))

                            lane_violation = True
                            trajectory_violation = True
                        else:
                            lane_violation = False
                            trajectory_violation = False

                        frame_capture += 1
                        if frame_number > 100:
                            if frame_number % 1000000000000 == 1:
                                dontShow = 0
                            else:
                                dontShow = 1

                            frame = TP._draw_line(frame, dontShow=dontShow)

                            # TP.get_roadside(object_id)
                            if not TP.is_in_trajectory(xyxy, TP.get_roadside(object_id)):
                                if args.SV:
                                    frame_capture = 0
                                    save_violation_snapshot(frame, frame_count, output_dir, 'not_in_trajectory')
                                lane_violation = True
                            else:
                                lane_violation = False
                        else:
                            lane_violation = False

                        if frame_capture <= args.FC_AFTER:
                            out.write(frame)

                        if frame_number % 1000 == 1:
                            print(f"{current_timestamp}")

                        # Calculate the Euclidean distance between the previous and current world coordinates
                        distance = np.linalg.norm(world_coordinates - prev_world_coords[object_id])
                        # Calculate the time elapsed between the previous and current frame
                        time_elapsed = current_timestamp - prev_timestamps[object_id]

                        # Calculate the speed (distance/time)
                        speed = 3.6 * distance / time_elapsed.total_seconds() if time_elapsed > timedelta(seconds=0) else 0
                        # todo grab speed averages
                        if speed < SPEED_THRESHOLD_MIN:
                            print(f'current time is {current_timestamp}')
                            print('duplicate')
                            FROZEN = 1
                            continue
                        else:
                            FROZEN = 0

                        if speed < SPEED_THRESHOLD_MIN or speed > SPEED_THRESHOLD_MAX:
                            continue

                        # Annotate the frame with the speed
                        class_name = model.names[int(cls)]
                        if class_name == 'bicycle':
                            if speed > SPEED_THRESHOLD_BIKE:
                                continue

                        if args.DS:
                            speed_text = f"ID {object_id}: Speed: {speed:.2f} km/hr"
                            cv2.putText(frame, speed_text, (int(xyxy[0]), int(xyxy[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                                        (0, 255, 0), 2)

                        if class_name in (
                                'motorbike',
                                'motorcycle'):  # Check if a motorbike is detected with high confidence
                            # filename = f"snapshot_{frame_count}.png"
                            filename_short = "snapshot_bike_{}_{}_{}_{}.png".format(current_timestamp.hour,
                                                                                    current_timestamp.minute,
                                                                                    current_timestamp.second, frame_count)
                            filename = os.path.join(output_dir, filename_short)

                            cv2.imwrite(filename, frame)
                            print(f"Snapshot saved as {filename_short}")

                        profiler_detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bounding_box': xyxy,
                            'object_id': object_id,
                            'speed_current': speed,
                            'distance_diff': distance,
                            'time_diff': time_elapsed,
                            'lane_violation': lane_violation,
                            'jerk_violation': jerk_vio,
                            'swerving_violation': swerving_vio,
                            'roadside': TP.get_region(object_id),
                            'speed': TP.get_speed(object_id),
                            'acceleration': TP.get_acceleration(object_id),
                            'jerk_value': TP.get_jerk_value(object_id),
                            'directions': TP.get_direction_value(object_id),
                            'world_x': world_x,
                            'world_y': world_y,
                            
                            'time_increase': 1/cap.get(cv2.CAP_PROP_FPS)




                        })

                        if args.DS:
                            bbox_text = f"Class: {class_name}: X: {world_x} : y: {world_y}"
                            text_location = (
                                int(xyxy[0]), int(xyxy[1]) + 10)  # Position the text above the top-left corner of the bbox
                            # Draw the bbox information text
                            cv2.putText(frame, bbox_text, text_location, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                        profiler.process_frame(current_timestamp, profiler_detections)

                    # Update the dictionary with the current world coordinates and timestamp for this object ID

                    prev_world_coords[object_id] = world_coordinates
                    prev_timestamps[object_id] = current_timestamp

                    prev_bounding_box[object_id] = xyxy

                for detection in detections:
                    TTC = TP.rear_end_detections(detection, detections)
                    if len(TTC) > 0:
                        TTC_array = np.array(TTC)
                        if np.any(TTC_array < 1000):
                            filename_short = "snapshot_TTC_{}_{}.png".format(np.min(TTC_array), frame_count)
                            profiler.record_TTC(min(TTC_array), frame_count)
                            filename = os.path.join(output_dir, filename_short)

                            cv2.imwrite(filename, frame)

                            #print(1)






            else:

                if frame_number % 1000 == 0 and args.ET:
                    current_timestamp = extract_timestamp(frame, current_timestamp)
                    print(current_timestamp)
            # Start recording when first frame is processed
            if record_end is not None:
                if current_timestamp >= record_end and args.WRITE:
                    out.write(frame)
            # show Frame, with annotations if possible
            if os.name == 'nt':
                try:
                    if not FAILED:
                        cv2.imshow('Frame', frame)
                    else:
                        plt.imshow(frame)
                except Exception as err:
                    print(err)
                    FAILED = 1
                    plt.imshow(frame)
                    print('cant display')

            if not HPC_CODE:
                key = cv2.waitKey(1) & 0xFF

                # Save a snapshot when 's' key is pressed
                if key == ord('s'):
                    filename = "snapshot_{}.png".format(frame_count)
                    cv2.imwrite(filename, frame)
                    print(f"Snapshot saved as {filename}")

                # Break the loop with 'q' key
                if key == ord('q'):
                    print('break key q activated')
                    break

        else:
            print('could not print frame')
            error_counter +=1
            if error_counter >= 10:
                break

    # After processing all frames, you can extract statistics from the profiler
    print('While Loop Has Finished..')
    profiler.get_all_detections_info()

    # Get motorcycle frequency stats
    profiler.get_motorcycle_frequency_stats()
    # Get motorcycle detections
    profiler.get_motorcycle_detections()
    # save detections to csv
    output_file_path = os.path.join(output_dir, 'detections.csv')
    output_file_path2 = os.path.join(output_dir, 'sumdetections.csv')
    output_file_path3 = os.path.join(output_dir, 'TTCdetections.csv')
    profiler.save_TTC(output_file_path3)
    profiler.save_detections_to_csv(output_file_path)





    object_types = list(profiler.object_types)
    # List to hold summary data
    summary_data = []

    for object_type in object_types:
        detections = profiler.get_detections_by_type(object_type)
        #print(f"Total {object_type.capitalize()} Detections: {len(detections)}")

        total_speed = 0
        total_time_diff = 0
        count_speed = 0

        for detection in detections:
            if detection['speed'] is not None:

                total_speed += detection['speed']
                if isinstance(detection['time_diff'], timedelta):
                    total_time_diff += detection['time_diff'].total_seconds()
                else:
                    total_time_diff += detection['time_diff']
                count_speed += 1

                # Print individual detection details (optional)
                if args.VD:
                    print(
                        f"{object_type.capitalize()} ID {detection['object_id']}, "
                        f"Speed: {detection['speed']:.2f} km/hr, Distance Measure {detection['distance_diff']}, "
                        f"Time Measured {detection['time_diff']}")

        # Calculate the mean speed and time diff if count is not zero to avoid division by zero
        mean_speed = total_speed / count_speed if count_speed else 0
        mean_time_diff = total_time_diff / count_speed if count_speed else 0

        # Add summary for this object type to the list
        summary_data.append({
            'object_type': object_type,
            'average_speed_km_hr': mean_speed,
            'average_time_diff_seconds': mean_time_diff,
            'total_detections': len(detections)
        })

    # Convert summary data to a pandas DataFrame
    summary_df = pd.DataFrame(summary_data)

    # Save the DataFrame to a CSV file
    summary_df.to_csv(output_file_path2, index=False)

    print("Summary data saved to summary_data.csv")
    TP.save(output_dir)

    # Release the video capture object and close all frames
    cap.release()
    if os.name == 'nt':
        cv2.destroyAllWindows()
    #sys.exit(0)


if __name__ == '__main__':
    """Loading in command line arguments.  """
    faulthandler.enable()
    parser = argparse.ArgumentParser(prog='main',
                                     description='Add in settings to visualise and process the video',
                                     epilog=main.__doc__)

    # Add argument for directory index
    parser.add_argument('-dir', '--directory', dest="dir",
                        type=int, default=6,
                        help='Index of the directory to select.')

    # Add argument for file index within the directory
    parser.add_argument('-file', type=int, default=50,
                        help='Index of the file within the directory to select.')
    parser.add_argument('-test_off', dest = 'TEST_OFF', action = 'store_true', default = False,
                        help = 'Turning off everything i dont want to run')

    parser.add_argument('-line', type=int, default=50,
                        help='Name of the file used.')
    parser.add_argument('-redefined-class', dest="RCNET", action='store_false',
                        default=False, help='for redefining the yolo classes')
    parser.add_argument('-FREEZE',
                        action='store_false', default=False, help='Detect froze')
    parser.add_argument('-frame_based', dest="FR_BASED", action='store_true', default=False)
    parser.add_argument('-verbose', dest='Verbose', action='store_true', default=False, help='For Printing Statements')

    parser.add_argument('-ocv', '--only_convert_video', dest='OCV', action='store_false', default=False,
                        help='Return once video is converted, more efficient i imagined')

    parser.add_argument('-extract_timestamp', dest='ET', action='store_false', default=False)

    parser.add_argument('-cv', '--convert_video', dest='CV', action='store_false', default=False,
                        help='help converts the video format')

    parser.add_argument('-vd', '--visualise_detection', dest='VD', action='store_false', default=False,
                        help='Visualise the Yulo Detections and displays info')

    parser.add_argument("-hm", '--heatmap', dest='HM', action='store_false', default=False,
                        help='For viewing heatmap options')

    parser.add_argument('-ds', '--display_speed', dest="DS", action='store_false', default=False,
                        help='For displaying the speed on the yulo frame.')

    parser.add_argument('-rv', '--reencode_video', dest='RV', action='store_false', default=False,
                        help='For displaying the speed on the yulo frame.')

    parser.add_argument('-snapshot_verbose', dest='SV', action='store_false', default=True,
                        help='For savinng the violations on the yulo frame.')

    parser.add_argument('-write', dest='WRITE', action='store_false', default=False,
                        help='For writing and saving the processed video footage.')

    parser.add_argument('-frame_capture_after', dest='FC_AFTER', type=int, default=12,
                        help='How many frames do I want to capture after the event for saving purposes.')

    parser.add_argument('-jerk_threshold', type=float, default=10,
                        help='Value for which throws the error for the jerk profile')
    parser.add_argument('-test_hpc', dest='TEST_HPC', action='store_false', default=False, help ='For testing the HPC detection')
    parser.add_argument('-max_file_count', dest='MFC', type = int,  default=10000)
    parser.print_help()
    args = parser.parse_args()
    print('Initial arguments:', args)
    # Now run


    while True:
        # Run the main function with the updated arguments
        main(args)

        # Update the file argument for the next iteration
        args.file += 1
        args.line += 1



        # Optionally, add a condition to exit the loop after a certain number of iterations
        if args.file > 100000:
            break

        # Delay between each iteration
        time.sleep(1)
