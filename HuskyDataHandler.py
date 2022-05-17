"""
This file aims to provide an easy interface with the data collected using the Husky robot.

Features that need to be implemented:

    - Bagfile to Dataset function
        - Argument parsing to select data

    - Data generators
        - Create generators to access data in the subfolders
    - Calibration file access

    bagfile extractor based off of bagfile to image script by Matt Church 2021
"""
import os
import dateutil.parser as dparser
import rosbag
import numpy as np
import pandas as pd
import cv2
import tqdm
import sys

sys.path.insert(0, '/opt/ros/melodic/lib/python2.7/dist-packages/')
import ros_numpy as rnp
import yaml


class Dataset_Handler():

    def __init__(self, data_path, low_memory=True, time_sync=True, time_tolerance=0.1, timestamp_resolution=1e-9):
        self.data_path = data_path
        self.low_memory = low_memory

        self.left_image_path = os.path.join(self.data_path, "image_00/data")
        self.right_image_path = os.path.join(self.data_path, "image_01/data")
        self.lidar_path = os.path.join(self.data_path, "velodyne_points/data")

        self.time_tolerance = time_tolerance
        self.timestamp_resolution = timestamp_resolution

        self.left_image_files = []
        self.right_image_files = []
        self.lidar_files = []

        with open(
                r'/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/aru-calibration/ZED/left.yaml') as file:
            left_vars = yaml.load(file, Loader=yaml.FullLoader)
        with open(
                r'/home/kats/Documents/My Documents/UCT/Masters/Code/PythonMeshManipulation/mesh_pydnet/aru-calibration/ZED/right.yaml') as file:
            right_vars = yaml.load(file, Loader=yaml.FullLoader)

        self.left_distortion_coefficients = np.array(left_vars['distortion_coefficients']['data']).reshape(
            (left_vars['distortion_coefficients']['rows'], left_vars['distortion_coefficients']['cols']))
        self.left_rectification_matrix = np.array(left_vars['rectification_matrix']['data']).reshape(
            (left_vars['rectification_matrix']['rows'], left_vars['rectification_matrix']['cols']))
        self.left_projection_matrix = np.array(left_vars['projection_matrix']['data']).reshape(
            (left_vars['projection_matrix']['rows'], left_vars['projection_matrix']['cols']))
        self.left_camera_matrix = np.array(left_vars['camera_matrix']['data']).reshape(
            (left_vars['camera_matrix']['rows'], left_vars['camera_matrix']['cols']))

        self.right_distortion_coefficients = np.array(right_vars['distortion_coefficients']['data']).reshape(
            (right_vars['distortion_coefficients']['rows'], right_vars['distortion_coefficients']['cols']))
        self.right_rectification_matrix = np.array(right_vars['rectification_matrix']['data']).reshape(
            (right_vars['rectification_matrix']['rows'], right_vars['rectification_matrix']['cols']))
        self.right_projection_matrix = np.array(right_vars['projection_matrix']['data']).reshape(
            (right_vars['projection_matrix']['rows'], right_vars['projection_matrix']['cols']))
        self.right_camera_matrix = np.array(right_vars['camera_matrix']['data']).reshape(
            (right_vars['camera_matrix']['rows'], right_vars['camera_matrix']['cols']))



        # Get names of files to iterate through
        assert os.path.exists(self.left_image_path), 'Error, left images not found'
        left_files = np.asarray(os.listdir(self.left_image_path), dtype=object)
        left_times = np.zeros(len(left_files))
        for i, f in enumerate(left_files):
            left_times[i] = int(os.path.splitext(f)[0])

        # Making an array to store matching right imgs and lidar imgs corresponding to each left.
        matches = -np.ones((len(left_files), 2), dtype=int)  # -1 if no match found

        # Load in the right files
        if os.path.exists(self.right_image_path):
            right_files = np.asarray(os.listdir(self.right_image_path), dtype=object)
            # sorted_files =np.zeros(len(right_files), dtype=object)
            right_times = -1 * np.ones(len(right_files))
            deltas = np.ones(len(right_files))
            for i, f in enumerate(right_files):
                right_times[i] = int(os.path.splitext(f)[0])
                difference = np.abs(left_times - right_times[i])
                deltas[i] = difference.min()  # Minimum time discrepancy
                l_ind = np.argmin(difference)
                if deltas[i] < (self.time_tolerance / self.timestamp_resolution):
                    # only append file to sorted list if less than tolerance
                    matches[l_ind, 0] = i

        if os.path.exists(self.lidar_path):
            lidar_files = np.asarray(os.listdir(self.lidar_path), dtype=object)
            lidar_times = -1 * np.ones(len(lidar_files))
            deltas = np.ones(len(lidar_files))
            for i, f in enumerate(lidar_files):
                lidar_times[i] = int(os.path.splitext(f)[0])
                difference = np.abs(left_times - lidar_times[i])
                deltas[i] = difference.min()  # Minimum time discrepancy
                l_ind = np.argmin(difference)
                if deltas[i] < (self.time_tolerance / self.timestamp_resolution):
                    # only append file to sorted list if less than tolerance
                    matches[l_ind, 1] = i

        if time_sync:
            valid = np.where(np.all(matches >= 0, axis=1))[0]  # only those with matches for all
            self.left_image_files = left_files[valid]
            self.times = left_times[valid]
            if os.path.exists(self.right_image_path):
                self.right_image_files = right_files[matches[valid, 0]]
                self.right_times = right_times[matches[valid, 0]]
            if os.path.exists(self.lidar_path):
                self.lidar_files = lidar_files[matches[valid, 1]]
                self.lidar_times = lidar_times[matches[valid, 1]]

        self.num_frames = len(self.left_image_files)

        temp_im = cv2.imread(os.path.join(self.left_image_path, self.left_image_files[0]))
        self.img_shape = temp_im.shape
        # For distortion:
        self.left_mapx, self.left_mapy = cv2.initUndistortRectifyMap(self.left_camera_matrix,
                                                                     self.left_distortion_coefficients.squeeze(), None,
                                                                     self.left_projection_matrix, self.img_shape[:2][::-1]
                                                                     ,5)

        self.right_mapx, self.right_mapy = cv2.initUndistortRectifyMap(self.right_camera_matrix,
                                                                     self.right_distortion_coefficients.squeeze(), None,
                                                                     self.right_projection_matrix,
                                                                     self.img_shape[:2][::-1]
                                                                     , 5)


        """
        # Calibration files
        # TODO
        self.calib_path = 
        self.calib.K_cam0 = 
        self.calib.K_cam1 = 
        self.calib.T_cam0_cam1 = 
        self.calib.T_cam0_velo = 
        self.calib.R_rect_00 =
        self.calib.P_rect_00 = 
        """

        """
        NOTES:
        
        - Still need to have a way of reading in calibration files for camera-to-camera calibration
            Also need to have lidar-to-camera0 calibration files. 
        
        - This could probably benefit from having generators built in. 
            Would allow iterating over images in loops. 
            You're welcome to add that in as a new function if you would like
        """

    def get_cam0(self, frame_num, rectify=False):
        r"""Returns the left image at the given frame"""
        assert frame_num <= self.num_frames, "Frame number greater than number of files"
        assert len(self.left_image_files) > 0, f"No image files found at {self.left_image_path}"

        im = os.path.join(self.left_image_path, self.left_image_files[frame_num])
        if not rectify:
            img = cv2.imread(im)
        else:
            img = cv2.remap(cv2.imread(im), self.left_mapx, self.left_mapy, cv2.INTER_LINEAR)
        return img


    def get_cam1(self, frame, rectify=False):
        r"""Returns the right image at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.right_image_files) > 0, f"No image files found at {self.right_image_path}"

        im = os.path.join(self.right_image_path, self.right_image_files[frame])
        if not rectify:
            img = cv2.imread(im)
        else:
            img = cv2.remap(cv2.imread(im), self.right_mapx, self.right_mapy, cv2.INTER_LINEAR)
        return img

    def get_lidar(self, frame):
        r"""Returns the lidar points at the given frame"""
        assert frame <= self.num_frames, "Frame number greater than number of files"
        assert len(self.lidar_files) > 0, f"No Lidar files found at {self.lidar_path}"
        file = os.path.join(self.lidar_path, self.lidar_files[frame])
        return np.load(file)

    def get_times(self):
        return self.times

    def get_num_frames(self):
        return self.num_frames



def timesync_dataset(path_to_data, simulate_removal = True, seconds_tolerance = 0.1, timestamp_resolution = 1e-9, verbose=True):
    r"""
    This is to check that an extracted dataset is timesynced.
    It will remove those files that do not sync to within the required tolerance.
    """
    import time
    # list the folders that contain data in the husky DatasetHandler() format
    import glob
    dataset_paths = glob.glob(path_to_data+"*/image_00/data/")

    # for each folder:
    for d in dataset_paths:
        set_dir = d.rsplit('/image_00/data/')[0]

        # make a timesynced DatasetHandler() object
        husky_var = Dataset_Handler(set_dir)

        # get the list of all the files
        l_imgs = os.listdir(husky_var.left_image_path)
        r_imgs = os.listdir(husky_var.right_image_path)
        lid_files = os.listdir(husky_var.lidar_path)

        # get the list of files that consitute the un-synced data
        l_diff = set(husky_var.left_image_files).symmetric_difference(set(os.listdir(husky_var.left_image_path)))
        r_diff = set(husky_var.right_image_files).symmetric_difference(set(os.listdir(husky_var.right_image_path)))
        lid_diff = set(husky_var.lidar_files).symmetric_difference(set(os.listdir(husky_var.lidar_path)))

        # if simulating:
        #       Print out the summary of files that do and do not sync
        #       make output file of the files to be removed
        #           -   Would be nice to have it as a shell script that could just be run to remove those files

        if simulate_removal:
            print(f"Simulating removal:\n"
                  f"Found:\n"
                  f"\t - {len(husky_var.left_image_files)} files that are synced.")
            if len(l_diff):
                  print(f"\t - {len(l_diff)} unsynced left files of {len(l_imgs)}.")
            if len(r_diff):
                  print(f"\t - {len(r_diff)} unsynced right files of {len(r_imgs)}.")
            if len(lid_diff):
                  print(f"\t - {len(lid_diff)} unsynced lidar files of {len(lid_files)}.")
            print(f"Run without the simulate_removal flag to remove.\n"
                  f"-------------------------------------------------------\n")

            print(f"Preview of removal:")
            if len(l_diff):
                  print(f"Removing left img   -> {os.path.join(husky_var.left_image_path, list(l_diff)[0])}")
            if len(r_diff):
              print(f"Removing right img  -> {os.path.join(husky_var.right_image_path, list(r_diff)[0])}")
            if len(lid_diff):
              print(f"Removing lidar scan -> {os.path.join(husky_var.lidar_path, list(lid_diff)[0])}")

        else:
            # If not simulating:
            #       remove those files that are not in the time synced list

            print(f"Preview of removal:")
            if len(l_diff):
                print(f"Removing left img   -> {os.path.join(husky_var.left_image_path, list(l_diff)[0])}")
            if len(r_diff):
                  print(f"Removing right img  -> {os.path.join(husky_var.right_image_path, list(r_diff)[0])}")
            if len(lid_diff):
                  print(f"Removing lidar scan -> {os.path.join(husky_var.lidar_path, list(lid_diff)[0])}")

            print("Press ctrl+c if this is wrong\n \t.... you have 5 seconds")
            time.sleep(5)

            if len(l_diff):
                print("Removing left files")
                print()
                for l in tqdm.tqdm(l_diff):
                    os.remove(os.path.join(husky_var.left_image_path, l))
                print()

            if len(r_diff):
                print("Removing right files")
                print()
                for r in tqdm.tqdm(r_diff):
                    os.remove(os.path.join(husky_var.right_image_path, r))
                print()

            if len(lid_diff):
                print("Removing lidar files")
                print()
                for li in tqdm.tqdm(lid_diff):
                    os.remove(os.path.join(husky_var.lidar_path, li))
                print()

            # print summary of what happened
            print("Summary of removal:")
            print(f"\t - Removed {len(l_imgs) - len(os.listdir(husky_var.left_image_path))} left images")
            print(f"\t - Removed {len(r_imgs) - len(os.listdir(husky_var.right_image_path))} left images")
            print(f"\t - Removed {len(lid_files) - len(os.listdir(husky_var.lidar_path))} left images")

    # repeat for the rest of the folders with data.


    return 0


def bagfile_to_dataset(path_bagfile, path_output_dataset, camera_mat=None, dist_mat=None, custom_date=False,
                       images=True,
                       lidar=True, verbose=True, overwrite=False):
    # Topics:
    #     / velodyne_points
    #     "/camera/image_left/image_raw"
    #     "/camera/image_right/image_raw"

    if images:
        left_topic = "/camera/image_left/image_raw"
        right_topic = "/camera/image_right/image_raw"
    if lidar:
        lidar_topic = "/velodyne_points/"

    if os.path.isdir(path_bagfile):
        bags = []
        for f in os.listdir(path_bagfile):
            _, extension = os.path.splitext(f)
            if extension == '.bag':
                bags.append(os.path.join(path_bagfile, f))

    if verbose:
        print(f"Bags for extraction: \n{bags}")

    for bag in bags:
        if custom_date:
            date = custom_date
        else:
            _, bag_name = os.path.split(bag)
            split = os.path.splitext(bag_name)[0].split('-')
            # date = dparser.parse(os.path.splitext(bag_name)[0], fuzzy=True).strftime("%Y_%m_%d_%H_%M")
            date = dparser.parse(os.path.splitext(bag_name)[0], fuzzy=True).strftime("%Y_%m_%d") + \
                   f"_{split[-3]}_{split[-2]}_{split[-1]}"

        # Check if folders exist
        """
        Kitti Data Structure:
        _______________________________
        data_dir
            <date>
                <date>_drive_000#_sync
                    image_00
                        data
                            <images.png>
                        timestamps.txt
                    image_01
                        ...
                    image_02
                        ...
                    image_03
                        ...
                    velodyne_points
                        data
                            #.bin
                        timestamps.txt
                calib_cam_to_cam.txt
                calib_imu_to_velo.txt
                calib_velo_to_cam.txt
        """

        if verbose:
            print("--------------------------------------------------------------------------------")
            print(f"\t\tChecking and Making folders for {bag_name}")
            print("--------------------------------------------------------------------------------")

        date_folder = os.path.join(path_output_dataset, date)
        if not os.path.exists(date_folder):
            os.makedirs(date_folder)
            if verbose: print(f"\t-Master folder created: {date_folder}")
        else:
            print("Folder already exists for this date and time.")
            if overwrite:
                print("\tOverwriting")
            else:
                print("\tSkipping")
                continue

        if not os.path.exists(os.path.join(date_folder, "calib.txt")):
            if verbose: print("\n\t- Making image calibration file.")
            if camera_mat is None:
                print("\t\tNo camera matrix supplied. Defaulting to preset.")
                camera_mat = np.array([[531.14774, 0.000000, 637.87114],
                                       [0.000000, 531.26312, 331.27469],
                                       [0.000000, 0.000000, 1.000000]])
            if dist_mat is None:
                print("\t\tNo distortion matrix supplied. Defaulting to preset.")
                dist_mat = np.array([-0.048045, 0.010493, -0.000281, -0.001232, 0.000000])

            file = open(os.path.join(date_folder, "calib.txt"), 'w')
            file.write(f"K={camera_mat}\n")
            file.write(f"dist={dist_mat}\n")
            file.close()
            if verbose: print("\t...... Done\n")

        if images:
            left_im_path = os.path.join(path_output_dataset, date, 'image_00', 'data')
            right_im_path = os.path.join(path_output_dataset, date, 'image_01', 'data')
            if not os.path.exists(left_im_path):
                # left camera dir does not exist
                os.makedirs(left_im_path)
                if verbose: print("\t-Left image data folder created")
            if not os.path.exists(right_im_path):
                # right camera dir does not exist
                os.makedirs(right_im_path)
                if verbose: print("\t-Right image data folder created")

        if lidar:
            velodyne_path = os.path.join(path_output_dataset, date, 'velodyne_points', 'data')
            if not os.path.exists(velodyne_path):
                os.makedirs(velodyne_path)
                if verbose: print("\t-Velodyne data folder created")

        if verbose: print("...Folders created")
        if verbose: print("_________________________________\n")

        """ 
        --------------------------------------------------------------------------------
          Extracting
        --------------------------------------------------------------------------------
        
        Now ready for extracting data
            
            - Will do images and/or lidar based on what is needed
            
            Adding progress bars for visualisation
            have to use the generator objects returned by rosbag for this to work. 
            Not ideal but shouldn't slow down much.

        """
        if verbose:
            print("--------------------------------------------------------------------------------")
            print(f"\t\tExtracting Data from {bag_name}")
            print("--------------------------------------------------------------------------------")

        if verbose: print("\tCreating rosbag variable. This may take a while")
        ros_bag = rosbag.Bag(bag)
        print("\tSTILL NEED TO CHECK IF THE TIMESTAMPS MATCH FOR IMAGESe")

        if images:
            if verbose: print("\t- Extracting Images")
            if verbose: print("\t\t- Extracting Left Images:")
            len_left = ros_bag.get_message_count(left_topic)
            len_right = ros_bag.get_message_count(right_topic)
            if (len_left == len_right): print("WARNING:Number of images on left and right cameras do not match")

            left_gen = ros_bag.read_messages(topics=left_topic)

            # Making the timestamp files:
            timestamp_path, _ = os.path.split(left_im_path)
            left_timestamps = open(os.path.join(timestamp_path, "timestamps.txt"), 'w')
            for l in tqdm.trange(len_left):
                # for topic, msg, t in ros_bag.read_messages(topics=left_topic):
                topic, msg, t = next(left_gen)
                left_timestamps.write(str(t) + '\n')
                bag_img = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
                if camera_mat is not None:
                    if dist_mat is None:
                        print("Please provide distortion coefficients")
                        break
                    else:
                        out_img = cv2.undistort(bag_img, camera_mat, dist_mat)
                else:
                    out_img = bag_img
                cv2.imwrite(os.path.join(left_im_path, str(t) + ".png"), out_img)

            # Remember to close files
            left_timestamps.close()

            if verbose: print("\t\t- Extracting Right Images:")
            right_gen = ros_bag.read_messages(topics=right_topic)

            print()
            print("NOTE: Need to clean the rest of right extraction.\n-- assert if timestamps are approx equal?")
            print()

            # Timestamp writer
            timestamp_path, _ = os.path.split(right_im_path)
            right_timestamps = open(os.path.join(timestamp_path, "timestamps.txt"), 'w')

            for right_index in tqdm.trange(len_right):
                # for topic, msg, t in ros_bag.read_messages(topics=topic_right):
                topic, msg, t = next(right_gen)

                right_timestamps.write(str(t) + '\n')

                bag_img = np.frombuffer(msg.data,
                                        dtype=np.uint8).reshape(msg.height,
                                                                msg.width,
                                                                -1)
                if camera_mat is not None:
                    if dist_mat is None:
                        print("Please provide distortion coefficients")
                        break
                    else:
                        out_img = cv2.undistort(bag_img, camera_mat, dist_mat)
                else:
                    out_img = bag_img
                cv2.imwrite(os.path.join(right_im_path, str(t) + ".png"), out_img)

            right_timestamps.close()

            if verbose: print("\n\t......Done extracting images\n")

        if lidar:
            if verbose: print("\n\t- Extracting Lidar Data")
            len_lidar = ros_bag.get_message_count(lidar_topic)
            lidar_gen = ros_bag.read_messages(topics=lidar_topic)

            for lidar_index in tqdm.trange(len_lidar):
                topic, msg, t = next(lidar_gen)
                pts = rnp.point_cloud2.pointcloud2_to_xyz_array(msg)

                lidar_file = os.path.join(velodyne_path, f"{t}.npy")
                np.save(lidar_file, pts, allow_pickle=False)

        if verbose:
            print("\t....Done extracting lidar data")
            print("\n\n")
            print("------------------------------------------")
            print(f"Summary for {bag_name}:")
            print("------------------------------------------")
            if images:
                print(f"\t- Extracted: {len_left} left images")
                print(f"\t- Extracted: {len_right} right images")
            if lidar:
                print(f"\t- Extracted: {len_lidar} lidar scans")

            print()
            print(f"\tFiles saved to {date_folder}")
            print()
            print("------------------------------------------")
            print()

    print("\n\n------------------------------------------")
    print("No other bagfiles found in folder.")
    print("Job is done.")
    print("Goodbye.")

    return 0


if __name__ == "__main__":
    print("Running tests")

    print("Test extraction")
    # bagfile_path = "/media/kats/Katsoulis3/Datasets/Husky/gucci_bags/Calibration and Lab"
    # output_path = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/Calibration and Lab"

    bagfile_path = '/media/kats/Katsoulis3/Datasets/Husky/gucci_bags/UCT Outdoor/old_zoo'
    output_path =  "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/"


    # bagfile_to_dataset(bagfile_path, output_path, lidar=True, overwrite=True)

    timesync_dataset(output_path, simulate_removal=False)
    pass

    # print("Testing data class")
    # data_path = "/media/kats/Katsoulis3/Datasets/Husky/extracted_data/old_zoo/Route C/2022_05_03_14_09_01/"
    # print(f"Looking for dataset in:{data_path}")
    # dataset = Dataset_Handler(data_path=data_path)
    #
    # import matplotlib.pyplot as plt
    # frame_list = np.arange(0, dataset.num_frames, dataset.num_frames//5)
    # for frame in frame_list:
    #     fig, ax = plt.subplots(1, 2)
    #     ax[0].imshow(dataset.get_cam0(frame))
    #     ax[1].imshow(dataset.get_cam1(frame))
    #
    #     ax[0].set_title("Left image")
    #     ax[1].set_title("Right image")
    #
    #     ax[0].axis('off')
    #     ax[1].axis('off')
    #
    #     fig.suptitle(f"Images for frame {frame}")
    #
    #     plt.show()
