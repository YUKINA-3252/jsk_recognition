#!/usr/bin/env python
# -*- coding:utf-8 -*-

import cv2
import cv_bridge
import geometry_msgs.msg
from image_geometry.cameramodels import PinholeCameraModel
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import numpy as np
import rospy
import sensor_msgs.msg
import std_msgs.msg
import tf
from tf.transformations import quaternion_from_matrix
from tf.transformations import unit_vector as normalize_vector
from pyquaternion import Quaternion


def outer_product_matrix(v):
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])


def cross_product(a, b):
    return np.dot(outer_product_matrix(a), b)


def rotation_matrix_from_axis(
        first_axis=(1, 0, 0), second_axis=(0, 1, 0), axes='xy'):
    if axes not in ['xy', 'yx', 'xz', 'zx', 'yz', 'zy']:
        raise ValueError("Valid axes are 'xy', 'yx', 'xz', 'zx', 'yz', 'zy'.")
    e1 = normalize_vector(first_axis)
    e2 = normalize_vector(second_axis - np.dot(second_axis, e1) * e1)
    if axes in ['xy', 'zx', 'yz']:
        third_axis = cross_product(e1, e2)
    else:
        third_axis = cross_product(e2, e1)
    e3 = normalize_vector(
        third_axis - np.dot(third_axis, e1) * e1 - np.dot(third_axis, e2) * e2)
    first_index = ord(axes[0]) - ord('x')
    second_index = ord(axes[1]) - ord('x')
    third_index = ((first_index + 1) ^ (second_index + 1)) - 1
    indices = [first_index, second_index, third_index]
    return np.vstack([e1, e2, e3])[np.argsort(indices)].T


def unit_normal(a, b, c):
    x = np.linalg.det([[1, a[1], a[2]],
                       [1, b[1], b[2]],
                       [1, c[1], c[2]]])
    y = np.linalg.det([[a[0], 1, a[2]],
                       [b[0], 1, b[2]],
                       [c[0], 1, c[2]]])
    z = np.linalg.det([[a[0], a[1], 1],
                       [b[0], b[1], 1],
                       [c[0], c[1], 1]])
    magnitude = (x**2 + y**2 + z**2)**.5
    return (x / magnitude, y / magnitude, z / magnitude)


def transform_position(ps_list, source_frame, target_frame):
    listener = tf.TransformListener()
    listener.waitForTransform(source_frame, target_frame, rospy.Time(0), rospy.Duration(3.0))
    (trans, rot) = listener.lookupTransform(source_frame, target_frame, rospy.Time(0))
    trans_np = np.asarray([trans[0], trans[1], trans[2]])
    q = Quaternion(rot[3], rot[0], rot[1], rot[2])
    rotation_matrix = q.rotation_matrix
    ps_list_transformed = np.dot(rotation_matrix, ps_list) + trans_np
    return ps_list_transformed

class EdgeDetector(object):

    def __init__(self,
                 length_threshold=10,
                 distance_threshold=1.41421356,
                 canny_th1=50.0,
                 canny_th2=50.0,
                 canny_aperture_size=3,
                 do_merge=False):
        self.lsd = cv2.ximgproc.createFastLineDetector(
            length_threshold,
            distance_threshold,
            canny_th1,
            canny_th2,
            canny_aperture_size,
            do_merge)

    def find_edges(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        line_segments = self.lsd.detect(gray)

        if line_segments is None:
            line_segments = []

        return line_segments


def draw_edges(image, edges):
    for edge in edges:
        x1, y1, x2, y2 = map(int, edge[0][:4])
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)


class EdgeFinder(ConnectionBasedTransport):

    def __init__(self):
        super(EdgeFinder, self).__init__()
        self.edge_detector = EdgeDetector()
        self.length_tolerance = rospy.get_param(
            '~length_tolerance', 0.04)
        self.target_frame = rospy.get_param('~target_frame', "odom")
        self.ps_list_a_x = rospy.get_param('~ps_list_a_x', 0)
        self.ps_list_a_y = rospy.get_param('~ps_list_a_y', 0)
        self.ps_list_a_z = rospy.get_param('~ps_list_a_z', 0)
        self.ps_list_b_x = rospy.get_param('~ps_list_b_x', 0)
        self.ps_list_b_y = rospy.get_param('~ps_list_b_y', 0)
        self.ps_list_b_z = rospy.get_param('~ps_list_b_z', 0)
        self.image_pub = self.advertise('~output/viz',
                                        sensor_msgs.msg.Image,
                                        queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.camera_info_msg = None
        self.cameramodel = None

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        if rospy.get_param('~with_depth', True):
            sub_img = message_filters.Subscriber(
                '~input',
                sensor_msgs.msg.Image,
                queue_size=1,
                buff_size=2**24)
            sub_depth = message_filters.Subscriber(
                '~input/depth',
                sensor_msgs.msg.Image,
                queue_size=1,
                buff_size=2**24)
            self.subs = [sub_img, sub_depth]
            self.sub_info = rospy.Subscriber(
                '~input/camera_info',
                sensor_msgs.msg.CameraInfo, self._cb_cam_info)

            if rospy.get_param('~approximate_sync', True):
                slop = rospy.get_param('~slop', 0.1)
                sync = message_filters.ApproximateTimeSynchronizer(
                    fs=self.subs, queue_size=queue_size, slop=slop)
            else:
                sync = message_filters.TimeSynchronizer(
                    fs=self.subs, queue_size=queue_size)
            sync.registerCallback(self._cb_with_depth)
        else:
            sub = rospy.Subscriber(
                '~input',
                sensor_msgs.msg.Image,
                callback=self._cb,
                queue_size=queue_size)
            self.subs = [sub]

    def unsubscribe(self):
        for s in self.subs:
            s.unregister()

    def _cb_cam_info(self, msg):
        self.camera_info_msg = msg
        self.cameramodel = PinholeCameraModel()
        self.cameramodel.fromCameraInfo(msg)
        self.sub_info.unregister()
        self.sub_info = None
        rospy.loginfo("Received camera info")

    def _cb(self, msg):
        bridge = self.bridge
        try:
            cv_image = bridge.imgmsg_to_cv2(
                msg, 'bgr8')
        except cv_bridge.CvBridgeError as e:
            rospy.logerr('{}'.format(e))
            return
        squares = self.edge_detector.find_edges(cv_image)

        if self.visualize:
            draw_squares(cv_image, squares)
            vis_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            vis_msg.header.stamp = msg.header.stamp
            self.image_pub.publish(vis_msg)

    def _cb_with_depth(self, img_msg, depth_msg):
        if self.camera_info_msg is None or self.cameramodel is None:
            rospy.loginfo("Waiting camera info ...")
            return
        bridge = self.bridge
        try:
            cv_image = bridge.imgmsg_to_cv2(img_msg, 'bgr8')
            depth_img = bridge.imgmsg_to_cv2(depth_msg, 'passthrough')

            if depth_msg.encoding == '16UC1':
                depth_img = np.asarray(depth_img, dtype=np.float32)
                depth_img /= 1000.0  # convert metric: mm -> m
            elif depth_msg.encoding != '32FC1':
                rospy.logerr('Unsupported depth encoding: %s' %
                             depth_msg.encoding)
        except cv_bridge.CvBridgeError as e:
            rospy.logerr('{}'.format(e))
            return
        height, width, _ = cv_image.shape
        cameramodel = self.cameramodel

        # model line
        ps_list_a = np.array([self.ps_list_a_x, self.ps_list_a_y, self.ps_list_a_z])
        ps_list_b = np.array([self.ps_list_b_x, self.ps_list_b_y, self.ps_list_b_z])
        source_frame = '/' + self.camera_info_msg.header.frame_id
        target_frame = self.target_frame
        ps_list_transformed_a = transform_position(ps_list_a, source_frame, target_frame)
        ps_list_transformed_b = transform_position(ps_list_b, source_frame, target_frame)

        ps_list_transformed_pixel_u_a = int((ps_list_transformed_a[0] / ps_list_transformed_a[2] * cameramodel.fx() + cameramodel.cx()))
        ps_list_transformed_pixel_v_a = int((ps_list_transformed_a[1] / ps_list_transformed_a[2] * cameramodel.fy() + cameramodel.cy()))
        ps_list_transformed_pixel_u_b = int((ps_list_transformed_b[0] / ps_list_transformed_b[2] * cameramodel.fx() + cameramodel.cx()))
        ps_list_transformed_pixel_v_b = int((ps_list_transformed_b[1] / ps_list_transformed_b[2] * cameramodel.fy() + cameramodel.cy()))
        model_line_len = (self.ps_list_b_x - self.ps_list_a_x) ** 2 + (self.ps_list_b_y - self.ps_list_a_y) ** 2 + (self.ps_list_b_z - self.ps_list_a_z) ** 2
        model_line_cetnter = np.array([(self.ps_list_a_x + self.ps_list_b_x) / 2, (self.ps_list_a_y + self.ps_list_b_y) / 2, (self.ps_list_a_z + self.ps_list_b_z) / 2])
        model_line_direction = np.array([self.ps_list_b_x - self.ps_list_a_x, self.ps_list_b_y - self.ps_list_a_y, self.ps_list_b_z - self.ps_list_a_z])

        edges = self.edge_detector.find_edges(cv_image)
        np_edges = np.array(edges, dtype=np.int32).reshape(-1, 4)
        np_edges_3d = np.zeros_like(np_edges)

        # edges len
        np_edges_3d[:, 0] = (np_edges[:, 0] - cameramodel.cx()) / cameramodel.fx()
        np_edges_3d[:, 2] = (np_edges[:, 2] - cameramodel.cx()) / cameramodel.fx()
        np_edges_3d[:, 1] = (np_edges[:, 1] - cameramodel.cy()) / cameramodel.fy()
        np_edges_3d[:, 3] = (np_edges[:, 3] - cameramodel.cy()) / cameramodel.fy()
        z1 = depth_img.reshape(-1)[np_edges[:, 1] * width + np_edges[:, 0]]
        z2 = depth_img.reshape(-1)[np_edges[:, 3] * width + np_edges[:, 2]]
        np_edges_3d[:, 0] = np_edges_3d[:, 0] * z1
        np_edges_3d[:, 1] = np_edges_3d[:, 1] * z1
        np_edges_3d[:, 2] = np_edges_3d[:, 2] * z2
        np_edges_3d[:, 3] = np_edges_3d[:, 3] * z2
        edges_len = np.sqrt((np_edges[:, 2] - np_edges[:, 0]) ** 2 + (np_edges[:, 3] - np_edges[:, 1]) ** 2 + (z2 - z1) ** 2)
        print(edges_len)

        # edges center
        cx = (np_edges_3d[:, 0] + np_edges_3d[:, 2]) / 2
        cy = (np_edges_3d[:, 1] + np_edges_3d[:, 3]) / 2
        cz = (z1 + z2) / 2
        np_edges_3d_center = np.column_stack((cx, cy, cz))

        # edges direction
        np_edges_3d_direction = (np_edges_3d[:, 2] - np_edges_3d[:, 0], np_edges_3d[:, 3] - np_edges_3d[:, 1], z2 - z1)

        if self.visualize:
            draw_edges(cv_image, edges)
            cv2.line(cv_image, (ps_list_transformed_pixel_u_a, ps_list_transformed_pixel_v_a), (ps_list_transformed_pixel_u_b, ps_list_transformed_pixel_v_b), (255, 0, 0), 5)
            vis_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            vis_msg.header.stamp = img_msg.header.stamp
            self.image_pub.publish(vis_msg)

    @property
    def visualize(self):
        return self.image_pub.get_num_connections() > 0


if __name__ == '__main__':
    rospy.init_node('edge_finder')
    act = EdgeFinder()
    rospy.spin()
