#!/usr/bin/env python
# -*- coding:utf-8 -*-

import copy
import cv2
import cv_bridge
import geometry_msgs.msg
from image_geometry.cameramodels import PinholeCameraModel
from jsk_recognition_msgs.msg import BoundingBox
from jsk_recognition_msgs.msg import BoundingBoxArray
from jsk_recognition_msgs.msg import PaperCorner
from jsk_topic_tools import ConnectionBasedTransport
import message_filters
import numpy as np
import rospy
import sensor_msgs.msg
import std_msgs.msg
from tf.transformations import quaternion_from_matrix
from tf.transformations import unit_vector as normalize_vector
from visualization_msgs.msg import Marker, MarkerArray


def image2cameramodel(cameramodel, coord_array, cv_img, depth_img):
    height, width, _ = cv_img.shape
    coord_array_int = coord_array.astype(int)
    # x = (coord_array[:, 0, 0] - cameramodel.cx()) / cameramodel.fx()
    # y = (coord_array[:, 0, 1] - cameramodel.cy()) / cameramodel.fy()
    # z = depth_img.reshape(-1)[coord_array_int[:, 0, 1] * width + coord_array_int[:, 0, 0]]
    x = (coord_array[:, 0, 0] - cameramodel.cx()) / cameramodel.fx()
    y = (coord_array[:, 0, 1] - cameramodel.cy()) / cameramodel.fy()
    z = depth_img.reshape(-1)[coord_array_int[:, 0, 1] * width + coord_array_int[:, 0, 0]]

    x *= z
    y *= z
    x = x.reshape(-1, 4, 1, 1)
    y = y.reshape(-1, 4, 1, 1)
    z = z.reshape(-1, 4, 1, 1)
    xyzs = np.concatenate([x, y, z], axis=3)

    return xyzs


class FeaturePointExtract(ConnectionBasedTransport):

    def __init__(self):
        super(FeaturePointExtract, self).__init__()

        self.feature_params = dict( maxCorners = 100,
                                    qualityLevel = 0.01,
                                    minDistance = 2,
                                    blockSize = 10 )

        self.lk_params = dict( winSize  = (15,15),
                               maxLevel = 2,
                               criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

        self.color = np.random.randint(0, 255, (100, 3))
        self.image_pub = self.advertise('~output/viz',
                                        sensor_msgs.msg.Image,
                                        queue_size=1)
        self.bridge = cv_bridge.CvBridge()
        self.corner_recog = False
        self.gray_image_prev = None
        self.gray_image_next = None
        self.feature = np.empty([0, 0, 0])
        self.xyzs = np.empty([0, 0, 0, 0])
        self.mask = None
        self.paper_corner_array = None
        self.camera_info_msg = None
        self.marker_paper_pub = self.advertise('~output/marker_array',
                                               MarkerArray,
                                               queue_size=1)

        self.marker_point_array = []

    def subscribe(self):
        queue_size = rospy.get_param('~queue_size', 10)
        rgb_img = message_filters.Subscriber(
            '~input/rgb_image',
            sensor_msgs.msg.Image,
            queue_size=1,
            buff_size=2**24)
        depth_img = message_filters.Subscriber(
            '~input/depth_image',
            sensor_msgs.msg.Image,
            queue_size=1,
            buff_size=2**24)
        # paper_corner_x = message_filters.Subscriber(
        #     '~paper_corner_x',
        #     PaperCorner,
        #     queue_size=1,
        #     buff_size=2**24)
        # paper_corner_y = message_filters.Subscriber(
        #     '~paper_corner_y',
        #     PaperCorner,
        #     queue_size=1,
        #     buff_size=2**24)
        # self.subs = [rgb_img, depth_img, paper_corner_x, paper_corner_y]
        self.subs = [rgb_img, depth_img]
        self.sub_info = rospy.Subscriber(
            '~input/camera_info',
            sensor_msgs.msg.CameraInfo, self._cb_cam_info)
        slop = rospy.get_param('~slop', 0.1)
        sync = message_filters.ApproximateTimeSynchronizer(
            fs=self.subs, queue_size=queue_size, slop=slop)
        sync.registerCallback(self._cb)
        self.marker_optical_flow_pub = self.advertise('~output/marker_array',
                                               MarkerArray,
                                               queue_size=1)


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

    def _cb(self, rgb_img, depth_msg):
        bridge = self.bridge
        cameramodel = self.cameramodel
        try:
            cv_image = bridge.imgmsg_to_cv2(rgb_img, 'bgr8')
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
        self.gray_image_next = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        # feature = cv2.goodFeaturesToTrack(gray_image, mask = None, **self.feature_params)

        marker_optical_flow_array_msg = MarkerArray()
        for i in range(4):
            marker_optical_flow = Marker()
            marker_optical_flow.header = rgb_img.header
            marker_optical_flow.ns = "optical_flow_{}".format(i)
            marker_optical_flow.id = 0
            marker_optical_flow.action = Marker.ADD
            marker_optical_flow.type = 4
            marker_optical_flow.scale.x = 0.005
            marker_optical_flow.color.r = 0.0
            marker_optical_flow.color.g = 1.0
            marker_optical_flow.color.b = 0.0
            marker_optical_flow.color.a = 1.0
            marker_optical_flow.lifetime = rospy.Duration()
            marker_optical_flow.points = []
            marker_optical_flow_array_msg.markers.append(marker_optical_flow)


        if (not self.corner_recog):
            # feature = np.empty([0, 0])
            paper_corner_x_len = False
            paper_corner_y_len = False
            while True:
                if (not paper_corner_x_len):
                    paper_corner_x = rospy.wait_for_message("/paper_finder/output/corner/x", PaperCorner)
                if (not paper_corner_y_len):
                    paper_corner_y = rospy.wait_for_message("/paper_finder/output/corner/y", PaperCorner)
                paper_corner = [len(paper_corner_x.corner), len(paper_corner_y.corner)]
                rospy.loginfo(paper_corner)
                if (len(paper_corner_x.corner) == 4):
                    paper_corner_x_len = True
                if (len(paper_corner_y.corner) == 4):
                    paper_corner_y_len = True
                if (paper_corner_x_len and paper_corner_y_len):
                    break
            self.feature = np.array([[[paper_corner_x.corner[0], paper_corner_y.corner[0]]],
                                     [[paper_corner_x.corner[1], paper_corner_y.corner[1]]],
                                     [[paper_corner_x.corner[2], paper_corner_y.corner[2]]],
                                     [[paper_corner_x.corner[3], paper_corner_y.corner[3]]]], dtype='float32')
            self.paper_corner_array = self.feature.copy()
            self.xyzs = image2cameramodel(cameramodel, self.feature, cv_image, depth_img)
            for i in range(4):
                cv_image = cv2.circle(cv_image, (self.feature[i][0][0], self.feature[i][0][1]), 5, self.color[i].tolist(), -1)
                marker_points = geometry_msgs.msg.Point()
                marker_points.x = self.xyzs[0][i][0][0]
                marker_points.y = self.xyzs[0][i][0][1]
                marker_points.z = self.xyzs[0][i][0][2]
                self.marker_point_array.append([marker_points])
                (marker_optical_flow_array_msg.markers[i]).points = copy.copy(self.marker_point_array[i])
            self.corner_recog = True
            self.mask = np.zeros_like(cv_image)
            self.gray_image_prev = self.gray_image_next.copy()

        if (self.corner_recog):
            feature_next, status, err = cv2.calcOpticalFlowPyrLK(self.gray_image_prev, self.gray_image_next, self.feature, None, **self.lk_params)
            xyzs_next = image2cameramodel(cameramodel, feature_next, cv_image, depth_img)
            for i in range(4):
                cv_image = cv2.circle(cv_image, (self.paper_corner_array[i][0][0], self.paper_corner_array[i][0][1]), 5, self.color[i].tolist(), -1)
            for i in range(4):
                diff = self.xyzs[0][i][0] - xyzs_next[0][i][0]
                distance = np.linalg.norm(diff)
                rospy.loginfo(distance)
                if distance > 0.05:
                    marker_points = geometry_msgs.msg.Point()
                    marker_points.x = xyzs_next[0][i][0][0]
                    marker_points.y = xyzs_next[0][i][0][1]
                    marker_points.z = xyzs_next[0][i][0][2]
                    self.marker_point_array[i].append(marker_points)
                    # (marker_optical_flow_array_msg.markers[i]).points = copy.copy(self.marker_point_array[i])
                else:
                    break
            for i, (next_point, prev_point) in enumerate(zip(self.feature, feature_next)):
                prev_x, prev_y = prev_point.ravel()
                next_x, next_y = next_point.ravel()
                self.mask = cv2.line(self.mask, (next_x, next_y), (prev_x, prev_y), (0, 0, 255), 2)
                # cv_image = cv2.circle(cv_image, (next_x, next_y), 5, self.color[i].tolist(), -1)
            cv_image = cv2.add(cv_image, self.mask)
            self.gray_image_prev = self.gray_image_next.copy()
            self.feature = feature_next.reshape(-1, 1, 2)
            self.xyzs = xyzs_next.copy()


        for i in range(4):
            marker_optical_flow_array_msg.markers[i].points = copy.copy(self.marker_point_array[i])
        self.marker_optical_flow_pub.publish(marker_optical_flow_array_msg)

        if self.visualize:
            vis_msg = bridge.cv2_to_imgmsg(cv_image, encoding='bgr8')
            vis_msg.header.stamp = rgb_img.header.stamp
            self.image_pub.publish(vis_msg)

    @property
    def visualize(self):
        return self.image_pub.get_num_connections() > 0


if __name__ == '__main__':
    rospy.init_node('feature_point_extract')
    act = FeaturePointExtract()
    rospy.spin()
