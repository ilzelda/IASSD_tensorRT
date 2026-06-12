import argparse
import math
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
TOOLS_DIR = ROOT_DIR / 'tools'
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import numpy as np
import rospy
import torch
from sensor_msgs import point_cloud2
from sensor_msgs.msg import CompressedImage, Image, PointCloud2, PointField
from visualization_msgs.msg import Marker, MarkerArray

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from iassd_postprocess import postprocess_outputs
from iassd_trt_runtime import IASSDTensorRTRunner


POINT_FIELD_TO_DTYPE = {
    PointField.INT8: np.int8,
    PointField.UINT8: np.uint8,
    PointField.INT16: np.int16,
    PointField.UINT16: np.uint16,
    PointField.INT32: np.int32,
    PointField.UINT32: np.uint32,
    PointField.FLOAT32: np.float32,
    PointField.FLOAT64: np.float64,
}


CLASS_COLORS = {
    1: (0.0, 0.8, 1.0, 0.45),
    2: (1.0, 0.7, 0.0, 0.45),
    3: (0.2, 1.0, 0.3, 0.45),
}


def parse_args():
    parser = argparse.ArgumentParser(description='ROS1 PointCloud2 입력을 받아 IA-SSD TensorRT 결과를 MarkerArray로 publish')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--engine_file', type=str, required=True)
    parser.add_argument('--plugin_library', type=str, required=True)
    parser.add_argument('--lidar_topic', type=str, default='/lidar_sync')
    parser.add_argument('--image_topic', type=str, default='/image_raw_sync')
    parser.add_argument('--boxes_topic', type=str, default='/iassd_trt/boxes')
    parser.add_argument('--publish_lidar_topic', type=str, default='/iassd_trt/lidar')
    parser.add_argument('--publish_image_topic', type=str, default='/iassd_trt/image_raw')
    parser.add_argument('--republish_lidar', type=str, choices=['raw', 'filtered', 'off'], default='raw')
    parser.add_argument('--republish_image', type=str, choices=['raw', 'compressed', 'off'], default='raw')
    parser.add_argument('--vis_point_stride', type=int, default=1)
    parser.add_argument('--num_points', type=int, default=16384)
    parser.add_argument('--queue_size', type=int, default=1)
    parser.add_argument('--seed', type=int, default=1024)
    parser.add_argument('--score_thresh', type=float, default=None)
    parser.add_argument('--nms_thresh', type=float, default=None)
    parser.add_argument('--nms_pre_maxsize', type=int, default=None)
    parser.add_argument('--nms_post_maxsize', type=int, default=None)
    parser.add_argument('--marker_lifetime', type=float, default=0.15)
    parser.add_argument('--latency_log_interval', type=int, default=30)
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER)
    return parser.parse_args(rospy.myargv(argv=sys.argv)[1:])


def resolve_path(path):
    path = Path(path)
    if path.is_absolute():
        return path
    root_path = ROOT_DIR / path
    if root_path.exists():
        return root_path
    return TOOLS_DIR / path


def load_config(args):
    cfg_file = resolve_path(args.cfg_file).resolve()
    old_cwd = Path.cwd()
    try:
        import os
        os.chdir(TOOLS_DIR)
        cfg_from_yaml_file(str(cfg_file), cfg)
    finally:
        os.chdir(old_cwd)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)
    return cfg_file


def pointcloud2_to_xyzi(msg):
    names = [field.name for field in msg.fields]
    for required_name in ('x', 'y', 'z'):
        if required_name not in names:
            raise ValueError(f'PointCloud2에 {required_name} 필드가 없습니다.')

    dtype_names = []
    dtype_formats = []
    dtype_offsets = []
    for field in sorted(msg.fields, key=lambda item: item.offset):
        field_dtype = POINT_FIELD_TO_DTYPE.get(field.datatype)
        if field_dtype is None:
            continue
        dtype_names.append(field.name)
        dtype_formats.append(field_dtype if field.count == 1 else (field_dtype, field.count))
        dtype_offsets.append(field.offset)

    dtype = np.dtype({
        'names': dtype_names,
        'formats': dtype_formats,
        'offsets': dtype_offsets,
        'itemsize': msg.point_step,
    })
    if msg.is_bigendian:
        dtype = dtype.newbyteorder('>')

    if msg.row_step == msg.point_step * msg.width:
        cloud = np.frombuffer(msg.data, dtype=dtype, count=msg.width * msg.height)
    else:
        # organized cloud에서 row padding이 있으면 row별 유효 point 영역만 읽는다.
        rows = []
        for row_index in range(msg.height):
            row_start = row_index * msg.row_step
            row_end = row_start + msg.point_step * msg.width
            rows.append(np.frombuffer(msg.data[row_start:row_end], dtype=dtype, count=msg.width))
        cloud = np.concatenate(rows, axis=0) if rows else np.zeros((0,), dtype=dtype)

    xyz = np.stack(
        [
            cloud['x'].astype(np.float32),
            cloud['y'].astype(np.float32),
            cloud['z'].astype(np.float32),
        ],
        axis=1,
    )
    if 'intensity' in cloud.dtype.names:
        intensity = cloud['intensity'].astype(np.float32).reshape(-1, 1)
    else:
        intensity = np.zeros((xyz.shape[0], 1), dtype=np.float32)
    points = np.concatenate([xyz, intensity], axis=1)
    finite_mask = np.isfinite(points).all(axis=1)
    return points[finite_mask]


def mask_points_by_range(points, point_cloud_range):
    point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)
    mask = (
        (points[:, 0] >= point_cloud_range[0])
        & (points[:, 0] <= point_cloud_range[3])
        & (points[:, 1] >= point_cloud_range[1])
        & (points[:, 1] <= point_cloud_range[4])
        & (points[:, 2] >= point_cloud_range[2])
        & (points[:, 2] <= point_cloud_range[5])
    )
    return points[mask]


def sample_points(points, num_points, rng):
    if points.shape[0] == 0:
        return np.zeros((num_points, 4), dtype=np.float32)

    if num_points < points.shape[0]:
        pts_depth = np.linalg.norm(points[:, 0:3], axis=1)
        far_idxs_choice = np.where(pts_depth >= 40.0)[0]
        near_idxs = np.where(pts_depth < 40.0)[0]
        if num_points > len(far_idxs_choice) and len(near_idxs) > 0:
            near_count = num_points - len(far_idxs_choice)
            near_idxs_choice = rng.choice(near_idxs, near_count, replace=False)
            choice = np.concatenate((near_idxs_choice, far_idxs_choice), axis=0)
        else:
            choice = rng.choice(np.arange(points.shape[0], dtype=np.int32), num_points, replace=False)
    else:
        choice = np.arange(points.shape[0], dtype=np.int32)
        if num_points > points.shape[0]:
            extra_choice = rng.choice(choice, num_points - points.shape[0], replace=True)
            choice = np.concatenate((choice, extra_choice), axis=0)

    rng.shuffle(choice)
    return points[choice].astype(np.float32)


def make_trt_points(points, num_points, point_cloud_range, rng):
    filtered = mask_points_by_range(points, point_cloud_range)
    sampled = sample_points(filtered, num_points, rng)
    batch_index = np.zeros((sampled.shape[0], 1), dtype=np.float32)
    return np.concatenate([batch_index, sampled], axis=1), filtered


def xyzi_to_pointcloud2(points, header):
    fields = [
        PointField(name='x', offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name='y', offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name='z', offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name='intensity', offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    return point_cloud2.create_cloud(header, fields, points[:, :4].astype(np.float32))


def yaw_to_quaternion(yaw):
    half_yaw = yaw * 0.5
    return 0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)


def make_delete_all_marker(frame_id, stamp):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = 'iassd_trt_boxes'
    marker.id = 0
    marker.action = Marker.DELETEALL
    return marker


def make_box_marker(frame_id, stamp, marker_id, box, score, label, lifetime):
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = 'iassd_trt_boxes'
    marker.id = marker_id
    marker.type = Marker.CUBE
    marker.action = Marker.ADD
    marker.pose.position.x = float(box[0])
    marker.pose.position.y = float(box[1])
    marker.pose.position.z = float(box[2])
    qx, qy, qz, qw = yaw_to_quaternion(float(box[6]))
    marker.pose.orientation.x = qx
    marker.pose.orientation.y = qy
    marker.pose.orientation.z = qz
    marker.pose.orientation.w = qw
    marker.scale.x = max(float(box[3]), 1e-3)
    marker.scale.y = max(float(box[4]), 1e-3)
    marker.scale.z = max(float(box[5]), 1e-3)
    marker.color.r, marker.color.g, marker.color.b, marker.color.a = CLASS_COLORS.get(int(label), (1.0, 1.0, 1.0, 0.45))
    marker.lifetime = rospy.Duration.from_sec(lifetime)
    marker.text = f'label={int(label)} score={float(score):.3f}'
    return marker


class IASSDTRTLidarNode:
    def __init__(self, args):
        self.args = args
        cfg_file = load_config(args)
        self.point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        post_cfg = cfg.MODEL.POST_PROCESSING
        nms_cfg = post_cfg.NMS_CONFIG
        self.score_thresh = float(args.score_thresh if args.score_thresh is not None else post_cfg.SCORE_THRESH)
        self.nms_thresh = float(args.nms_thresh if args.nms_thresh is not None else nms_cfg.NMS_THRESH)
        self.nms_pre_maxsize = int(args.nms_pre_maxsize if args.nms_pre_maxsize is not None else nms_cfg.NMS_PRE_MAXSIZE)
        self.nms_post_maxsize = int(args.nms_post_maxsize if args.nms_post_maxsize is not None else nms_cfg.NMS_POST_MAXSIZE)
        self.rng = np.random.default_rng(args.seed)
        self.runner = IASSDTensorRTRunner(resolve_path(args.engine_file), resolve_path(args.plugin_library))
        self.box_publisher = rospy.Publisher(args.boxes_topic, MarkerArray, queue_size=1)
        self.lidar_publisher = None
        self.image_publisher = None
        if args.republish_lidar != 'off':
            self.lidar_publisher = rospy.Publisher(args.publish_lidar_topic, PointCloud2, queue_size=1)
        if args.republish_image == 'raw':
            self.image_publisher = rospy.Publisher(args.publish_image_topic, Image, queue_size=1)
        elif args.republish_image == 'compressed':
            self.image_publisher = rospy.Publisher(args.publish_image_topic, CompressedImage, queue_size=1)
        self.frame_count = 0

        rospy.Subscriber(args.lidar_topic, PointCloud2, self.lidar_callback, queue_size=args.queue_size, buff_size=2**24)
        if args.republish_image == 'raw':
            rospy.Subscriber(args.image_topic, Image, self.image_callback, queue_size=args.queue_size, buff_size=2**24)
        elif args.republish_image == 'compressed':
            rospy.Subscriber(args.image_topic, CompressedImage, self.image_callback, queue_size=args.queue_size, buff_size=2**24)
        rospy.loginfo(
            'IA-SSD TensorRT lidar node ready: cfg=%s lidar=%s boxes=%s repub_lidar=%s repub_image=%s',
            cfg_file,
            args.lidar_topic,
            args.boxes_topic,
            args.republish_lidar,
            args.republish_image,
        )

    def image_callback(self, msg):
        if self.image_publisher is not None:
            self.image_publisher.publish(msg)

    def publish_visual_lidar(self, msg, filtered_points):
        if self.lidar_publisher is None:
            return
        if self.args.republish_lidar == 'raw':
            self.lidar_publisher.publish(msg)
            return

        stride = max(1, int(self.args.vis_point_stride))
        vis_points = filtered_points[::stride]
        self.lidar_publisher.publish(xyzi_to_pointcloud2(vis_points, msg.header))

    def lidar_callback(self, msg):
        started = time.perf_counter()
        decode_started = time.perf_counter()
        points = pointcloud2_to_xyzi(msg)
        decode_ms = (time.perf_counter() - decode_started) * 1000.0

        preprocess_started = time.perf_counter()
        trt_points_np, filtered_points = make_trt_points(points, self.args.num_points, self.point_cloud_range, self.rng)
        trt_points = torch.from_numpy(trt_points_np).cuda()
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        infer_started = time.perf_counter()
        outputs = self.runner.infer(trt_points)
        infer_ms = (time.perf_counter() - infer_started) * 1000.0

        post_started = time.perf_counter()
        detections = postprocess_outputs(
            outputs,
            score_thresh=self.score_thresh,
            nms_thresh=self.nms_thresh,
            nms_pre_maxsize=self.nms_pre_maxsize,
            nms_post_maxsize=self.nms_post_maxsize,
        )
        post_ms = (time.perf_counter() - post_started) * 1000.0

        publish_started = time.perf_counter()
        self.publish_visual_lidar(msg, filtered_points)
        marker_array = MarkerArray()
        marker_array.markers.append(make_delete_all_marker(msg.header.frame_id, msg.header.stamp))
        for index, (box, score, label) in enumerate(
            zip(detections['pred_boxes'], detections['pred_scores'], detections['pred_labels']),
            start=1,
        ):
            marker_array.markers.append(
                make_box_marker(msg.header.frame_id, msg.header.stamp, index, box, score, label, self.args.marker_lifetime)
            )
        self.box_publisher.publish(marker_array)
        publish_ms = (time.perf_counter() - publish_started) * 1000.0
        total_ms = (time.perf_counter() - started) * 1000.0

        self.frame_count += 1
        if self.frame_count % self.args.latency_log_interval == 0:
            rospy.loginfo(
                'IA-SSD TRT frame=%d raw=%d filtered=%d det=%d decode=%.2fms prep=%.2fms infer=%.2fms post=%.2fms pub=%.2fms total=%.2fms',
                self.frame_count,
                points.shape[0],
                filtered_points.shape[0],
                detections['pred_scores'].shape[0],
                decode_ms,
                preprocess_ms,
                infer_ms,
                post_ms,
                publish_ms,
                total_ms,
            )


def main():
    rospy.init_node('iassd_trt_lidar_node', anonymous=False)
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')
    IASSDTRTLidarNode(args)
    rospy.spin()


if __name__ == '__main__':
    main()
