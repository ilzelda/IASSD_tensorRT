import argparse
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
import torch.nn as nn
from sensor_msgs.msg import CompressedImage, Image, PointCloud2
from visualization_msgs.msg import MarkerArray

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils
from iassd_postprocess import postprocess_outputs
from ros_iassd_trt_lidar_node import (
    make_box_marker,
    make_delete_all_marker,
    make_trt_points,
    pointcloud2_to_xyzi,
    xyzi_to_pointcloud2,
)


def install_spconv_import_stub():
    try:
        import spconv  # noqa: F401
        return
    except ImportError:
        pass

    class SparseConvolution(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            raise RuntimeError('이 환경에는 spconv이 설치되어 있지 않아 sparse convolution 모델은 실행할 수 없습니다.')

    class SparseModule(nn.Module):
        pass

    class SparseConvTensor:
        def __init__(self, *args, **kwargs):
            raise RuntimeError('이 환경에는 spconv이 설치되어 있지 않아 SparseConvTensor를 사용할 수 없습니다.')

    import types
    spconv_stub = types.ModuleType('spconv')
    spconv_pytorch_stub = types.ModuleType('spconv.pytorch')
    conv_stub = types.SimpleNamespace(SparseConvolution=SparseConvolution)

    for module in (spconv_stub, spconv_pytorch_stub):
        module.conv = conv_stub
        module.SparseModule = SparseModule
        module.SparseSequential = nn.Sequential
        module.SparseConvTensor = SparseConvTensor
        module.SubMConv3d = SparseConvolution
        module.SparseConv3d = SparseConvolution
        module.SparseInverseConv3d = SparseConvolution

    sys.modules['spconv'] = spconv_stub
    sys.modules['spconv.pytorch'] = spconv_pytorch_stub


install_spconv_import_stub()

from pcdet.models import build_network


class RosInferenceDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, root_path, logger):
        super().__init__(
            dataset_cfg=dataset_cfg,
            class_names=class_names,
            training=False,
            root_path=root_path,
            logger=logger,
        )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise NotImplementedError('ROS 실시간 노드는 topic callback에서 직접 입력을 구성합니다.')


class IASSDTorchRunner:
    def __init__(self, cfg_file, ckpt_file, set_cfgs):
        self.cfg_file = self._resolve_path(cfg_file).resolve()
        self.ckpt_file = self._resolve_path(ckpt_file).resolve()
        if not self.ckpt_file.exists():
            raise FileNotFoundError(f'checkpoint 파일이 없습니다: {self.ckpt_file}')

        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(TOOLS_DIR)
            cfg_from_yaml_file(str(self.cfg_file), cfg)
        finally:
            os.chdir(old_cwd)
        if set_cfgs is not None:
            cfg_from_list(set_cfgs, cfg)

        self.logger = common_utils.create_logger()
        self.dataset = RosInferenceDataset(
            dataset_cfg=cfg.DATA_CONFIG,
            class_names=cfg.CLASS_NAMES,
            root_path=ROOT_DIR,
            logger=self.logger,
        )
        self.model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=self.dataset)
        self.model.load_params_from_file(filename=str(self.ckpt_file), logger=self.logger, to_cpu=True)
        self.model.cuda()
        self.model.eval()

    @staticmethod
    def _resolve_path(path):
        path = Path(path)
        if path.is_absolute():
            return path
        root_path = ROOT_DIR / path
        if root_path.exists():
            return root_path
        return TOOLS_DIR / path

    def infer_raw(self, points):
        if not points.is_cuda:
            points = points.cuda()
        if points.dtype != torch.float32:
            points = points.float()

        batch_dict = {
            'points': points.contiguous(),
            'batch_size': 1,
        }
        with torch.no_grad():
            for cur_module in self.model.module_list:
                batch_dict = cur_module(batch_dict)
        torch.cuda.synchronize()
        return {
            'batch_cls_preds': batch_dict['batch_cls_preds'].detach().cpu().numpy(),
            'batch_box_preds': batch_dict['batch_box_preds'].detach().cpu().numpy(),
        }


def parse_args():
    parser = argparse.ArgumentParser(description='ROS1 PointCloud2 입력을 받아 IA-SSD PyTorch 결과를 MarkerArray로 publish')
    parser.add_argument('--cfg_file', type=str, default='tools/cfgs/kitti_models/IA-SSD.yaml')
    parser.add_argument('--ckpt', type=str, default='tools/IA-SSD.pth')
    parser.add_argument('--lidar_topic', type=str, default='/lidar_sync')
    parser.add_argument('--image_topic', type=str, default='/image_raw_sync')
    parser.add_argument('--boxes_topic', type=str, default='/iassd_torch/boxes')
    parser.add_argument('--publish_lidar_topic', type=str, default='/iassd_torch/lidar')
    parser.add_argument('--publish_image_topic', type=str, default='/iassd_torch/image_raw')
    parser.add_argument('--republish_lidar', type=str, choices=['raw', 'filtered', 'off'], default='off')
    parser.add_argument('--republish_image', type=str, choices=['raw', 'compressed', 'off'], default='off')
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


class IASSDTorchLidarNode:
    def __init__(self, args):
        self.args = args
        self.runner = IASSDTorchRunner(args.cfg_file, args.ckpt, args.set_cfgs)
        self.point_cloud_range = np.asarray(cfg.DATA_CONFIG.POINT_CLOUD_RANGE, dtype=np.float32)
        post_cfg = cfg.MODEL.POST_PROCESSING
        nms_cfg = post_cfg.NMS_CONFIG
        self.score_thresh = float(args.score_thresh if args.score_thresh is not None else post_cfg.SCORE_THRESH)
        self.nms_thresh = float(args.nms_thresh if args.nms_thresh is not None else nms_cfg.NMS_THRESH)
        self.nms_pre_maxsize = int(args.nms_pre_maxsize if args.nms_pre_maxsize is not None else nms_cfg.NMS_PRE_MAXSIZE)
        self.nms_post_maxsize = int(args.nms_post_maxsize if args.nms_post_maxsize is not None else nms_cfg.NMS_POST_MAXSIZE)
        self.rng = np.random.default_rng(args.seed)
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
            'IA-SSD PyTorch lidar node ready: cfg=%s ckpt=%s lidar=%s boxes=%s',
            self.runner.cfg_file,
            self.runner.ckpt_file,
            args.lidar_topic,
            args.boxes_topic,
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
        torch_points_np, filtered_points = make_trt_points(points, self.args.num_points, self.point_cloud_range, self.rng)
        torch_points = torch.from_numpy(torch_points_np).cuda()
        preprocess_ms = (time.perf_counter() - preprocess_started) * 1000.0

        infer_started = time.perf_counter()
        outputs = self.runner.infer_raw(torch_points)
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
                'IA-SSD Torch frame=%d raw=%d filtered=%d det=%d decode=%.2fms prep=%.2fms infer=%.2fms post=%.2fms pub=%.2fms total=%.2fms',
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
    rospy.init_node('iassd_torch_lidar_node', anonymous=False)
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA를 사용할 수 없습니다.')
    IASSDTorchLidarNode(args)
    rospy.spin()


if __name__ == '__main__':
    main()
