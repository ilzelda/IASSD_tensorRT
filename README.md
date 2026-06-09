# IA-SSD 실시간 추론 변환 프로젝트

이 저장소는 CVPR 2022 IA-SSD 구현을 기반으로, 3D LiDAR 객체 검출 모델을 실시간 추론 파이프라인으로 배포하기 위한 변환/검증 작업을 진행하는 프로젝트다.

현재 목표는 기존 PyTorch IA-SSD 추론 경로를 기준 성능으로 유지하면서, 같은 모델을 ONNX로 변환하고 ONNX Runtime 및 TensorRT 경로에서 실행할 수 있게 만드는 것이다. 최종적으로는 ROS1 `sensor_msgs/PointCloud2` 토픽에서 들어오는 포인트클라우드를 입력으로 받아 PyTorch, ONNX Runtime CUDA EP, ONNX Runtime TensorRT EP 또는 TensorRT plugin 경로의 latency와 throughput을 비교한다.

## 프로젝트 목표

1. PyTorch IA-SSD 모델을 ONNX로 변환한다.
2. ONNX Runtime custom op로 IA-SSD PointNet2 계열 연산을 실행한다.
3. ONNX Runtime CUDA EP와 TensorRT EP 경로를 검증한다.
4. ROS1 포인트클라우드 입력을 기존 IA-SSD 전처리 형식과 맞춘다.
5. 변환 전후 추론 속도를 같은 입력, 같은 후처리 기준으로 비교한다.

이 프로젝트의 변경 작업은 다음 네 영역 중 하나 이상을 지원한다.

- 모델 변환
- ROS1 포인트클라우드 입력
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

## 현재 기준 범위

현재 변환 작업의 1차 기준은 KITTI IA-SSD 설정이다.

- Config: `tools/cfgs/kitti_models/IA-SSD.yaml`
- 입력 포인트 형식: `(x, y, z, intensity)`
- 모델 입력 tensor: batch index가 포함된 `points`, 예시 shape `(N, 5)`
- 기본 point count: `16384`
- Batch size: 우선 `1`
- ONNX 출력: NMS 전 raw prediction
  - `batch_cls_preds`
  - `batch_box_preds`
- 후처리/NMS: 초기 단계에서는 ONNX graph 밖의 기존 Python/PyTorch 경로와 맞춘다.

상세한 custom op 변환 계획과 진행 기록은 [docs/KITTI_ONNX_CUSTOM_OPS_PLAN.md](docs/KITTI_ONNX_CUSTOM_OPS_PLAN.md)를 기준 문서로 관리한다.

## 저장소 구조

```text
IA-SSD
├── pcdet/                         # IA-SSD/OpenPCDet 기반 모델, dataset, CUDA extension
├── tools/
│   ├── cfgs/                      # KITTI/Waymo/NuScenes 모델 및 dataset config
│   ├── export_onnx.py             # PyTorch IA-SSD -> ONNX export
│   ├── validate_iassd_ort_model.py # PyTorch와 ORT raw output 비교
│   ├── benchmark_iassd_ort_cuda.py # PyTorch와 ORT CUDA EP raw forward benchmark
│   ├── iassd_ort_ops/             # ONNX Runtime custom op library
│   └── iassd_trt_plugins/         # TensorRT plugin 실험 코드
├── docs/
│   ├── INSTALL.md                 # 원본 IA-SSD 설치 참고 문서
│   ├── DEMO.md                    # 원본 IA-SSD demo 참고 문서
│   └── KITTI_ONNX_CUSTOM_OPS_PLAN.md
└── README.md
```

## 환경 준비

원본 IA-SSD/OpenPCDet 환경은 `docs/INSTALL.md`를 참고한다. 이 프로젝트의 변환/추론 작업에서는 아래 항목이 특히 중요하다.

- CUDA, GPU driver, PyTorch, ONNX, ONNX Runtime, TensorRT 버전
- `pcdet` 개발 설치 상태
- PointNet2, iou3d_nms, roiaware_pool3d 등 CUDA extension 빌드 상태
- ONNX Runtime GPU/TensorRT provider 설치 여부
- TensorRT plugin을 빌드할 수 있는 CUDA/TensorRT header 및 library 경로

기본 설치 예시는 다음과 같다.

```bash
pip install -r requirements.txt
python setup.py develop
```

재현성을 위해 benchmark report에는 hardware, CUDA device, Python, PyTorch, ONNX Runtime, point count, warmup, iteration 수를 함께 기록한다.

## PyTorch 기준 추론

기존 PyTorch 경로는 변환 후 성능과 정확도 비교를 위한 기준선이다.

```bash
cd tools
python test.py \
  --cfg_file cfgs/kitti_models/IA-SSD.yaml \
  --batch_size 16 \
  --ckpt IA-SSD.pth \
  --set MODEL.POST_PROCESSING.RECALL_MODE 'speed'
```

이 경로는 기존 OpenPCDet 평가/후처리 흐름을 그대로 사용한다.

## ONNX export

KITTI synthetic 입력 기준으로 shape와 raw output만 먼저 확인하려면 다음 명령을 사용한다.

```bash
cd tools
python export_onnx.py \
  --cfg_file cfgs/kitti_models/IA-SSD.yaml \
  --ckpt IA-SSD.pth \
  --output_file ../onnx_exports/stage1/ia_ssd_kitti.onnx \
  --num_points 16384 \
  --device cuda \
  --shape_report_file ../onnx_exports/stage1/kitti_shape_report.json \
  --dump_raw_output_file ../onnx_exports/stage1/kitti_raw_outputs.npz \
  --skip_export
```

IA-SSD custom ONNX op placeholder 경로로 export하려면 다음처럼 실행한다.

```bash
cd tools
python export_onnx.py \
  --cfg_file cfgs/kitti_models/IA-SSD.yaml \
  --ckpt IA-SSD.pth \
  --output_file ../onnx_exports/stage2/ia_ssd_kitti.onnx \
  --num_points 16384 \
  --device cuda \
  --use_iassd_custom_ops
```

현재 1차 custom op 범위는 다음과 같다.

- `IASSD::FarthestPointSampling`
- `IASSD::GatherPoints`
- `IASSD::BallQuery`
- `IASSD::GroupPoints`

## ONNX Runtime custom op 빌드

ONNX Runtime CUDA EP에서 IA-SSD custom node를 실행하기 위해 `libiassd_ort_ops.so`를 빌드한다.

```bash
cmake -S tools/iassd_ort_ops -B tools/iassd_ort_ops/build
cmake --build tools/iassd_ort_ops/build -j
```

단위 테스트 예시는 다음과 같다.

```bash
python tools/test_iassd_ort_fps.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda

python tools/test_iassd_ort_gather.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda

python tools/test_iassd_ort_ball_query.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda

python tools/test_iassd_ort_group.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda
```

## ONNX Runtime 모델 검증

PyTorch raw output과 ONNX Runtime output을 비교한다.

```bash
cd tools
python validate_iassd_ort_model.py \
  --cfg_file cfgs/kitti_models/IA-SSD.yaml \
  --ckpt IA-SSD.pth \
  --onnx_file ../onnx_exports/stage2/ia_ssd_kitti.onnx \
  --ort_op_library iassd_ort_ops/build/libiassd_ort_ops.so \
  --providers CUDAExecutionProvider \
  --num_points 16384 \
  --report_file ../onnx_exports/stage5/kitti_ort_cuda_validation.json
```

TensorRT EP session 생성만 빠르게 확인할 때는 `--session_only`와 provider 목록을 사용한다.

```bash
cd tools
python validate_iassd_ort_model.py \
  --onnx_file ../onnx_exports/stage2/ia_ssd_kitti.onnx \
  --ort_op_library iassd_ort_ops/build/libiassd_ort_ops.so \
  --trt_plugin_library iassd_trt_plugins/build/libiassd_trt_plugins.so \
  --providers TensorrtExecutionProvider,CUDAExecutionProvider \
  --session_only
```

현재 전체 IA-SSD graph의 TensorRT EP 통합은 blocker 분석 중이며, 단기 benchmark는 검증된 ORT CUDA EP custom op 경로를 우선 기준으로 둔다.

## Benchmark

PyTorch raw forward와 ORT CUDA EP raw forward를 같은 입력으로 비교한다. 현재 benchmark 범위는 전처리와 후처리를 제외하고 `batch_cls_preds`, `batch_box_preds` 생성까지다.

```bash
cd tools
python benchmark_iassd_ort_cuda.py \
  --cfg_file cfgs/kitti_models/IA-SSD.yaml \
  --ckpt IA-SSD.pth \
  --onnx_file ../onnx_exports/stage2/ia_ssd_kitti.onnx \
  --ort_op_library iassd_ort_ops/build/libiassd_ort_ops.so \
  --providers CUDAExecutionProvider \
  --num_points 16384 \
  --warmup 20 \
  --iterations 100 \
  --report_file ../onnx_exports/stage5/kitti_ort_cuda_benchmark.json
```

측정 report에는 다음 정보를 포함한다.

- hardware 및 CUDA device
- Python, PyTorch, ONNX Runtime 버전
- config, checkpoint, ONNX 파일, custom op library 경로
- provider 요청/활성 목록
- point count, warmup, iteration 수
- PyTorch raw forward latency
- ORT CUDA `session.run` latency
- 선택 시 ORT CUDA IO binding latency

최종 비교에서는 ROS subscribe, PointCloud2 decode, 전처리, 모델 추론, 후처리 시간을 가능한 한 분리해서 기록한다.

## TensorRT plugin

TensorRT plugin 실험 코드는 `tools/iassd_trt_plugins/`에 있다. 현재 `IASSD::FarthestPointSampling` plugin 1차 구현과 작은 단위 ONNX 모델 기준 parser/engine 실행 검증을 포함한다.

```bash
cmake -S tools/iassd_trt_plugins -B tools/iassd_trt_plugins/build
cmake --build tools/iassd_trt_plugins/build -j

python tools/test_iassd_trt_fps_plugin.py \
  --plugin_library tools/iassd_trt_plugins/build/libiassd_trt_plugins.so
```

전체 IA-SSD TensorRT EP 경로는 아직 최종 경로가 아니며, ORT TensorRT EP의 graph partition/resolve 문제와 나머지 custom op plugin 범위를 분리해서 진행한다.

## ROS1 입력 계획

최종 입력은 ROS1 `sensor_msgs/PointCloud2` 토픽을 우선 대상으로 한다. ROS node는 다음 단계를 분리해서 측정할 수 있게 작성한다.

- ROS subscribe 대기 및 callback 진입 시간
- `PointCloud2` decode 시간
- IA-SSD 입력 형식 `(x, y, z, intensity)` 변환 시간
- point range filtering 및 sampling 등 전처리 시간
- PyTorch 또는 ORT/TensorRT 모델 추론 시간
- score threshold, box decode, NMS 등 후처리 시간

ROS 입력 경로에서도 기존 KITTI config의 좌표계, point feature 순서, score threshold, output box 형식을 가능한 한 유지한다.

## 데이터와 weight 관리

- 학습된 weight 파일과 dataset 파일은 수정하지 않는다.
- KITTI 데이터는 기존 OpenPCDet/IA-SSD 구조와 호환되게 배치한다.
- 변환 산출물과 benchmark report는 `onnx_exports/` 아래에 저장하는 것을 기본으로 한다.
- TensorRT engine과 target machine에서 빌드한 binary는 CUDA, TensorRT, GPU architecture에 민감하므로 재현 정보를 함께 남긴다.

## 원본 IA-SSD

이 저장소의 기반 모델은 IA-SSD 공식 구현이다.

**Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds**<br>
Yifan Zhang, Qingyong Hu, Guoquan Xu, Yanxin Ma, Jianwei Wan, Yulan Guo<br>
CVPR 2022 Oral

- Paper: https://arxiv.org/abs/2203.11139
- Original repository: https://github.com/yifanzhang713/IA-SSD

```bibtex
@inproceedings{zhang2022not,
  title={Not All Points Are Equal: Learning Highly Efficient Point-based Detectors for 3D LiDAR Point Clouds},
  author={Zhang, Yifan and Hu, Qingyong and Xu, Guoquan and Ma, Yanxin and Wan, Jianwei and Guo, Yulan},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  year={2022}
}
```

## License

원본 IA-SSD 구현은 [Apache 2.0 license](LICENSE)를 따른다.
