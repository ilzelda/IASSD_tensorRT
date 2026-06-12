# KITTI IA-SSD ONNX Custom Ops 변환 계획

## 목적

`tools/cfgs/kitti_models/IA-SSD.yaml`을 기준 설정으로 삼아 IA-SSD 모델을 ONNX로 변환하고, 이후 ONNX Runtime 및 TensorRT 경로에서 실행할 수 있게 custom op 변환 체계를 구축한다.

이 문서는 다음 작업 영역을 지원한다.

- 모델 변환
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

ROS1 포인트클라우드 입력은 이번 문서의 직접 범위는 아니지만, 최종 입력 형식은 ROS1 `sensor_msgs/PointCloud2`에서 얻은 KITTI 형식 포인트 `(x, y, z, intensity)`와 호환되도록 유지한다.

## 기준 설정

- Config: `tools/cfgs/kitti_models/IA-SSD.yaml`
- Dataset base config: `tools/cfgs/dataset_configs/kitti_dataset.yaml`
- 입력 feature: `x, y, z, intensity`
- 모델 입력 tensor: batch index가 붙은 `points`, shape 예시는 `(N, 5)`
- 기본 point count: `16384`
- Batch size: 우선 `1`
- Export output: NMS 전 raw prediction
  - `batch_cls_preds`
  - `batch_box_preds`
- Post-processing/NMS: 초기에는 ONNX 밖에서 PyTorch/Python 경로로 유지

KITTI config를 먼저 쓰는 이유는 Waymo config보다 입력 포인트 수와 sampling 수가 작아 custom op 디버깅, parity test, TensorRT plugin 검증이 가볍기 때문이다. 단, KITTI config에는 KITTI로 학습된 checkpoint가 필요하다. Waymo checkpoint를 그대로 쓰면 shape가 일부 맞더라도 class 의미, feature 수, point cloud range, box mean size가 맞지 않는다.

## 실행 환경 경계

- 서버에서 수행해도 되는 범위:
  - PyTorch checkpoint를 ONNX로 export
  - ONNX graph/checker 검증
  - custom op source 구조 정리
  - custom op 단위 테스트용 소스 작성
- target machine에서 수행하는 것이 안전한 범위:
  - ONNX Runtime custom op 공유 라이브러리 최종 빌드
  - TensorRT plugin 최종 빌드
  - TensorRT engine/cache 생성
  - ORT CUDA EP, ORT TensorRT EP, TensorRT plugin 로딩 검증
  - 최종 latency benchmark

이유는 ONNX Runtime, CUDA, TensorRT, GPU architecture, driver ABI가 runtime binary와 TensorRT engine 생성 결과에 직접 영향을 주기 때문이다. 다만 source-level kernel 공용화는 target-specific binary를 만들지 않으므로 서버에서 진행해도 된다.

## 현재 blocker

`torch.onnx.export(..., dynamo=True)`는 IA-SSD의 PointNet2 PyTorch extension 호출을 표준 ONNX graph로 변환하지 못한다.

대표 실패 지점:

```text
pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda.PyCapsule.farthest_point_sampling_wrapper
```

따라서 단순한 export 명령 수정으로는 해결되지 않는다. PyTorch export 시점에는 custom ONNX node를 생성하고, 실행 시점에는 ONNX Runtime custom op 또는 TensorRT plugin이 해당 node를 처리해야 한다.

## 목표 구조

```text
PyTorch IA-SSD
  -> ONNX graph + IA-SSD custom nodes
  -> ONNX Runtime custom op library
  -> TensorRT plugin library
  -> TensorRT engine
```

## 1차 custom op 범위

KITTI IA-SSD inference 경로에서 우선 필요한 PointNet2 batch op를 1차 범위로 둔다.

| Op | PyTorch 위치 | 예상 ONNX custom op | 비고 |
| --- | --- | --- | --- |
| FarthestPointSampling | `pointnet2_batch/pointnet2_utils.py` | `IASSD::FarthestPointSampling` | 첫 blocker |
| GatherPoints | `pointnet2_batch/pointnet2_utils.py` | `IASSD::GatherPoints` | sampled index로 xyz/features gather |
| BallQuery | `pointnet2_batch/pointnet2_utils.py` | `IASSD::BallQuery` | radius/nsample attribute 필요 |
| GroupPoints | `pointnet2_batch/pointnet2_utils.py` | `IASSD::GroupPoints` | grouping된 feature 생성 |

2차 후보:

- `FurthestPointSamplingWithDist`
- `ThreeNN`
- `ThreeInterpolate`
- `BallQueryDilated`
- `iou3d_nms` 계열 NMS op

초기 export는 NMS 전 raw prediction을 출력하므로 `iou3d_nms`는 바로 ONNX graph에 넣지 않는다.

## Op schema 초안

### `IASSD::FarthestPointSampling`

Inputs:

- `xyz`: `float32[B, N, 3]`

Attributes:

- `npoint`: int

Outputs:

- `idx`: `int32[B, npoint]`

### `IASSD::GatherPoints`

Inputs:

- `features`: `float32[B, C, N]`
- `idx`: `int32[B, npoint]`

Outputs:

- `output`: `float32[B, C, npoint]`

### `IASSD::BallQuery`

Inputs:

- `xyz`: `float32[B, N, 3]`
- `new_xyz`: `float32[B, npoint, 3]`

Attributes:

- `radius`: float
- `nsample`: int

Outputs:

- `idx`: `int32[B, npoint, nsample]`

### `IASSD::GroupPoints`

Inputs:

- `features`: `float32[B, C, N]`
- `idx`: `int32[B, npoint, nsample]`

Outputs:

- `output`: `float32[B, C, npoint, nsample]`

## 단계별 실행 계획

### 1단계: KITTI export 재현 기준 고정

산출물:

- KITTI용 export 명령 문서화
- synthetic input 기준 shape 기록
- PyTorch raw output 저장 스크립트 또는 옵션

검증:

- `tools/export_onnx.py --cfg_file cfgs/kitti_models/IA-SSD.yaml --num_points 16384` 실행 시 첫 custom op blocker가 재현되어야 한다.
- PyTorch forward에서 NMS 전 `batch_cls_preds`, `batch_box_preds` shape를 기록한다.

진행 기록:

- 상태: 완료
- 완료일: 2026-06-04
- `tools/export_onnx.py`에 `--shape_report_file`, `--dump_raw_output_file`, `--skip_export` 옵션을 추가했다.
- KITTI synthetic 입력으로 raw forward를 실행해 shape report와 raw output NPZ를 생성했다.
- 검증 산출물은 `onnx_exports/stage1/` 아래에 생성하며, 이 디렉터리는 git 추적 대상에서 제외한다.

검증 명령:

```bash
docker run --rm --gpus all \
  -v /home/tisc/IASSD_tensorRT:/workspace/IA-SSD \
  ia-ssd-export \
  "cd /workspace/IA-SSD/tools && python3 export_onnx.py \
    --cfg_file cfgs/kitti_models/IA-SSD.yaml \
    --ckpt IA-SSD.pth \
    --output_file ../onnx_exports/stage1/ia_ssd_kitti.onnx \
    --num_points 16384 \
    --device cuda \
    --opset_version 17 \
    --shape_report_file ../onnx_exports/stage1/kitti_shape_report.json \
    --dump_raw_output_file ../onnx_exports/stage1/kitti_raw_outputs.npz \
    --skip_export"
```

기록된 shape:

- 입력 `points`: `(16384, 5)`, `torch.float32`, `cuda:0`
- `batch_cls_preds`: `(256, 3)`, `torch.float32`, `cuda:0`
- `batch_box_preds`: `(256, 7)`, `torch.float32`, `cuda:0`

ONNX export blocker 재현:

- 같은 KITTI config로 `--skip_export` 없이 실행했을 때 `FarthestPointSampling` 내부의 `pointnet2_batch_cuda.PyCapsule.farthest_point_sampling_wrapper`에서 graph break가 재현됐다.
- PyTorch exporter가 TorchScript fallback까지 시도한 뒤 프로세스가 exit code `139`로 종료됐다.
- 다음 단계는 이 PyCapsule 호출을 직접 tracing하지 않고 `IASSD::FarthestPointSampling` custom ONNX node로 내보내는 것이다.

### 2단계: PyTorch export 중 custom ONNX node 생성

산출물:

- export 중 PointNet2 CUDA extension을 직접 호출하지 않는 symbolic 경로
- `IASSD::FarthestPointSampling` node가 들어간 ONNX 생성
- 이후 `GatherPoints`, `BallQuery`, `GroupPoints` node 추가

검증:

- ONNX graph가 생성되어야 한다.
- `onnx.checker.check_model`이 통과해야 한다.
- custom op node의 input/output shape와 attribute가 문서 schema와 일치해야 한다.

진행 기록:

- 상태: 완료
- 완료일: 2026-06-04
- PyTorch 2.5.1의 공개 `torch.onnx.export(..., dynamo=True)` 경로에는 `custom_translation_table`과 `torch.onnx.ops`가 없어, `torch.onnx._internal.exporter._core.export`와 내부 `ONNXRegistry`를 사용하는 placeholder 경로를 추가했다.
- `pcdet/ops/onnx_custom_ops.py`에 `iassd::farthest_point_sampling` PyTorch custom op와 `IASSD::FarthestPointSampling` ONNXScript 변환 함수를 추가했다.
- `FarthestPointSampling.forward`는 일반 PyTorch 실행에서는 기존 CUDA extension을 그대로 사용하고, export placeholder context 안에서만 custom op placeholder를 반환한다.
- `tools/export_onnx.py`에 `--use_iassd_custom_ops` 옵션을 추가해 IA-SSD custom op registry를 사용하는 export 경로를 선택할 수 있게 했다.
- `pcdet/ops/onnx_custom_ops.py`에 `iassd::gather_points` PyTorch custom op와 `IASSD::GatherPoints` ONNXScript 변환 함수를 추가했다.
- `GatherOperation.forward`는 일반 PyTorch 실행에서는 기존 CUDA extension을 그대로 사용하고, export placeholder context 안에서만 custom op placeholder를 반환한다.
- `pcdet/ops/onnx_custom_ops.py`에 `iassd::ball_query` PyTorch custom op와 `IASSD::BallQuery` ONNXScript 변환 함수를 추가했다.
- `BallQuery.forward`는 일반 PyTorch 실행에서는 기존 CUDA extension을 그대로 사용하고, export placeholder context 안에서만 custom op placeholder를 반환한다.
- `pcdet/ops/onnx_custom_ops.py`에 `iassd::group_points` PyTorch custom op와 `IASSD::GroupPoints` ONNXScript 변환 함수를 추가했다.
- `GroupingOperation.forward`는 일반 PyTorch 실행에서는 기존 CUDA extension을 그대로 사용하고, export placeholder context 안에서만 custom op placeholder를 반환한다.
- `IASSD_backbone.py`의 `SAVE_SAMPLE_LIST` 설정 조회를 `__init__` 시점의 plain bool 속성으로 옮겨 forward 중 `EasyDict.get(...)` 호출을 제거했다.
- `IASSD_head.py`의 `self.forward_ret_dict = ret_dict` 모듈 속성 mutation은 export placeholder context 안에서만 건너뛰게 했다.
- `PointnetSAModuleMSG_WithSampling`에서 sampling index가 하나뿐인 경우 불필요한 `torch.cat([sample_idx], dim=-1)`를 생략해 ONNX 변환 단계의 single-element `Concat` 문제를 우회했다.
- ONNXScript lowering 함수에 `FLOAT[...]`, `INT32[...]` 입출력 annotation을 추가해 custom op 출력이 후속 표준 ONNX op의 tensor 입력으로 정상 연결되게 했다.
- PyTorch 2.5.1과 ONNXScript 0.1.0 조합에서 `ReduceSum.keepdims=False`가 bool-valued INT attribute로 생성되어 저장이 실패하므로, ONNX IR 저장 직전에 bool-valued INT attribute를 `0/1`로 보정한다.
- ONNX Runtime이 local function wrapper를 inline할 때 `IASSD` domain opset import가 필요하므로, 저장 직전에 top-level model opset import에 `IASSD=1`을 추가한다.

검증 명령:

```bash
docker run --rm --gpus all \
  -v /home/tisc/IASSD_tensorRT:/workspace/IA-SSD \
  ia-ssd-export \
  "cd /workspace/IA-SSD/tools && python3 export_onnx.py \
    --cfg_file cfgs/kitti_models/IA-SSD.yaml \
    --ckpt IA-SSD.pth \
    --output_file ../onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset.onnx \
    --num_points 16384 \
    --device cuda \
    --opset_version 17 \
    --shape_report_file ../onnx_exports/stage2/kitti_with_iassd_opset_shape_report.json \
    --use_iassd_custom_ops"
```

검증 결과:

- `FarthestPointSampling`의 기존 blocker였던 `pointnet2_batch_cuda.PyCapsule.farthest_point_sampling_wrapper`는 더 이상 첫 실패 지점이 아니다.
- `GatherPoints`의 기존 blocker였던 `pointnet2_batch_cuda.PyCapsule.gather_points_wrapper`도 더 이상 첫 실패 지점이 아니다.
- `BallQuery`의 기존 blocker였던 `pointnet2_batch_cuda.PyCapsule.ball_query_wrapper`도 더 이상 첫 실패 지점이 아니다.
- `GroupPoints`의 기존 blocker였던 `pointnet2_batch_cuda.PyCapsule.group_points_wrapper`도 더 이상 첫 실패 지점이 아니다.
- 1차 custom op 범위의 PyCapsule blocker는 모두 지나갔다.
- `IASSD_backbone.py`의 `self.model_cfg.SA_CONFIG.get('SAVE_SAMPLE_LIST', False)` blocker도 더 이상 첫 실패 지점이 아니다.
- `IASSD_head.py`의 `self.forward_ret_dict = ret_dict` 모듈 속성 mutation blocker도 더 이상 첫 실패 지점이 아니다.
- `torch.export` graph capture는 내부 Dynamo 경로에서 성공했고, exporter는 ONNX graph 변환 단계로 진입했다.
- ONNX graph 변환과 파일 저장이 완료됐다.
- `onnx.checker.check_model`이 통과했다.
- 생성된 ONNX output shape는 `batch_cls_preds: [256, 3]`, `batch_box_preds: [256, 7]`이다.
- 현재 graph는 main graph에서 `this::onnx_*` local function을 호출하고, 각 function body 안에 `IASSD::FarthestPointSampling`, `IASSD::GatherPoints`, `IASSD::BallQuery`, `IASSD::GroupPoints` custom node가 들어가는 구조다.
- top-level opset import에 `IASSD=1`이 포함된다.
- ONNX Runtime 1.20.1에서 custom op library 없이 session을 생성하면 `IASSD:FarthestPointSampling(-1) is not a registered function/op`로 실패한다. 이는 wrapper/opset 문제가 아니라 custom op kernel 미등록 단계까지 도달했다는 뜻이다.
- 따라서 ORT custom op 경로에서는 즉시 wrapper 평탄화를 진행하지 않고, custom op library 구현 단계로 넘어간다.
- TensorRT plugin 경로에서 ONNX parser가 local function wrapper를 처리하지 못하면 그때 `this::onnx_*` wrapper를 main graph의 직접 `IASSD::*` custom node로 평탄화한다.

주의:

- PyTorch 2.5 exporter에서 custom translation API가 부족하면 Docker base를 PyTorch 2.6 이상으로 올리는 별도 커밋을 검토한다.
- legacy exporter fallback은 장기 목표가 아니며, TensorRT plugin까지 고려하면 custom node schema를 명시하는 방향을 우선한다.

### 3단계: CUDA kernel 공용화

산출물:

- PyTorch extension, ONNX Runtime custom op, TensorRT plugin이 공유할 CUDA kernel 인터페이스
- 기존 `pcdet/ops/pointnet2/pointnet2_batch/src/*.cu` kernel 호출부 분리

검증:

- 기존 PyTorch IA-SSD forward 결과가 변경 전과 동일해야 한다.
- extension rebuild가 Docker image build 단계에서 통과해야 한다.

진행 기록:

- 상태: 부분 완료
- 진행일: 2026-06-04
- `pcdet/ops/pointnet2/pointnet2_batch/src/sampling_gpu_raw.h`를 추가해 sampling 계열 포인터 기반 CUDA launcher 선언을 Torch 헤더 의존성 없이 분리했다.
- `sampling_gpu.h`는 기존 PyTorch wrapper API를 유지하면서 새 raw 헤더를 include하도록 정리했다.
- `sampling_gpu.cu`는 더 이상 `sampling_gpu.h`를 직접 include하지 않고 `sampling_gpu_raw.h`만 include한다.
- 이 변경으로 `FarthestPointSampling` ORT custom op와 이후 TensorRT plugin에서 기존 `farthest_point_sampling_kernel_launcher(...)`를 Torch 의존성 없이 참조할 수 있는 기반을 만들었다.

검증 결과:

- Docker `ia-ssd-export` 이미지에서 `python3 setup.py develop --no-deps`가 통과했다.
- `import torch` 이후 `pcdet.ops.pointnet2.pointnet2_batch.pointnet2_batch_cuda` import가 통과했다.
- KITTI synthetic 입력으로 `tools/export_onnx.py --skip_export` forward 검증이 통과했다.
- forward 검증 shape는 `batch_cls_preds: (256, 3)`, `batch_box_preds: (256, 7)`로 기존 기준과 동일하다.
- 이번 검증은 기존 PyTorch extension 빌드/로드 안정성 확인이며, ORT custom op `.so` 빌드는 다음 단계에서 별도 수행한다.

### 4단계: ONNX Runtime custom op 구현

산출물:

- `libiassd_ort_ops.so`
- `RegisterCustomOps` entrypoint
- ORT CUDAExecutionProvider용 custom op kernel
- custom op별 단위 테스트

검증:

- PyTorch custom op 출력과 ORT custom op 출력 비교
- index 출력은 exact match
- float 출력은 tolerance 기반 비교
- `onnxruntime.InferenceSession(..., providers=["CUDAExecutionProvider"])`에서 custom op library 로드

진행 기록:

- 상태: 완료
- 진행일: 2026-06-04
- `tools/iassd_ort_ops/`에 ONNX Runtime custom op 공유 라이브러리 소스 골격을 추가했다.
- `tools/iassd_ort_ops/farthest_point_sampling_op.cc`에 `IASSD::FarthestPointSampling` CUDAExecutionProvider custom op를 추가했다.
- 해당 custom op는 `npoint_i` attribute를 우선 읽고, 수동 작성 ONNX graph 호환을 위해 `npoint` fallback도 지원한다.
- FPS kernel 실행은 3단계에서 분리한 `sampling_gpu_raw.h`의 `farthest_point_sampling_kernel_launcher(...)`를 재사용한다.
- `tools/test_iassd_ort_fps.py`에 단일 custom node ONNX graph를 생성하고 PyTorch extension 출력과 ORT custom op 출력을 exact match로 비교하는 단위 테스트를 추가했다.
- target `ia-ssd-target:latest` 이미지에서 `IASSD::FarthestPointSampling` ORT custom op 단위 테스트가 통과했다.
- 같은 shared library에 `IASSD::GatherPoints` CUDAExecutionProvider custom op를 추가했다.
- GatherPoints kernel 실행은 3단계에서 분리한 `sampling_gpu_raw.h`의 `gather_points_kernel_launcher_fast(...)`를 재사용한다.
- `tools/test_iassd_ort_gather.py`에 단일 custom node ONNX graph를 생성하고 PyTorch extension 출력과 ORT custom op 출력을 exact match로 비교하는 단위 테스트를 추가했다.
- target `ia-ssd-target:latest` 이미지에서 `IASSD::FarthestPointSampling` regression과 `IASSD::GatherPoints` ORT custom op 단위 테스트가 모두 통과했다.
- `pcdet/ops/pointnet2/pointnet2_batch/src/ball_query_gpu_raw.h`를 추가해 BallQuery 계열 포인터 기반 CUDA launcher 선언을 Torch 헤더 의존성 없이 분리했다.
- 같은 shared library에 `IASSD::BallQuery` CUDAExecutionProvider custom op를 추가했다.
- BallQuery kernel 실행은 새 raw header의 `ball_query_kernel_launcher_fast(...)`를 재사용한다.
- `tools/test_iassd_ort_ball_query.py`에 단일 custom node ONNX graph를 생성하고 PyTorch extension 출력과 ORT custom op 출력을 exact match로 비교하는 단위 테스트를 추가했다.
- target `ia-ssd-target:latest` 이미지에서 `IASSD::FarthestPointSampling`, `IASSD::GatherPoints`, `IASSD::BallQuery` ORT custom op 단위 테스트가 모두 통과했다.
- `pcdet/ops/pointnet2/pointnet2_batch/src/group_points_gpu_raw.h`를 추가해 GroupPoints 계열 포인터 기반 CUDA launcher 선언을 Torch 헤더 의존성 없이 분리했다.
- 같은 shared library에 `IASSD::GroupPoints` CUDAExecutionProvider custom op를 추가했다.
- GroupPoints kernel 실행은 새 raw header의 `group_points_kernel_launcher_fast(...)`를 재사용한다.
- `tools/test_iassd_ort_group.py`에 단일 custom node ONNX graph를 생성하고 PyTorch extension 출력과 ORT custom op 출력을 exact match로 비교하는 단위 테스트를 추가했다.
- target `ia-ssd-target:latest` 이미지에서 1차 custom op 범위인 `IASSD::FarthestPointSampling`, `IASSD::GatherPoints`, `IASSD::BallQuery`, `IASSD::GroupPoints` 단위 테스트가 모두 통과했다.
- target GPU는 Orin, compute capability는 `8.7`이다. 기존 Docker 기본값에는 `8.7`이 없어 PyTorch extension 실행 시 `no kernel image is available for execution on the device`가 발생했으므로 `docker/Dockerfile.target`과 `docker/Dockerfile`의 `TORCH_CUDA_ARCH_LIST` 기본값에 `8.7`을 추가했다.
- `tools/iassd_ort_ops/CMakeLists.txt`는 `TORCH_CUDA_ARCH_LIST`를 읽어 CMake CUDA architecture 값으로 변환한다.
- Jetson용 `onnxruntime-gpu==1.16.0` wheel은 import와 `CUDAExecutionProvider`, `TensorrtExecutionProvider` 확인이 통과했지만 pip wheel 안에 custom op 빌드용 C++ header와 `libonnxruntime.so`가 없었다.
- custom op 빌드는 ONNX Runtime `v1.16.0` source header를 `ONNXRUNTIME_INCLUDE_DIR`로 넘기고, `libonnxruntime.so` 명시 링크 없이 ORT가 전달하는 `OrtApi`로 동작하게 했다.
- `tools/iassd_ort_ops/farthest_point_sampling_op.cc`는 `ORT_API_MANUAL_INIT`과 `Ort::InitApi(api)`를 사용해 pip wheel 환경의 `OrtGetApiBase` 미해결 심볼 문제를 해결했다.
- `pcdet/ops/onnx_custom_ops.py`는 target PyTorch 2.0 환경에서 `onnxscript`와 `torch.library.custom_op`가 없어도 일반 PyTorch forward가 동작하도록 export 전용 의존성을 fallback 처리했다.
- `tools/test_iassd_ort_fps.py`는 repo root에서 직접 실행해도 `pcdet`를 import할 수 있게 경로 설정을 자체 처리한다.

빌드 및 검증 명령:

```bash
cd /workspace/IA-SSD
git config --global --add safe.directory /workspace/IA-SSD
wget -q https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl \
  -O /tmp/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
python3 -m pip install /tmp/onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl
wget -q https://github.com/microsoft/onnxruntime/archive/refs/tags/v1.16.0.tar.gz \
  -O /tmp/onnxruntime-v1.16.0.tar.gz
mkdir -p /tmp/ort_src
tar -xzf /tmp/onnxruntime-v1.16.0.tar.gz -C /tmp/ort_src --strip-components=1
TORCH_CUDA_ARCH_LIST=8.7 MAX_JOBS=2 python3 setup.py build_ext --force --inplace
TORCH_CUDA_ARCH_LIST=8.7 cmake -S tools/iassd_ort_ops -B /tmp/iassd_ort_ops_build \
  -DONNXRUNTIME_INCLUDE_DIR=/tmp/ort_src/include/onnxruntime/core/session
cmake --build /tmp/iassd_ort_ops_build -j2
python3 tools/test_iassd_ort_fps.py \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --batch_size 1 \
  --num_points 512 \
  --npoint 64 \
  --device cuda
python3 tools/test_iassd_ort_gather.py \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --batch_size 1 \
  --channels 4 \
  --num_points 512 \
  --npoint 64 \
  --device cuda
python3 tools/test_iassd_ort_ball_query.py \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --batch_size 1 \
  --num_points 512 \
  --npoint 64 \
  --radius 0.2 \
  --nsample 16 \
  --device cuda
python3 tools/test_iassd_ort_group.py \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --batch_size 1 \
  --channels 4 \
  --num_points 512 \
  --npoint 64 \
  --nsample 16 \
  --device cuda
```

검증 결과:

- 실행일: 2026-06-04
- Docker image: `ia-ssd-target:latest`
- PyTorch: `2.0.0a0+ec3941ad.nv23.02`
- ONNX Runtime GPU: `1.16.0`
- ORT providers: `TensorrtExecutionProvider`, `CUDAExecutionProvider`, `CPUExecutionProvider`
- GPU: Orin, compute capability `(8, 7)`
- 테스트 입력: `input_shape=(1, 512, 3)`, `npoint=64`
- 테스트 출력: `output_shape=(1, 64)`
- 결과: `IASSD::FarthestPointSampling ORT custom op 테스트 통과`
- GatherPoints 테스트 입력: `features_shape=(1, 4, 512)`, `idx_shape=(1, 64)`
- GatherPoints 테스트 출력: `output_shape=(1, 4, 64)`
- 결과: `IASSD::GatherPoints ORT custom op 테스트 통과`
- BallQuery 테스트 입력: `xyz_shape=(1, 512, 3)`, `new_xyz_shape=(1, 64, 3)`, `radius=0.2`, `nsample=16`
- BallQuery 테스트 출력: `idx_shape=(1, 64, 16)`
- 결과: `IASSD::BallQuery ORT custom op 테스트 통과`
- GroupPoints 테스트 입력: `features_shape=(1, 4, 512)`, `idx_shape=(1, 64, 16)`
- GroupPoints 테스트 출력: `output_shape=(1, 4, 64, 16)`
- 결과: `IASSD::GroupPoints ORT custom op 테스트 통과`

주의:

- 테스트는 `CUDAExecutionProvider`를 요구하므로 Python 패키지는 CPU 전용 `onnxruntime`이 아니라 CUDA EP가 포함된 ONNX Runtime 빌드 또는 `onnxruntime-gpu` 계열이어야 한다.
- custom op ABI는 ONNX Runtime 버전의 영향을 받으므로 최종 `.so`는 target machine의 ONNX Runtime/CUDA 환경에서 다시 빌드한다.

### 5단계: 모델 단위 ORT 검증

산출물:

- KITTI IA-SSD ONNX graph
- ORT 실행 스크립트
- raw prediction parity test

검증:

- PyTorch raw `batch_cls_preds`, `batch_box_preds`와 ORT output 비교
- shape 일치
- max/mean absolute error 기록
- 최종 post-processing은 Python/PyTorch 경로로 붙여 box 결과 비교

진행 기록:

- 상태: 진행 중
- 진행일: 2026-06-04
- `tools/validate_iassd_ort_model.py`를 추가해 KITTI IA-SSD PyTorch raw output과 ONNX Runtime output의 shape 및 absolute error를 비교할 수 있게 했다.
- target image에는 `spconv`이 없지만 IA-SSD KITTI 경로는 sparse convolution을 직접 만들지 않으므로, 검증 스크립트에서 import 전용 `spconv` stub을 설치해 `pcdet.models` import를 통과시킨다.
- OpenPCDet config의 `_BASE_CONFIG_`가 `tools/` 기준 상대경로를 사용하므로 config load 시 작업 디렉터리를 일시적으로 `tools/`로 전환한다.
- 전체 모델 ORT 실행 중 `GatherPoints`의 index 입력이 깨져 CUDA illegal memory access가 발생했다. 원인은 `TopK`/`Cast`에서 만들어진 index tensor를 custom CUDA op에서 device pointer로 가정한 것이다.
- `IASSD::GatherPoints`와 `IASSD::GroupPoints`는 index 입력을 `OrtMemTypeCPUInput`으로 명시하고, kernel 호출 직전에 CPU index를 int32 CUDA buffer로 복사하도록 수정했다.
- 이 수정 이후 1차 custom op 단위 테스트 4개가 모두 다시 통과했고, KITTI IA-SSD ONNX graph가 `CUDAExecutionProvider`에서 끝까지 실행됐다.
- `tools/trace_iassd_ort_parity.py`를 추가해 ONNX graph에 중간 tensor를 임시 output으로 붙이고 PyTorch `encoder_xyz`, `encoder_features`, `sa_ins_preds`, `TopK` index와 비교할 수 있게 했다.
- 중간 tensor trace 결과, 첫 두 FPS/Gather sampling과 첫 SA feature는 exact 또는 매우 작은 오차로 일치했다.
- 최초로 큰 좌표 차이가 생기는 지점은 `TopK` 기반 sampling 이후의 `gather_points_2`이다.
- `sa_ins_score_1`과 ONNX `sigmoid`는 max abs `1.2194737792015076e-05`, mean abs `5.297447280838696e-08`로 거의 일치하지만, `topk_indices_0`과 ONNX `_to_copy_1`은 달랐다.
- 따라서 현재 큰 box 오차의 직접 원인은 custom op kernel의 큰 수치 오류보다는 PyTorch `torch.topk`와 ONNX Runtime `TopK`가 동점 또는 근접 score에서 다른 index를 선택하는 sampling divergence로 판단한다.
- `pcdet/ops/pointnet2/pointnet2_batch/pointnet2_modules.py`의 `TopK` sampling 직전에 index 기반 tie-breaker를 score에서 빼는 deterministic index 우선순위를 추가했다.
- target image의 PyTorch `2.0.0a0+ec3941ad.nv23.02`에는 stage2 export에서 사용한 `torch.onnx._internal.exporter` API가 없어, target 안에서는 tie-breaker가 포함된 ONNX를 재export할 수 없었다.
- 기존 ONNX graph에 사후로 `Sub(score, index_bias)`를 삽입하는 방식도 시험했지만, 첫 FPS/Gather output까지 깨지는 부작용이 확인되어 폐기했다.
- PyTorch 2.5 계열 export 환경에서 tie-breaker가 포함된 stage2 ONNX를 재생성했고, target에서 `tools/trace_iassd_ort_parity.py`로 확인했다.
- `eps=1e-7`에서는 수치 차이를 충분히 이기지 못해 `topk_indices_0/_to_copy_1`, `topk_indices_1/_to_copy_2`가 여전히 달랐다.
- tie-breaker 계수를 `1e-5`로 올린 뒤 stage2 ONNX를 다시 생성했다.
- 새 ONNX에는 `sigmoid -> Sub -> TopK`, `sigmoid_1 -> Sub -> TopK` 경로가 반영되어 있다.
- `eps=1e-5` 재export ONNX의 trace 결과, `topk_indices_0/_to_copy_1`, `topk_indices_1/_to_copy_2`는 exact match가 됐다.
- 일반 ONNX graph에서는 trace graph보다 final output 오차가 크게 나오는 현상이 있었다. 원인은 ORT CUDA EP 표준 op가 ORT compute stream에서 실행되는 반면, custom CUDA launcher가 기본 stream에 kernel을 launch해 input readiness 순서가 보장되지 않는 race로 판단했다.
- `sampling_gpu_raw.h`, `ball_query_gpu_raw.h`, `group_points_gpu_raw.h`와 각 CUDA launcher에 `cudaStream_t` 인자를 추가하고, ORT custom op에서 `Ort::KernelContext::GetGPUComputeStream()`을 전달하도록 수정했다.
- PyTorch extension wrapper는 기본 stream 인자를 유지해 기존 호출 형태와 호환된다.

검증 명령:

```bash
cd /workspace/IA-SSD
IASSD_DEBUG_RANGE_CHECK=1 python3 tools/validate_iassd_ort_model.py \
  --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
  --ckpt tools/IA-SSD.pth \
  --onnx_file onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset.onnx \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --num_points 16384 \
  --device cuda \
  --providers CUDAExecutionProvider \
  --report_file onnx_exports/stage5/kitti_ort_cuda_parity_report.json
```

중간 tensor trace 명령:

```bash
cd /workspace/IA-SSD
IASSD_DEBUG_RANGE_CHECK=1 python3 tools/trace_iassd_ort_parity.py \
  --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
  --ckpt tools/IA-SSD.pth \
  --onnx_file onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset.onnx \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --num_points 16384 \
  --providers CUDAExecutionProvider \
  --report_file onnx_exports/stage5/kitti_ort_cuda_trace_report.json
```

검증 결과:

- 실행일: 2026-06-04
- Docker image: `ia-ssd-target:latest`
- 입력 `points`: `(16384, 5)`, `torch.float32`, `cuda:0`
- `batch_cls_preds`: PyTorch `(256, 3)`, ORT `(256, 3)`, shape 일치
- `batch_cls_preds` 오차: max abs `2.86238956451416`, mean abs `0.03077808301895857`
- `batch_box_preds`: PyTorch `(256, 7)`, ORT `(256, 7)`, shape 일치
- `batch_box_preds` 오차: max abs `50.89475631713867`, mean abs `0.12930782358307624`
- 결과 해석: shape와 runtime은 통과했지만 TopK index divergence 때문에 numerical parity 오차가 컸다.

재검증 결과:

- 실행일: 2026-06-05
- Docker image: `ia-ssd-target:latest`
- ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset.onnx`
- custom op library: `/tmp/iassd_ort_ops_build/libiassd_ort_ops.so`
- 입력 `points`: `(16384, 5)`, `torch.float32`, `cuda:0`
- 단위 테스트: `IASSD::FarthestPointSampling`, `IASSD::GatherPoints`, `IASSD::BallQuery`, `IASSD::GroupPoints` 모두 통과
- trace: `topk_indices_0/_to_copy_1`, `topk_indices_1/_to_copy_2` exact match
- trace final `batch_cls_preds` 오차: max abs `0.0108184814453125`, mean abs `0.000836263100306193`
- trace final `batch_box_preds` 오차: max abs `0.0018192529678344727`, mean abs `8.585689855473382e-05`
- stream 수정 후 일반 graph final `batch_cls_preds` 오차: max abs `0.00971221923828125`, mean abs `0.0008820537477731705`
- stream 수정 후 일반 graph final `batch_box_preds` 오차: max abs `0.0028738975524902344`, mean abs `9.923223218980379e-05`
- report: `onnx_exports/stage5/kitti_ort_cuda_trace_report_after_eps1e5.json`
- report: `onnx_exports/stage5/kitti_ort_cuda_parity_report_stream_fixed.json`

남은 작업:

- CUDA EP raw output parity는 다음 단계로 넘어갈 수 있는 수준으로 개선됐다.
- index CPU 왕복 복사는 정확도 검증 우선의 임시 안정화 경로다. benchmark 단계 전에는 ORT CUDA tensor memory 처리 방식에 맞춰 device index 경로를 최적화할지 결정한다.
- custom op 내부의 `cudaDeviceSynchronize()`는 정확도 검증 중 안정성을 위해 남아 있다. benchmark 전에는 ORT stream dependency를 유지하면서 op별 불필요한 device-wide sync를 줄이는 최적화를 검토한다.

### 6단계: TensorRT plugin 구현 및 direct TensorRT engine 빌드

산출물:

- `libiassd_trt_plugins.so`
- TensorRT plugin creator
- op별 plugin serialization/deserialization
- TensorRT parser용 ONNX sanitizer
- FP32 TensorRT engine

검증:

- TensorRT parser에서 plugin library 로드
- IA-SSD custom op가 TensorRT plugin으로 매핑되는지 확인
- direct TensorRT engine build 성공
- ORT CUDA path와 TensorRT path output 비교
- FP16은 FP32 parity가 안정화된 뒤 진행

진행 기록:

- 상태: direct TensorRT FP32 engine build 성공
- 진행일: 2026-06-05
- target image의 ONNX Runtime GPU wheel에서 작은 표준 ONNX `MatMul` 모델은 `TensorrtExecutionProvider,CUDAExecutionProvider`로 session 생성과 실행이 통과했다.
- `IASSD::FarthestPointSampling` 하나만 포함한 작은 custom op ONNX 모델도 `TensorrtExecutionProvider,CUDAExecutionProvider`로 session 생성과 실행이 통과했다. 이 경우 TensorRT EP는 실행할 subgraph가 없다고 판단하고 CUDA custom op fallback으로 실행한다.
- 따라서 target의 TensorRT EP 설치 자체와 custom op fallback 조합은 최소 모델에서는 동작한다.
- 실제 IA-SSD ONNX는 TensorRT EP session 생성 단계에서 `graph_build.Resolve().IsOK() was false`로 실패한다.
- 실패 원인은 TensorRT plugin 부재 하나로 단정하기 어렵고, 현재 export graph에 남아 있는 ONNXScript local function과 scalar helper subgraph가 TensorRT EP partition/resolve 단계와 충돌하는 것으로 판단한다.
- `tools/iassd_trt_plugins/`를 추가해 TensorRT plugin shared library `libiassd_trt_plugins.so`를 빌드할 수 있게 했다.
- `IASSD::FarthestPointSampling`에 대응하는 TensorRT `IPluginV2DynamicExt` plugin과 creator를 구현했다.
- plugin enqueue는 기존 raw CUDA launcher `farthest_point_sampling_kernel_launcher(...)`를 재사용하고, TensorRT workspace를 FPS temp buffer로 사용한다.
- `tools/test_iassd_trt_fps_plugin.py`를 추가했다. 이 테스트는 작은 `IASSD::FarthestPointSampling` ONNX를 생성하고, TensorRT ONNX parser가 plugin으로 engine을 빌드한 뒤 TensorRT 실행 결과를 PyTorch FPS index와 비교한다.
- 검증 결과: `batch_size=1`, `num_points=512`, `npoint=64`에서 TensorRT parser가 `PLUGIN_V2: node_of_idx`로 plugin layer를 생성했고, TensorRT 실행 output index가 PyTorch FPS output index와 정확히 일치했다.
- `tools/build_iassd_trt_engine.py`를 추가해 ONNX Runtime을 거치지 않고 TensorRT Python API로 ONNX parser와 serialized engine build를 직접 수행할 수 있게 했다.
- `tools/sanitize_onnx_for_trt.py`를 추가해 PyTorch 2.5 exporter가 만든 ONNX graph 중 TensorRT parser가 싫어하는 helper graph를 표준 ONNX 또는 initializer로 정리한다.
- sanitizer는 `Constant/Cast/Reshape/Concat/Unsqueeze/Squeeze/Size/Shape/Equal/If/Expand/Tile` 상수 folding, `CastLike` lowering, `IsScalar` lowering, `aten_squeeze_dim` lowering, 미사용 branch 제거, `MaxPool` indices output 제거, `SplitToSequence/SequenceAt` lowering을 수행한다.
- `IASSD::GatherPoints`, `IASSD::BallQuery`, `IASSD::GroupPoints` TensorRT plugin을 추가했다.
- TensorRT plugin들은 기존 raw CUDA launcher인 `gather_points_kernel_launcher_fast`, `ball_query_kernel_launcher_fast`, `group_points_kernel_launcher_fast`를 재사용한다.
- `tools/iassd_trt_plugins/CMakeLists.txt`에 `gather_points_plugin.cu`, `ball_query_plugin.cu`, `group_points_plugin.cu` 및 대응 CUDA source를 추가했다.
- direct TensorRT parser 기준으로 `FarthestPointSampling`, `GatherPoints`, `BallQuery`, `GroupPoints`가 모두 `PLUGIN_V2` layer로 생성됨을 확인했다.
- `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_except_squeeze_noid_trt_sanitized.onnx`로 direct TensorRT FP32 engine build가 성공했다.
- 저장된 FP32 engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_fp32.engine`
- engine build 결과: `parse_ok=true`, `build_ok=true`, input `points: [16384, 5]`, outputs `batch_cls_preds: [256, 3]`, `batch_box_preds: [256, 7]`, layer 수 `447`, engine 크기 `11745848` bytes.

빌드 및 검증 명령:

```bash
cd /workspace
TORCH_CUDA_ARCH_LIST=8.7 cmake -S tools/iassd_trt_plugins -B /host_tmp/iassd_trt_plugins_build
cmake --build /host_tmp/iassd_trt_plugins_build -j2

python3 tools/sanitize_onnx_for_trt.py \
  --input onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_except_squeeze_noid.onnx \
  --output onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_except_squeeze_noid_trt_sanitized.onnx \
  --skip_check

python3 tools/build_iassd_trt_engine.py \
  --onnx_file onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_except_squeeze_noid_trt_sanitized.onnx \
  --engine_file onnx_exports/stage2/ia_ssd_kitti_direct_trt_fp32.engine \
  --plugin_library /host_tmp/iassd_trt_plugins_build/libiassd_trt_plugins.so \
  --workspace_mb 4096 \
  --log_level error
```

주의:

- 현재 direct TensorRT engine은 raw output `batch_cls_preds`, `batch_box_preds`까지만 포함한다. NMS/post-processing은 아직 graph 밖 Python 경로로 유지한다.
- generated ONNX sanitizer는 TensorRT parser 호환성을 위한 사후 정리 도구다. 장기적으로는 export 단계에서 helper op가 덜 생기도록 낮추는 것이 더 좋다.
- FP32 engine build는 성공했지만, 아직 TensorRT runtime inference parity/latency 스크립트는 추가하지 않았다.
- FP16 engine은 다음 단계에서 FP32 direct TensorRT inference parity를 확인한 뒤 생성한다.

### 7단계: direct TensorRT runtime 통합

산출물:

- TensorRT engine 로드 및 실행 스크립트
- TensorRT plugin library 로드 옵션
- host/device buffer 관리
- PyTorch/ORT CUDA/direct TensorRT output 비교

검증:

- TensorRT engine deserialization 성공
- direct TensorRT output shape 확인
- ORT CUDA path와 direct TensorRT raw output 비교
- PyTorch, ORT CUDA, direct TensorRT latency 비교

진행 기록:

- 상태: ORT TensorRT EP는 보류하고 direct TensorRT 경로로 전환
- 진행일: 2026-06-05
- 원본 stage2 ONNX의 top-level node domain은 `ai.onnx` 658개, `pkg.onnxscript.torch_lib` 80개, `this` 31개였다.
- `this::onnx_farthest_point_sampling`, `this::onnx_gather_points`, `this::onnx_ball_query`, `this::onnx_group_points`는 실제 `IASSD::*` custom op를 한 번 더 감싼 ONNXScript local function wrapper다.
- `tools/flatten_onnx_functions.py`를 추가해 ONNX local function call을 내부 node로 평탄화할 수 있게 했다.
- `--domains this`로 IA-SSD custom op wrapper만 평탄화한 `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx`는 CUDA EP parity를 유지했다.
- `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx` CUDA EP final `batch_cls_preds` 오차: max abs `0.008821487426757812`, mean abs `0.0009585299218694369`
- `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx` CUDA EP final `batch_box_preds` 오차: max abs `0.0035059452056884766`, mean abs `0.00010579687659628689`
- 그러나 `TensorrtExecutionProvider,CUDAExecutionProvider` session 생성은 여전히 `graph_build.Resolve().IsOK() was false`로 실패했다.
- 전체 local function flatten도 시험했지만 `aten_squeeze_dim`의 제어흐름 helper를 펼치며 CUDA 실행 중 `Concat` shape mismatch가 발생해 폐기했다.
- `aten_squeeze_dim`, `Rank`, `IsScalar`만 남기고 나머지 local function을 평탄화한 모델은 TensorRT EP가 실제 TensorRT parser까지 진행했지만, 여전히 `graph_build.Resolve().IsOK() was false`로 실패했다.
- `trt_dump_subgraphs=True`로 얻은 실패 subgraph는 scalar `Identity` 하나짜리 graph였다: `node_aten_max_683__result -> node_aten_max_683__result_4`, shape `[]`.
- top-level `Identity` node를 제거한 모델에서도 같은 scalar `Identity` subgraph가 TensorRT EP 내부 partition 과정에서 다시 생성되어 실패했다.
- `aten_max`, `aten_min`을 function으로 남겨도 TensorRT EP session 생성은 같은 resolve 오류로 실패했다.
- `IASSD::FarthestPointSampling` TensorRT plugin library를 `ctypes.CDLL(..., RTLD_GLOBAL)`로 로드하고, ORT custom op library도 함께 등록한 상태에서 `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx`를 `TensorrtExecutionProvider,CUDAExecutionProvider`로 다시 검증했다.
- 결과: TensorRT FPS plugin이 있어도 전체 IA-SSD ORT TensorRT EP session 생성은 여전히 `graph_build.Resolve().IsOK() was false`로 실패했다.
- 따라서 현재 blocker는 FPS plugin 부재가 아니라 ORT TensorRT EP의 graph partition/resolve 단계에 남아 있다.
- `tools/validate_iassd_ort_model.py`에 `--session_only`, `--provider_options`, `--graph_optimization_level` 옵션을 추가해 PyTorch 비교 없이 ORT/TensorRT session 생성만 빠르게 재현할 수 있게 했다.
- `tools/flatten_onnx_functions.py`에 `--remove_identity_outputs`를 추가해 graph output에 연결된 `Identity`까지 제거할 수 있게 했다.
- `ia_ssd_kitti_with_iassd_opset_iassd_flat_noid_outputs.onnx`를 생성해 top-level `Identity`가 0개임을 확인했지만, `TensorrtExecutionProvider,CUDAExecutionProvider` session 생성은 여전히 첫 subgraph `graph_build.Resolve().IsOK() was false`로 실패했다.
- `ia_ssd_kitti_with_iassd_opset_except_squeeze_noid.onnx`와 `ia_ssd_kitti_with_iassd_opset_trt_probe.onnx`는 TensorRT parser 단계까지 도달하지만, recursive partition 중 동일한 scalar `Identity` subgraph에서 다시 실패했다.
- 최신 `trt_dump_subgraphs=True` dump도 `Identity(node_aten_max_683__result -> node_aten_max_683__result_4)` 하나짜리 scalar graph였다. 원본 top-level graph에는 이 `Identity`가 없고 `node_aten_max_683__result`는 `ReduceMax` 출력이므로, 이 `Identity`는 ORT TensorRT EP partition 과정에서 생성된 alias subgraph로 판단한다.
- `--graph_optimization_level ORT_DISABLE_ALL`로 graph optimization을 꺼도 같은 scalar `Identity` partition 실패가 유지됐다.
- `trt_max_partition_iterations=0`은 ORT가 허용하지 않고 warning 후 기본값 `1000`으로 되돌려, provider option만으로 recursive partition을 끄는 우회는 사용할 수 없었다.

현재 판단:

- ORT TensorRT EP 1.16에서 이 export graph를 그대로 TensorRT/CUDA fallback으로 실행하려면, 단순히 IA-SSD custom op wrapper를 평탄화하는 것만으로는 부족하다.
- 직접 TensorRT parser 경로에서는 sanitizer와 1차 custom op plugin 4개를 통해 FP32 engine build까지 성공했다.
- 따라서 이후 가속 경로는 ONNX Runtime TensorRT EP가 아니라 direct TensorRT runtime을 우선한다.
- 다음 작업은 `ia_ssd_kitti_direct_trt_fp32.engine`을 로드해 실제 inference를 수행하고 ORT CUDA raw output과 비교하는 runtime script를 추가하는 것이다.

### 8단계: benchmark 정리

산출물:

- KITTI config 기준 benchmark 스크립트
- 측정 로그 포맷
- 결과 문서

기록 항목:

- hardware
- CUDA/PyTorch/ONNX Runtime/TensorRT 버전
- config path
- checkpoint path
- batch size
- point count
- precision mode
- warmup 횟수
- iteration 수
- 전처리 포함 여부
- post-processing 포함 여부
- pure model latency
- custom op별 latency 가능 여부

진행 기록:

- 상태: ORT CUDA EP raw benchmark 1차 완료
- 진행일: 2026-06-05
- `tools/benchmark_iassd_ort_cuda.py`를 추가해 같은 입력에서 PyTorch raw forward와 ORT CUDA EP `session.run` latency를 비교할 수 있게 했다.
- 현재 benchmark는 전처리와 후처리를 제외하고, `batch_cls_preds`, `batch_box_preds` raw output 생성까지를 측정한다.
- ORT 측정은 Python `onnxruntime.InferenceSession.run`에 numpy 입력을 넣고 numpy 출력을 받는 경로라 H2D/D2H 및 Python call overhead가 포함된다.
- `tools/benchmark_iassd_ort_cuda.py`에 ORT CUDA IO binding 측정을 추가해 CUDA input 재사용 및 CUDA output 유지 경로를 별도로 측정할 수 있게 했다.
- `tools/benchmark_iassd_ort_cuda.py`에 direct TensorRT engine 측정을 추가해 PyTorch raw forward, ORT CUDA, direct TensorRT를 같은 입력 tensor로 비교할 수 있게 했다.
- custom op 내부에는 정확도 검증 안정성을 위해 `cudaDeviceSynchronize()`가 아직 남아 있고, `GatherPoints`/`GroupPoints` index 입력은 CPU 왕복 복사를 사용한다.

측정 명령:

```bash
cd /workspace/IA-SSD
IASSD_DEBUG_RANGE_CHECK=1 python3 tools/benchmark_iassd_ort_cuda.py \
  --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
  --ckpt tools/IA-SSD.pth \
  --onnx_file onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset.onnx \
  --ort_op_library /tmp/iassd_ort_ops_build/libiassd_ort_ops.so \
  --num_points 16384 \
  --warmup 20 \
  --iterations 100 \
  --providers CUDAExecutionProvider \
  --report_file onnx_exports/stage5/kitti_ort_cuda_benchmark_raw_100.json
```

측정 결과:

- Hardware: Jetson Orin, `aarch64`, Linux `5.10.120-tegra`
- PyTorch: `2.0.0a0+ec3941ad.nv23.02`, CUDA `11.4`
- ONNX Runtime: `1.16.0`
- 입력 `points`: `(16384, 5)`, `torch.float32`, `cuda:0`
- warmup: 20
- iterations: 100
- PyTorch raw forward: mean `47.96612024307251 ms`, median `47.850911505520344 ms`, p95 `48.69451932609081 ms`, FPS(mean) `20.848048475307415`
- ORT CUDA `session.run`: mean `63.68923461064696 ms`, median `63.65747284144163 ms`, p95 `64.33191150426865 ms`, FPS(mean) `15.70124065885429`
- report: `onnx_exports/stage5/kitti_ort_cuda_benchmark_raw_100.json`

추가 측정 및 최적화 실험:

- `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx`는 2026-06-05 재검증에서 CUDA EP parity를 유지했다.
  - `batch_cls_preds`: max abs `0.012142181396484375`, mean abs `0.0008731304357449213`
  - `batch_box_preds`: max abs `0.005153656005859375`, mean abs `0.0001013898872770369`
  - report: `onnx_exports/stage5/kitti_ort_cuda_parity_report_iassd_flat_recheck.json`
- 같은 flat graph 기준 100회 benchmark 결과:
  - ORT CUDA `session.run`: mean `63.75184828415513 ms`, median `63.742563128471375 ms`, p95 `64.30155970156193 ms`, FPS(mean) `15.685819735653684`
  - ORT CUDA IO binding run: mean `65.97143437713385 ms`, median `66.01518765091896 ms`, p95 `66.77856110036373 ms`, FPS(mean) `15.158075755688085`
  - IO binding output `copy_outputs_to_cpu`: mean `0.45840539038181305 ms`
  - report: `onnx_exports/stage5/kitti_ort_cuda_benchmark_iassd_flat_100.json`
- 결론: 현재 병목은 numpy 출력 복사나 Python `session.run`의 단순 D2H 비용이 아니라 graph/custom op 실행 내부에 있다. IO binding은 이 모델/ORT 1.16 경로에서 속도 개선이 아니므로 기본 benchmark는 `session.run` 결과를 기준으로 유지한다.
- custom op 내부 `cudaDeviceSynchronize()`를 `cudaStreamSynchronize()` 또는 무동기화 경로로 줄이는 실험은 full model parity가 크게 깨져 보류했다. ORT 1.16 custom op와 CUDA EP 표준 노드 사이의 stream dependency를 직접 보장할 수 있을 때 다시 검토한다.
- `GatherPoints`/`GroupPoints` index 입력을 device tensor로 직접 받는 실험은 단위 테스트에서는 통과했지만, full IA-SSD graph parity가 크게 깨져 보류했다. 실제 graph에서는 해당 index 입력의 memory placement가 custom op 기대와 일치한다고 가정하면 안 된다.
- `tools/iassd_ort_ops/farthest_point_sampling_op.cc`에 `IASSD_ORT_PROFILE=1` custom op별 CUDA event 누적 계측을 추가했다. 이 옵션은 custom op 라이브러리 언로드/프로세스 종료 시 stderr에 op별 호출 수, 총 시간, 평균 시간을 출력한다.
- flat graph, warmup 2, iterations 3의 짧은 계측 결과:
  - `FarthestPointSampling`: count `10`, total `161.548 ms`, mean `16.1548 ms`
  - `GatherPoints`: count `25`, total `2.608 ms`, mean `0.10432 ms`
- second-only sorted TopK direct TensorRT engine 기준으로 실제 sample `seoulsi_yeoui/training/velodyne/000000.bin` 30회 raw forward benchmark를 수행했다.
  - report: `onnx_exports/stage7/seoulsi_yeoui_000000_second_topk_direct_trt_benchmark_30.json`
  - warmup: 10, iterations: 30
  - PyTorch raw forward: mean `86.5383829921484 ms`, median `82.0612357929349 ms`, p95 `97.37789630889893 ms`, FPS(mean) `11.555566043922148`
  - ORT CUDA `session.run`: mean `147.19489819059768 ms`, median `145.86011972278357 ms`, p95 `169.06576231122017 ms`, FPS(mean) `6.79371372440595`
  - ORT CUDA IO binding run: mean `146.44864338139692 ms`, median `145.35456523299217 ms`, p95 `163.2617563009262 ms`, FPS(mean) `6.828332287078243`
  - direct TensorRT engine run: mean `75.79354674865802 ms`, median `77.62681506574154 ms`, p95 `96.0360188037157 ms`, FPS(mean) `13.193735389058116`
  - 해석: 이 실제 sample의 짧은 raw benchmark에서는 direct TensorRT가 PyTorch 대비 약 `1.14x`, ORT CUDA IO binding 대비 약 `1.93x` 빠르다. 측정 variance가 있으므로 다음 단계에서는 sample 수와 iteration 수를 늘려 재측정한다.
  - `BallQuery`: count `40`, total `59.448 ms`, mean `1.4862 ms`
  - `GroupPoints`: count `80`, total `43.568 ms`, mean `0.5446 ms`
  - 같은 실행의 ORT CUDA `session.run`: mean `64.80249700446923 ms`
- 계측 해석: IA-SSD 1회 추론 기준 custom op 대략 비용은 FPS 2회 약 `32 ms`, BallQuery 8회 약 `12 ms`, GroupPoints 16회 약 `9 ms`, GatherPoints 5회 약 `0.5 ms` 수준이다. 따라서 다음 속도 최적화의 1순위는 `FarthestPointSampling`이다.
- FPS 내부 세분화 계측을 추가했다. flat graph, warmup 2, iterations 3의 짧은 계측 결과:
  - `FarthestPointSampling`: count `10`, total `173.545 ms`, mean `17.3545 ms`
  - `FarthestPointSampling.cudaMalloc`: count `10`, total `0.104 ms`, mean `0.0104 ms`
  - `FarthestPointSampling.tempInit`: count `10`, total `0.333 ms`, mean `0.0333 ms`
  - `FarthestPointSampling.kernel`: count `10`, total `172.083 ms`, mean `17.2083 ms`
  - `FarthestPointSampling.kernel.large`: count `5`, total `159.224 ms`, mean `31.8448 ms`
  - `FarthestPointSampling.kernel.small`: count `5`, total `12.719 ms`, mean `2.5438 ms`
  - `FarthestPointSampling.cudaFree`: count `10`, total `0.073 ms`, mean `0.0073 ms`
- 계측 해석: FPS 병목은 temp buffer 할당/초기화가 아니라 kernel 본체다. 특히 KITTI config의 첫 D-FPS 단계 `16384 -> 4096`이 평균 `31.8448 ms`로 지배적이고, 두 번째 D-FPS 단계 `4096 -> 1024`는 평균 `2.5438 ms` 수준이다.

다음 최적화 후보:

- `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx`를 benchmark 기준 graph로 우선 사용한다.
- 첫 D-FPS `16384 -> 4096`의 exact FPS 비용을 줄이는 방법을 검토한다. 단순 메모리 관리 최적화로는 효과가 거의 없으므로 kernel 알고리즘/구현 자체를 다뤄야 한다.
- 정확도 유지가 최우선이면 FPS kernel의 reduction 구현, block size, warp-level reduction 최적화를 검토한다.
- 속도 우선 모드가 허용되면 첫 D-FPS에 대해 approximate FPS, sector/grid 기반 pre-sampling, 또는 `NPOINT_LIST[0]` 축소를 별도 옵션으로 실험하고 PyTorch 기준 output/parity 및 최종 detection 품질 영향을 측정한다.
- `cudaDeviceSynchronize()` 제거는 단순 stream sync가 아니라 ORT CUDA EP와 명시 이벤트 또는 allocator/stream contract를 맞춘 뒤 재시도한다.
- `GatherPoints`/`GroupPoints` device index 경로는 ORT node assignment와 input memory info를 확인할 수 있는 진단 로그를 먼저 추가한 뒤 재시도한다.
- 정확도 리포트와 benchmark 리포트를 같은 실행에서 묶는 통합 benchmark를 작성한다.

### Direct TensorRT engine 경로 진행 기록

- 진행일: 2026-06-11
- ORT TensorRT EP의 recursive partition 문제를 우회하기 위해 ONNX Runtime을 거치지 않고 TensorRT parser/build/runtime을 직접 사용하는 경로를 우선 검증했다.
- `tools/build_iassd_trt_engine.py`를 추가해 TensorRT plugin library를 `ctypes.CDLL(..., RTLD_GLOBAL)`로 로드한 뒤 ONNX parser와 builder로 FP32 engine을 생성할 수 있게 했다.
- `tools/validate_iassd_trt_engine.py`를 추가해 PyTorch CUDA tensor pointer를 TensorRT binding에 직접 연결하고, direct TensorRT engine output을 PyTorch 및 ORT CUDA output과 비교할 수 있게 했다.
- `tools/test_iassd_trt_pointnet2_plugins.py`를 추가해 `FarthestPointSampling`, `GatherPoints`, `BallQuery`, `GroupPoints` TensorRT plugin을 작은 ONNX graph로 PyTorch extension과 단위 비교할 수 있게 했다.
- `tools/sanitize_onnx_for_trt.py`에 TensorRT parser가 처리하지 못하는 일부 ONNXScript local function call lowering을 추가했다.
  - `aten_unsqueeze -> Unsqueeze`
  - `_aten_native_batch_norm_inference_onnx -> Sub/Add/Sqrt/Div/Mul/Add` 산술식
  - `aten_repeat -> Tile` 및 constant folding
  - `aten_where -> Where`
  - `aten_split`/`aten_getitem -> Slice`
  - `aten_exp -> Exp`
  - `ReduceMax(axis=-1, keepdims=0) -> Slice/Squeeze/Max`
  - `--disabled_lowerings` 진단 옵션을 추가해 lowering별 parity 영향을 분리할 수 있게 했다.

검증 결과:

- `ia_ssd_kitti_with_iassd_opset_except_squeeze_flat_trt_sanitized.onnx` 기준 direct TensorRT FP32 engine build가 성공했다.
  - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_except_squeeze_flat_fp32.engine`
  - output shape: `batch_cls_preds=(256, 3)`, `batch_box_preds=(256, 7)`
  - 하지만 `except_squeeze_flat` ONNX 자체가 PyTorch 대비 평균 오차가 커 최종 기준 graph로 부적합하다.
    - ORT CUDA vs PyTorch: cls mean abs `0.6604945206393799`, box mean abs `1.8488085775503091`
    - direct TRT vs PyTorch: cls mean abs `3.278387589380145`, box mean abs `8.293494494002516`
- `ia_ssd_kitti_with_iassd_opset_iassd_flat.onnx`는 direct TensorRT parser가 처음에는 남은 ONNXScript local function에서 실패했으나, sanitizer lowering 추가 후 FP32 engine build까지 성공했다.
  - sanitized ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized.onnx`
  - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32.engine`
  - engine size: `11907121` bytes
  - output shape: `batch_cls_preds=(256, 3)`, `batch_box_preds=(256, 7)`
- BatchNorm lowering 문제는 분리 및 수정했다.
  - `iassd_flat` 원본 ONNX는 ORT CUDA 기준 평균 오차가 작다.
    - cls mean abs `0.013319187797605991`, box mean abs `0.0778676547924988`
  - `--disabled_lowerings batch_norm`으로 sanitizer를 돌리면 ORT CUDA parity가 크게 회복되어 기존 `BatchNormalization` 단순 치환이 blocker임을 확인했다.
    - cls mean abs `0.0009747681518395742`, box mean abs `0.00010998599464073777`
  - BatchNorm을 ONNX `BatchNormalization` 노드가 아니라 산술식으로 lowering한 뒤 full sanitized ONNX의 ORT CUDA parity가 회복됐다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_trt_sanitized_ort_cuda_parity_report.json`
    - cls mean abs `0.0014383929471174877`, box mean abs `0.00015485349077997462`
- direct TensorRT engine 정확도 blocker는 아직 남아 있다.
  - 같은 sanitized ONNX로 만든 direct TensorRT engine은 ORT CUDA와 크게 어긋난다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_trt_engine_parity_report.json`
    - ORT vs TRT: cls mean abs `3.6359202976649008`, box mean abs `8.513163616697577`
  - `FarthestPointSampling`, `GatherPoints`, `BallQuery`, `GroupPoints` plugin 단위 테스트는 통과했다.
  - custom op 중간 출력을 debug output으로 노출해 비교한 결과, `group_points_7`까지는 ORT/TRT가 거의 동일하고 `gather_points_2`부터 크게 벌어진다.
    - debug ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_debug_iassd_outputs.onnx`
    - debug engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_debug_iassd_outputs.engine`
    - debug report: `onnx_exports/stage7/kitti_iassd_flat_trt_debug_iassd_outputs_report.json`
    - `group_points_7`: max abs `0.00012743473052978516`, mean abs `5.6868662943673765e-08`
    - `gather_points_2`: max abs `79.75126647949219`, mean abs `17.563795119490653`
  - `gather_points_2`의 index 입력은 custom FPS가 아니라 표준 ONNX `TopK` 경로에서 생성된다.
    - 주변 graph: `ReduceMax -> Sigmoid -> Range/Reshape/Mul -> Sub -> TopK -> Cast -> GatherPoints`
    - `TopK` attribute: `axis=-1`, `largest=1`, `sorted=1`, `k=512`
  - `TopK` 입력 `sub_4`, 출력 `topk__0/getitem_56`, Cast 출력 `_to_copy_1`을 debug output으로 추가해 비교했다.
    - debug ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_debug_topk.onnx`
    - debug engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_debug_topk.engine`
    - debug report: `onnx_exports/stage7/kitti_iassd_flat_trt_debug_topk_report.json`
    - `sub_4`: max abs `0.10756140784360468`, mean abs `0.10488297685424186`
    - `getitem_56`: max abs `1023.0`, mean abs `161.072265625`
  - 추가 debug에서 `sub_4`의 원인은 `ReduceMax` axes 입력 처리로 확인됐다. TensorRT가 `ReduceMax` 출력 `getitem_53`와 그 뒤 `sigmoid`를 `[1, 1024]`가 아니라 scalar `[]`로 해석했다.
    - debug ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_debug_score.onnx`
    - debug engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_debug_score.engine`
    - debug report: `onnx_exports/stage7/kitti_iassd_flat_trt_debug_score_report.json`
    - `slice_14`: max abs `0.007523536682128906`, mean abs `0.002597086519623796`
    - `getitem_53`: shape mismatch, ORT `[1, 1024]`, TRT scalar `[]`
    - `mul`: max abs `0.0`, mean abs `0.0`
  - sanitizer에 `ReduceMax(axis=-1, keepdims=0)` lowering을 추가했다. ONNX-vs-ONNX 비교에서는 reference sanitized graph와 lowered graph의 `slice_14`, `getitem_53`, `sigmoid`, `sub_4`, `slice_17`, `getitem_85`, final raw output이 모두 `0` 오차였다.
  - `ReduceMax` lowering 후 direct TensorRT FP32 engine을 재빌드했고 ORT-vs-TRT final output 오차가 개선됐다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_reducemax_lower_trt_engine_parity_report.json`
    - ORT vs TRT: cls mean abs `0.6882749106734991`, box mean abs `1.9815277494973478`
  - `ReduceMax` fix가 반영된 sanitized ONNX 기준으로 `TopK -> GatherPoints(gather_points_2)`를 다시 debug했다.
    - debug ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_reducemax_debug_topk.onnx`
    - debug engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_reducemax_debug_topk.engine`
    - debug report: `onnx_exports/stage7/kitti_iassd_flat_reducemax_lower_trt_debug_topk_report.json`
    - `sub_4`: max abs `0.0001620948314666748`, mean abs `4.020788132663711e-06`
    - `topk__0`: max abs `0.0001620948314666748`, mean abs `5.855142703126148e-06`
    - `getitem_56`: max abs `918.0`, mean abs `68.6328125`
    - `gather_points_2`: max abs `70.53976821899414`, mean abs `4.926960854048957`
  - score head 내부 debug 결과, `Slice`/`Transpose` 자체가 아니라 마지막 score Conv 출력 `convolution_15`에서 `slice_14` 오차가 시작된다.
    - debug ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_debug_score_head.onnx`
    - debug engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_debug_score_head.engine`
    - debug report: `onnx_exports/stage7/kitti_iassd_flat_trt_debug_score_head_report.json`
    - `convolution_14`: max abs `0.0029694437980651855`, mean abs `0.00032028598027800115`
    - `relu_14`: max abs `0.002256035804748535`, mean abs `0.00011539230770714337`
    - `convolution_15`: max abs `0.013176918029785156`, mean abs `0.0027147420526792607`
    - `slice_14`: max abs `0.013176918029785156`, mean abs `0.0027147420526792607`
  - TensorRT FP32 builder의 TF32 tactic 영향도 분리했다. `tools/build_iassd_trt_engine.py`에 `--disable_tf32` 옵션을 추가해 `trt.BuilderFlag.TF32`를 clear할 수 있게 했다.
    - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_no_tf32.engine`
    - report: `onnx_exports/stage7/kitti_iassd_flat_trt_engine_no_tf32_parity_report.json`
    - ORT vs TRT: cls mean abs `0.8386426344513893`, box mean abs `2.1050358138670617`
    - 해석: no-TF32 engine이 기본 FP32 engine보다 개선되지 않아, 현재 blocker를 TF32 tactic만으로 설명하기 어렵다.
  - `TopK` tie-breaker 강도를 키우는 사후 ONNX 실험을 수행했다. `val_233`은 첫 `TopK` 앞의 `index * epsilon` bias이며 기존 값은 `1e-5`다.
    - `epsilon=2e-4`: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_topk_eps2e4.onnx`
      - ORT original vs ORT eps2e4 report: `onnx_exports/stage7/kitti_iassd_flat_topk_eps2e4_ort_vs_base_report.json`
      - cls mean abs `3.5745936573172608`, box mean abs `8.531867323796698`
    - `epsilon=5e-5`: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_topk_eps5e5.onnx`
      - ORT original vs ORT eps5e5 report: `onnx_exports/stage7/kitti_iassd_flat_topk_eps5e5_ort_vs_base_report.json`
      - cls mean abs `3.2999298836415014`, box mean abs `8.195081846069245`
    - 해석: tie-breaker 강화는 TensorRT/ORT index 안정화 후보일 수 있지만, 원본 ORT 출력 자체를 크게 바꿔 현재 정확도 보존 전략으로는 부적합하다.
  - TensorRT tactic source 제한 실험을 위해 `tools/build_iassd_trt_engine.py`에 `--tactic_sources` 옵션을 추가했다.
    - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_tactics_cublas_cudnn.engine`
    - report: `onnx_exports/stage7/kitti_iassd_flat_trt_engine_tactics_cublas_cudnn_parity_report.json`
    - ORT vs TRT: cls mean abs `0.9251846577972174`, box mean abs `2.562146398513505`
    - 해석: `CUBLAS,CUDNN` 제한도 기본 FP32 engine보다 개선되지 않아 tactic source 제한만으로는 blocker가 해결되지 않았다.
  - score head 주변의 BatchNorm 산술 차이를 줄이기 위해 `Conv + _aten_native_batch_norm_inference_onnx` weight folding을 sanitizer에 실험 옵션으로 추가했다.
    - 옵션: `tools/sanitize_onnx_for_trt.py --enable_conv_batch_norm_fold`
    - 기본 sanitizer 경로에서는 `folded_conv_batch_norm_nodes=0`, `lowered_batch_norm_nodes=35`로 기존 동작을 유지한다.
    - 실험 ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_bn_fold.onnx`
    - ORT original sanitized vs ORT BN-fold report: `onnx_exports/stage7/kitti_iassd_flat_bn_fold_ort_vs_base_report.json`
    - ORT vs ORT: cls mean abs `0.8670467597742876`, box mean abs `2.2347811788703047`
    - 해석: Conv/BN fold는 수식상 등가에 가깝지만 FP32 산술 순서가 바뀌면서 조밀한 score 분포의 `TopK` 선택을 다시 흔든다. 원본 ORT raw output 보존 전략으로는 부적합하므로 기본 변환에서는 비활성화한다.
  - `tools/validate_iassd_trt_engine.py`에 `--dump_outputs_npz` 옵션을 추가해 ORT/TRT/PyTorch 중간 output을 NPZ로 저장할 수 있게 했다.
  - `tools/analyze_iassd_topk_margin.py`를 추가해 저장된 `sub_4` score와 `getitem_56` TopK index를 분석했다.
    - NPZ dump: `onnx_exports/stage7/kitti_iassd_flat_reducemax_lower_trt_debug_topk_outputs.npz`
    - 분석 report: `onnx_exports/stage7/kitti_iassd_flat_reducemax_lower_trt_topk_margin_report.json`
    - `getitem_56` TopK index set overlap은 `512/512`로, ORT와 TensorRT가 고른 index 집합은 같았다.
    - 다만 index 순서가 달라 `GatherPoints` output 순서와 final raw prediction row 순서가 크게 달라졌다.
    - TopK boundary gap은 `2.0126113668084145e-05`, top-k 내부 adjacent score gap 최솟값은 `4.423782229423523e-09`였다.
    - `<=1e-5` adjacent gap이 `245`개, `<=5e-5` adjacent gap이 `434`개로 score가 매우 조밀해 작은 산술 차이만으로 정렬 순서가 흔들린다.
  - sanitizer에 `--sort_topk_indices` 실험 옵션을 추가했다.
    - TopK index output을 float로 cast한 뒤 `TopK(largest=0, sorted=1)`를 한 번 더 적용해 selected index를 오름차순으로 정렬하고, 다시 int64로 cast해 기존 index consumer에 연결한다.
    - 이 옵션은 기본 변환에서는 비활성화한다.
    - 실험 ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_sort_topk.onnx`
    - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk.engine`
    - build 결과: `parse_ok=true`, `build_ok=true`, `num_layers=849`, `engine_size_bytes=11928962`
  - sorted TopK ONNX와 sorted TopK TensorRT engine을 비교하면 direct TensorRT parity가 크게 개선됐다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_trt_engine_sort_topk_parity_report.json`
    - ORT vs TRT: cls mean abs `0.013567300513386726`, max abs `0.09396934509277344`
    - ORT vs TRT: box mean abs `0.003995210289271459`, max abs `3.1938645243644714`
  - 하지만 sorted TopK ONNX 자체를 원래 sanitized ONNX와 비교하면 raw output이 크게 달라졌다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_sort_topk_ort_vs_base_report.json`
    - ORT original vs ORT sorted: cls mean abs `3.5885217680285373`, max abs `17.64932870864868`
    - ORT original vs ORT sorted: box mean abs `7.967100935988128`, max abs `79.17637634277344`
    - 해석: index set은 같아도 raw output row 순서가 바뀌기 때문에 단순 element-wise parity 기준에서는 큰 오차로 보인다. NMS 전 raw prediction의 순서가 의미 없는 후보 목록이라면 후처리 전 canonical sort를 별도로 정의해 비교할 수 있지만, 원본 raw tensor element-wise 보존 기준으로는 기본 적용할 수 없다.
  - `tools/compare_iassd_postprocess_outputs.py`를 추가해 base ORT, candidate ORT, candidate TensorRT의 raw output을 기존 IA-SSD `post_processing`으로 통과시켜 비교할 수 있게 했다.
    - 기본 KITTI `SCORE_THRESH=0.1`과 synthetic 입력에서는 세 경로 모두 detection count가 `0`이었다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_sort_topk_postprocess_compare_report.json`
    - threshold를 `0.0`으로 낮추는 synthetic 실험은 NMS CUDA extension에서 exit code `139`로 실패했다. synthetic 입력에서 너무 낮은 threshold로 무리하게 후보를 살리는 방식은 안정적인 판단 기준으로 쓰지 않는다.
  - 같은 스크립트에 `--skip_postprocess`를 추가해 NMS 없이 raw 후보를 sigmoid score 기준으로 canonical sort한 뒤 비교할 수 있게 했다.
    - report: `onnx_exports/stage7/kitti_iassd_flat_sort_topk_raw_candidates_compare_report.json`
    - base ORT vs sorted TopK ORT: 후보 count는 `256/256`으로 같지만 canonical sort 후 label sequence가 일치하지 않는다.
    - base ORT vs sorted TopK ORT: score mean abs `3.539101091871999e-05`, box mean abs `8.003834749738287`
    - base ORT vs sorted TopK TensorRT: 후보 count는 `256/256`으로 같지만 canonical sort 후 label sequence가 일치하지 않는다.
    - base ORT vs sorted TopK TensorRT: score mean abs `3.766309249514377e-05`, box mean abs `7.984905568597371`
    - 해석: sorted TopK는 TensorRT와 sorted ORT 사이의 runtime parity를 크게 개선하지만, 원본 ORT 후보 집합 보존 측면에서는 아직 충분하지 않다. 단순 row-order 차이만으로 설명하기 어렵고, TopK index 순서를 바꾸는 것이 downstream feature aggregation과 최종 후보 자체를 바꾼다.
  - 실제 sample 입력으로 `/mnt/NAS2/pjt_AI/00-git-archive/lidar-pytorch/data/seoulsi_yeoui/training/velodyne/000000.bin`을 사용해 재검증했다.
    - 원본 파일은 float32 4 feature `(x, y, z, intensity)` 형식이며, `000000.bin`은 `26938` points다.
    - KITTI config의 `sample_points`를 통해 ONNX 입력은 기존과 같은 `[16384, 5]`로 맞춰진다.
  - 실제 sample에서도 기본 post-processing/NMS 비교는 exit code `139`로 실패했다.
    - report 파일은 생성되지 않았다: `onnx_exports/stage7/seoulsi_yeoui_000000_sort_topk_postprocess_compare_report.json`
    - synthetic threshold lowering 때와 같은 계열의 NMS CUDA extension 불안정으로 판단한다.
  - 실제 sample에서 NMS를 생략한 raw 후보 비교를 수행했다.
    - report: `onnx_exports/stage7/seoulsi_yeoui_000000_sort_topk_raw_candidates_compare_report.json`
    - base ORT max score는 `0.7118441462516785`, sorted TopK ORT max score는 `0.5143852233886719`, sorted TopK TensorRT max score는 `0.514943540096283`이다.
    - base ORT vs sorted TopK ORT: 후보 count는 `256/256`으로 같지만 canonical sort 후 label sequence가 일치하지 않는다.
    - base ORT vs sorted TopK ORT: score mean abs `0.008218067133835216`, box mean abs `3.873906783430097`
    - base ORT vs sorted TopK TensorRT: score mean abs `0.008212141999519251`, box mean abs `3.912777145459196`
    - candidate sorted ORT vs candidate sorted TensorRT: score mean abs `1.6630406288201116e-05`, box mean abs `1.2487147216327554` by canonical sort. 이 값은 canonical sort가 작은 score/box 차이로 후보 순서를 다시 흔들어 과장된 것으로 보인다.
  - 실제 sample에서 sorted TopK ONNX와 sorted TopK TensorRT engine의 element-wise raw parity도 별도로 측정했다.
    - report: `onnx_exports/stage7/seoulsi_yeoui_000000_sort_topk_trt_engine_parity_report.json`
    - ORT vs TRT: cls mean abs `0.011454129802586976`, max abs `0.39871978759765625`
    - ORT vs TRT: box mean abs `0.005034685983056468`, max abs `3.1150981783866882`
    - 해석: direct TensorRT engine은 sorted TopK candidate graph 자체는 비교적 잘 실행한다. 남은 큰 문제는 candidate graph가 원본 ORT의 후보 생성 정책을 바꾸는 점과, 현재 환경에서 GPU NMS post-processing 비교가 exit code `139`로 불안정한 점이다.
  - `tools/compare_iassd_postprocess_outputs.py`에 비교 전용 `--postprocess_mode numpy_nms`를 추가했다.
    - 이 모드는 CUDA NMS extension을 호출하지 않고, heading에 따라 `dx/dy`를 바꾼 axis-aligned BEV box로 CPU NMS를 수행한다.
    - 실제 CUDA rotated NMS와 완전 동일한 구현은 아니므로 production 후처리가 아니라 post-processing 비교 안정화 용도다.
  - 실제 sample에서 NumPy NMS fallback으로 최종 detection 비교를 수행했다.
    - report: `onnx_exports/stage7/seoulsi_yeoui_000000_sort_topk_numpy_nms_compare_report.json`
    - detection count: base ORT `3`, sorted TopK ORT `1`, sorted TopK TensorRT `1`
    - base ORT vs sorted TopK ORT: count mismatch. 공통 1개 detection은 score/box 차이 `0`으로 일치한다.
    - base ORT vs sorted TopK TensorRT: count mismatch. 공통 1개 detection은 score mean/max abs `0.000558316707611084`, box mean abs `0.0004656144550868443`, max abs `0.0011203289031982422`
    - sorted TopK ORT vs sorted TopK TensorRT: count match, label match. score mean/max abs `0.000558316707611084`, box mean abs `0.0004656144550868443`, max abs `0.0011203289031982422`
    - 해석: direct TensorRT engine은 sorted TopK candidate graph의 최종 검출까지는 거의 같은 결과를 낸다. 하지만 sorted TopK candidate graph 자체가 원본 ORT 대비 검출 수를 `3 -> 1`로 바꾸므로, 정확도 보존 관점의 남은 blocker는 TensorRT plugin 산술보다 TopK/order 정책 변경이다.
  - 최종 기준을 PyTorch inference로 두기 위해 `tools/compare_iassd_postprocess_outputs.py`에 `--include_pytorch` 옵션을 추가했다.
    - PyTorch 모델을 같은 sampled points로 raw inference하고, base ORT/candidate ORT/candidate TensorRT와 같은 `numpy_nms` fallback 후처리로 비교한다.
    - report: `onnx_exports/stage7/seoulsi_yeoui_000000_pytorch_vs_ort_trt_numpy_nms_compare_report.json`
    - detection count: PyTorch `3`, base ORT `3`, sorted TopK ORT `1`, sorted TopK TensorRT `1`
    - PyTorch vs base ORT: count match, label match. score mean abs `0.00011474887530008952`, max abs `0.0002493858337402344`, box mean abs `9.053803625560942e-05`, max abs `0.0003581047058105469`
    - PyTorch vs sorted TopK ORT: count mismatch. 공통 1개 detection은 score/box 차이 `0`으로 일치한다.
    - PyTorch vs sorted TopK TensorRT: count mismatch. 공통 1개 detection은 score mean/max abs `0.000558316707611084`, box mean abs `0.0004656144550868443`, max abs `0.0011203289031982422`
    - 해석: base ORT는 PyTorch inference 기준으로 최종 detection을 잘 보존한다. 현재 정확도 blocker는 TensorRT engine 자체보다 sorted TopK candidate graph가 PyTorch/base ORT 대비 detection을 `3 -> 1`로 줄이는 점이다.
  - 실제 입력의 `sample_points` 랜덤성을 제거하기 위해 `tools/compare_iassd_postprocess_outputs.py`에 `--seed` 옵션을 추가했다.
    - 같은 seed `1024`에서 first-only, second-only, both sorted TopK variant를 다시 비교했다.
    - first-only ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_sort_topk_first.onnx`
    - second-only ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_sort_topk_second.onnx`
    - reports:
      - `onnx_exports/stage7/seoulsi_yeoui_000000_pytorch_vs_sort_topk_first_seed1024_numpy_nms_compare_report.json`
      - `onnx_exports/stage7/seoulsi_yeoui_000000_pytorch_vs_sort_topk_second_seed1024_numpy_nms_compare_report.json`
      - `onnx_exports/stage7/seoulsi_yeoui_000000_pytorch_vs_sort_topk_both_seed1024_numpy_nms_compare_report.json`
    - PyTorch/base ORT는 세 실험 모두 detection count `2/2`, label match, score mean abs `0.00011610984802246094`, box mean abs `3.255690847124372e-05`로 일치했다.
    - first-only sorted TopK는 detection count와 label은 유지하지만, PyTorch 대비 score mean abs `0.04930570721626282`, max abs `0.09861141443252563`, box mean abs `0.02584254741668701`, max abs `0.1643381118774414`로 오차가 커진다.
    - second-only sorted TopK는 PyTorch/base ORT와 같은 수준으로 일치한다.
    - both sorted TopK는 first-only와 같은 오차를 보인다.
    - 해석: divergence는 첫 번째 score-based sampling TopK `node_TopK_347`의 index output `getitem_56`을 정렬해 후속 `GatherPoints` 입력 순서를 바꾸는 지점에서 시작한다. 두 번째 TopK `node_TopK_471`/`getitem_88` 정렬은 이 sample/seed에서는 최종 detection에 영향을 주지 않았다.
  - second-only sorted TopK ONNX로 direct TensorRT FP32 engine을 생성하고 같은 실제 sample/seed로 PyTorch/base ORT/candidate ORT/candidate TensorRT를 비교했다.
    - ONNX: `onnx_exports/stage2/ia_ssd_kitti_with_iassd_opset_iassd_flat_trt_sanitized_sort_topk_second.onnx`
    - engine: `onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine`
    - report: `onnx_exports/stage7/seoulsi_yeoui_000000_pytorch_vs_sort_topk_second_trt_seed1024_numpy_nms_compare_report.json`
    - detection count: PyTorch `2`, base ORT `2`, second-only sorted TopK ORT `2`, second-only sorted TopK TensorRT `2`
    - PyTorch vs base ORT: count match, label match. score mean abs `0.00011610984802246094`, max abs `0.00023221969604492188`, box mean abs `3.255690847124372e-05`, max abs `0.000118255615234375`
    - PyTorch vs second-only sorted TopK ORT: count match, label match. score/box 오차는 base ORT와 동일했다.
    - PyTorch vs second-only sorted TopK TensorRT: count match, label match. score mean abs `0.0010794103145599365`, max abs `0.0013611912727355957`, box mean abs `0.0003993298326219831`, max abs `0.002277851104736328`
    - base ORT vs second-only sorted TopK TensorRT: count match, label match. score mean abs `0.0009633004665374756`, max abs `0.0011289715766906738`, box mean abs `0.0003667729241507394`, max abs `0.002159595489501953`
    - 해석: second-only sorted TopK는 이번 실제 sample/seed에서 원본 PyTorch/base ORT의 최종 detection을 보존하면서 direct TensorRT engine까지 연결된다. TRT 오차는 FP32 tactic/연산 순서 차이로 볼 수 있는 1e-3 수준이며, 최종 detection count와 label은 유지된다.
- 해석: 현재 direct TensorRT 경로의 1차 목표인 parser/build/runtime 연결은 달성했다. `ReduceMax` scalar 해석 문제를 제거했고, 전체 sorted TopK는 원본 output을 바꾸지만 두 번째 TopK만 정렬하는 제한적 변환은 이번 sample/seed에서 정확도 보존과 TensorRT runtime 연결을 동시에 만족했다.

다음 액션:

- score head `Conv+BatchNorm` weight-folding은 원본 ORT 출력 보존에 불리하므로 기본 경로에서 제외한다.
- `TopK` index 안정화는 단순 epsilon 증폭이 원본 출력을 크게 바꾸므로 보류한다. 필요하면 score quantization/thresholded TopK/plugin을 별도 정확도 정책으로 검토한다.
- `--sort_topk_indices` 전체 적용은 원본 raw output과 detection을 바꾸므로 기본 경로에 넣지 않는다. 대신 두 번째 TopK만 정렬하는 제한 변환을 candidate TensorRT 경로로 유지하고 더 많은 sample에서 검증한다.
- GPU NMS post-processing 비교는 현재 target 환경에서 exit code `139`로 실패하므로, 정확한 rotated NMS parity는 별도 안정화 대상으로 둔다. 당장은 `--postprocess_mode numpy_nms`로 최종 detection 변화 방향을 추적한다.
- 다음 단계는 second-only sorted TopK direct TensorRT engine을 여러 실제 sample에서 검증하고, PyTorch/base ORT/direct TensorRT의 latency benchmark를 같은 입력 조건에서 측정하는 것이다.

### Jetson ROS 앱 1단계: TensorRT 단일 프레임 smoke

지원 영역:

- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 완료
- 완료일: 2026-06-12
- Jetson target 컨테이너 `ia-ssd-target:latest`에서 TensorRT plugin을 빌드하고 direct TensorRT engine 단일 프레임 실행을 확인했다.
- Jetson의 CMake 3.16 + CUDA 11.4 조합은 `CUDA17` 표준 플래그를 알지 못하므로, `tools/iassd_trt_plugins/CMakeLists.txt`의 CUDA 표준을 14로 낮췄다. plugin C++ host 표준은 기존처럼 C++17로 유지한다.
- host `python3`에는 `torch`가 없어 직접 smoke는 실패했지만, target Docker 환경에서는 PyTorch/TensorRT 의존성이 잡혀 검증이 통과했다.
- 기존 build cache는 host 경로(`/home/tisc/...`)와 컨테이너 경로(`/workspace/IA-SSD`)가 달라 충돌하므로, 컨테이너에서는 `tools/iassd_trt_plugins/build_target`을 사용했다.

검증 명령:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "cd /workspace/IA-SSD && \
   cmake -S tools/iassd_trt_plugins -B tools/iassd_trt_plugins/build_target && \
   cmake --build tools/iassd_trt_plugins/build_target -j && \
   python3 tools/validate_iassd_trt_engine.py \
     --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
     --engine_file onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine \
     --plugin_library tools/iassd_trt_plugins/build_target/libiassd_trt_plugins.so \
     --num_points 16384 \
     --skip_torch \
     --skip_ort \
     --report_file onnx_exports/stage7/jetson_step1_trt_smoke_report.json"
```

검증 결과:

- report: `onnx_exports/stage7/jetson_step1_trt_smoke_report.json`
- 입력 `points`: `[16384, 5]`, `torch.float32`, `cuda:0`
- TensorRT 출력:
  - `batch_cls_preds`: `[256, 3]`, `float32`
  - `batch_box_preds`: `[256, 7]`, `float32`
- 이 단계는 ROS subscribe와 시각화 전송 전, Jetson TensorRT runtime/plugin/engine 연결이 동작함을 확인하는 smoke 기준이다.

### Jetson ROS 앱 2단계: `/lidar_sync` 입력과 MarkerArray publish

지원 영역:

- ROS1 포인트클라우드 입력
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 완료
- 완료일: 2026-06-12
- `tools/iassd_trt_runtime.py`를 추가해 TensorRT engine/plugin 로딩과 `points` 입력 추론을 앱용 class로 분리했다.
- `tools/iassd_postprocess.py`를 추가해 TensorRT raw output에서 score/label/box 후보를 만들고, CPU axis-aligned BEV NMS fallback으로 최종 detection을 만든다. 이 NMS는 ROS smoke용이며 정확한 CUDA rotated NMS parity는 별도 안정화 대상이다.
- `tools/ros_iassd_trt_lidar_node.py`를 추가했다. `/lidar_sync` `sensor_msgs/PointCloud2`를 구독하고, IA-SSD TensorRT 추론 후 `/iassd_trt/boxes` `visualization_msgs/MarkerArray`를 publish한다.
- PointCloud2 decoding은 `x, y, z` 필드를 필수로 보고 `intensity`가 없으면 0으로 채운다. field offset, `point_step`, row padding을 반영한다.
- 전처리는 KITTI config의 `POINT_CLOUD_RANGE`를 적용하고 OpenPCDet `sample_points`와 같은 near/far sampling 규칙으로 `16384` points를 맞춘 뒤, batch index를 붙여 `[N, 5]` TensorRT 입력을 만든다.
- 노드는 단계별 latency를 주기적으로 ROS log로 출력한다: decode, preprocess, TensorRT inference, postprocess, marker publish, total.

검증:

- host Python 문법 검사 통과:

```bash
python3 -m py_compile \
  tools/iassd_trt_runtime.py \
  tools/iassd_postprocess.py \
  tools/ros_iassd_trt_lidar_node.py
```

- target Docker에서 ROS/TensorRT 노드 import 통과:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 -c 'import tools.ros_iassd_trt_lidar_node; print(\"node import ok\")'"
```

- 분리한 TensorRT runner 단일 입력 실행 통과:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 -c 'import torch; from tools.iassd_trt_runtime import IASSDTensorRTRunner; r=IASSDTensorRTRunner(\"onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine\", \"tools/iassd_trt_plugins/build_target/libiassd_trt_plugins.so\"); x=torch.zeros((16384,5), dtype=torch.float32, device=\"cuda\"); y=r.infer(x); print({k:v.shape for k,v in y.items()})'"
```

실행 예시:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 tools/ros_iassd_trt_lidar_node.py \
     --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
     --engine_file onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine \
     --plugin_library tools/iassd_trt_plugins/build_target/libiassd_trt_plugins.so \
     --lidar_topic /lidar_sync \
     --boxes_topic /iassd_trt/boxes \
     --num_points 16384"
```

다음 단계:

- Jetson에서 `/image_raw_sync`, `/lidar_sync`, `/iassd_trt/boxes`를 원격 시각화 머신 `192.168.0.241`에서 볼 수 있게 topic 재전송 정책을 추가한다.
- 원격 RViz 기준으로 `/iassd_trt/lidar`, `/iassd_trt/image_raw` 또는 compressed image, `/iassd_trt/boxes`를 확인한다.

### Jetson ROS 앱 3단계: 원격 시각화용 topic 재전송

지원 영역:

- ROS1 포인트클라우드 입력
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 완료
- 완료일: 2026-06-12
- `tools/ros_iassd_trt_lidar_node.py`에 원격 RViz 확인용 topic 재전송 옵션을 추가했다.
- LiDAR 재전송:
  - `--republish_lidar raw`: 입력 `/lidar_sync` `PointCloud2`를 `/iassd_trt/lidar`로 그대로 publish한다.
  - `--republish_lidar filtered`: KITTI `POINT_CLOUD_RANGE`로 필터링한 point cloud를 `/iassd_trt/lidar`로 publish한다.
  - `--republish_lidar off`: LiDAR 재전송을 끈다.
  - `--vis_point_stride`로 filtered visual point cloud만 stride downsample할 수 있다.
- 이미지 재전송:
  - `--republish_image raw`: 입력 `/image_raw_sync` `Image`를 `/iassd_trt/image_raw`로 그대로 publish한다.
  - `--republish_image off`: 이미지 재전송을 끈다.
- `/iassd_trt/boxes`는 기존처럼 `MarkerArray`로 publish한다.
- 이 단계의 목적은 Jetson에서 추론한 결과를 display가 연결된 시각화 머신 `192.168.0.241`의 RViz에서 확인하는 것이다. ROS1 네트워크 연결은 같은 `ROS_MASTER_URI`와 각 머신의 `ROS_IP` 설정에 의존한다.

검증:

- host Python 문법 검사 통과:

```bash
python3 -m py_compile \
  tools/iassd_trt_runtime.py \
  tools/iassd_postprocess.py \
  tools/ros_iassd_trt_lidar_node.py
```

- target Docker에서 ROS 노드 import 통과:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 -c 'import tools.ros_iassd_trt_lidar_node; print(\"node import ok\")'"
```

Jetson 실행 예시:

```bash
export ROS_MASTER_URI=http://<jetson_ip>:11311
export ROS_IP=<jetson_ip>

docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 tools/ros_iassd_trt_lidar_node.py \
     --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
     --engine_file onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine \
     --plugin_library tools/iassd_trt_plugins/build_target/libiassd_trt_plugins.so \
     --lidar_topic /lidar_sync \
     --image_topic /image_raw_sync \
     --boxes_topic /iassd_trt/boxes \
     --publish_lidar_topic /iassd_trt/lidar \
     --publish_image_topic /iassd_trt/image_raw \
     --republish_lidar raw \
     --republish_image raw \
     --num_points 16384"
```

시각화 머신 `192.168.0.241` 설정 예시:

```bash
export ROS_MASTER_URI=http://<jetson_ip>:11311
export ROS_IP=192.168.0.241
rviz
```

RViz 표시 항목:

- `PointCloud2`: `/iassd_trt/lidar`
- `MarkerArray`: `/iassd_trt/boxes`
- `Image`: `/iassd_trt/image_raw`

남은 제약:

- raw image와 raw point cloud를 모두 전송하면 네트워크 대역폭이 커질 수 있다. 필요하면 다음 단계에서 `image_transport/compressed` 또는 `sensor_msgs/CompressedImage` publish 옵션을 추가한다.
- 이미지 위 3D box overlay는 camera-LiDAR calibration과 projection 코드가 필요하므로 별도 단계로 둔다.

### Jetson ROS 앱 4단계: PyTorch 실시간 baseline 노드

지원 영역:

- ROS1 포인트클라우드 입력
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 완료
- 완료일: 2026-06-12
- `tools/ros_iassd_torch_lidar_node.py`를 추가했다.
- 이 노드는 `/lidar_sync` `sensor_msgs/PointCloud2`를 구독하고, TensorRT 노드와 같은 전처리/후처리/MarkerArray publish 구조로 PyTorch IA-SSD baseline 결과를 `/iassd_torch/boxes`에 publish한다.
- 입력 전처리는 TensorRT 노드와 같은 helper를 재사용한다:
  - `x, y, z, intensity` 추출
  - KITTI `POINT_CLOUD_RANGE` 필터
  - OpenPCDet `sample_points`와 같은 near/far sampling
  - batch index를 붙인 `[N, 5]` 입력 생성
- 후처리는 TensorRT 노드와 같은 `tools/iassd_postprocess.py` CPU NMS fallback을 사용한다. 따라서 ROS 실시간 비교에서는 PyTorch와 TensorRT의 post-processing 차이를 줄이고 raw forward 경로 차이를 보기 쉽다.
- OpenPCDet import 경로가 `spconv`를 먼저 찾으므로, IA-SSD처럼 sparse convolution을 쓰지 않는 모델을 위해 검증 스크립트와 같은 spconv import stub을 포함했다.

검증:

- host Python 문법 검사 통과:

```bash
python3 -m py_compile tools/ros_iassd_torch_lidar_node.py
```

- target Docker에서 노드 import 통과:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 -c 'import tools.ros_iassd_torch_lidar_node; print(\"torch node import ok\")'"
```

- target Docker에서 checkpoint 로딩과 단일 raw forward 통과:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 -c 'import torch; from tools.ros_iassd_torch_lidar_node import IASSDTorchRunner; r=IASSDTorchRunner(\"tools/cfgs/kitti_models/IA-SSD.yaml\", \"tools/IA-SSD.pth\", None); x=torch.zeros((16384,5), dtype=torch.float32, device=\"cuda\"); y=r.infer_raw(x); print({k:v.shape for k,v in y.items()})'"
```

검증 결과:

- checkpoint: `tools/IA-SSD.pth`
- loaded params: `221/221`
- PyTorch 출력:
  - `batch_cls_preds`: `[256, 3]`
  - `batch_box_preds`: `[256, 7]`

Jetson 실행 예시:

```bash
docker run -it --rm \
  --name iassd_torch \
  --runtime nvidia \
  --net host \
  -e ROS_MASTER_URI=http://192.168.0.190:11311 \
  -e ROS_IP=192.168.0.190 \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 tools/ros_iassd_torch_lidar_node.py \
     --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
     --ckpt tools/IA-SSD.pth \
     --lidar_topic /lidar_sync \
     --image_topic /image_raw_sync \
     --boxes_topic /iassd_torch/boxes \
     --republish_lidar off \
     --republish_image off \
     --score_thresh 0.01 \
     --latency_log_interval 1 \
     --num_points 16384"
```

RViz 비교 topic:

- TensorRT boxes: `/iassd_trt/boxes`
- PyTorch boxes: `/iassd_torch/boxes`
- PointCloud2: `/iassd_trt/lidar` 또는 원본 `/lidar_sync`

비교 시 주의:

- PyTorch baseline과 TensorRT 노드를 동시에 실행하면 같은 `/lidar_sync`를 각각 처리하므로 GPU 부하가 합산된다. 속도 비교는 단독 실행과 동시 실행을 구분해 기록해야 한다.
- 두 노드는 같은 seed 기본값 `1024`를 쓰지만, callback마다 sampling RNG 상태가 진행되므로 엄밀한 frame-level parity 비교가 필요하면 sampling index를 고정하거나 입력 point count가 항상 `16384`가 되도록 별도 deterministic 전처리를 추가한다.

### Jetson profile: PyTorch vs direct TensorRT raw forward

지원 영역:

- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 완료
- 완료일: 2026-06-12
- `tools/profile_iassd_torch_trt.py`를 추가해 ROS callback, PointCloud2 decode, topic publish를 제외한 PyTorch raw forward와 direct TensorRT 실행 시간을 분리 측정했다.
- 측정 항목:
  - `pytorch_raw_forward`: PyTorch IA-SSD raw forward
  - `direct_trt_gpu_only`: TensorRT engine 실행만 측정, output GPU->CPU copy 제외
  - `direct_trt_wrapper_with_cpu_copy`: ROS TensorRT 노드가 쓰는 wrapper 경로, output GPU->CPU copy 포함
  - `pytorch_numpy_postprocess`, `trt_numpy_postprocess`: 공통 NumPy postprocess fallback

검증 명령:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-target \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD && \
   python3 tools/profile_iassd_torch_trt.py \
     --cfg_file tools/cfgs/kitti_models/IA-SSD.yaml \
     --ckpt tools/IA-SSD.pth \
     --engine_file onnx_exports/stage2/ia_ssd_kitti_direct_trt_from_iassd_flat_fp32_sort_topk_second.engine \
     --plugin_library tools/iassd_trt_plugins/build_target/libiassd_trt_plugins.so \
     --num_points 16384 \
     --warmup 10 \
     --iterations 30 \
     --score_thresh 0.01 \
     --report_file onnx_exports/stage7/jetson_profile_torch_trt_raw_30.json"
```

측정 환경:

- platform: `Linux-5.10.120-tegra-aarch64-with-glibc2.29`
- CUDA device: `Orin`
- PyTorch: `2.0.0a0+ec3941ad.nv23.02`
- CUDA: `11.4`
- input: `[16384, 5]`, `torch.float32`, `cuda:0`
- warmup: `10`
- iterations: `30`

측정 결과:

| 항목 | mean ms | fps |
| --- | ---: | ---: |
| PyTorch raw forward | 47.97 | 20.85 |
| direct TensorRT GPU-only | 43.36 | 23.06 |
| direct TensorRT wrapper + CPU output copy | 44.14 | 22.65 |
| PyTorch NumPy postprocess | 0.40 | 2484.68 |
| TensorRT NumPy postprocess | 0.30 | 3359.98 |

해석:

- ROS를 제외한 pure raw forward 기준에서는 TensorRT가 PyTorch보다 빠르다.
- TensorRT GPU-only와 wrapper+CPU copy 차이는 약 `0.78ms`라 output CPU copy는 이번 측정에서 큰 병목은 아니다.
- PyTorch가 실시간 실행에서 더 빨라 보인다면 raw inference보다는 ROS callback 구성 차이가 원인일 가능성이 높다.
  - TensorRT 노드는 `/iassd_trt/lidar`, `/iassd_trt/image_raw/compressed`, `/iassd_trt/boxes`를 함께 publish하는 경우가 많다.
  - PyTorch 노드는 비교 실행 예시에서 `--republish_lidar off`, `--republish_image off`로 둔 상태라 publish 비용이 더 작다.
  - 두 노드를 동시에 실행하면 GPU와 ROS callback queue 부하가 합산되어 단독 latency와 달라진다.

다음 액션:

- TensorRT와 PyTorch 노드를 같은 publish 조건으로 맞춘 뒤 ROS callback latency를 비교한다.
  - 공정한 inference 비교: 둘 다 `--republish_lidar off --republish_image off`
  - 원격 시각화 포함 end-to-end 비교: 둘 다 같은 LiDAR/Image publish 정책 사용
- 실제 `/lidar_sync` 입력에서 `decode`, `preprocess`, `infer`, `post`, `pub`, `total` 로그를 CSV로 저장하는 옵션을 추가한다.

### Jetson profile: pointcloud-3d-detector-tensorrt submodule

지원 영역:

- ROS1 포인트클라우드 입력
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교

진행 기록:

- 상태: 부분 완료
- 완료일: 2026-06-12
- submodule `pointcloud-3d-detector-tensorrt`를 추가해 HAV/GridBallQuery 기반 IA-SSD TensorRT 구현을 검토했다.
- Jetson target Docker에서 빌드되도록 다음을 수정했다.
  - `plugins/CMakeLists.txt`: Jetson CUDA library 경로 `${CUDA_TOOLKIT_ROOT_DIR}/targets/aarch64-linux/lib`를 search hint에 추가했다.
  - `plugins/CMakeLists.txt`: Jetson CUDA 11.4 빌드 호환을 위해 plugin CUDA 표준을 14로 낮췄다.
  - `src/point_detector.cpp`: ROS topic과 config를 런타임 private param으로 받도록 변경했다.
    - `_config`
    - `_input_topic`
    - `_output_topic`
  - `src/profile_detector.cpp`: ROS 없이 detector latency를 측정하는 profiler를 추가했다.
  - `config/trt_hvcsx2.yaml`: GQ가 아닌 `iassd_hvcsx2_4x8_80e_kitti_3cls(export).onnx` 기반 FP16 config를 추가했다.
  - `docker/Dockerfile.havtrt`: `ia-ssd-target`에 `libyaml-cpp-dev`를 더한 파생 image를 정의했다.

빌드:

```bash
docker build -f docker/Dockerfile.havtrt -t ia-ssd-havtrt .

docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-havtrt \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD/pointcloud-3d-detector-tensorrt && \
   cmake -S . -B build_iassd_target \
     -DCMAKE_BUILD_TYPE=Release \
     -DTRT_QUANTIZE=FP16 \
     -DTENSORRT_DIR=/usr \
     -DCUDNN_DIR=/usr && \
   cmake --build build_iassd_target -j"
```

프로파일:

```bash
docker run --runtime nvidia --rm --net host \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-havtrt \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD/pointcloud-3d-detector-tensorrt && \
   ./build_iassd_target/devel/lib/point_detection/profile_detector \
     --config config/trt_hvcsx2.yaml \
     --warmup 10 \
     --iterations 30"
```

측정 결과:

| 항목 | mean ms | fps |
| --- | ---: | ---: |
| PyTorch raw forward | 47.97 | 20.85 |
| 기존 direct TensorRT wrapper + CPU output copy | 44.14 | 22.65 |
| submodule HAV TensorRT FP16 detector | 17.07 | 58.59 |

해석:

- submodule의 non-GQ HAV TensorRT FP16 detector는 synthetic 입력 기준 기존 PyTorch/직접 변환 TensorRT보다 뚜렷하게 빠르다.
- submodule detector 측정은 입력 host->device copy, TensorRT 실행, output device->host copy, plugin NMS output까지 포함하는 `TRTDetector3D::operator()` 기준이다.
- 현재 프로젝트의 기존 direct TensorRT는 원본 IA-SSD 구조와 parity를 유지하는 쪽이고, submodule은 HAV sampler와 plugin NMS를 포함한 별도 모델/ONNX라 정확도와 출력 의미는 별도 검증이 필요하다.
- 기본 `config/trt.yaml`의 GQ variant(`iassd_hvcsx2_gq_...`)는 Orin에서 engine build 후 첫 inference가 30초 timeout 안에 반환되지 않았다. 따라서 현재 Jetson 실험 기본값은 `config/trt_hvcsx2.yaml`로 둔다.

ROS 실행 예시:

```bash
docker run -it --rm \
  --name iassd_havtrt \
  --runtime nvidia \
  --net host \
  -e ROS_MASTER_URI=http://192.168.0.190:11311 \
  -e ROS_IP=192.168.0.190 \
  -v /home/tisc/data/sangjune/IA-SSD:/workspace/IA-SSD \
  ia-ssd-havtrt \
  "source /opt/ros/noetic/setup.bash && \
   cd /workspace/IA-SSD/pointcloud-3d-detector-tensorrt && \
   ./build_iassd_target/devel/lib/point_detection/point_detector \
     _config:=config/trt_hvcsx2.yaml \
     _input_topic:=/lidar_sync \
     _output_topic:=/iassd_havtrt/boxes"
```

RViz 비교 topic:

- 기존 TensorRT: `/iassd_trt/boxes`
- PyTorch: `/iassd_torch/boxes`
- submodule HAV TensorRT: `/iassd_havtrt/boxes`

남은 작업:

- 실제 `/lidar_sync` 입력에서 submodule HAV TensorRT의 marker가 좌표계/크기/heading 기준으로 정상 표시되는지 확인한다.
- submodule 모델은 별도 ONNX/checkpoint 계열이므로, 현재 `tools/IA-SSD.pth` 기반 PyTorch baseline과 detection 품질을 직접 비교하려면 동일 sample에 대한 정성/정량 검증이 필요하다.
- GQ variant hang 원인을 별도 분석한다. 후보는 GridBallQuery plugin의 Orin/TensorRT 8.5 호환성, CUDA arch flag, 입력 분포, plugin workspace/shape 처리다.

## 개발 우선순위

1. KITTI raw PyTorch output shape 저장
2. `IASSD::FarthestPointSampling` placeholder로 첫 export blocker 우회
3. `IASSD::GatherPoints` placeholder로 두 번째 export blocker 우회
4. `IASSD::BallQuery` placeholder로 세 번째 export blocker 우회
5. `IASSD::GroupPoints` placeholder로 네 번째 export blocker 우회
6. `IASSD_backbone.py`의 export 비친화적인 `EasyDict.get(...)` 설정 조회 정리
7. `IASSD_head.py`의 export 비친화적인 `forward_ret_dict` 모듈 속성 mutation 정리
8. custom op lowering 출력 메타데이터/시퀀스 처리 문제 정리
9. ONNX graph checker 통과
10. ONNX local function wrapper 구조와 ORT custom op 로딩 실패 지점 확인
11. ORT custom op로 `FarthestPointSampling` 단위 테스트 통과
12. 1차 custom op 전체 ORT 단위 테스트 통과
13. KITTI IA-SSD ORT raw output parity test
14. TensorRT plugin으로 동일 op 이식
15. 필요 시 TensorRT plugin 경로용 ONNX custom node 평탄화
16. ORT TensorRT EP 통합
17. benchmark 작성

## 주요 리스크

- PyTorch exporter API 버전에 따라 custom node 생성 방식이 달라질 수 있다.
- ONNX Runtime custom op와 TensorRT plugin의 shape inference 및 memory layout을 동일하게 유지해야 한다.
- FPS, BallQuery는 index 결과가 후속 feature aggregation에 크게 영향을 주므로 exact parity가 중요하다.
- TensorRT plugin 구현은 ORT custom op보다 비용이 크다.
- KITTI checkpoint가 없으면 의미 있는 정확도 비교는 어렵고, 우선 shape/export/runtime 검증만 가능하다.

## 다음 액션

다음 작업은 second-only sorted TopK direct TensorRT engine을 여러 실제 sample에서 PyTorch/base ORT와 비교한 뒤, 같은 입력 조건으로 latency benchmark를 수행하는 것이다. 첫 번째 score-based sampling `TopK`는 정렬 시 정확도 보존성이 깨졌으므로 현재 기본 변환 대상에서 제외한다.

최종 TensorRT binary와 benchmark는 target machine에서 다시 빌드/검증한다.
