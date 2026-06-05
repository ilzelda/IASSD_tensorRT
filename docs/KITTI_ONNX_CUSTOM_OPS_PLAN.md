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
  -> ONNX Runtime TensorRT Execution Provider 또는 TensorRT engine
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
- 새 ONNX에는 `sigmoid -> Sub -> TopK`, `sigmoid_1 -> Sub -> TopK` 경로가 반영되어 있으며, `TopK` 입력인 `topk_input_0/sub_4`는 max abs `3.0174851417541504e-06`, mean abs `3.858489350250238e-08`로 거의 일치했다.
- 그러나 `eps=1e-7`에서는 이 수치 차이를 충분히 이기지 못해 `topk_indices_0/_to_copy_1`, `topk_indices_1/_to_copy_2`가 여전히 달랐다.
- tie-breaker 계수를 `1e-5`로 올렸으므로, 다음 검증은 PyTorch 2.5 계열 export 환경에서 ONNX를 다시 생성한 뒤 같은 trace를 반복한다.

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

남은 작업:

- shape와 runtime은 통과했지만 numerical parity 오차가 아직 크므로, 다음 단계에서 PyTorch `topk`와 ONNX Runtime `TopK`의 tie-breaking 차이를 줄이는 방식을 결정한다.
- 1차 후보는 이미 core에 반영한 deterministic index tie-breaker를 포함해 ONNX를 재export하는 방식이다. `eps=1e-5` 재검증에서도 index가 맞지 않으면 `TopK` sampling도 `IASSD::TopKSampling` custom op로 분리하는 방식을 검토한다.
- index CPU 왕복 복사는 정확도 검증 우선의 임시 안정화 경로다. benchmark 단계 전에는 ORT CUDA tensor memory 처리 방식에 맞춰 device index 경로를 최적화할지 결정한다.

### 6단계: TensorRT plugin 구현

산출물:

- `libiassd_trt_plugins.so`
- TensorRT plugin creator
- op별 plugin serialization/deserialization
- FP32 plugin 우선 구현

검증:

- `trtexec` 또는 ORT TensorRT EP에서 plugin library 로드
- ORT CUDA path와 TensorRT path output 비교
- FP16은 FP32 parity가 안정화된 뒤 진행

### 7단계: ORT TensorRT EP 통합

산출물:

- ONNX Runtime TensorRT Execution Provider 실행 스크립트
- TensorRT plugin library 로드 옵션
- engine/cache 디렉터리 관리

검증:

- TensorRT EP session 생성 성공
- unsupported op가 plugin으로 매핑되는지 확인
- PyTorch, ORT CUDA, ORT TensorRT latency 비교

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

다음 작업은 PyTorch 2.5 계열 export 환경에서 `eps=1e-5` deterministic TopK tie-breaker가 포함된 KITTI IA-SSD ONNX graph를 재생성하고, target에서 `tools/trace_iassd_ort_parity.py`로 TopK index와 final raw output parity를 다시 확인하는 것이다.

최종 ORT/TensorRT binary와 benchmark는 target machine에서 다시 빌드/검증한다.
