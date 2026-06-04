# IA-SSD ONNX Runtime Custom Ops

KITTI IA-SSD ONNX 변환 4단계에서 사용하는 ONNX Runtime custom op 라이브러리 소스다.

현재 포함된 op:

- `IASSD::FarthestPointSampling`
- `IASSD::GatherPoints`
- `IASSD::BallQuery`
- `IASSD::GroupPoints`

빌드 예시:

```bash
cmake -S tools/iassd_ort_ops -B tools/iassd_ort_ops/build
cmake --build tools/iassd_ort_ops/build -j
```

단위 테스트 예시:

```bash
python3 tools/test_iassd_ort_fps.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda
python3 tools/test_iassd_ort_gather.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda
python3 tools/test_iassd_ort_ball_query.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda
python3 tools/test_iassd_ort_group.py \
  --ort_op_library tools/iassd_ort_ops/build/libiassd_ort_ops.so \
  --device cuda
```

이 라이브러리는 `pcdet/ops/pointnet2/pointnet2_batch/src/*_raw.h`의 포인터 기반 CUDA launcher를 재사용한다.

`GatherPoints`와 `GroupPoints`의 index 입력은 현재 `OrtMemTypeCPUInput`으로 받아 CUDA kernel 호출 직전에 device buffer로 복사한다. 전체 모델 검증에서 `TopK`/`Cast`가 만든 index tensor를 안정적으로 처리하기 위한 경로이며, 최종 benchmark 전에는 device index 경로 최적화 여부를 다시 확인한다.
