# Docker 환경

이 프로젝트는 역할에 따라 Docker 환경을 분리한다.

- `Dockerfile.export`: AMD/x86_64 서버에서 PyTorch 모델을 ONNX로 export하기 위한 환경
- `Dockerfile.target`: 현재 Jetson/L4T target machine에서 ROS1 입력, TensorRT 실행, benchmark를 수행하기 위한 환경
- `Dockerfile`: 현재는 target 환경과 같은 용도로 유지한다.

## ONNX Export 환경

AMD/x86_64 서버에서 빌드한다.

```shell
docker build -f docker/Dockerfile.export -t ia-ssd-export .
```

실행 시 소스코드는 bind mount해서 수정 사항이 바로 반영되게 사용한다.

```shell
docker run --gpus all -it --rm \
  -v $(pwd):/workspace/IA-SSD \
  ia-ssd-export
```

컨테이너 안에서 필요 시 마운트된 소스 기준으로 extension을 빌드한다.

```shell
python3 setup.py develop
```

## Target 환경

현재 Jetson/L4T target machine에서 빌드한다.

```shell
docker build -f docker/Dockerfile.target -t ia-ssd-target .
```

실행 시 ROS 통신을 위해 host network를 사용하고, 소스코드는 bind mount한다.

```shell
docker run --runtime nvidia -it --rm \
  --net host \
  -v $(pwd):/workspace/IA-SSD \
  ia-ssd-target
```

Jetson용 ONNX Runtime GPU wheel이 준비되어 있으면 빌드 시 URL을 넘긴다.

```shell
docker build -f docker/Dockerfile.target \
  --build-arg ONNXRUNTIME_GPU_WHEEL_URL=<wheel-url-or-local-path> \
  -t ia-ssd-target .
```

현재 Orin target에서는 `onnxruntime-gpu==1.16.0` Jetson wheel이 import와 CUDA/TensorRT provider 확인을 통과했다. NVIDIA Box URL처럼 URL의 마지막 파일명이 wheel 규칙과 맞지 않는 경우가 있어 Dockerfile은 URL을 `ONNXRUNTIME_GPU_WHEEL_NAME` 이름으로 내려받은 뒤 설치한다.

```shell
docker build -f docker/Dockerfile.target \
  --build-arg ONNXRUNTIME_GPU_WHEEL_URL=https://nvidia.box.com/shared/static/iizg3ggrtdkqawkmebbfixo7sce6j365.whl \
  --build-arg ONNXRUNTIME_GPU_WHEEL_NAME=onnxruntime_gpu-1.16.0-cp38-cp38-linux_aarch64.whl \
  -t ia-ssd-target .
```

Orin의 compute capability는 `8.7`이므로 target extension과 ORT custom op는 `TORCH_CUDA_ARCH_LIST=8.7` 또는 기본값에 포함된 `8.7`로 빌드해야 한다.

## 산출물 흐름

```text
AMD/x86_64 서버:
  PyTorch checkpoint + config
  -> ONNX export
  -> ia_ssd.onnx

현재 Jetson target:
  ia_ssd.onnx
  -> TensorRT engine/cache 생성
  -> ROS1 PointCloud2 입력 추론
  -> PyTorch baseline과 속도 비교
```
