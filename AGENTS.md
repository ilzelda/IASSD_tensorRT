# AGENTS.md

## 프로젝트 맥락

이 저장소는 OpenPCDet 계열 구조를 기반으로 한 IA-SSD 3D LiDAR 객체 검출 모델을 포함한다. 현재 프로젝트의 목표는 이 모델을 실시간 추론 파이프라인으로 배포할 수 있게 변환하고, 변환 전후의 추론 성능을 비교하는 것이다.

## 장기 목표

다음 흐름의 추론 및 평가 워크플로를 구축한다.

1. 기존 PyTorch IA-SSD 모델을 ONNX로 변환한다.
2. 변환된 모델을 ONNX Runtime으로 추론한다.
3. ONNX Runtime의 TensorRT execution provider를 사용해 추론을 가속한다.
4. ROS1 토픽을 subscribe하여 포인트클라우드 입력을 받는다.
5. 기존 PyTorch 모델과 ONNX Runtime/TensorRT 경로의 추론 속도를 비교한다.

## 주요 가정

- 입력 데이터는 LiDAR 포인트클라우드이며, ROS1의 `sensor_msgs/PointCloud2` 토픽에서 들어오는 형태를 우선 고려한다.
- target machine은 현재 개발 중인 이 머신이다.
- 기존 PyTorch/OpenPCDet 방식의 추론 경로는 기준 성능 측정을 위해 계속 실행 가능해야 한다.
- 모델 변환 과정에서는 전처리, 후처리, 좌표계 관례, score threshold, 출력 box 형식을 가능한 한 기존 모델과 동일하게 유지한다.
- 속도 비교 시 가능하면 모델 순수 추론 시간과 ROS subscribe, 포인트클라우드 디코딩, 전처리, 후처리 시간을 분리해서 측정한다.

## 예상 작업 영역

- PyTorch 모델을 ONNX로 export하는 스크립트.
- ONNX 그래프 검증 및 ONNX Runtime 추론 스크립트.
- TensorRT execution provider 설정.
- 포인트클라우드 입력을 받는 ROS1 subscriber 노드.
- 변환 전후 latency 및 throughput 비교를 위한 benchmark 스크립트 또는 로깅 유틸리티.
- 환경 요구사항, 변환 절차, 실행 명령어, 알려진 제약사항 문서화.

## 개발 메모

- 새 스크립트는 특별한 이유가 없으면 `tools/` 아래에 추가한다.
- 기존 IA-SSD/OpenPCDet 프로젝트 구조와 호환되도록 변경한다.
- 학습된 weight 파일이나 dataset 파일은 수정하지 않는다.
- KITTI config 기반 ONNX custom ops 변환 작업은 `docs/KITTI_ONNX_CUSTOM_OPS_PLAN.md`를 기준 계획으로 참고한다.
- ONNX custom ops 변환 계획, 범위, 우선순위, 리스크, 검증 결과가 바뀌면 관련 코드 변경과 함께 `docs/KITTI_ONNX_CUSTOM_OPS_PLAN.md`를 갱신한다.
- CUDA, PyTorch, ONNX, ONNX Runtime, TensorRT, ROS 버전은 재현성에 중요한 정보로 취급한다.
- TensorRT engine 생성과 최종 benchmark는 현재 머신에서 수행한다.
- benchmark 시 hardware, batch size, point count, precision mode, warmup 횟수, 측정 iteration 수, 전처리 및 후처리 포함 여부를 함께 기록한다.
- 앞으로 새로 작성하거나 수정하는 코드의 주석은 모두 한글로 작성한다.

## 소통 방식

이 프로젝트는 단계적으로 진행한다. 변경 작업을 할 때는 각 작업이 다음 항목 중 무엇을 지원하는지 함께 설명한다.

- 모델 변환
- ROS1 포인트클라우드 입력
- ONNX Runtime/TensorRT 추론
- 변환 전후 추론 속도 비교
