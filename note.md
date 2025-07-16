# 笔记
## 配置
### 外网调试
```sh
docker build -t dzp_waymax:test --network=host --progress=plain .

xhost +

docker run -itd --privileged --gpus all --net=host \
  -e DISPLAY=$DISPLAY \
  --name dzp-waymax-test \
  dzp_waymax:test \
  /bin/bash

docker exec -it dzp-waymax-test /bin/bash
```
### 内网使用
```sh
docker run -itd --privileged --gpus all --net=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro -v /dev/shm:/dev/shm \
  -v /disk:/disk -v /disk1:/disk1 -v /mnt:/mnt \
  --name dzp-waymax-test \
  dzp_waymax:test \
  /bin/bash

docker exec -it dzp-waymax-test /bin/bash

cd /workspace/ScenarioMax

# num_worker不超过原有tfrecord块数
python scenariomax/convert_dataset.py \
  --waymo_src /disk/deepdata/dataset/waymo_motion/waymo/uncompressed/scenario/small_valid \
  --dst /workspace/V-Max/womd_valid \
  --target_format tfexample \
  --num_workers 3 \
  --tfrecord_name valid

python -m vmax.scripts.evaluate.evaluate --scenario_indexes 0 --sdc_actor expert --render True --path_dataset womd_valid --batch_size 1
```

## 核心问题
1. At the moment we don't support multi step prediction and trajectory action.