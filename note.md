# 笔记
## 配置
### 外网调试
```sh
docker build -t dzp_waymax:0717 --network=host --progress=plain .

xhost +

docker run -itd --privileged --gpus all --net=host -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix:ro --shm-size=4g \
  -v /home/dzp/Public/tiny_waymo:/workspace/tiny_waymo \
  --name dzp-waymax-0717 \
  dzp_waymax:0717 \
  /bin/bash

docker exec -it dzp-waymax-0717 /bin/bash

cd /workspace/ScenarioMax

# num_worker不超过原有tfrecord块数
python scenariomax/convert_dataset.py \
  --waymo_src /workspace/tiny_waymo \
  --dst /workspace/V-Max/ \
  --target_format tfexample \
  --num_workers 3 \
  --tfrecord_name womd_valid

cd /workspace/V-Max

python vmax/scripts/training/train.py total_timesteps=10 path_dataset=womd_valid.tfrecord algorithm=ppo network/encoder=mlp

# python -m vmax.scripts.evaluate.evaluate --sdc_actor ai --path_model name_of_the_run --path_dataset womd_valid --batch_size 8

python -m vmax.scripts.evaluate.evaluate --scenario_indexes 0 --sdc_actor expert --render True --path_dataset womd_valid.tfrecord --batch_size 1
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

cd /workspace/V-Max

python -m vmax.scripts.evaluate.evaluate --scenario_indexes 0 --sdc_actor expert --render True --path_dataset womd_valid --batch_size 1
```

## 核心问题
1. At the moment we don't support multi step prediction and trajectory action.
2. https://github.com/valeoai/V-Max/blob/459ac92b19a9d6fa5da05b6967174971d92dde4a/docs/training.md