# 笔记
## 配置
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
