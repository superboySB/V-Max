FROM nvcr.io/nvidia/jax:24.04-py3

LABEL maintainer="Zipeng Dai <daizipeng@bit.edu.cn>"

ENV DEBIAN_FRONTEND=noninteractive

# 如果需要走代理
ENV http_proxy=http://127.0.0.1:8889
ENV https_proxy=http://127.0.0.1:8889

RUN apt-get update && apt-get install -y --no-install-recommends \
    git tmux vim gedit curl wget libgl1 ffmpeg libpng-dev libjpeg-dev

WORKDIR /workspace

RUN git clone https://github.com/superboySB/ScenarioMax
RUN cd ScenarioMax && pip install -e .
RUN cd ScenarioMax && pip install -e devkit/nuplan-devkit && \
    pip install -r devkit/nuplan-devkit/requirements.txt

RUN git clone https://github.com/superboySB/V-Max
RUN cd V-Max && pip install --upgrade pip && pip install -r requirements.txt
RUN cd V-Max && pip install -e .

# 如需清理代理，取消注释
# ENV http_proxy=
# ENV https_proxy=
# ENV no_proxy=
# RUN rm -rf /var/lib/apt/lists/* && apt-get clean

CMD ["/bin/bash"]