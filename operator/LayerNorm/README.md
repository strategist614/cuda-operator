## LayerNorm

### 安装 nsys
```
apt update
apt install -y --no-install-recommends gnupg lsb-release ca-certificates wget software-properties-common

echo "deb http://developer.download.nvidia.com/devtools/repos/ubuntu$(source /etc/lsb-release; echo "$DISTRIB_RELEASE" | tr -d .)/$(dpkg --print-architecture) /" \
  > /etc/apt/sources.list.d/nvidia-devtools.list

apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub

apt update
apt install -y nsight-systems-cli
```
