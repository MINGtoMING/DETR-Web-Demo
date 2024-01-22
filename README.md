# <center>DETR基于Sophon BM1684X TPU的部署</center>
本项目在搭载**Sophon BM168X TPU**的开发板的linux环境中实现了**DETR**目标检测模型的部署，并可用**Gradio Web Demo**的形式进行使用。

## 快速使用
请在搭载Sophon BM168X TPU的soc的linux环境中进行如下步骤：

下载sophon-sail的python依赖包，并进行安装。
```shell
pip3 install path/to/sophon_arm-3.6.0-py3-none-any.whl
```

克隆本项目到soc的linux环境的相应目录下。
```shell
git clone 
```

安装相关依赖：

```shell
pip install -r requirements.txt
```

下载转化好的DETR的模型权重压缩包，并解压到本项目的目录下：
```shell
tar -zxvf /path/to/checkpoints.tar.gz
```
查看本设备的ip地址, 假设为`192.168.11.3`：
```shell
ifconfig
```

运行gradio demo:
```shell
python3 app.py --server-name="0.0.0.0" --server-port=8888
```

最后在与本设备在同一局域网下的主机浏览器中输入`192.168.11.3:8888`进行访问即可。
