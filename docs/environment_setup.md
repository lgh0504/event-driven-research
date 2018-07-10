### 如何设置环境

1. 利用 conda 或者 virtualenv 创建虚拟环境, 安装需要安装的包即可

2. 安装时要选择tensorflow-gpu(1.8.0)的版本, 如果要采用这个版本,需要做以下复杂的工作:

   * 根据操作系统的版本, 安装cuda(应该是需要9.0版本), 选择它的自动化安装模式(runtime file), 在运行时设置路径, 当然之后的cudnn的安装和环境变量的设置会因此有所改变

   * 安装cudnn, 我目前用我的微信注册了账号,以后可以用微信登录, 选择版本时要当心对应cuda的版本

     ```bash
     # install cuda first
     sh cuda_script.sh
     
     # then install cudnn
     tar -xzvf cudnn.tgz
     cp cuda/include/cudnn.h path/to/cuda/include
     cp cuda/lib64/libcudnn* path/to/cuda/lib64
     chmod a+r path/to/cuda/include/cudnn.h path/to/cuda/lib64/libcudnn*
     ```


​     