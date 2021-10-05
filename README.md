# URLCD
###########环境依赖
python 3.7以上

####依赖库
tensorflow2.0-gpu

###########部署步骤
确保拥有python环境以及tensorflow2.0，以便能够更好复现本文结果
（作者的环境：Python3.7.5,CPU为i7-9750H,GPU为GeForce GTX 1660Ti with Max-Q Design(内存6G),单个程序占用的最大显存为2.51GIB,电脑内存为16G。）


###########目录结构描述
├── Readme.md                   // help
├── data                         // ft数据集
├── data2                     // it数据集
├── result                         // 代码每次运行会记录全部数据结果，每次运行会覆盖
├── GF.py           //复现的GF
├── LINE.py         //复现的LINE
├── BiNE.py         //复现的BiNE
├── DeppWalk.py         //复现的DeepWalk
├── URLCD.py         // 我们自己提出的模型
