训练集放在datasets文件夹下
lfw的图片数据放在lfw文件夹下
logs用于存放训练产生的模型等信息
nice_logs是我训练得到的比较好的模型的权值文件


训练请先运行txt_annotation.py生成训练集记录文件路径的txt，再运行train.py

在lfw上通过lfw提供的lfw_pair.txt进行验证请运行eval_LFW.py
要使用预测功能请运行predict.py