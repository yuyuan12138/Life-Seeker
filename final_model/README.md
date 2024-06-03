# 最终模型（干扰卷积 + 带权Transformer + 预训练模型）

## 文件说明

* **data/\***：原始数据（来自iDNA-MS）
* **combined_data/\***：对预训练模型进行训练采用的数据，由data文件夹整合而来
* **4mC.pth**、**5hmC.pth**、**6mA.pth**：对应甲基化类型的预训练模型文件，不带Transformer权重（对应model.py）
* **4mC_dzy.pth**、**5hmC_dzy.pth**、**6mA_dzy.pth**：带Tranformer权重的预训练模型（对应model_dzy.py，use_spectic_conv1d=True, use_spectic_transformer=True）
* **combine_data.py**：脚本文件，用于生成预训练数据

## config.py新增参数说明
* self.combined：是否对预训练模型进行训练；True时读入combined_data，写入预训练模型；False时以预训练模型为基础，读入data进行模型微调
* self.model：当前数据对应的预训练模型文件名称；当self.combined为True时写入到该文件中，否则从该文件中读取预训练模型

## 注意事项
* 程序不会覆盖存在的预训练模型文件。对预训练模型进行训练前，需要删除self.model对应的同名文件。