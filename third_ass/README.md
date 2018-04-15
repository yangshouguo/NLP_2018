# CRF 作业说明

1. data_preprocess.py 用于从原始语料中提取特征，生成交叉验证用的5个训练集和测试集 （原始语料见第二题）
2. 使用train_test_assistant.py 进行交叉验证，得到5个result文件
3. 使用statistics.py 文件进行统计结果，得到 Precision Recall 和 F1 Score
4. 运行结果 ![Alt text](result_screenshot.png)