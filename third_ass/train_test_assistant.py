
import os
# 训练模型，然后测试数据，得到结果
for i in range(5):
	train_order = 'crf_learn.exe -f 5  template_goodone.txt %dtrain_data.txt model%d' % (i,i)
	print(train_order)
	os.system(train_order)

	test_order = 'crf_test.exe -m .\model%d %dtest_data.txt -o %d_result' % (i,i,i)
	print(test_order)
	os.system(test_order)