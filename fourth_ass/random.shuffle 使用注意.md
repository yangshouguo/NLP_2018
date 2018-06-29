# 有关random.shuffle函数的局限性



刚刚踩的坑，记录一下，万一对别人有帮助呢。

本文重点： random.shuffle千万不要用于二维numpy.array（也就是矩阵）!!!

首先上证据

如下代码：

	import random
	import numpy as np
	a = np.array([[1,2,3,4],
	              [5,6,7,8]])
	random.shuffle(a)
	print(a)

***可能***对应输出：

	[[1 2 3 4]
	 [1 2 3 4]]
 
得到一个错误输出！
查看random.shuffle源码：

	    def shuffle(self, x, random=None):
        """Shuffle list x in place, and return None.

        Optional argument random is a 0-argument function returning a
        random float in [0.0, 1.0); if it is the default None, the
        standard random.random will be used.

        """

        if random is None:
            randbelow = self._randbelow
            for i in reversed(range(1, len(x))):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = randbelow(i+1)
                x[i], x[j] = x[j], x[i]
        else:
            _int = int
            for i in reversed(range(1, len(x))):
                # pick an element in x[:i+1] with which to exchange x[i]
                j = _int(random() * (i+1))
                x[i], x[j] = x[j], x[i]

只需要关注其中交换元素操作为 ***x[i], x[j] = x[j], x[i]*** （好骚的操作，我之前不知道有这种写法）。

自己写个代码测试一下，这种交换方式对于二维numpy.array会发生什么事情：



	import numpy as np
	a = np.array([[1,2,3,4],
	              [5,6,7,8]])
	a[0], a[1] = a[1], a[0]
	print(a)

输出：

	[[5 6 7 8]
	 [5 6 7 8]]
	
显然这种方式不适合numpy.array的行交换（但是二维list就可以使用这种交换方式。可以自行证明，至于原因暂时不知道，求解答）。

打乱numpy.array正确的姿势当然是使用numpy自带的*numpy.random.shuffle()*
	
	def shuffle(x): # real signature unknown; restored from __doc__
    """
    shuffle(x)
    
            Modify a sequence in-place by shuffling its contents.
    
            This function only shuffles the array along the first axis of a
            multi-dimensional array. The order of sub-arrays is changed but
            their contents remains the same.
    
            Parameters
            ----------
            x : array_like
                The array or list to be shuffled.
    
            Returns
            -------
            None
    
            Examples
            --------
            >>> arr = np.arange(10)
            >>> np.random.shuffle(arr)
            >>> arr
            [1 7 5 2 9 4 3 6 0 8]
    
            Multi-dimensional arrays are only shuffled along the first axis:
    
            >>> arr = np.arange(9).reshape((3, 3))
            >>> np.random.shuffle(arr)
            >>> arr
            array([[3, 4, 5],
                   [6, 7, 8],
                   [0, 1, 2]])
    """
    pass

注释中Multi-dimensional arrays are only shuffled along the first axis: 多维向量只是沿着第一个坐标轴进行重新排序。