# import logging
# logging.basicConfig(level=logging.DEBUG)
# # 默认是warning级别，只输出warning级别后的面内容
# logging.debug('this is a debug')
# logging.info('this is a info')
# logging.error('this is a error')
# logging.warning('this is a warning')
# logging.critical('this is a critical')
b = []
import torch
a = torch.arange(10)
# print(a.reshape(-1,1))
a = a.reshape(-1,1)
b = b.append(a)
print(b)

