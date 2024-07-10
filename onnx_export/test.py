import test_all
import numpy as np



t = np.random.randn(690, 640,3)
res  = test_all.padding_resize(t,(640,640))
print(res[0].shape)