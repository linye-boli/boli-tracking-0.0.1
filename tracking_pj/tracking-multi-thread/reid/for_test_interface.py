import reid_interface
import numpy as np
import torch
import os
from PIL import Image 
import time
img = Image.open('/file/reid_eval/0001_c3s1_000551_00.jpg')
img =  img.convert('RGB')
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
# Extract feature
since = time.time()
start = time.clock()
with torch.no_grad():
    count =0
    for i in range(101):
        reid = reid_interface.ReID(is_folder=False)
        query_feature = reid.get_feature(img)
        count += 1
        if count == 100:
            break
    end  = time.clock()-start
    print('Extract features complete in {:.5f}s'.format(end))
    print('per img : {:.5f}s'.format(float(end)/(float(count))))

    time_elapsed = time.time() - since
    print('Extract features complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
# tensor([[ 0.0188,  0.0483, -0.0457,  ...,  0.0165, -0.0991, -0.0041]])
print(query_feature.shape)
print('----------------------')

query_feature = query_feature.cpu().numpy()
np.savetxt("temp_result4test.txt",query_feature)

