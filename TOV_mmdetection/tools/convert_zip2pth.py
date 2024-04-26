import torch

zip_file = '/home/ubuntu/Guo/TOV_mmdetection-main/work-dir/Tinyperson/centernet_neck51_MT/epoch_150.pth'
statedict = torch.load(zip_file)
torch.save(statedict, '/home/ubuntu/Guo/TOV_mmdetection-main/work-dir/Tinyperson/centernet_neck51_MT/epoch_150_zip.pth',_use_new_zipfile_serialization=False)
