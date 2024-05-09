import numpy as np
import mmcv
import os
import cv2

def show_fea(fea, img_file=None, name=None):
        """
        Args:
            fea: list[tensor]
            img_file: image path, to read image
            name: image name
        """
        save_dir = '/home/ubuntu/Guo/TOV_mmdetection-main/work-dir/Tinyperson/centernet_adaptive_fpn/visua/'
        assert os.path.exists(save_dir)
        if img_file is not None:
            img = mmcv.imread(img_file)
            img_name = img_file.split('/')[-1]
            print(img_name)
        # b, c, h, w = fea[0].shape

        for i, out in enumerate(fea):
            att = out.detach().cpu().numpy()[0]

            att = np.mean(att, 0)
            att = np.maximum(att, 0)
            # print(att.shape)
            att /= np.max(att)
            
            # att = att * 255
            if img_file is not None:
                att = cv2.resize(att, (img.shape[1], img.shape[0]))
            else:
                att = cv2.resize(att, (640, 512))
            att = np.uint8(att * 255)
            att = cv2.applyColorMap(att, cv2.COLORMAP_JET)
            if img_file is not None:
                att = 0.5 * att + 0.5 * img
    
            # mmcv.imshow(att)
            if img_file is not None:
                cv2.imwrite(save_dir + img_name + '_' + name + str(i) + '.png', att)
            else:
                cv2.imwrite(save_dir + name + str(i) + '.png', att)
