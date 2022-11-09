import numpy as np
# add your imports
import os

def segmentate(images):
    """
​
    :param np.ndarray images: N x 12288 array containing N 64x64x3 images flattened into vectors
    :return: np.ndarray
    """
    os.system("pip install segmentation-models")
    import segmentation_models as sm
    N = images.shape[0]
    batch_size = 64
    batch_num = int(N / batch_size)
    end_idx = batch_size * batch_num
    # define network parameters
    # case for binary and multiclass segmentation
    # create model
    model = sm.Unet('efficientnetb3', classes=11, activation='softmax')
    model.load_weights('best_model.h5')
    pred_seg = np.empty((N, 4096), dtype=np.int32)
    for i in range(batch_num):
        start_idx = i * batch_size
        end_idx = (i+1)* batch_size
        mask = model.predict(images[start_idx:end_idx].reshape([batch_size, 64,64,3]))
        # mask shape [batch_size, 64, 64, 11]
        mask = mask[:,:,:, :-1]
        mask = np.argmax(mask, 3).reshape([batch_size, 64*64])
        pred_seg[start_idx : end_idx] = mask

    if end_idx != N:
        mask = model.predict(images[end_idx: N].reshape([N-end_idx, 64, 64, 3]))
        mask = mask[:, :, :, :-1]
        mask = np.argmax(mask, 3).reshape([batch_size, 64 * 64])
        pred_seg[end_idx:] = mask
    return pred_seg
