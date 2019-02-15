import os
import random
import numpy as np
import scipy.ndimage
import skimage.measure
from matplotlib import pyplot as plt
import keras.backend as K
import utils as ut
import argparse

def load_data(imgpath, dims=None, pad=0, normalize=False):
    '''
    dims: desired output shape
    pad (int): pixels of mean padding to include on each border
    normalize: if True, return image in range [0,1]
    '''
    img = scipy.misc.imread(imgpath, mode='RGB')
    if normalize:
        img = img/255.
    if dims:
        imgdims = (dims[0]-pad*2, dims[1]-pad*2, dims[2])
        img = scipy.misc.imresize(img, (imgdims[0], imgdims[1]))
        if pad:
            padded_im = np.zeros(dims)
            padded_im[:] = np.mean(img, axis=(0, 1))
            padded_im[pad:imgdims[0]-pad, pad:imgdims[1]-pad, :] = img

    return img


def load_dotlabel(lbpath, imgdims, pad=0):
    '''
    load labels stored as dot annotation maps
    imgdims: output size
    pad (int): pixels of zero padding to include on each border
    '''

    lb = scipy.misc.imread(lbpath, mode='RGB')

    # resize dot labels
    lb = np.asarray(lb[:, :, 0] > 230)
    coords = np.column_stack(np.where(lb == 1))
    new_lb = np.zeros((imgdims[0], imgdims[1]), dtype='float32')

    zx = (imgdims[0]-2*pad)/lb.shape[0]
    zy = (imgdims[1]-2*pad)/lb.shape[1]

    for c in range(coords.shape[0]):
        new_lb[pad+int(coords[c,0]*zx),pad+int(coords[c, 1]*zy)] = 1

    return new_lb

def max_pooling(img, stride=(2, 2)):
    return skimage.measure.block_reduce(img, block_size=stride, func=np.max)


def affine_transform_Image(img, matrix, offset):
    #padX = [img.shape[1] - pivot[0], pivot[0]]
    #padY = [img.shape[0] - pivot[1], pivot[1]]
    #imgP = np.pad(img, [padY, padX, [0,0]], 'reflect')
    imgR = scipy.ndimage.affine_transform(img, matrix, offset=offset, mode='nearest', order=5)
    return imgR


def flip_axis(array, axis):
    """
    Flip the given axis of an array.  Note that the ordering follows the
    numpy convention and may be unintuitive; that is, the first axis
    flips the axis horizontally, and the second axis flips the axis vertically.
    :param array: The array to be flipped.
    :type array: `ndarray`
    :param axis: The axis to be flipped.
    :type axis: `int`
    :returns: The flipped array.
    :rtype: `ndarray`
    """

    # Rearrange the array so that the axis of interest is first.
    array = np.asarray(array).swapaxes(axis, 0)
    # Reverse the elements along the first axis.
    array = array[::-1, ...]
    # Put the array back and return.
    return array.swapaxes(0, axis)



def affine_image_with_python(img, target_shape=None, xy=np.array([0.0, 0.0]), rt=0.0, zm=1.0):
    # This is specifically designed for the stn face project.
    xy_mat = np.array([1.0, 1.0, 1.0, 1.0])
    rt_mat = np.array([np.cos(rt), np.sin(rt), -np.sin(rt), np.cos(rt)])
    zm_mat = np.array([zm, zm, zm, zm])
    transform_mat = np.reshape((xy_mat * rt_mat) * zm_mat, (2, 2))
    c_in = 0.5*np.array(img.shape[:2])
    c_out = c_in
    offset = c_in - c_out.dot(transform_mat)
    trans_img_c0 = affine_transform_Image(img[:, :, 0], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img_c1 = affine_transform_Image(img[:, :, 1], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img_c2 = affine_transform_Image(img[:, :, 2], transform_mat.T, offset=offset+xy*(target_shape[:2]//2))
    trans_img = np.stack((trans_img_c0, trans_img_c1, trans_img_c2), -1)
    return trans_img



def augment_data(img, opt={}, prob=.9):
    '''
    performs a random horizontal flip
    and a random affine transform with probability prob
    Args:
        opt: options for adjusting amount of translation, rotation, zoom
    '''

    xy = opt.get('xy', -0.03)
    rt = opt.get('rt', [8, 20])
    zm = opt.get('zm', [.95, 1.05])

    if random.random() > .5:
        img = flip_axis(img, 1)

    if random.random() < prob:
        rand_xy = xy * np.random.random((2,))
        rand_rt = np.pi / random.randint(rt[0], rt[1])
        rand_zm = np.random.uniform(zm[0], zm[1])
        target_shape = np.array(img.shape)

        img = affine_image_with_python(img, target_shape, xy=rand_xy, rt=rand_rt, zm=rand_zm)
    return img


def sample_exemplar(inputs, patchdims, augment):
    '''
    Samples an exemplar patch from an input image.
    Args:
        inputs: tuple of (img, lb)
            img: input image
            lb: dot annotations of instances (same size as img)
        patchdims: desired size of exemplar patch
        augment: whether to do data augmentation on patch
    '''
    img,lb = inputs
    imgdims = img.shape

    # get coordinates of potential exemplars
    coords = np.column_stack(np.where(lb == 1.0))
    valid_coords = np.array([c for c in coords
                             if (c[0] > patchdims[0]//2) and c[1] > (patchdims[1]//2)
                             and c[0] < (imgdims[0] - patchdims[0]//2)
                             and c[1] < (imgdims[1] - patchdims[1]//2)])

    if valid_coords.shape[0] == 0:
        # TODO: different way of handling this case
        # no objects, so choose patch at center of image to match to itself
        valid_coords = np.array([[imgdims[0] // 2, imgdims[1] // 2]], 'int')
        lb[:] = 0
        lb[valid_coords[0][0], valid_coords[0][1]] = 1

    patch_coords = valid_coords[random.randint(0, valid_coords.shape[0]-1)]
    ex_patch = img[patch_coords[0] - patchdims[0] // 2: patch_coords[0] + patchdims[0] // 2,
                   patch_coords[1] - patchdims[1] // 2: patch_coords[1] + patchdims[1] // 2, ]

    output_map = max_pooling(lb, (4, 4))  # resize to output size
    output_map = 100 * scipy.ndimage.gaussian_filter(
            output_map, sigma=(2, 2), mode='constant')

    if augment:
        opt = {'xy': -0.05, 'rt': [1, 20], 'zm': [0.9, 1.1]}
        ex_patch = augment_data(ex_patch, opt)
    return (ex_patch, output_map)



def predict(imgdict):

    # ==> gpu configuration
    ut.initialize_GPU(args)

    # ==> set up model path and log path.
    model_path, log_path = ut.set_path(args)

    # ==> import library
    import keras
    import data_loader
    import model_factory
    import data_generator

    # ==> get dataset information
    # trn_config = data_loader.get_config(args)

    # params = {'cg': trn_config,
    #           'processes': 12,
    #           'batch_size': args.batch_size
    #           }

    # trn_gen, val_gen = data_generator.setup_generator(**params)

    # ==> load networks
    
    class C(object):
    	"""docstring for C"""
    	patchdims = (64,64,3)
    	imgdims = (800, 800,3)
    	def __init__(self, arg):
    		super(C, self).__init__()
    		self.arg = arg
    		
    gmn = model_factory.two_stream_matching_networks(C, sync=False, adapt=False)
    # model = model_factory.two_stream_matching_networks(C, sync=False, adapt=True)

    # ==> attempt to load pre-trained model
    if args.resume:
        if os.path.isfile(args.resume):
            model.load_weights(os.path.join(args.resume), by_name=True)
            print('==> successfully loading the model: {}'.format(args.resume))
        else:
            print("==> no checkpoint found at '{}'".format(args.resume))

    # ==> attempt to load pre-trained GMN
    elif args.gmn_path:
        if os.path.isfile(args.gmn_path):
            gmn.load_weights(os.path.join(args.gmn_path), by_name=True)
            print('==> successfully loading the model: {}'.format(args.gmn_path))
        else:
            print("==> no checkpoint found at '{}'".format(args.gmn_path))

    # ==> print model summary
    # model.summary()

    # ==> transfer weights from gmn to new model (this step is slow, but can't seem to avoid it)
    # for i,layer in enumerate(gmn.layers):
    #     if isinstance(layer, model.__class__):
    #         for l in layer.layers:
    #             weights = l.get_weights()
    #             if len(weights) > 0:
    #                 #print('{}'.format(l.name))
    #                 model.layers[i].get_layer(l.name).set_weights(weights)
    #     else:
    #         weights = layer.get_weights()
    #         if len(weights) > 0:
    #             #print('{}'.format(layer.name))
    #             model.get_layer(layer.name).set_weights(weights)

    return gmn.predict(imgdict) #model
    # print('xxxx',x.shape)

def preprocess_input(x, dim_ordering='default'):
    '''
    imagenet preprocessing
    '''
    if dim_ordering == 'default':
        dim_ordering = K.image_dim_ordering()
    assert dim_ordering in {'tf', 'th'}

    if dim_ordering == 'th':
        x[:, 0, :, :] -= 103.939
        x[:, 1, :, :] -= 116.779
        x[:, 2, :, :] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, ::-1, :, :]
    else:
        x[:, :, :, 0] -= 103.939
        x[:, :, :, 1] -= 116.779
        x[:, :, :, 2] -= 123.68
        # 'RGB'->'BGR'
        x = x[:, :, :, ::-1]
    return x


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', default='', type=str)
    parser.add_argument('--net', default='resnet50', choices=['resnet50'], type=str)
    parser.add_argument('--optimizer', default='adam', choices=['adam'], type=str)
    parser.add_argument('--mode', default='pretrain', choices=['pretrain', 'adapt'], type=str,
                    help='pretrain on tracking data or adapt to specific dataset.')
    parser.add_argument('--dataset', default='imagenet',
                    choices=['imagenet', 'vgg_cell', 'hela_cell', 'car'],
                    type=str, help='pretrain on tracking data or adapt to specific dataset.')
    parser.add_argument('--lr', default=0.0005, type=float)
    parser.add_argument('--warmup_ratio', default=0, type=float)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--gmn_path', default='', type=str,
                    help='path to pretrained GMN, used for "adapt" mode')
    parser.add_argument('--batch_size', default=24, type=int)
    parser.add_argument('--data_path', default='', type=str)
    parser.add_argument('--epochs', default=36, type=int,
                    help='number of total epochs to run')

    global args
    args = parser.parse_args()	
    import cv2
    path = './src/cells/001cell.png'
    lbpath = './src/cells/001dots.png'
    img_dim = (800,800,3)
    patch_dim = (64,64,3)
    img = load_data(path,img_dim)
    # load the image data
    lbl_img = load_dotlabel(path,img_dim)
    # load the dot label image i.e image with dot markings
    result = sample_exemplar((img, lbl_img), patch_dim, True)
    # sample the exemplar patch from the image
    ex_patch = result[0].reshape(1, result[0].shape[0], result[0].shape[1], result[0].shape[2])
    # reshape the ex_patch into batch size, dimensions of the patch
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    #reshape the image to batchsize, img dimensions
    inp_img = preprocess_input(np.array(img, dtype='float32'))
    ex_patch = preprocess_input(np.array(ex_patch, dtype='float32'))
    imgdict = {'image_patch':ex_patch, 'image':inp_img}
    x = predict(imgdict)
    #predicted image
    # print(x)
    print(x.shape)
    # remove the batch dimension
    x  =x.reshape(200,200)
    # x[:,:] = 1
    for i in range(200):
        for j in range(200):
            if x[i,j] > 0:
                x[i,j] = 255.0
                print(x[i,j])
    # 	print()
    cv2.imwrite('xxxxx.png', x)
    imm = cv2.imread('xxxxx.png',0)
    print(imm.shape)
    
    cv2.imshow('ss',imm)
    cv2.waitKey()

    # cv2.imsave(x,'xxx.jpg')


if __name__ == '__main__':
    main()