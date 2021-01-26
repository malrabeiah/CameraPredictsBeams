"""
Script could be used as a template for preparing the image-beam data structure required to train
the modified ResNet on beam or blockage prediction. NOTE: script might need modification based on the
dataset you generated and where it is saved. PLEASE READ THROUGH.
"""
import numpy as np
import h5py as h5
import pickle
import scipy.io as sciio
import skimage.io as imio
import os
from shutil import copyfile

root_img_dir = '' # Where ViWi images are
codebook = sciio.loadmat('DFT_codebook64')['W']# Every column is a beam (codebook must be generated beforehand, see UPA_codebook_generator_DFT.m)

def getMATLAB(matlab_file=None, save_mode=False,pickle_name=None):
    '''
    It converts MAT data into a python dictionary of numpy arrays.
    :param matlab_file: path to the matlab data structure to read. The structure is expected
			to have two fields: wireless channels and user locations. The channel field
			should be a 4D array: #of antennas, # of subcarriers, # of user positions, # of BS
			user location field should be a 2D array: 3, # of user positions
    :param save_mode: whether the numpy data needs to be saved or not
    :param pickle_name: name of the pickle file where the data is stored
    :return: dictionary of numpy arrays containing the raw channel and location data.
    -----------------------------------------------------------------------------------
    NOTE:
    The MATLAB data structure has to be prepared following the generation of the wireless data using the ViWi data-generation
    script.
    '''
    # Read MATLAB structure:
    f = h5.File(matlab_file, 'r')
    key1 = list(f.keys())
    raw_data = f[key1[1]]
    key2 = list(raw_data.keys())
    channels = raw_data[key2[0]][:] # Wireless data field
    loc = raw_data[key2[1]][:] # Loc data field
    s1 = channels.shape
    s2 = loc.shape

    # Construct and store numpy dictionary
    X = channels.view(np.double).reshape(s1+(2,))
    X = X.astype(np.float32) # This is necessary to reduce the precision
    if len(s1) == 4:
        X_comp = X[:, :, :, :, 0] + X[:, :, :, :, 1] * 1j # size: # of BSs X # of users X sub-carriers X # of antennas
    else:
        X_comp = X[:, :, :, 0] + X[:, :, :, 1] * 1j  # size: # of users X sub-carriers X # of antennas

    # Normalize channels
    rec_pow = np.mean(np.abs(X_comp)**2)
    X_comp = X_comp/np.sqrt(rec_pow)

    raw_data = {'ch': X_comp,
                'loc': loc,
                'norm_fact': rec_pow}
    print(raw_data['ch'].shape)
    if save_mode:
        f = open(pickle_name, 'wb')
        pickle.dump(raw_data, f, protocol=4)
        f.close()
    return raw_data

def beamPredStruct(raw_data,codebook,val_per,image_path=None):
    '''
    This function prepares an image data structure for training and testing
    a CNN on mmWave beam prediction. The function is designed for the
    direct distributed-camera scenario.
    :param raw_data: Wireless data dictionary with the keys: ch, loc, and
                     norm_fact.
    :param codebook: Beamforming DFT matrix
    :param val_per: Precentage of validation (test) data
    :param image_path: Path to the ViWi IMAGE folder
    :return:
    '''
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    shuf_ind = np.random.permutation(len(image_names))
    loc = raw_data['loc'][:, 0:2]# User coordinates as output by ViWi
    num_train = len(image_names) - np.ceil( val_per*len(image_names) )
    count = 0
    train_list = []
    test_list = []
    for i in shuf_ind:
        # Find the coordinates in the image: (NOTE an image is tagged with the coordinates of its single user)
        split_name = image_names[i].split('_')
        x_axis = float( split_name[2] )
        y_axis = float( split_name[3][:-4] )
        coord = np.array([x_axis, y_axis])
        cam_num = int( split_name[1] )-1

        # Find the channel of those coordinates:
        diff = np.sum( np.abs(loc - coord), axis=1 )
        user = np.argmin(diff)
#        print('coord {} and locatio {}'.format( coord,loc[user] ))
        h = raw_data['ch'][cam_num,user,:,:] # Channel for img_name

        # Finding the best beamforming vector:
        codebook_H = codebook.conj()
        rec_pow_sub = np.power(np.abs(np.matmul(h, codebook_H)), 2)  # per subcarrier
        rate_per_sub = np.log2( 1+rec_pow_sub )
        print(rec_pow_sub.shape)
        ave_rate_per_beam = np.mean(rate_per_sub, axis=0)  # averaged over subcarriers
        beam_ind = np.argmax(ave_rate_per_beam)+1
        print('image name {} and beam index {}'.format(image_names[i], beam_ind))
        
        # Dividing images into folders
        count += 1
        if count <= num_train:
            if not os.path.exists('train_images'):
                os.mkdir('train_images')
            sub_dir_name = beam_ind
            sub_dir_path = os.getcwd()+'/train_images/'+str(sub_dir_name)
            if not os.path.exists( sub_dir_path ):
                os.mkdir( sub_dir_path )
                copyfile( image_path+'/'+image_names[i], sub_dir_path+'/'+image_names[i] )
            else:
                copyfile( image_path+'/'+image_names[i], sub_dir_path+'/'+image_names[i] )
            train_list.append(image_names[i])
        else:
            if not os.path.exists('test_images'):
                os.mkdir('test_images')
            sub_dir_name = beam_ind
            sub_dir_path = os.getcwd()+'/test_images/'+str(sub_dir_name)
            if not os.path.exists( sub_dir_path ):
                os.mkdir( sub_dir_path )
                copyfile( image_path+'/'+image_names[i], sub_dir_path+'/'+image_names[i] )
            else:
                copyfile( image_path+'/'+image_names[i], sub_dir_path+'/'+image_names[i] )
            test_list.append(image_names[i])

    # return [train_list,test_list]

def blockagePredStruct(raw_data,codebook,val_per,image_path=None):
    '''
    This function prepares an image data structure for training and testing
    a CNN on blockage prediction. The function is designed for the
    blocked colocated-camera scenario.
    :param raw_data: Wireless data dictionary with the keys: ch, loc, and
                     abs_max.
    :param codebook: Beamforming DFT matrix
    :param val_per: Precentage of validation (test) data
    :param image_path: Path to the ViWi IMAGE folder
    :return:
    '''
    image_names = os.listdir(image_path)
    image_names = sorted(image_names)
    shuf_ind = np.random.permutation(len(image_names))
    loc = raw_data['loc'][:, 0:2]  # User coordinates as output by ViWi
    num_train = len(image_names) - np.ceil(val_per * len(image_names))
    count = 0
    train_list = []
    test_list = []
    for i in shuf_ind:
        # Find the coordinates in the image:
        split_name = image_names[i].split('_')
        x_axis = float(split_name[2])
        y_axis = float(split_name[3])
        blk_status = int(split_name[4][:-4])
        coord = np.array([x_axis, y_axis])
        cam_num = int(split_name[1]) - 1

        # Dividing images into folders
        count += 1
        if count <= num_train:
            if not os.path.exists('train_images_blk'):
                os.mkdir('train_images_blk')
            sub_dir_name = blk_status
            sub_dir_path = os.getcwd() + '/train_images_blk/' + str(sub_dir_name)
            if not os.path.exists(sub_dir_path):
                os.mkdir(sub_dir_path)
                copyfile(image_path + '/' + image_names[i], sub_dir_path + '/' + image_names[i])
            else:
                copyfile(image_path + '/' + image_names[i], sub_dir_path + '/' + image_names[i])
            train_list.append(image_names[i])
        else:
            if not os.path.exists('test_images_blk'):
                os.mkdir('test_images_blk')
            sub_dir_name = blk_status
            sub_dir_path = os.getcwd() + '/test_images_blk/' + str(sub_dir_name)
            if not os.path.exists(sub_dir_path):
                os.mkdir(sub_dir_path)
                copyfile(image_path + '/' + image_names[i], sub_dir_path + '/' + image_names[i])
            else:
                copyfile(image_path + '/' + image_names[i], sub_dir_path + '/' + image_names[i])
            test_list.append(image_names[i])


if __name__ == '__main__':
    raw_data = getMATLAB(matlab_file='raw_data_dist_D_64ULA_5p_64s.mat',
                         save_mode=False,
                         pickle_name=None)
    beamPredStruct(raw_data,codebook,0.3,root_img_dir)
    # blockagePredStruct(raw_data,codebook,0.3,root_img_dir)
    # print('Training samples: {} and testing samples: {}'.format( len(train_list), len(test_list) ))
    print('break')
