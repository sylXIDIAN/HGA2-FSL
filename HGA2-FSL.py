import torch.utils.data as dataf
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os
import math
import argparse
import pickle
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
import time
import utils as UT
from einops import rearrange
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv as SAGE


parser = argparse.ArgumentParser(description="Few Shot Visual Recognition")
parser.add_argument("-f","--feature_dim",type = int, default = 160)
parser.add_argument("-c","--src_input_dim",type = int, default = 3)
parser.add_argument("-d","--tar_input_dim",type = int, default = 155) #bay=224 river=198 farmland=155
parser.add_argument("-n","--n_dim",type = int, default = 100)
parser.add_argument("-w","--class_num",type = int, default = 2)
parser.add_argument("-s","--shot_num_per_class",type = int, default = 1)
parser.add_argument("-b","--query_num_per_class",type = int, default = 15)
# parser.add_argument("-sum","--sum_num_per_class",type = int, default = 17)
parser.add_argument("-e","--episode",type = int, default= 1000)
parser.add_argument("-t","--test_episode", type = int, default = 600)
parser.add_argument("-l","--learning_rate", type = float, default = 0.001)
parser.add_argument("-g","--gpu",type=int, default=0)
parser.add_argument("-u","--hidden_unit",type=int,default=10)
parser.add_argument("-can","--hyperparameter",type=int,default=1)

parser.add_argument("-m" ,"--test_class_num",type=int, default=2)
parser.add_argument("-z","--test_lsample_num_per_class",type=int,default=5, help='5 4 3 2 1')

args = parser.parse_args(args=[])

# Hyper Parameters
FEATURE_DIM = args.feature_dim
SRC_INPUT_DIMENSION = args.src_input_dim
TAR_INPUT_DIMENSION = args.tar_input_dim
N_DIMENSION = args.n_dim
CLASS_NUM = args.class_num
SHOT_NUM_PER_CLASS = args.shot_num_per_class
QUERY_NUM_PER_CLASS = args.query_num_per_class
SUM_NUM_PER_CLASS = SHOT_NUM_PER_CLASS + QUERY_NUM_PER_CLASS
EPISODE = args.episode
TEST_EPISODE = args.test_episode
LEARNING_RATE = args.learning_rate
GPU = args.gpu
HIDDEN_UNIT = args.hidden_unit
CAN = args.hyperparameter
print(CAN)
TEST_CLASS_NUM = args.test_class_num
TEST_LSAMPLE_NUM_PER_CLASS = args.test_lsample_num_per_class

patchsize = 9

UT.same_seeds(0)
def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('classificationMap'):
        os.makedirs('classificationMap')
_init_()

# load source domain data set
with open(os.path.join('datasets',  'MSI157_9.pickle'), 'rb') as handle:
    source_imdb = pickle.load(handle)
print(source_imdb.keys())
print(source_imdb['Labels'])

# process source domain data set
data_train = source_imdb['data']
labels_train = source_imdb['Labels']
keys_all_train = sorted(list(set(labels_train)))
# print(keys_all_train)
label_encoder_train = {}
for i in range(len(keys_all_train)):
    label_encoder_train[keys_all_train[i]] = i
print(label_encoder_train)

train_set = {}
for class_, path in zip(labels_train, data_train):
    if label_encoder_train[class_] not in train_set:
        train_set[label_encoder_train[class_]] = []
    train_set[label_encoder_train[class_]].append(path)
print(train_set.keys())
data = train_set
del train_set
del keys_all_train
del label_encoder_train

print("Num classes for source domain datasets: " + str(len(data)))

data = UT.sanity_check500(data)
print("Num classes of the number of class larger than 500 in dataset: " + str(len(data)))

for class_ in data:
    for i in range(len(data[class_])):
        image_transpose = np.transpose(data[class_][i], (2, 0, 1))
        data[class_][i] = image_transpose

# source few-shot classification data
metatrain_data= data
print(len(metatrain_data.keys()), metatrain_data.keys())
del data

# source domain adaptation data
print(source_imdb['data'].shape)
source_imdb['data'] = source_imdb['data'].transpose((1, 2, 3, 0))
print(source_imdb['data'].shape)
print(source_imdb['Labels'])
source_dataset = UT.matcifar(source_imdb, train=True, d=3, medicinal=0)
source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)


## target domain data set
test_data1 = 'datasets/farm/farm06.mat'
test_data2 = 'datasets/farm/farm07.mat'
test_label = 'datasets/farm/farm_gt_nonzero.mat'



Data_Band_Scaler1,GroundTruth = UT.load_data(test_data1, test_label)
Data_Band_Scaler2,GroundTruth = UT.load_data1(test_data2, test_label)
Data_Band_Scaler = Data_Band_Scaler2-Data_Band_Scaler1



# get train_loader and test_loader
def get_train_test_loader(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    print(Data_Band_Scaler.shape)
    [nRow, nColumn, nBand] = Data_Band_Scaler.shape

    '''label start'''
    num_class = int(np.max(GroundTruth))
    data_band_scaler = UT.flip(Data_Band_Scaler)
    groundtruth = UT.flip(GroundTruth)
    del Data_Band_Scaler
    del GroundTruth

    HalfWidth = 4
    G = groundtruth[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn+ HalfWidth]
    data = data_band_scaler[nRow - HalfWidth:2 * nRow + HalfWidth, nColumn - HalfWidth:2 * nColumn + HalfWidth,:]

    [Row, Column] = np.nonzero(G)
    del data_band_scaler
    del groundtruth

    nSample = np.size(Row)
    print('number of sample', nSample)

    # Sampling samples
    train = {}
    test = {}
    da_train = {} # Data Augmentation
    m = int(np.max(G))
    nlabeled =TEST_LSAMPLE_NUM_PER_CLASS
    print('labeled number per class:', nlabeled)
    print((200 - nlabeled) / nlabeled + 1)
    print(math.ceil((200 - nlabeled) / nlabeled) + 1)

    for i in range(m):
        indices = [j for j, x in enumerate(Row.ravel().tolist()) if G[Row[j], Column[j]] == i + 1]
        np.random.shuffle(indices)
        nb_val = shot_num_per_class
        train[i] = indices[:nb_val]
        da_train[i] = []
        for j in range(math.ceil((200 - nlabeled) / nlabeled) + 1):
            da_train[i] += indices[:nb_val]
        test[i] = indices[nb_val:]

    train_indices = []
    test_indices = []
    da_train_indices = []
    for i in range(m):
        train_indices += train[i]
        test_indices += test[i]
        da_train_indices += da_train[i]
    np.random.shuffle(test_indices)

    print('the number of train_indices1:', len(train_indices))
    print('the number of test_indices1:', len(test_indices))
    print('the number of train_indices1 after data argumentation:', len(da_train_indices))
    print('labeled sample indices1:',train_indices)

    nTrain = len(train_indices)
    nTest = len(test_indices)
    da_nTrain = len(da_train_indices)

    imdb = {}
    imdb['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, nTrain + nTest], dtype=np.float32)
    imdb['Labels'] = np.zeros([nTrain + nTest], dtype=np.int64)
    imdb['set'] = np.zeros([nTrain + nTest], dtype=np.int64)

    RandPerm = train_indices + test_indices

    RandPerm = np.array(RandPerm)

    for iSample in range(nTrain + nTest):
        imdb['data'][:, :, :, iSample] = data[Row[RandPerm[iSample]] - HalfWidth:  Row[RandPerm[iSample]] + HalfWidth + 1,
                                         Column[RandPerm[iSample]] - HalfWidth: Column[RandPerm[iSample]] + HalfWidth + 1, :]
        imdb['Labels'][iSample] = G[Row[RandPerm[iSample]], Column[RandPerm[iSample]]].astype(np.int64)

    imdb['Labels'] = imdb['Labels'] - 1
    imdb['set'] = np.hstack((np.ones([nTrain]), 3 * np.ones([nTest]))).astype(np.int64)
    print('Data is OK.')


    train_dataset = UT.matcifar(imdb, train=True, d=3, medicinal=0)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=36,shuffle=False, num_workers=0)
    del train_dataset

    test_dataset = UT.matcifar(imdb, train=False, d=3, medicinal=0)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=0)
    del test_dataset
    del imdb

    # Data Augmentation for target domain for training
    imdb_da_train = {}
    imdb_da_train['data'] = np.zeros([2 * HalfWidth + 1, 2 * HalfWidth + 1, nBand, da_nTrain],  dtype=np.float32)
    imdb_da_train['Labels'] = np.zeros([da_nTrain], dtype=np.int64)
    imdb_da_train['set'] = np.zeros([da_nTrain], dtype=np.int64)

    da_RandPerm = np.array(da_train_indices)
    for iSample in range(da_nTrain):
        imdb_da_train['data'][:, :, :, iSample] =UT.radiation_noise(data[Row[da_RandPerm[iSample]] - HalfWidth:  Row[da_RandPerm[iSample]] + HalfWidth + 1,
                                                       Column[da_RandPerm[iSample]] - HalfWidth: Column[da_RandPerm[iSample]] + HalfWidth + 1, :])
        imdb_da_train['Labels'][iSample] = G[Row[da_RandPerm[iSample]], Column[da_RandPerm[iSample]]].astype(np.int64)


    imdb_da_train['Labels'] = imdb_da_train['Labels'] - 1
    imdb_da_train['set'] = np.ones([da_nTrain]).astype(np.int64)
    print('imdb_da_train ok')

    return train_loader, test_loader, imdb_da_train ,G,RandPerm,Row, Column,nTrain

def get_target_dataset(Data_Band_Scaler, GroundTruth, class_num, shot_num_per_class):
    train_loader, test_loader, imdb_da_train,G,RandPerm,Row, Column,nTrain = get_train_test_loader(Data_Band_Scaler=Data_Band_Scaler,  GroundTruth=GroundTruth, \
                                                                     class_num=class_num,shot_num_per_class=shot_num_per_class)
    train_datas, train_labels = train_loader.__iter__().next()
    print('train labels:', train_labels)
    print('size of train datas:', train_datas.shape)

    print(imdb_da_train.keys())
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    del Data_Band_Scaler, GroundTruth

    # target data with data augmentation
    target_da_datas = np.transpose(imdb_da_train['data'], (3, 2, 0, 1))
    print(target_da_datas.shape)
    target_da_labels = imdb_da_train['Labels']
    print('target data augmentation label:', target_da_labels)

    # metatrain data for few-shot classification
    target_da_train_set = {}
    for class_, path in zip(target_da_labels, target_da_datas):
        if class_ not in target_da_train_set:
            target_da_train_set[class_] = []
        target_da_train_set[class_].append(path)
    target_da_metatrain_data = target_da_train_set
    print(target_da_metatrain_data.keys())

    # target domain : batch samples for domian adaptation
    print(imdb_da_train['data'].shape)
    print(imdb_da_train['Labels'])
    target_dataset = UT.matcifar(imdb_da_train, train=True, d=3, medicinal=0)
    target_loader = torch.utils.data.DataLoader(target_dataset, batch_size=SUM_NUM_PER_CLASS*CLASS_NUM, shuffle=True, num_workers=0)
    del target_dataset


    return train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain


def getGraphdata_Da(bs, data, atten_score,k, target=True):
    data_edge = torch.tensor(getEdge_Da(data, bs, atten_score,k)).t().contiguous()
    graph = Data(x=data,edge_index=data_edge).cuda()

    return graph

def getEdge_Da(image, segments,atte_score,k, compactness=300, sigma=3.):
    coo = []
    getcoo = []
    for i in range(0, segments):
        count = atte_score[i].cpu().detach().numpy()
        p = np.argsort(-count)
        coo.append((p[0:k]))
    coo = np.array(coo)

    for j in range(0,segments):
        for t in range(0,k):
            if j != coo[j][t]:
              getcoo.append((j,coo[j][t]))

    getcoo = np.asarray(list(getcoo))
    return getcoo

class CrossGraphDA(nn.Module):
    def __init__(self):
        super(CrossGraphDA, self).__init__()
        self.w_q = nn.Linear(160, 160)
        self.w_k = nn.Linear(160, 160)
        self.gcn = Global_graph(FEATURE_DIM, CLASS_NUM)
        self.Layer1 = nn.Linear(32, 160, bias=True)
        self.Layer2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        self.Loss = UT.MMD_loss(kernel_type='linear')


    def forward(self,x1,x2,x3,x4):
        x1 = self.Layer1(x1)
        x2 = self.Layer1(x2)
        Q1 = self.w_q(x1)
        K1 = self.w_k(x1)
        Q2 = self.w_q(x2)
        K2 = self.w_k(x2)
        bs_da = x1.shape[0]
        k=6 ##k=int(bs_da/6)  k2

        attention_scores1 = torch.matmul(Q1, K1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(160)
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        attention_scores2 = torch.matmul(Q2, K2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(160)
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        graph_feature1 = getGraphdata_Da(bs_da, x1, attention_probs1, k)
        src_gcn_feature1 = self.gcn(graph_feature1)

        graph_feature2 = getGraphdata_Da(bs_da, x2, attention_probs2, k)
        src_gcn_feature2 = self.gcn(graph_feature2)

        All_GraFeature= torch.cat([src_gcn_feature1,src_gcn_feature2],dim=1)
        All_GraFeature = rearrange(All_GraFeature, 'b hw -> b hw 1 1')
        All_GraFeature = self.Layer2(All_GraFeature)
        All_GraFeature = rearrange(All_GraFeature, 'b hw 1 1 -> b hw')


        x3_1 = x3 - All_GraFeature
        x4_1 = x4 - All_GraFeature

        x3 = x3_1+x3
        x4 = x4_1+x4


        loss_MMD = self.Loss(x3,x4)


        return loss_MMD


def conv3x3x3(in_channel, out_channel):
    layer = nn.Sequential(
        nn.Conv3d(in_channels=in_channel,out_channels=out_channel,kernel_size=3, stride=1,padding=1,bias=False),
        nn.BatchNorm3d(out_channel),
        # nn.ReLU(inplace=True)
    )
    return layer

class residual_block(nn.Module):

    def __init__(self, in_channel,out_channel):
        super(residual_block, self).__init__()

        self.conv1 = conv3x3x3(in_channel,out_channel)
        self.conv2 = conv3x3x3(out_channel,out_channel)
        self.conv3 = conv3x3x3(out_channel,out_channel)

    def forward(self, x):
        x1 = F.relu(self.conv1(x), inplace=True)
        x2 = F.relu(self.conv2(x1), inplace=True)
        x3 = self.conv3(x2)

        out = F.relu(x1+x3, inplace=True)
        return out

class D_Res_3d_CNN(nn.Module):
    def __init__(self, in_channel, out_channel1, out_channel2):
        super(D_Res_3d_CNN, self).__init__()

        self.block1 = residual_block(in_channel,out_channel1)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(4,2,2),padding=(0,1,1),stride=(4,2,2))
        self.block2 = residual_block(out_channel1,out_channel2)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(4,2,2),stride=(4,2,2), padding=(2,1,1))
        self.conv = nn.Conv3d(in_channels=out_channel2,out_channels=32,kernel_size=3, bias=False)

        self.final_feat_dim = 160

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.maxpool1(x)
        x = self.block2(x)
        x = self.maxpool2(x)
        x = self.conv(x)
        x = x.view(x.shape[0],-1)
        return x

class Mapping_NEW(nn.Module):
    def __init__(self, in_dimension, out_dimension):
        super(Mapping_NEW, self).__init__()
        self.preconv = nn.Conv2d(in_dimension, 264, 1, 1, bias=False)
        self.preconv1 = nn.Conv2d(264, out_dimension, 1, 1, bias=False)
        self.preconv_bn1 = nn.BatchNorm2d(264)
        self.preconv_bn = nn.BatchNorm2d(out_dimension)

    def forward(self, x):
        x = self.preconv(x)
        x = self.preconv_bn1(x)
        x = self.preconv1(x)
        x = self.preconv_bn(x)
        return x

def getGraphdata_classwise(bs, data, target=True):
    segments = bs
    data_edge = torch.tensor(getEdge_classwise(data, segments)).t().contiguous().long()
    graph = Data(x=data,edge_index=data_edge).cuda()

    return graph

def getEdge_classwise(image, segments, compactness=300, sigma=3.):
    coo = []
    for i in range(0, segments):
        for j in range(i, segments):
            if i!=j:
                coo.append((i,j))

    coo = np.asarray(list(coo))
    return coo


class GlobalGraphAttention(nn.Module):
    def __init__(self,input_dim):
        super(GlobalGraphAttention, self).__init__()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)

    def forward(self, x,input_dim):
        Q = self.w_q(x)
        K = self.w_k(x)

        attention_scores = torch.matmul(Q, K.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(input_dim)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        return attention_probs


def getGraphdata_globalatten(bs, data, atten_score,k, target=True):
    data_edge = torch.tensor(getEdge_globalatten(data, bs, atten_score,k)).t().contiguous()
    graph = Data(x=data,edge_index=data_edge).cuda()

    return graph

def getEdge_globalatten(image, segments,atte_score,k, compactness=300, sigma=3.):
    coo = []
    getcoo = []
    for i in range(0, segments):
        count = atte_score[i].cpu().detach().numpy()
        p = np.argsort(-count)
        coo.append((p[0:k]))
    coo = np.array(coo)

    for j in range(0,segments):
        for t in range(0,k):
            if j != coo[j][t]:
              getcoo.append((j,coo[j][t]))

    getcoo = np.asarray(list(getcoo))
    return getcoo

class Topology_Extraction(torch.nn.Module):
    def __init__(self, in_channels,num_classes,dropout=0.5):
        super(Topology_Extraction, self).__init__()
        self.graph1 = SAGE(in_channels, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.graph2 = SAGE(64, 32)
        self.bn2 = nn.BatchNorm1d(32)

        self.mlp_classifier = nn.Sequential(
            nn.Linear(32, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1024, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, num_classes, bias=True)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.graph1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x_temp_1 = x
        x = self.graph2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x_temp_2 = x
        x = self.mlp_classifier(x)
        return F.softmax(x, dim=1), x, x_temp_1, x_temp_2

class Global_graph(nn.Module):
    def __init__(self, in_channels, num_classes=2):
        super(Global_graph, self).__init__()
        self.sharedNet_src = Topology_Extraction(in_channels,num_classes)
        self.sharedNet_tar = Topology_Extraction(in_channels,num_classes)

    def forward(self, graphFeature):
        out = self.sharedNet_src(graphFeature)
        p_source, source, source_share_1, source_share_2 = out[0], out[1], out[2], out[3]
        return source_share_2


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.feature_encoder = D_Res_3d_CNN(1,8,16)
        self.final_feat_dim = FEATURE_DIM
        self.classifier = nn.Linear(in_features=self.final_feat_dim, out_features=CLASS_NUM)
        self.target_mapping = Mapping_NEW(TAR_INPUT_DIMENSION, N_DIMENSION)
        self.source_mapping = Mapping_NEW(SRC_INPUT_DIMENSION, N_DIMENSION)
        self.gcn = Global_graph(FEATURE_DIM, CLASS_NUM)
        self.Conv = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=(1,1))
        self.Drop =nn.Dropout(0.1)
        self.Gatten = GlobalGraphAttention(FEATURE_DIM)

    def forward(self,x1, x2,y1,y2, domain='source', condition='train'):
        if domain == 'target':
            x1 = self.target_mapping(x1)
            x2 = self.target_mapping(x2)
        elif domain == 'source':
            x1 = self.source_mapping(x1)
            x2 = self.source_mapping(x2)
        if condition == 'train':
            feature1 = self.feature_encoder(x1)
            feature2 = self.feature_encoder(x2)

            '''Global-wise Graph'''
            features = torch.cat([feature1, feature2], dim=0)

            ##Get Graph Structure Data
            bs_global = features.shape[0]
            inputdim_global = features.shape[1]
            k=14
            atten_score = self.Gatten(features,inputdim_global)

            graph_feature = getGraphdata_globalatten(bs_global, features,atten_score,k)
            src_gcn_feature = self.gcn(graph_feature)


            '''Class-wise Graph'''
            count0=[]
            count1=[]
            featureGraph0=feature1[0:SHOT_NUM_PER_CLASS]
            featureGraph1=feature1[SHOT_NUM_PER_CLASS:SHOT_NUM_PER_CLASS * 2]
            for i in range (QUERY_NUM_PER_CLASS*2):
                if y2[i]==0:
                    count0.append(i)
                elif y2[i]==1:
                    count1.append(i)

            for j in range (QUERY_NUM_PER_CLASS):
                featureGraph0= torch.cat([featureGraph0,feature2[count0[j]].reshape(1,160)],dim=0)
                featureGraph1 = torch.cat([featureGraph1, feature2[count1[j]].reshape(1,160)], dim=0)

            ##Get Graph Structure Data
            bs = featureGraph0.shape[0]
            graph_feature0 = getGraphdata_classwise(bs, featureGraph0)
            graph_feature1 = getGraphdata_classwise(bs, featureGraph1)
            src_gcn_feature0 = self.gcn(graph_feature0)
            src_gcn_feature1 = self.gcn(graph_feature1)

            src_gcn_featureSupport = torch.cat([src_gcn_feature0[0:SHOT_NUM_PER_CLASS],src_gcn_feature1[0:SHOT_NUM_PER_CLASS]],dim=0)
            src_gcn_featureQuery = torch.cat([src_gcn_feature0[SHOT_NUM_PER_CLASS:], src_gcn_feature1[SHOT_NUM_PER_CLASS:]], dim=0)
            src_Classgcn_feature =  torch.cat([src_gcn_featureSupport[0:],src_gcn_featureQuery[0:]],dim=0)
            all_feature = torch.cat([src_Classgcn_feature[0:], src_gcn_feature[0:]],dim=1)
            all_feature = rearrange(all_feature, 'b hw -> b hw 1 1')
            all_feature = self.Conv(all_feature)
            all_feature = rearrange(all_feature, 'b hw 1 1 -> b hw')
            all_feature = self.Drop(all_feature)
            src_allgcn_featureSupport = all_feature[0:SHOT_NUM_PER_CLASS*2]
            src_allgcn_featureQuery = all_feature[SHOT_NUM_PER_CLASS*2:]


        elif condition == 'test':
            features = self.feature_encoder(x1)
            # make the Graph_data
            bs = features.shape[0]
            k = int(bs/4)     ##k = int(bs/4)
            inputdim_global = features.shape[1]
            atten_score = self.Gatten(features, inputdim_global)
            graph_feature = getGraphdata_globalatten(bs, features,atten_score,k)
            src_allgcn_featureSupport = self.gcn(graph_feature)
            src_allgcn_featureQuery = src_allgcn_featureSupport


        return src_allgcn_featureSupport, src_allgcn_featureQuery



crossEntropy = nn.CrossEntropyLoss().cuda()
domain_criterion = nn.BCEWithLogitsLoss().cuda()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Conv3d') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())


def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits


nDataSet =1
acc = np.zeros([nDataSet, 1])
A = np.zeros([nDataSet, CLASS_NUM])
k = np.zeros([nDataSet, 1])
best_predict_all = []
best_acc_all = 0.0
best_kappa = 0.0
best_G,best_RandPerm,best_Row, best_Column,best_nTrain = None,None,None,None,None


seeds = [1228,1236,1330,1338,1320,1330,1235,1228,1334,1334,1226,1220,1336, 1229]  ##bay1330 Santa1236 Farm1228
for iDataSet in range(nDataSet):
    # load target domain data for training and testing
    np.random.seed(seeds[iDataSet])
    print('seeds:', seeds[iDataSet])
    train_loader, test_loader, target_da_metatrain_data, target_loader,G,RandPerm,Row, Column,nTrain = get_target_dataset(
        Data_Band_Scaler=Data_Band_Scaler, GroundTruth=GroundTruth,class_num=TEST_CLASS_NUM, shot_num_per_class=TEST_LSAMPLE_NUM_PER_CLASS)

    # model
    feature_encoder = Network()
    CroG_DA = CrossGraphDA()


    # total1 = sum(p.numel() for p in feature_encoder.parameters() if p.requires_grad)
    # total2 = sum(p.numel() for p in CroG_DA.parameters() if p.requires_grad)
    # print("Numer of parameter:%.2fM" % ((total1+total2)/1e6))

    feature_encoder.apply(weights_init)
    CroG_DA.apply(weights_init)


    feature_encoder.cuda()
    CroG_DA.cuda()


    feature_encoder.train()
    CroG_DA.train()

    # optimizer
    feature_encoder_optim = torch.optim.Adam(feature_encoder.parameters(), lr=args.learning_rate)
    CroG_DA_optim = torch.optim.Adam(CroG_DA.parameters(), lr=args.learning_rate)  ##lr=0.001


    print("Training...")
    tic1 = time.perf_counter()


    last_accuracy = 0.0
    last_kappa = 0.0
    best_episdoe = 0
    train_loss = []
    test_acc = []
    running_D_loss, running_F_loss = 0.0, 0.0
    running_label_loss = 0
    running_domain_loss = 0
    total_hit, total_num = 0.0, 0.0
    total_hit_sor, total_num_sor, total_hit_tar, total_num_tar = 0.0, 0.0, 0.0, 0.0
    test_acc_list = []


    source_iter = iter(source_loader)
    target_iter = iter(target_loader)
    len_dataloader = min(len(source_loader), len(target_loader))
    train_start = time.time()

    train_total = 0.0
    test_total = 0.0
    for episode in range(2000):
        # get domain adaptation data from  source domain and target domain
        try:
            source_data, source_label = source_iter.next()
        except Exception as err:
            source_iter = iter(source_loader)
            source_data, source_label = source_iter.next()

        try:
            target_data, target_label = target_iter.next()
        except Exception as err:
            target_iter = iter(target_loader)
            target_data, target_label = target_iter.next()


        '''Few-shot claification for source domain data set'''
        task_sor = UT.Task(metatrain_data, CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        support_dataloader_sor = UT.get_HBKC_data_loader(task_sor, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
        query_dataloader_sor = UT.get_HBKC_data_loader(task_sor, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)


        task_tar = UT.Task(target_da_metatrain_data, TEST_CLASS_NUM, SHOT_NUM_PER_CLASS,QUERY_NUM_PER_CLASS)
        support_dataloader_tar = UT.get_HBKC_data_loader(task_tar, num_per_class=SHOT_NUM_PER_CLASS, split="train",shuffle=False)
        query_dataloader_tar = UT.get_HBKC_data_loader(task_tar, num_per_class=QUERY_NUM_PER_CLASS, split="test",shuffle=True)

        supports_sor, support_labels_sor = support_dataloader_sor.__iter__().next()
        querys_sor, query_labels_sor = query_dataloader_sor.__iter__().next()

        supports_tar, support_labels_tar = support_dataloader_tar.__iter__().next()
        querys_tar, query_labels_tar = query_dataloader_tar.__iter__().next()


        support_features_sor, query_features_sor = feature_encoder(supports_sor.cuda(), querys_sor.cuda(),support_labels_sor.cuda(),query_labels_sor.cuda(), condition='train')
        support_features_tar, query_features_tar = feature_encoder(supports_tar.cuda(), querys_tar.cuda(),support_labels_tar.cuda(),query_labels_tar.cuda(),domain='target', condition='train')

        # Prototype network
        if SHOT_NUM_PER_CLASS > 1:
            support_proto_sor = support_features_sor.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
            support_proto_tar = support_features_tar.reshape(CLASS_NUM, SHOT_NUM_PER_CLASS, -1).mean(dim=1)
        else:
            support_proto_sor = support_features_sor
            support_proto_tar = support_features_tar

        '''few-shot learning'''
        logits1 = euclidean_metric(query_features_sor, support_proto_sor)
        f_loss_sor = crossEntropy(logits1, query_labels_sor.cuda().long())
        f_loss1 = f_loss_sor

        logits2 = euclidean_metric(query_features_tar, support_proto_tar)
        f_loss_tar = crossEntropy(logits2, query_labels_tar.cuda().long())
        f_loss2 = f_loss_tar

        loss = f_loss1 + f_loss2

        '''Domain adpation'''
        DA_source = torch.cat([support_features_sor, query_features_sor], dim=0)
        DA_target = torch.cat([support_features_tar, query_features_tar], dim=0)
        DaCro_source = torch.cat([support_features_sor, query_features_tar], dim=0)
        DaCro_target = torch.cat([support_features_tar, query_features_sor], dim=0)
        loss_da = CroG_DA(DaCro_source,DaCro_target,DA_source,DA_target)


        Fin_loss= loss+0.001*loss_da


        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        # Update parameters
        feature_encoder.zero_grad()
        CroG_DA_optim.zero_grad()
        Fin_loss.backward()
        feature_encoder_optim.step()
        CroG_DA_optim.step()

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        train_total += (time.perf_counter() - t0)


        total_hit_sor += torch.sum(torch.argmax(logits1, dim=1).cpu() == query_labels_sor.long()).item()
        total_num_sor += querys_sor.shape[0]
        total_hit_tar += torch.sum(torch.argmax(logits2, dim=1).cpu() == query_labels_tar.long()).item()
        total_num_tar += querys_tar.shape[0]


        if (episode + 1) % 10== 0:  # display
            # train_loss.append(loss.item())
            print(
                'episode {:>3d}:   fsl loss: {:6.4f},  acc_src {:6.4f}, acc_tar {:6.4f}, loss: {:6.4f}'.format(episode + 1,
                                                                                                                f_loss1.item(),
                                                                                                                total_hit_sor / total_num_sor,
                                                                                                                total_hit_tar / total_num_tar,
                                                                                                                f_loss2.item()))


        '''----TEST----'''
        if (episode + 1) % 50 == 0:
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            print("Testing ...")
            train_end = time.time()
            feature_encoder.eval()
            total_rewards = 0
            counter = 0
            accuracies = []
            predict = np.array([], dtype=np.int64)
            labels = np.array([], dtype=np.int64)

            train_datas, train_labels = train_loader.__iter__().next()

            train_features, _ = feature_encoder(Variable(train_datas).cuda(), Variable(train_datas).cuda(),train_labels.cuda(),train_labels.cuda(),domain='target',condition='test')

            max_value = train_features.max()
            min_value = train_features.min()
            print(max_value.item())
            print(min_value.item())
            train_features = (train_features - min_value) * 1.0 / (max_value - min_value)

            KNN_classifier = KNeighborsClassifier(n_neighbors=1)
            KNN_classifier.fit(train_features.cpu().detach().numpy(), train_labels)
            for test_datas, test_labels in test_loader:
                batch_size = test_labels.shape[0]

                test_features, _ = feature_encoder(Variable(test_datas).cuda(),Variable(test_datas).cuda(),test_labels.cuda(),test_labels.cuda(), domain='target',condition='test')
                test_features = (test_features - min_value) * 1.0 / (max_value - min_value)
                predict_labels = KNN_classifier.predict(test_features.cpu().detach().numpy())
                test_labels = test_labels.numpy()
                rewards = [1 if predict_labels[j] == test_labels[j] else 0 for j in range(batch_size)]

                total_rewards += np.sum(rewards)
                counter += batch_size

                predict = np.append(predict, predict_labels)
                labels = np.append(labels, test_labels)

                accuracy = total_rewards / 1.0 / counter
                accuracies.append(accuracy)

            test_accuracy = 100. * total_rewards / len(test_loader.dataset)

            print('\t\tAccuracy: {}/{} ({:.2f}%)\n'.format( total_rewards, len(test_loader.dataset),100. * total_rewards / len(test_loader.dataset)))
            test_end = time.time()
            print('seeds:', seeds[iDataSet])

            # Training mode
            feature_encoder.train()

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            test_total += (time.perf_counter() - t1)

            if test_accuracy > last_accuracy:
                last_accuracy = test_accuracy
                best_episdoe = episode


                acc[iDataSet] = 100. * total_rewards / len(test_loader.dataset)
                OA = acc
                C = metrics.confusion_matrix(labels, predict)
                A[iDataSet, :] = np.diag(C) / np.sum(C, 1, dtype=np.float64)
                best_predict_all = predict
                best_G, best_RandPerm, best_Row, best_Column, best_nTrain = G, RandPerm, Row, Column, nTrain
                k[iDataSet] = metrics.cohen_kappa_score(labels, predict)

            print('best episode:[{}], best accuracy={}'.format(best_episdoe + 1, last_accuracy))


    print('iter:{} best episode:[{}], best accuracy={}'.format(iDataSet, best_episdoe + 1, last_accuracy))
    print('***********************************************************************************')
    for i in range(len(best_predict_all)):
        best_G[best_Row[best_RandPerm[best_nTrain + i]]][best_Column[best_RandPerm[best_nTrain + i]]] = best_predict_all[i] + 1


AA = np.mean(A, 1)

AAMean = np.mean(AA,0)
AAStd = np.std(AA)

AMean = np.mean(A, 0)
AStd = np.std(A, 0)

OAMean = np.mean(acc)
OAStd = np.std(acc)

kMean = np.mean(k)
kStd = np.std(k)
print ("train time per DataSet(s): " + "{:.5f}".format(train_end-train_start))
print("test time per DataSet(s): " + "{:.5f}".format(test_end-train_end))
print ("average OA: " + "{:.2f}".format( OAMean) + " +- " + "{:.2f}".format( OAStd))
print ("average AA: " + "{:.2f}".format(100 * AAMean) + " +- " + "{:.2f}".format(100 * AAStd))
print ("average kappa: " + "{:.4f}".format(100 *kMean) + " +- " + "{:.4f}".format(100 *kStd))
print ("accuracy for each class: ")
print(f"pure train time (updates only): {train_total:.5f} s")
print(f"test time (evaluation only):    {test_total:.5f} s")
print(f"total time (train+test):        {train_total + test_total:.5f} s")
print(f"avg train time / episode:       {train_total / 2000:.6f} s")
for i in range(CLASS_NUM):
    print ("Class " + str(i) + ": " + "{:.2f}".format(100 * AMean[i]) + " +- " + "{:.2f}".format(100 * AStd[i]))


best_iDataset = 0
for i in range(len(acc)):
    print('{}:{}'.format(i, acc[i]))
    if acc[i] > acc[best_iDataset]:
        best_iDataset = i
print('best acc all={}'.format(acc[best_iDataset]))

#################classification map################################

##Bay Santa
# hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
# for i in range(best_G.shape[0]):
#     for j in range(best_G.shape[1]):
#         if best_G[i][j] == 0:
#             hsi_pic[i, j, :] = [0, 0, 0]
#         if best_G[i][j] == 1:
#             hsi_pic[i, j, :] = [0.5, 0.5, 0.5]
#         if best_G[i][j] == 2:
#             hsi_pic[i, j, :] = [1, 1, 1]


#Farmland
hsi_pic = np.zeros((best_G.shape[0], best_G.shape[1], 3))
for i in range(best_G.shape[0]):
    for j in range(best_G.shape[1]):
        if best_G[i][j] == 2:
            hsi_pic[i, j, :] = [0,0,0]
        if best_G[i][j] == 1:
            hsi_pic[i, j, :] = [1,1,1]

UT.classification_map(hsi_pic[4:-4, 4:-4, :], best_G[4:-4, 4:-4], 24,  "classificationMap/HGA_FSL_Farmland_{}shot.png".format(TEST_LSAMPLE_NUM_PER_CLASS))
