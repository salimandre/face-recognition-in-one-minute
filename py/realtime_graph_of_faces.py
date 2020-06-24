import cv2
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
import time
import random
import shutil
import datetime
import sys

from pcanet import *

# -----------------------------------------------------------------------------------------------------------------------------------------------

# path working directory
PWD='/Users/mac/Desktop/python/reco_login_app/'

# Create the haar cascade for Viola-Jones algo
faceCascade = cv2.CascadeClassifier(PWD+'cascades/haarcascade_frontalface_default.xml')

PATH_LFW_FACES=PWD+'lfw_faces/'

# -----------------------------------------------------------------------------------------------------------------------------------------------

def realTimeLabeling(camera, recording_time=10, snapshots_freq=1):

    # perform snapshots during a time 'recording_time' at a rate 'snapshots_freq'  

    # recording time in seconds
    # snapshots_freq : number of snapshots taken per second

    fps = int(camera.get(cv2.CAP_PROP_FPS))

    data=[]
    cnt_frames = 0
    cnt_snap = 0

    while( camera.isOpened()):
        _, frame = camera.read() 
        cv2.putText(img=frame,text='PRESS', org=(200,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(0, 0, 0), thickness=5)
        cv2.putText(img=frame,text='S', org=(100,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(103, 204, 0), thickness=5)
        cv2.putText(img=frame,text=' TO START', org=(140,200), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(0, 0, 0), thickness=5)
        cv2.putText(img=frame,text='Q', org=(100,300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(20, 50, 200), thickness=5)
        cv2.putText(img=frame,text=' TO QUIT', org=(140,300), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(0, 0, 0), thickness=5)        
        cv2.imshow("Snapshots", frame)
        if cv2.waitKey(1) & 0xFF == ord('s') :
            break
        if cv2.waitKey(1) & 0xFF == ord('q') :
            sys.exit('You chose to quit :(')

    while( camera.isOpened()):
        _, frame = camera.read() 

        Width = frame.shape[1]
        Height = frame.shape[0]

        cnt_frames += 1

        if cnt_frames%(fps//snapshots_freq)==0:
            cnt_snap += 1
            data+=[frame]
            #cv2.imwrite(PATH_WEBCAM_TRUE+TAG_TRUE+str(0)+str(cnt_true)+'.jpg', frame)
            #nb_to_show = cnt_true#nb_seconds_step_1*snapshot_freq_1 - cnt_frames//fps
            cv2.putText(img=frame,text='SNAP '+str(cnt_snap), org=(50,100), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(103, 204, 0), thickness=5)


        cv2.imshow("Snapshots", frame)

        if cnt_frames>=recording_time*fps:
            #for i in range(len(data)):
            #    cv2.imshow("True Face", data[i])
            #    cv2.waitKey(0)
            break

        if cv2.waitKey(1) & 0xFF == ord('q') :
            break

    cv2.destroyAllWindows()
    return data


def make_NewFaceReco(target_name, path_model, n_snapshots=7, n_false=5, sampling=10):

        # target name: label to display on top of bounding box if true
        # path_model: path to folder where to save new face reco model
        # n_false: number of false labelled images in the graph
        # sampling: number of times GrapOfFaces model is sampled
        # n_snapshots: number of snapshots to get true face

        # create paths
        if os.path.exists(path_model):
            ans = input(path_model+" already exists. Remove it? ")
            if ans in {'y', 'yes', 'ok', 'k', 'oui'}:
                shutil.rmtree(path_model)
                os.mkdir(path_model)
            else:
                ans = input('save new face reco model as '+path_model+"_last ? ")
                if ans in {'y', 'yes', 'ok', 'k', 'oui'}:
                    path_model=path_model+"_last"
                    os.mkdir(path_model)
                else:
                    return 0
        else:
            os.mkdir(path_model)

        # face recognition engine pipeline
        GOFFW_ARGS = {'target_name': target_name, \

            'face_shape': (50,37), 'channel':  'gray', 'verbose': 0, \
            
                'similarity': s_Manhattan, 'model': None, 'n_features': 1920}

        face_reco = GraphOfFaces(**GOFFW_ARGS)

        camera = cv2.VideoCapture(0)
        camera.set(3, 480)
        camera.set(4, 480)

        imgs_true = realTimeLabeling(camera, recording_time=n_snapshots*2, snapshots_freq=0.5)
        if target_name is None:
            face_reco.add_nodes(label='true', list_imgs=imgs_true)
        else:
            face_reco.add_nodes(label='true', list_imgs=imgs_true,list_tags=[target_name+'_'+str(i+1) for i in range(len(imgs_true))])

        face_reco.search_best_gof(n_imgs=n_false, path_to_imgs=PATH_LFW_FACES, n_tries=sampling, start=0, end=100, shuffle=True)

        face_reco.boosting(max_n_boost_nodes=40, path_imgs_to_pick=PATH_LFW_FACES, start=0, end=500,shuffle=True)

        face_reco.summary()

        # save face reco
        face_reco.save(path_model)

def load_FaceReco(path_model, verbose=1):

    #create instance
    GOFFW_ARGS = {'target_name': None, \

        'face_shape': (50,37), 'channel':  'gray', 'verbose': 0, \
        
            'similarity': s_Manhattan, 'model': None, 'n_features': 1920}

    face_reco = GraphOfFaces(**GOFFW_ARGS)

    face_reco.load(path_model, verbose)

    return face_reco

# -----------------------------------------------------------------------------------------------------------------------------------------------

def get_random_imgs(n_imgs, path_to_imgs):
    list_imgs=[f for f in os.listdir( path_to_imgs )]
    random.shuffle(list_imgs)
    bag_imgs=[]
    bag_tags=[]
    for filename in list_imgs[:n_imgs]:

        if filename.endswith('.jpg'):

            img = cv2.imread(path_to_imgs+filename, cv2.IMREAD_COLOR)
            bag_imgs+=[img]
            bag_tags+=[filename]

    return bag_imgs, bag_tags

# shift pixel values into [0, maxValue] interval
def shift_pixel_values(I, maxValue, dtype):
    maxpv = np.max(I)
    minpv = np.min(I)
    I = maxValue * np.divide((I - np.min(I)), (np.max(I)-np.min(I)))
    return np.round(I).astype(dtype)

# maps global index from list of lists to (local_index, sub_list)
def phi(i, list_1, list_2, list_3):
    n_1, n_2, n_3 = len(list_1), len(list_2), len(list_3)
    
    if i<n_1:
        return list_1[i]

    if i>=n_1 and i<n_1+n_2:
        return list_2[i-n_1]

    if i>=n_1+n_2:
        return list_3[i-n_1-n_2]

class FaceNode:

    def __init__(self, path_face=None, path_img=None, img=None, label=None, value=None, tag=None):

        self.tag=tag
        self.label=label
        self.value=value
        self.features=None
        self.coord=None

        if img is not None or path_img is not None:
            if path_img is not None:
                img = cv2.imread(path_img, cv2.IMREAD_COLOR)

            #np.stack([cv2.equalizeHist(img[:,:,0]), cv2.equalizeHist(img[:,:,1]),cv2.equalizeHist(img[:,:,2])], axis=2),
            faces = faceCascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30),
                flags = cv2.CASCADE_SCALE_IMAGE
            )

            '''
            if len(faces) == 1:
                (x, y, w, h) = faces[0]
                self.face = img[y:y+w, x:x+h]
                self.coord=(x, y, w, h)
            '''
            if len(faces) > 0:
                faces = sorted(faces, key=lambda tup: tup[2]*tup[3])
                (x, y, w, h) = faces[-1]
                self.face = img[y:y+w, x:x+h]
                self.coord=(x, y, w, h)
            else:
                self.face = None

        if path_face is not None:
            self.face = cv2.imread(path_face, cv2.IMREAD_GRAYSCALE)

    def display_face(self, delay=0):

        if self.tag is None:
            cv2.imshow("Face of Node", self.face)
        else:
            cv2.imshow(self.tag, self.face)
        cv2.waitKey(delay)
        pass

    def notEmpty(self):
        if self.face is not None:
            return True
        else:
            return False

    def get_features(self, model, n_features=1920):
        self.features = model.predict(np.array([self.face]))
        self.features = self.features[0,:n_features]



class GraphOfFaces:

    def __init__(self, similarity, model=None, target_name=None, channel='green', face_shape=(100,100), n_features=1920, verbose=0):

        self.verbose = verbose

        if target_name is not None:
            self.target_name=target_name
        else:
            self.target_name='TRUE'

        self.channel=channel
        self.face_shape=face_shape

        self.model=model
        self.n_features=n_features

        self.face_nodes={'true': [],'false': [],'no_label': [], 'support': []}

        self.value_nodes={'true': None, 'false': -1, 'no_label': None, 'support': None}

        self._update_stats()

        self.simMat=np.zeros((self.n_nodes, self.n_nodes))
        self.similarity=similarity

        self.use_boosting=False

    def summary(self):
        print('\n***** Graph Of Faces *****\n')
        print('# total Nodes = ', self.n_nodes)
        print('# True Nodes = ', self.n_true)
        print('# False Nodes = ', self.n_false)
        print('# Unlabelled Nodes = ', self.n_unlab)
        print('# Edges = ',self.n_edges)
        print('Face_shape = ', self.face_shape)
        print('# features = ', self.n_features)
        if self.use_boosting:
            print('Boosting mode = ON')
        else:
            print('Boosting mode = OFF')
        print('# Support Nodes = ', len(self.face_nodes['support']))
        print('\n**************************\n\n')

    def save(self, path_model):

        # save faces from graph
        path_false=path_model+'/false/'
        path_true=path_model+'/true/'
        path_support=path_model+'/support/'
        if not os.path.exists(path_false):
            os.mkdir(path_false)
        if not os.path.exists(path_true):
            os.mkdir(path_true)        
        if not os.path.exists(path_support):
            os.mkdir(path_support) 

        # save false face
        for node in self.face_nodes['false']:
            filename=node.tag+'.jpg'
            cv2.imwrite(path_false+filename, node.face)

        # save true faces
        for node in self.face_nodes['true']:
            filename=node.tag+'.jpg'
            cv2.imwrite(path_true+filename, node.face)

        # save support faces
        for node in self.face_nodes['support']:
            filename=node.tag+'.jpg'
            cv2.imwrite(path_support+filename, node.face)

        self.model.save(path_model=path_model, target_name=self.target_name)

    def load(self, path_model, verbose=1):

        # load pcanet
        self.model = PcaNet()
        self.model.load(path_model=path_model)
        if verbose!=0:
            self.model.summary()

        # load faces
        path_false=path_model+'/false/'
        path_true=path_model+'/true/'
        path_support=path_model+'/support/'

        for filename in os.listdir( path_false ):
            if filename.endswith('.jpg'):
                node = FaceNode(path_face=path_false+filename, label='false', value=self.value_nodes['false'], tag=filename.split('.')[0])
                node.get_features(self.model)
                self.face_nodes['false']+=[node]

        for filename in os.listdir( path_true ):
            if filename.endswith('.jpg'):  
                node = FaceNode(path_face=path_true+filename, label='true', value=self.value_nodes['true'], tag=filename.split('.')[0])
                node.get_features(self.model)            
                self.face_nodes['true']+=[node]

        for filename in os.listdir( path_support ):
            if filename.endswith('.jpg'):
                node = FaceNode(path_face=path_support+filename, label='support', value=self.value_nodes['support'], tag=filename.split('.')[0])
                node.get_features(self.model)
                self.face_nodes['support']+=[node]

        # load target name
        with open(path_model+'/parameters/summary.json', 'r') as outfile:
            dict_summary=json.load(outfile)

        self.target_name=dict_summary['target_name']

        self._update_stats()

        if verbose!=0:
            self.summary()

    def _update_stats(self):

        self.n_true = len(self.face_nodes['true'])
        self.n_false = len(self.face_nodes['false'])
        self.n_unlab = len(self.face_nodes['no_label'])
        self.n_nodes = self.n_true + self.n_false + self.n_unlab
        self.n_edges = int(0.5*self.n_nodes*(self.n_nodes-1))

        if self.n_true>0:
            self.value_nodes['true'] = - self.value_nodes['false']*self.n_false/self.n_true



    def add_nodes(self, label=None, value=None, path_to_imgs=None, list_imgs=None, list_tags=None):

        tic = time.time()

        if path_to_imgs is not None:

            face_nodes=[]
            for filename in os.listdir( path_to_imgs ):

                if filename.endswith('.jpg'):

                    # create a Face Node
                    node = FaceNode(path_to_imgs+filename, label=label, value=self.value_nodes[label], tag=filename.split('.')[0])          

                    if node.notEmpty():

                        # preprocessing step
                        if self.channel == 'blue':
                            node.face = node.face[:,:,0]
                        if self.channel == 'green':
                            node.face = node.face[:,:,1]
                        if self.channel == 'red':
                            node.face = node.face[:,:,2]
                        if self.channel == 'gray':
                            node.face = cv2.cvtColor(node.face, cv2.COLOR_BGR2GRAY)

                        node.face = cv2.resize(node.face, self.face_shape[::-1], interpolation=cv2.INTER_LINEAR)

                        if self.model is not None:
                            # infer features by applying model on face
                            node.get_features(self.model, self.n_features)

                        if self.verbose>0:
                            node.display_face(delay=100)

                        face_nodes += [node]

            self.face_nodes[label] += face_nodes

            if self.verbose>0:
                cv2.destroyAllWindows()

        else:

            if list_imgs is not None:


                face_nodes=[]
                for i, img in enumerate(list_imgs):

                        if list_tags is None:
                            # create a Face Node
                            node = FaceNode(img = img, label=label, value=self.value_nodes[label], tag=label+str(i+1))          
                        else:
                            # create a Face Node
                            node = FaceNode(img = img, label=label, value=self.value_nodes[label], tag=list_tags[i])    

                        if node.notEmpty():

                            # preprocessing step
                            if self.channel == 'blue':
                                node.face = node.face[:,:,0]
                            if self.channel == 'green':
                                node.face = node.face[:,:,1]
                            if self.channel == 'red':
                                node.face = node.face[:,:,2]
                            if self.channel == 'gray':
                                node.face = cv2.cvtColor(node.face, cv2.COLOR_BGR2GRAY)

                            node.face = cv2.resize(node.face, self.face_shape[::-1], interpolation=cv2.INTER_LINEAR)

                            if self.model is not None:
                                # infer features by applying model on face
                                node.get_features(self.model, self.n_features)

                            if self.verbose>0:
                                node.display_face(delay=100)

                            face_nodes += [node]

                self.face_nodes[label] += face_nodes

                if self.verbose>0:
                    cv2.destroyAllWindows()

        self._update_stats()

        toc = time.time()
        if self.verbose>0:
            print('adding nodes time: ',toc-tic)

        return len(face_nodes)


    def pop(self, label):

        self.face_nodes[label].pop()

        self._update_stats()

    def replace_model(self, new_model):

        self.model=new_model

        get_node = lambda i: phi(i, self.face_nodes['false'], self.face_nodes['true'], self.face_nodes['no_label'])
        for i in range(self.n_nodes):

            node = get_node(i)
            node.get_features(self.model, self.n_features)

    def init_pcanet_model(self):

        get_node = lambda i: phi(i, self.face_nodes['false'], self.face_nodes['true'], self.face_nodes['no_label'])

        X_faces = []
        for i in range(self.n_nodes):

            node = get_node(i)
            X_faces += [node.face]

        X_faces = np.array(X_faces)

        layer_1 = Layer_PcaNet(n_filters=10, filter_shape=(4,4))
        layer_2 = Layer_PcaNet(n_filters=3, filter_shape=(4,4))
        network = PcaNet(layers=[layer_1, layer_2], shape_in_blocks = (6,4))

        network.summary()

        # training PcaNet model

        network.train(X_faces, n_patches_per_img=150)

        if self.verbose>0:
            for layer_name in network.layers.keys():
                network.layers[layer_name].display_filters()

        # inference through PcaNet model

        outputs = network.inference(X_faces)

        # hashing step

        hashed_outputs = network.binary_hashing_step(outputs, sparsity=0.5)

        # histogram block step 

        feature_map = network.histogram_encoding_step(hashed_outputs)

        for i in range(self.n_nodes):

            node = get_node(i)
            node.features = feature_map[i,:]

        self.model = network


    def compute_simMat(self):

        tic = time.time()

        self.simMat=np.zeros((self.n_nodes, self.n_nodes))

        get_node = lambda i: phi(i, self.face_nodes['false'], self.face_nodes['true'], self.face_nodes['no_label'])

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if j>i:
                    node_i = get_node(i)
                    node_j = get_node(j)
                    self.simMat[i,j] = self.similarity(node_i.features, node_j.features)
                    self.simMat[j,i] = self.simMat[i,j]

        toc = time.time()
        if self.verbose>0:
            print('compute_simMat time: ',toc-tic)

    def display_simMat_custom(self,col_true, col_false, col_unlab, p_value_max=0.95):

        get_node = lambda i: phi(i, self.face_nodes['false'], self.face_nodes['true'], self.face_nodes['no_label'])

        white = np.array([255.,255.,255])

        col_true = np.array(col_true) #[26, 188, 156])
        col_false = np.array(col_false) #[205, 92, 92])
        col_unlab = np.array(col_unlab) #[46, 134, 193])


        tags = [get_node(i).tag for i in range(self.n_nodes)]

        q_max_thresh = np.quantile(self.simMat, p_value_max)
        I_simMat = self.simMat.clip(0, q_max_thresh)
        I_simMat = shift_pixel_values(I_simMat, 255., 'int')
        I_simMat = np.stack([I_simMat]*3, axis=2)

        for i in range(self.n_nodes):
            for j in range(self.n_nodes):
                if j-i!=0:
                    node_i = get_node(i)
                    node_j = get_node(j)

                    if node_i.label is not None:
                        if node_i.label == 'false':
                            I_simMat[i,j,:] = white - np.multiply(I_simMat[i,j,0]/255., white - col_false) #[0., 0., 255.] [RGB(205, 92, 92)] [50, 163, 163]

                        if node_i.label == 'true':
                            I_simMat[i,j,:] = white - np.multiply(I_simMat[i,j,0]/255., white - col_true) #[255., 0., 0.]

                    if node_i.label == 'no_label':
                        I_simMat[i,j,:] = white - np.multiply(I_simMat[i,j,0]/255., white - col_unlab) #[138.,43.,226.] [117.,212.,29.]

        plt.imshow(I_simMat)
        plt.xticks([], [])
        plt.yticks(range(self.n_nodes), tags)
        plt.show()  

    def __get_W_ul(self):

        W_ul = np.zeros((self.n_unlab, self.n_true+self.n_false))

        for i in range(W_ul.shape[0]):
            for j in range(W_ul.shape[1]):
                    node_i = self.face_nodes['no_label'][i]
                    node_j = (self.face_nodes['false']+self.face_nodes['true'])[j]
                    W_ul[i,j] = self.similarity(node_i.features, node_j.features)

        return W_ul

    def __get_W_uu(self):

        W_uu = np.zeros((self.n_unlab, self.n_unlab))

        for i in range(len(W_uu)):
            for j in range(len(W_uu)):
                if j>i:
                    node_i = self.face_nodes['no_label'][i]
                    node_j = self.face_nodes['no_label'][j]
                    W_uu[i,j] = self.similarity(node_i.features, node_j.features)
                    W_uu[j,i] = W_uu[i,j]

        return W_uu

    def harmonic_solver(self, gamma_reg=0):

        tic = time.time()

        if self.use_boosting:
            self.face_nodes['no_label']+=self.face_nodes['support']
            self._update_stats()

        W_uu = self.__get_W_uu()

        W_ul = self.__get_W_ul()

        D_u = np.sum(np.concatenate((W_ul,W_uu),axis=1), axis=1)
        D_u = np.diag(D_u)

        L_u = D_u - W_uu

        old_values = np.array([self.value_nodes['false']]*self.n_false+[self.value_nodes['true']]*self.n_true)

        new_values = np.linalg.inv(L_u+gamma_reg*np.eye(len(D_u)))@W_ul@old_values

        toc = time.time()
        if self.verbose>0:
            print('harmonic_solver time: ',toc-tic)

        nodes_new_label=[]
        for i, node in enumerate(self.face_nodes['no_label']):
            node.value = new_values[i]
            if node.value>0:
                node.label='true'
                #self.face_nodes['true']+=[node]
            if node.value<0:
                node.label='false'
            nodes_new_label+=[node]
            #print(node.tag.ljust(20),'\t',node.label.ljust(8),'\t',node.value)

        self.face_nodes['no_label']=[]
        self._update_stats()

        if self.use_boosting and len(self.face_nodes['support'])>0:
            return nodes_new_label[:-len(self.face_nodes['support'])]
        else:
            return nodes_new_label

    def evaluate(self, path_imgs_to_pick, start=0, end=1000, shuffle=False, sensitivity='tf'):

        # evaluate recall on images which have wrong face
        # -> n_true_negative / n_negative
        # sensitivity = 'tf' (true false) or 'tp' (true positive)

        list_of_imgs=[f for f in os.listdir( path_imgs_to_pick )]
        if shuffle:
            random.shuffle(list_of_imgs)

        labels=np.zeros(end-start)
        cnt_faces = 0
        cnt_imgs = 0
        cnt_true=0
        cnt_false=0
        print('\n> evaluation of GraphOfFaces...\n')
        for filename in list_of_imgs:

            if filename.endswith('.jpg'):

                cnt_imgs+=1

                if cnt_imgs>=start and cnt_imgs<=end:

                    target = cv2.imread(path_imgs_to_pick+filename, cv2.IMREAD_COLOR)

                    self.add_nodes(label='no_label', list_imgs=[target], list_tags=[filename.split('.')[0]])
                    new_label_node = self.harmonic_solver()

                    if len(new_label_node)>0 and new_label_node[0].label=='true':
                        labels[cnt_faces]=1
                        cnt_true+=1
                        cnt_faces+=1

                    if len(new_label_node)>0 and new_label_node[0].label=='false':
                        labels[cnt_faces]=0
                        cnt_false+=1
                        cnt_faces+=1

                    if cnt_faces>0:
                        if sensitivity=='tf':
                            acc=round((1.-cnt_true/(cnt_true+cnt_false))*100., 1)
                            str_update = 'TRUE FALSE: '+str(acc)+'%\t\t# NODES: '+str(self.n_nodes)+'\t\tBOOSTING: '+str(self.use_boosting)+'\t\t# FACES: '+str(cnt_faces)+'\t\t# IMGS: '+str(cnt_imgs)
                            print(str_update,end='\r    ',flush=True)
                        if sensitivity=='tp' and len(new_label_node)>0:
                            acc=round(100.*cnt_true/(cnt_true+cnt_false), 1)
                            str_update = 'TRUE POS: '+str(acc)+'%\t\t# NODES: '+str(self.n_nodes)+'\t\tBOOSTING: '+str(self.use_boosting)+'\t\tIMG: '+filename+'\t\t# INFERENCE: '+new_label_node[0].label
                            #print(str_update,end='\r    ',flush=True) 
                            print(str_update)                           

                if cnt_imgs>end:
                    print(str_update)
                    return acc


    def boosting(self,max_n_boost_nodes, path_imgs_to_pick, start=0, end=1000, shuffle=False):

        self.set_boosting_mode('on')

        list_of_imgs=[f for f in os.listdir( path_imgs_to_pick )]
        if shuffle:
            random.shuffle(list_of_imgs)

        labels=np.zeros(end-start)
        cnt_faces = 0
        cnt_imgs = 0
        cnt_true=0
        cnt_false=0

        tags_picked=[]
        print('\n> boosting...\n')
        for filename in list_of_imgs:

            if filename.endswith('.jpg'):

                if len(tags_picked)<max_n_boost_nodes:

                    cnt_imgs+=1

                    if cnt_imgs>=start and cnt_imgs<=end:

                        target = cv2.imread(path_imgs_to_pick+filename, cv2.IMREAD_COLOR)

                        self.add_nodes(label='no_label', list_imgs=[target], list_tags=[filename.split('.')[0]])
                        new_label_node = self.harmonic_solver()


                        if len(new_label_node)>0 and new_label_node[0].label=='true':
                            labels[cnt_faces]=1
                            cnt_true+=1
                            cnt_faces+=1

                        if len(new_label_node)>0 and new_label_node[0].label=='false':
                            labels[cnt_faces]=0
                            cnt_false+=1
                            cnt_faces+=1
                            if np.random.rand()<1-(np.abs(new_label_node[0].value)/0.1) and len(tags_picked)<max_n_boost_nodes:
                                tags_picked += [filename]
                                new_label_node[0].label='no_label'
                                self.face_nodes['support']+=[new_label_node[0]]

                        if cnt_faces>0:
                            acc=round((1.-cnt_true/(cnt_true+cnt_false))*100., 1)
                            str_update = 'ACCURACY: '+str(acc)+'%\t\t# NODES: '+str(self.n_nodes)+'\t\t# BOOSTING NODES: '+str(len(self.face_nodes['support']))+'\t\t# FACES: '+str(cnt_faces)+'\t\t# IMGS: '+str(cnt_imgs)                            
                            print(str_update, end='\r     ',flush=True)
                    
                    if cnt_imgs>end:
                        print(str_update)
                        return tags_picked

                else:
                    print(str_update)
                    return tags_picked

    def set_boosting_mode(self, mode='off'):
        if mode=='on':
            self.use_boosting=True
        else:
            self.use_boosting=False

    def search_best_gof(self, n_imgs, path_to_imgs, n_tries, start, end, shuffle=True):

        best_acc=0
        best_model=None
        best_false_nodes=[]
        for i in range(n_tries):
            bag_imgs, bag_tags = get_random_imgs(n_imgs, path_to_imgs)
            n_added = self.add_nodes(label='false', value=-1, list_imgs=bag_imgs, list_tags=bag_tags)

            self.init_pcanet_model()
            acc = self.evaluate(path_to_imgs,start,end,shuffle, sensitivity='tf')
            if acc > best_acc:
                best_acc=acc
                best_model=self.model
                best_imgs=bag_imgs
                best_tags=bag_tags
            
            print('try # '+str(i+1)+'\t\tACC = '+str(acc)+'%\t\tBEST ACC = '+str(best_acc)+'%')
            
            for i in range(n_added):
                self.pop(label='false')

        self.replace_model(new_model=best_model)
        self.add_nodes(label='false', value=-1, list_imgs=best_imgs, list_tags=best_tags)

        print('\nbest acc out of '+str(n_tries)+': '+str(best_acc)+'%')

    def webcamFaceRecognition(self, recording_time=10, snapshots_freq=1):

        # perform real time face recogntion by doing label inference on each frame during 
        # a time 'recording_time' at a rate 'snapshots_freq'  

        # recording time in seconds
        # snapshots_freq : number of snapshots taken per second

        camera = cv2.VideoCapture(0)
        camera.set(3, 480)
        camera.set(4, 480)

        if self.model is not None:

            fps = int(camera.get(cv2.CAP_PROP_FPS))

            cnt_frames = 0
            cnt_snaps = 0
            
            while( camera.isOpened()):
                _, frame = camera.read() 

                Width = frame.shape[1]
                Height = frame.shape[0]

                cnt_frames += 1

                if cnt_frames%(fps//snapshots_freq)==0:
                    cnt_snaps += 1

                    # update graph of nodes
                    self.add_nodes(label='no_label', list_imgs=[frame], list_tags=['gautier_'+str(cnt_snaps)])

                    frame_node = self.harmonic_solver()
                    if len(frame_node)>0 and frame_node[0].label=='true':
                        coord = frame_node[0].coord
                        cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0,255,0), 3)
                        cv2.putText(img=frame,text=self.target_name, org=(coord[0]-20, coord[1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3, color=(0, 255, 0), thickness=5)

                    if len(frame_node)>0 and frame_node[0].label=='false':
                        coord = frame_node[0].coord
                        cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0,0,255), 3)
                        cv2.putText(img=frame,text='UNKNOWN', org=(coord[0]-20, coord[1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(0, 0, 255), thickness=3)

                cv2.imshow("object inference", frame)

                if cnt_frames>=recording_time*fps:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q') :
                    break
            
        else:
            print('model of GraphOfFaces is None! \nYou need to initialize it by running methods search_best_gof() or init_pcanet_model()')
            return 0

    def loginRecognitionTest(self, path_snaps_refused, success_score=5, recording_time=10, snapshots_freq=1, showcam=False):

        # perform real time face recogntion by doing label inference on each frame during 
        # a time 'recording_time' at a rate 'snapshots_freq'  

        # recording time in seconds
        # snapshots_freq : number of snapshots taken per second

        camera = cv2.VideoCapture(0)
        camera.set(3, 480)
        camera.set(4, 480)

        score=0
        status='FAILED'
        remaining_time=round(float(recording_time),1)
        if self.model is not None:

            fps = int(camera.get(cv2.CAP_PROP_FPS))

            cnt_frames = 0
            cnt_snaps = 0
            
            while( camera.isOpened()):
                _, frame = camera.read() 

                Width = frame.shape[1]
                Height = frame.shape[0]

                cnt_frames += 1


                if cnt_frames%(fps//snapshots_freq)==0:
                    cnt_snaps += 1
                    remaining_time = round(float(recording_time) - cnt_snaps/snapshots_freq,1)

                    # update graph of nodes
                    self.add_nodes(label='no_label', list_imgs=[frame], list_tags=['gautier_'+str(cnt_snaps)])

                    frame_node = self.harmonic_solver()
                    if len(frame_node)>0 and frame_node[0].label=='true':
                        coord = frame_node[0].coord
                        cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0,255,0), 3)
                        cv2.putText(img=frame,text=self.target_name, org=(coord[0]-20, coord[1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 3, color=(0, 255, 0), thickness=5)
                        score+=1
                        if score>=success_score:
                            status='SUCCESS'
                            break

                    if len(frame_node)>0 and frame_node[0].label=='false':
                        coord = frame_node[0].coord
                        cv2.rectangle(frame, (coord[0], coord[1]), (coord[0]+coord[2], coord[1]+coord[3]), (0,0,255), 3)
                        cv2.putText(img=frame,text='UNKNOWN', org=(coord[0]-20, coord[1]-20), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(0, 0, 255), thickness=3)

                cv2.putText(img=frame,text='Shutdown - '+str(remaining_time), org=(20, 60), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale = 2, color=(255, 0, 0), thickness=3)
                
                if showcam:
                    cv2.imshow("object inference", frame)

                if cnt_frames>=recording_time*fps:
                    break

                if cv2.waitKey(1) & 0xFF == ord('q') :
                    break
            print(status)

            if status=='FAILED':
                # take a snapshot and shutdown computer
                ts = datetime.datetime.now()
                filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))
                p = os.path.sep.join((path_snaps_refused, filename))
                # save the file
                cv2.imwrite(p, frame.copy())
                print("[INFO] saved {}".format(filename))

                # shutdown computer (MAC only) if recognition failed
                os.system("bash /Users/mac/Desktop/python/reco_login_app/shutdown.sh")
            
        else:
            print('model of GraphOfFaces is None! \nYou need to initialize it by running methods search_best_gof() or init_pcanet_model()')
            return 0

# -----------------------------------------------------------------------------------------------------------------------------------------------




d_Manhattan = lambda x,y: np.sum(np.abs(x-y))

s_Manhattan = lambda x,y: (d_Manhattan(x,y)/5000)**-4

d_Euclidean = lambda x,y: np.sqrt(np.sum((x-y)**2))

s_Cosine = lambda x,y: 10*np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

d_Canberra = lambda x,y: np.sum( np.abs(x-y)/(np.abs(x)+np.abs(y)+1) )

s_Canberra = lambda x,y: d_Canberra(x,y)**-3


def d_Hassanat(x,y):
    d = 0
    for i in range(len(x)):

        mini = min(x[i],y[i])
        maxi = max(x[i],y[i])

        if mini >= 0:
            d += 1 - (1+mini)/(1+maxi)
        else:
            d+= 1 - (1+mini+np.abs(mini))/(1+maxi+np.abs(mini))

    return d

s_Hassanat = lambda x,y: d_Hassanat(x,y)**-3



# -----------------------------------------------------------------------------------------------------------------------------------------------

