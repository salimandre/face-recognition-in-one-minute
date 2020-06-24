
from realtime_graph_of_faces import *

# -----------------------------------------------------------------------------------------------------------------------------------------------

# path working directory
PWD='/Users/mac/Desktop/python/reco_login_app/'

# where to save models
PATH_MODELS=PWD+'models'

PATH_LFW_FACES=PWD+'lfw_faces/'

# where to save snapshots for login recognition function when failed
PATH_LOG_IN_REFUSED=PWD+'log_in_refused'

# -----------------------------------------------------------------------------------------------------------------------------------------------


if __name__ == '__main__':

    make_NewFaceReco(target_name='Gautier', path_model=PATH_MODELS+'/model_3', n_snapshots=7, n_false=5, sampling=10)

    face_reco = load_FaceReco(PATH_MODELS+'/model_3',verbose=1)

    face_reco.set_boosting_mode('on')

    #face_reco.loginRecognitionTest(path_snaps_refused=PATH_LOG_IN_REFUSED, success_score=2, recording_time=30, snapshots_freq=3, showcam=True)

    face_reco.webcamFaceRecognition(recording_time=120, snapshots_freq=4)

    #face_reco.evaluate( path_imgs_to_pick=PATH_LFW_FACES, start=0, end=1000, shuffle=True, sensitivity='tf')
