import os
#PASCAL_PATH =os.path.join(DATA_PATH, 'pascal_voc')
TRAINING_json_DATA = 'data/training_data'
TESTING_json_DATA = 'data/testing_data'#'data/Inputs'

TRAINING_key_DATA = 'key_ori/trining_keys'
TESTING_key_DATA = 'key_ori/testing_keys'

W2V_PATH = '/Users/abhipubali/Public/DropBox/sem2_s18/chenhao_course/word_embeddings_benchmarks/scripts/word2vec_wikipedia/wiki_w2v_300.txt'

LABEL_data_TRAINING = 'cluster/train_separate.cluster'
label_data_testing = 'cluster/testing1.cluster'
IND_MODEL_PATH = 'trained_model/model_arg_TD_2.h5'
OUTPUT_KEY = 'output/key'
CROSS_MODEL_PATH ='cross_doc_model/model3_mse500.h5'
MODEL_TO_SAVE_AT = [150,500,1000]
epch = 500
