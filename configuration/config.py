import os
#PASCAL_PATH =os.path.join(DATA_PATH, 'pascal_voc')
TRAINING_json_DATA = '../ECAJsonData/training_data'
TESTING_json_DATA = 'data/testing_data'#'data/Inputs'

TRAINING_key_DATA = 'key_ori/trining_keys'
TESTING_key_DATA = 'key_ori/testing_keys'

LABEL_data_TRAINING = 'cluster/train_separate.cluster'
LABEL_data_TRAINING = 'cluster/tiny_train_separate.cluster'

label_data_testing = 'cluster/testing1.cluster'
IND_MODEL_PATH = 'trained_model/model_arg_TD_2.h5'
OUTPUT_KEY = 'output/key'
CROSS_MODEL_PATH ='cross_doc_model/model3_mse500.h5'
MODEL_TO_SAVE_AT = [150,500,1000]
epch = 500

REO = 'ontology_processing/reo.json'
ERE2REO = 'ontology_processing/ere2reo.json'
