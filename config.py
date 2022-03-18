

config_dict = {
    'size': 256,                       # height is equal to width

    'is_single': False,
    'load': False,                     # load pretrained classifier model

    'train_batch_size': 2,
    'test_batch_size': 1,

    'save_dir': 'outputs',
    'class_name': 'textile',

    'distance_metric':'mahalonobis',    # 'mahalonobis_no_sqrt', 'euclidean'

    'train_dataset_path': '/home/ubuntu/dataset/train/',
    'test_dataset_path': '/home/ubuntu/dataset/test/', #image001.jpg',
    
    'train_feature_filepath': 'model/padim_model_train.pkl',         # trained model path
    'classifier_model_filepath': 'model/xgb_model.pkl'               # trained classifier
}
