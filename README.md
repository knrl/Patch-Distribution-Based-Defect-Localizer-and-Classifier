# Patch Distribution Based Defect Localizer and Classifier

## Directory Hierarchy

```bash
  dataset
    |__train_dataset_folder
        |__ ....jpg
    |__test_dataset_folder
        |__ good     
            |__ ....jpg
        |__ defect
            |__ ....jpg
  model
    |__model.pb/h5/ckpt
  classification_model
    |__ xgboost.py
  dataset_manager
    |__dataset_manager.py
  train.py
  test.py
  eval.py
  config.py
```

* model directory: Represents the folder with the previously trained model where model outputs are saved. Model extensions .h5, .pb, .ckpt etc. may form.

* dataset_manager directory: The folder containing the files that make the dataset ready for training and testing.

* train.py : The file that performs the training of both the padim model and the classification model from data preparation to the end of the training phase. Here it will be done by calling the classes and methods of the files inside the folders.

* test.py : The file that performs the testing of both the pad and the classification model from data preparation to the end of the training phase. Here it will be done by calling the classes and methods of the files inside the folders.

* config.py: File that containing the training or test parameters of the models. With this file, instead of terminal argument, train.py and test.py files import the parameters from here.

## Citation
```
@inproceedings{defard2021padim,
  title={Padim: a patch distribution modeling framework for anomaly detection and localization},
  author={Defard, Thomas and Setkov, Aleksandr and Loesch, Angelique and Audigier, Romaric},
  booktitle={International Conference on Pattern Recognition},
  pages={475--489},
  year={2021},
  organization={Springer}
}
```
```
@inproceedings{Chen:2016:XST:2939672.2939785,
 author = {Chen, Tianqi and Guestrin, Carlos},
 title = {{XGBoost}: A Scalable Tree Boosting System},
 booktitle = {Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining},
 series = {KDD '16},
 year = {2016},
 isbn = {978-1-4503-4232-2},
 location = {San Francisco, California, USA},
 pages = {785--794},
 numpages = {10},
 url = {http://doi.acm.org/10.1145/2939672.2939785},
 doi = {10.1145/2939672.2939785},
 acmid = {2939785},
 publisher = {ACM},
 address = {New York, NY, USA},
 keywords = {large-scale machine learning},
}
```