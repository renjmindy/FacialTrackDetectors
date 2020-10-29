# Face Detectors

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here 
![GitHub repo size](https://img.shields.io/github/repo-size/scottydocs/README-template.md)
![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)
![GitHub stars](https://img.shields.io/github/stars/scottydocs/README-template.md?style=social)
![GitHub forks](https://img.shields.io/github/forks/scottydocs/README-template.md?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/scottydocs?style=social) --->

## Motivation

Workforce shortage, heavy workload, and salary disparities have never been more poignant in healthcare than they are today. It’s safe to say that doctors and nurses can’t deliver good care to their patients if they neglect their own health. In the circumstances, vendors and healthcare advisors and workers have high hopes of integrating technology into healthcare. **Technology can ease the workload of healthcare workers and transform the way patients are diagnosed and treated.**

Slowly but surely, Artificial Intelligence (AI), Deep Learning, Machine Learning, the Internet of Things, Natural Language Processing (NLP), and Face Recognition are becoming the new norm in the healthcare sector. Today, healthcare providers can deliver excellent patient care by blending their competencies and innovation.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/facial-recognition-for-healthcare-disruption-1.png)

The total number of data breaches reported went up to 36.12% annually, from 371 breaches in 2018 to 505 breaches in 2019.

Take data in healthcare. It’s diverse and complex. And on top of that, it’s hard to store and keep safe. Since the healthcare sector generates plenty of data that is an easy target for hackers, AI offers its watchful eye. According to a 2019 Ponemon report, [73% of organizations are understaffed](https://www.domaintools.com/content/2019-ponemon-report-staffing-it-age-automation.pdf) and fail to detect and respond to data breaches fast. [HIPAA Journal](https://www.hipaajournal.com/december-2019-healthcare-data-breach-report/#:~:text=That's%20195.61%25%20more%20than%202018,to%20505%20breaches%20in%202019) stated that 2019 brought about an increase in data breaches. 41,232,527 patient records were disclosed and stolen in 2019.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/facial-recognition-for-healthcare-disruption-2.png)

With millions of healthcare records breached so far, hospitals are seeking a robust solution to guard the patients’ data. This is where AI comes to aid. Today, one of the key objectives of artificial intelligence in healthcare is to safeguard and transfer sensitive data securely. AI provides solutions that have the power to automate malware analysis and threat intelligence.

Face recognition is a subset of AI. It is used in healthcare for plenty of purposes. Take a closer look at the key technology use cases:

### **Patient Check-in and Check-Out Procedures:**

Patient identification solutions have recently gained momentum. They simplify the whole patient check-in process and free hospital personnel from paperwork.

### **Diagnosing Diseases and Conditions Using Face Recognition:**

Face recognition has hovered almost all healthcare domains and the diagnostic process is no exсeption. Healthcare evangelists and advisors claim that in the coming years, [health mirrors](https://www.theguardian.com/technology/2012/jan/22/medical-mirror-ming-zher-poh) will be in high gear. 

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/facial-recognition-for-healthcare-disruption-5.jpg)

### **Face Recognition Against COVID-19:**

As the pandemic sweeps the world, healthcare advisors are seeking a way to stop the spread of the deadly virus. AI and face recognition technologies are at the frontline to make it happen. If you want to understand how can AI be used in healthcare to win the war against the coronavirus, take a closer look:

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/facial-recognition-for-healthcare-disruption-3.png)

### **Emotion Detection in Mental Therapy:**

   - fair patient emotion assessment
   - personalized therapy
   - access to mental care (app check ups instead of seeing a mental healthcare provider)
   - real-time harmful habit detection (lip biting, cheek biting, eye rubbing, etc)

Another use case of facial recognition technology in healthcare is real-time emotion tracking. Using facial recognition for mental health purposes, patients can get personalized, patient-centered, efficient, and timely care. The next-gen technology is used to track facial landmarks and cues to interpret the patient’s inner feelings. Face-to-face therapy has a lot to offer as listed above.

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You should install `keras`, `matplotlib`, `numpy`, `scikit-learn` and `scikit-image`
* You have a `<Windows/Linux/Mac>` machine.
* Do you have modern Nvidia [GPU](https://en.wikipedia.org/wiki/Graphics_processing_unit)? 
  There is your video-card model in [list](https://developer.nvidia.com/cuda-gpus) and CUDA capability >= 3.0?

   - Yes. You can use it for fast deep learning! In this work using tensorflow backend with GPU is recommended. Read [installation notes] (https://www.tensorflow.org/install/) with attention to gpu section, install all requirements and then install GPU version `tensorflow-gpu`.
   - No. CPU is enough for this task, but we have to use only simple model. Read [installation notes](https://www.tensorflow.org/install/) and install CPU version `tensorflow`.
   
* It is worth noting that there is a framing from all sides in most of the images. This framing can appreciably worsen the quality of channels alignment. Here, borders on the plates should be detected using `Canny edge detector`, so that we can crop the images according to these edges. The example of using Canny detector implemented in skimage library can be found [here](https://scikit-image.org/docs/dev/auto_examples/edges/plot_canny.html).
* Important thing in deep learning is `augmentation`. Sometimes, if your model are complex and cool, you can increase quality by using good augmentation.

   - Keras provide good [images preprocessing and augmentation](https://keras.io/api/preprocessing/image/). This preprocessing executes online (on the fly) while learning.
   - Of course, if you want using samplewise and featurewise `center and std normalization` you should run this transformation on predict stage. But you will use this classifier to fully convolution detector, in this case such transformation quite complicated. As such, it's not recommended to use them in classifier.
   - For heavy augmentation you can use library [imgaug](https://github.com/aleju/imgaug). If you need, you can use this library in offline manner (simple way) and online manner (hard way). However, hard way is not so hard: you only have to write [python generator](https://wiki.python.org/moin/Generators), which returns image batches, and pass it to [fit_generator](https://keras.io/api/models/model/#fit_generator).
   
* For fitting you can use one of Keras optimizer algorithms. [Good overview](https://ruder.io/optimizing-gradient-descent/).

    - To choose best learning rate strategy you should read about `EarlyStopping` and `ReduceLROnPlateau` or `LearningRateScheduler` on [callbacks](https://keras.io/callbacks/) page of keras documentation, it's very useful in deep learning.
    - If you repeat architecture from some paper, you can find information about good optimizer algorithm and learning rate strategy in this paper. For example, every [keras application](https://keras.io/api/applications/) has link to paper, that describes suitable learning procedure for this specific architecture.    
    - After learning model weights saves in folder data/checkpoints/. Use model.load_weights(fname) to load best weights. If you use Windows and Model Checkpoint doesn't work on your configuration, you should implement [your own Callback](https://keras.io/api/callbacks/#create-a-callback) to save best weights in memory and then load it back.

## Installing Face Detectors

To install Face Detectors, follow these steps:

### Linux and macOS:

```
To install Docker container with all necessary software installed, follow
```

[instructions](https://hub.docker.com/r/zimovnov/coursera-aml-docker) After that you should see a Jupyter page in your browser.

### Windows:

```
We highly recommend to install docker environment, but if it's not an option, you can try to install the necessary python modules with Anaconda.
```

* First, install Anaconda with **Python 3.5+** from [here](https://www.anaconda.com/products/individual).
* Download `conda_requirements.txt` from [here](https://github.com/ZEMUSHKA/coursera-aml-docker/blob/master/conda_requirements.txt).
* Open terminal on Mac/Linux or "Anaconda Prompt" in Start Menu on Windows and run:

```
conda config --append channels conda-forge
conda config --append channels menpo
conda install --yes --file conda_requirements.txt
```

To start Jupyter Notebooks run `jupyter notebook` on Mac/Linux or "Jupyter Notebook" in Start Menu on Windows.

After that you should see a Jupyter page in your browser.

## Getting started with using Face Detectors

To use Face Detectors, clone repositories (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)) as follows:

### [Face detection detector](https://github.com/renjmindy/FaceDetectors/tree/master/RegionDetector)

* **Usage**

Clone source and data from [clone Face detection detector](https://github.com/renjmindy/FaceDetectors/tree/master/RegionDetector), open ./[Face_Detection.ipynb](https://github.com/renjmindy/FaceDetectors/blob/master/RegionDetector/Face_Detection.ipynb) and do task. It's easy.
  
* **[Procedures](https://github.com/renjmindy/FaceDetectors/blob/master/RegionDetector/Face_Detection.ipynb)**

  - Prepare data: extract positive and negative samples from data.
  - Classifier training: 
    + build convolutional neural network model by adding layers into model.
    + run fitting and validation accuracy expected to exceed 90%.
    + select epoch with best validation loss and load this epoch weight.
  - FCNN (Fully Connected Neural Network) model: build fcnn model, write `copy_weight` function and visualized `activation heat map`.
  - Detector: write `get_bboxes_and_decision_function` and visualized `predicted bboxes`.
  - Precision/recall curve: implement precision/recall curve and plot it.
  - Threshold: 
    + find point that corresponds to recall=0.85 
    + Precision/recall graph should stop at recall=0.85
  - Detector score: on test dataset detection score (in graph header) should be 0.85 or greater.

* **Files** This [repository](https://github.com/renjmindy/FaceDetectors/tree/master/RegionDetector) consist of multiple files:
  
  - `Face_Detection.ipynb` -- main task, read and do.
  - `get_data.py` -- script to download data for task, run automatically from main task. You don't need download data manually.
  - `scores.py` -- scores, which are using in main task.
  - `graph.py` -- graph plotting and image showing functions.
  - `prepare_data.ipynb` -- prepare data to train and test, you may run this script and repeat learning-test procedure to make sure that model haven't over-fitting.   
  
* **Dataset**

  - Raw Data is being kept [FDDB dataset](http://vis-www.cs.umass.edu/fddb/).
  - Data pre-processing/transformation scripts are being kept, defined in ./[prepare_data.ipynb](https://github.com/renjmindy/FaceDetectors/blob/master/RegionDetector/prepare_data.ipynb) and explained in ./[Face_Detection.ipynb](https://github.com/renjmindy/FaceDetectors/blob/master/RegionDetector/Face_Detection.ipynb)

### [Face Recognition detector](https://github.com/renjmindy/FaceDetectors/tree/master/TrackDetector)

* **Usage**

Clone source and data from [clone Face recognition detector](https://github.com/renjmindy/FaceDetectors/tree/master/TrackDetector), open ./[Face_Recognition.ipynb](https://github.com/renjmindy/FaceDetectors/blob/master/TrackDetector/Face_Recognition_task.ipynb) and do task. It's easy.

* **[Procedures](https://github.com/renjmindy/FaceDetectors/blob/master/TrackDetector/Face_Recognition_task.ipynb)**

  - Prepare data: [unpack](https://github.com/renjmindy/FaceDetectors/blob/master/TrackDetector/get_data.py) Face_Recognition_data.zip to extract [Face Recognition Images](https://github.com/renjmindy/FaceDetectors/tree/master/TrackDetector/Face_Recognition_data). implement `load_image_data` and `load_video_data` funtions. visualize training and testing samples, respectively.
  - Prepare training: implement `preprocess_imgs` function to detect face, find facial keypoints, and then crop and normalize the image according to these keypoints. 
  - Classifier training: 
    + load pre-trained model in which we can select the last hidden layer as feature extractor to get descriptors of the faces.
    + build `kNN Classifier` class that include `initialization`, `fit`, `classify_image` and `classify_video`. 
    + run fitting and validation accuracy expected to exceed 90%.
 
* **Files** This [repository](https://github.com/renjmindy/FaceDetectors/tree/master/TrackDetector) consist of multiple files:
  
  - `Face_Reconition_task.ipynb` -- main task, read and do.
  - `get_data.py` -- script to download data for task, run automatically from main task. You don't need download data manually.
  - `Face_Recognition_data` -- image raw data used for image and video classification as well

## Contributing to Face Detectors
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to Face Detectors, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin FaceDetectors/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

### Methods Used

* Object Key-point Regression
* Semantic Segmentation
* Object Detection/Classification
* Multi-layered Perceptrons (MLPs)
* Convolution Neural Networks (CNN), e.g. VGG16, Inception-V3, ResNet, MobileNet, etc
* Transfer Learning
* Gradient Descent
* Backpropagation
* Overfitting
* Probability

### Technologies

* Traditional Image Processing Techniques, e.g. use of OpenCV, skimage
* Python
* Pandas, jupyter
* Keras
* TensorFlow
* Scikit-learn
* Dlib from both Python and C++ 
* Matplotlib 
* NumPy

## Project Overview

Face detectors are used for `calibration`, `classification`, `detection`, `recognition`, `tracking` and `generation` that allow `classification` being converted into `detection`, `recognition`, `tracking` and `generation`.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_1.png)

One of the central problems in `computer vision` is the object detection task. The goal of object detection is to detect the presence of object from a certain set of classes, and locate the exact position in the image. We can informally divide all objects into two big groups: `things` and `stuff`. Things are objects of certain size and shape like cars, bicycles, people, animals, planes. We can specify where object is located in image with a bounding box. Stuff is more likely a region of image which correspond to objects like road, or grass, or sky, or water. It is easier to specify the location of a sky by marking the region in an image, not by a bounding box. Unlike the detection of things, to detect a stuff, it is better to use `semantic image segmentation` methods. Compared to image classification, the output of the detector is each structured object that is usually marked with a bounding box and class label. Object position and class are annotated in ground truth data. To check whether the detection is correct, we compare the predicted bounding box with ground truth one. The metric is intersection over union (aka: **IoU**). It is the ratio of area of intersection of predicted in ground truth bounding boxes to the area of the union on these boxes as shown on the slide. Either IoU is larger than the threshold, then the detection is correct. The larger the threshold, the more precisely detector should localize objects.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_2.png)

## Needs of Face Detectors

### [Face detection detector](https://github.com/renjmindy/FaceDetectors/tree/master/RegionDetector)

* data exploration/descriptive statistics

```
print(len(train_images),len(train_bboxes),len(train_shapes))
print(X_train.shape,Y_train.shape)
918 1051 918
(1916, 32, 32, 3) (1916, 2)
```
```
print(len(val_images),len(val_bboxes),len(val_shapes.shape))
print(X_val.shape,Y_val.shape)
306 357 2
(652, 32, 32, 3) (652, 2)
```

* data processing/cleaning

  - label == 1 (w/ face)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_5.png)
  - label == 0 (w/o face)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_6.png)

* statistical modeling

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_7.png)

```
Model: "functional_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 32, 32, 3)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 32, 32, 32)        896       
_________________________________________________________________
re_lu (ReLU)                 (None, 32, 32, 32)        0         
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 16, 16, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 16, 16, 64)        18496     
_________________________________________________________________
re_lu_1 (ReLU)               (None, 16, 16, 64)        0         
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 8, 8, 64)          0         
_________________________________________________________________
flatten (Flatten)            (None, 4096)              0         
_________________________________________________________________
dense (Dense)                (None, 128)               524416    
_________________________________________________________________
re_lu_2 (ReLU)               (None, 128)               0         
_________________________________________________________________
dropout (Dropout)            (None, 128)               0         
_________________________________________________________________
dense_1 (Dense)              (None, 64)                8256      
_________________________________________________________________
re_lu_3 (ReLU)               (None, 64)                0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 64)                0         
_________________________________________________________________
dense_2 (Dense)              (None, 2)                 130       
_________________________________________________________________
softmax (Softmax)            (None, 2)                 0         
=================================================================
Total params: 552,194
Trainable params: 552,194
Non-trainable params: 0
_________________________________________________________________
```

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_8.png)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_9.png)

```
results_train = model.evaluate(X_train, Y_train)
60/60 [==============================] - 2s 27ms/step - loss: 0.0504 - accuracy: 0.9849
results_val = model.evaluate(X_val, Y_val)
21/21 [==============================] - 1s 29ms/step - loss: 0.0570 - accuracy: 0.9785
```

* Transfer Learning

If one classification architecture with high validation score has been prepared, we can use this architecture for detection. Convert classification architecture to fully convolution neural network (FCNN), that returns heatmap of activation. **Now we should replace fully-connected layers with 1×1 convolution layers.** In brief:

1. 1×1 convolution layer is equivalent of fully connected layer.
2. 1×1 convolution layers can be used to get activation map of classification network in `sliding window` manner.

We propose replace last fully connected layer with `softmax` actiovation to convolution layer with `linear` activation. It will be usefull to find good threshold. Of course, we can use `softmax` activation that follows by fully convolution neural network.

##### Head before convert
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_10.png)

##### Head after convert
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_11.png)

```
Model: "functional_3"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_2 (InputLayer)         [(None, 176, 176, 3)]     0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 176, 176, 32)      896       
_________________________________________________________________
re_lu_4 (ReLU)               (None, 176, 176, 32)      0         
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 88, 88, 32)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 88, 88, 64)        18496     
_________________________________________________________________
re_lu_5 (ReLU)               (None, 88, 88, 64)        0         
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 44, 44, 64)        0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 37, 37, 128)       524416    
_________________________________________________________________
re_lu_6 (ReLU)               (None, 37, 37, 128)       0         
_________________________________________________________________
dropout_2 (Dropout)          (None, 37, 37, 128)       0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 37, 37, 64)        8256      
_________________________________________________________________
re_lu_7 (ReLU)               (None, 37, 37, 64)        0         
_________________________________________________________________
dropout_3 (Dropout)          (None, 37, 37, 64)        0         
_________________________________________________________________
conv2d_6 (Conv2D)            (None, 37, 37, 2)         130       
=================================================================
Total params: 552,194
Trainable params: 552,194
Non-trainable params: 0
_________________________________________________________________
```

Require weight being transferred from fully connected layers to fully convolution layers: Then we should write function that copy weights from classification model to fully convolution model. Convolution weights may be obtained from copy without modification, fully-connected layer weights should be reshaped before being copied.

* writeup/reporting

* detector score

  - Training samples:

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_12.png)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_3.png)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_4.png)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_13.png)

  - Testing samples:

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_14.png)
![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_15.png)

## Featured Notebooks/Analysis/Deliverables

* [Blog Post](link)

## Contact

If you want to contact me you can reach me at <jencmhep@gmail.com>.
