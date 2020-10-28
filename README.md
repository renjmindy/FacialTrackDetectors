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

* **Patient Check-in and Check-Out Procedures:**

...Patient identification solutions have recently gained momentum. They simplify the whole patient check-in process and free hospital personnel from paperwork.

* **Diagnosing Diseases and Conditions Using Face Recognition:**

...Face recognition has hovered almost all healthcare domains and the diagnostic process is no exсeption. Healthcare evangelists and advisors claim that in the coming years, [health mirrors](https://www.theguardian.com/technology/2012/jan/22/medical-mirror-ming-zher-poh) will be in high gear. 

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/facial-recognition-for-healthcare-disruption-5.jpg)

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

## Using Face Detectors

To use Face Detectors, follow these steps:

### [Face detection detector](https://github.com/renjmindy/FaceDetectors/tree/master/RegionDetector)

* **Usage**
  
* **Procedures**
  
* **Files**
  
* **Dataset**

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
* Deep Learning
* Data Visualization
* Predictive Modeling
* etc.

### Technologies
* Python
* Pandas, jupyter
* etc. 

## Project Overview

Face detectors are used for `calibration`, `classification`, `detection`, `recognition`, `tracking` and `generation` that allow `classification` being converted into `detection`, `recognition`, `tracking` and `generation`.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_1.png)

One of the central problems in `computer vision` is the object detection task. The goal of object detection is to detect the presence of object from a certain set of classes, and locate the exact position in the image. We can informally divide all objects into two big groups: `things` and `stuff`. Things are objects of certain size and shape like cars, bicycles, people, animals, planes. We can specify where object is located in image with a bounding box. Stuff is more likely a region of image which correspond to objects like road, or grass, or sky, or water. It is easier to specify the location of a sky by marking the region in an image, not by a bounding box. Unlike the detection of things, to detect a stuff, it is better to use `semantic image segmentation` methods. Compared to image classification, the output of the detector is each structured object that is usually marked with a bounding box and class label. Object position and class are annotated in ground truth data. To check whether the detection is correct, we compare the predicted bounding box with ground truth one. The metric is intersection over union (aka: **IoU**). It is the ratio of area of intersection of predicted in ground truth bounding boxes to the area of the union on these boxes as shown on the slide. Either IoU is larger than the threshold, then the detection is correct. The larger the threshold, the more precisely detector should localize objects.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_2.png)

## Needs of this project

- frontend developers
- data exploration/descriptive statistics
- data processing/cleaning
- statistical modeling
- writeup/reporting
- etc. (be as specific as possible)

## Getting Started

1. Clone this repo (for help see this [tutorial](https://help.github.com/articles/cloning-a-repository/)).
2. Raw Data is being kept [here](Repo folder containing raw data) within this repo.

    *If using offline data mention that and how they may obtain the data from the froup)*
    
3. Data processing/transformation scripts are being kept [here](Repo folder containing data processing scripts/notebooks)
4. etc...

*If your project is well underway and setup is fairly complicated (ie. requires installation of many packages) create another "setup.md" file and link to it here*  
5. Follow setup [instructions](Link to file)

## Featured Notebooks/Analysis/Deliverables

* [Notebook/Markdown/Slide Deck Title](link)
* [Notebook/Markdown/Slide DeckTitle](link)
* [Blog Post](link)

## Contact

If you want to contact me you can reach me at <jencmhep@gmail.com>.
