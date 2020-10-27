# Face Detectors

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here 
![GitHub repo size](https://img.shields.io/github/repo-size/scottydocs/README-template.md)
![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)
![GitHub stars](https://img.shields.io/github/stars/scottydocs/README-template.md?style=social)
![GitHub forks](https://img.shields.io/github/forks/scottydocs/README-template.md?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/scottydocs?style=social) --->

Face detectors are used for `calibration`, `classification`, `detection`, `recognition`, `tracking` and `generation` that allow `classification` being converted into `detection`, `recognition`, `tracking` and `generation`.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_1.png)

One of the central problems in `computer vision` is the object detection task. The goal of object detection is to detect the presence of object from a certain set of classes, and locate the exact position in the image. We can informally divide all objects into two big groups: `things` and `stuff`. Things are objects of certain size and shape like cars, bicycles, people, animals, planes. We can specify where object is located in image with a bounding box. Stuff is more likely a region of image which correspond to objects like road, or grass, or sky, or water. It is easier to specify the location of a sky by marking the region in an image, not by a bounding box. Unlike the detection of things, to detect a stuff, it is better to use `semantic image segmentation` methods. Compared to image classification, the output of the detector is each structured object that is usually marked with a bounding box and class label. Object position and class are annotated in ground truth data. To check whether the detection is correct, we compare the predicted bounding box with ground truth one. The metric is intersection over union (aka: **IoU**). It is the ratio of area of intersection of predicted in ground truth bounding boxes to the area of the union on these boxes as shown on the slide. Either IoU is larger than the threshold, then the detection is correct. The larger the threshold, the more precisely detector should localize objects.

![fd](https://github.com/renjmindy/FaceDetectors/blob/master/images/face_2.png)

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



Add run commands and examples you think users will find useful. Pr

## Contributing to Face Detectors
<!--- If your README is long or you have some specific process or steps you want contributors to follow, consider creating a separate CONTRIBUTING.md file--->
To contribute to <project_name>, follow these steps:

1. Fork this repository.
2. Create a branch: `git checkout -b <branch_name>`.
3. Make your changes and commit them: `git commit -m '<commit_message>'`
4. Push to the original branch: `git push origin <project_name>/<location>`
5. Create the pull request.

Alternatively see the GitHub documentation on [creating a pull request](https://help.github.com/en/github/collaborating-with-issues-and-pull-requests/creating-a-pull-request).

## Contributors

Thanks to the following people who have contributed to this project:

* [@scottydocs](https://github.com/scottydocs) 📖
* [@cainwatson](https://github.com/cainwatson) 🐛
* [@calchuchesta](https://github.com/calchuchesta) 🐛

You might want to consider using something like the [All Contributors](https://github.com/all-contributors/all-contributors) specification and its [emoji key](https://allcontributors.org/docs/en/emoji-key).

## Contact

If you want to contact me you can reach me at <your_email@address.com>.

## License
<!--- If you're not sure which open license to use see https://choosealicense.com/--->

This project uses the following license: [<license_name>](<link>).
