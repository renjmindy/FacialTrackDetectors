# Face (Object) Detectors

<!--- These are examples. See https://shields.io for others or to customize this set of shields. You might want to include dependencies, project status and licence info here 
![GitHub repo size](https://img.shields.io/github/repo-size/scottydocs/README-template.md)
![GitHub contributors](https://img.shields.io/github/contributors/scottydocs/README-template.md)
![GitHub stars](https://img.shields.io/github/stars/scottydocs/README-template.md?style=social)
![GitHub forks](https://img.shields.io/github/forks/scottydocs/README-template.md?style=social)
![Twitter Follow](https://img.shields.io/twitter/follow/scottydocs?style=social) --->

Face detectors are used for `calibration`, `classification`, `detection`, `recognition`, `tracking` and `generation` that allow `classification` being converted into `detection`, `recognition` and `tracking`.

One of the central problems in `computer vision` is the object detection task. The goal of object detection is to detect the presence of object from a certain set of classes, and locate the exact position in the image. We can informally divide all objects into two big groups: `things` and `stuff`. Things are objects of certain size and shape like cars, bicycles, people, animals, planes. We can specify where object is located in image with a bounding box. Stuff is more likely a region of image which correspond to objects like road, or grass, or sky, or water. It is easier to specify the location of a sky by marking the region in an image, not by a bounding box. Unlike the detection of things, to detect a stuff, it is better to use semantic image segmentation methods. Compared to image classification, the output of the detector is each structured object that is usually marked with a bounding box and class label. Object position and class are annotated in ground truth data. To check whether the detection is correct, we compare the predicted bounding box with ground truth one. The metric is intersection over union (aka: **IoU**). It is the ratio of area of intersection of predicted in ground truth bounding boxes to the area of the union on these boxes as shown on the slide. Either IoU is larger than the threshold, then the detection is correct. The larger the threshold, the more precisely detector should localize objects.

## Prerequisites

Before you begin, ensure you have met the following requirements:
<!--- These are just example requirements. Add, duplicate or remove as required --->
* You have installed the latest version of `<coding_language/dependency/requirement_1>`
* You have a `<Windows/Linux/Mac>` machine. State which OS is supported/which is not.
* You have read `<guide/link/documentation_related_to_project>`.

## Installing <project_name>

To install <project_name>, follow these steps:

Linux and macOS:
```
<install_command>
```

Windows:
```
<install_command>
```
## Using <project_name>

To use <project_name>, follow these steps:

```
<usage_example>
```

Add run commands and examples you think users will find useful. Provide an options reference for bonus points!

## Contributing to <project_name>
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
