# Core-module
**Module for car detection**

- Implemented to detect cars in [CARLA simulator](http://carla.org/) (version: 0.8.4)

**How to start car detection as stand alone application:**
- Navigate to [config file](https://github.com/affinis-lab/car-detection-module/blob/master/config.json)
- For training: 
  - Set path to dataset with car images from CARLA (_images_ parameter)
  - Set path to annotations file for car images (_annotations_file_ parameter)
  - Set parameter _training_ to **true**
- For prediction:
  - Set path to folder with car images you want to test application on (_test_images_ parameter) 
  - Set parameter _training_ to **false**
  
- Run main.py

**Prediction samples:**

![Gif 1](https://github.com/affinis-lab/car-detection-module/blob/master/images/car_image1.png)
![Gif 2](https://github.com/affinis-lab/car-detection-module/blob/master/images/car_image2.png)
