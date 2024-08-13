## Mining Mine - A Computer Vision Approach  
---
#### 1. Overview  
+ In addition to inferring mining activities by analyzing tabular data, we are also curious about whether the image data can be useful for making more accurate decisions. With this motivation, we plan to apply **Computer Vision** to the mining inference process.  
+ Here are the steps:  
1. Raw Image Data Collection from [Google Earth Pro](https://earth.google.com/web).  **Done✅**  
2. Data Labeling  **Done✅***  
3. Model Exploration.  **TODO⛳️**  
4. Fine Tuning.  **TODO⛳️**  
5. Evaluation.  **TODO⛳️**    
---

#### 2. Raw Images Collection

+ Code Base: [./data_collection](./data_collection)

+ How to run collector?

  + Please make sure you have set the coordinates in `google_earth_collector.py` according to your computer before running the code.

  ```bash
  cd data_collection
  
  python3 google_earth_collector.py
  ```

+ The collected images are stored in [./raw_image_dataset](https://drive.google.com/drive/folders/1brilaUXeCxpNdUa3mDqOWD7ymUnd3M3q)  
---

#### 3. Data Labeling  
  
+ Code Base: [./labeling_website](https://github.com/QinPR/Mining_Website)  
  
+ The labeled images dataset are stored [here](https://drive.google.com/drive/folders/1iUSdqqA9NaACCcjkW1Bhvv48YYSnHwoa)  
---
  
#### 4. Model Exploration  
  
+ Code Baseline: [./model](./model)  

+ We are exploring the fowlloing models:  
  1. CNN  


