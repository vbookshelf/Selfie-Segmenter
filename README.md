## V_1.0 - Selfie-Segmenter
Ai powered web app that takes as input the half-length portrait of a person and outputs a segmented image showing the human separated from the background - Tensorflow.js

Live Web App: http://selfieseg.test.woza.work/


This is a prototype for an online tool that can take as input a portrait image of a person and output a segmented image that has separated the person from the background. The quality of the output segmentation is not very good. This is because the model was trained using only 4000 of the 24,479 images in the dataset. My main reason for building this app was to understand the workflow involved in building an end-to-end web based solution that ouputs an image. The same workflow can also be used for other fun machine learning applications like style transfer. Or on a more serious note, imagine having a freely available app like this that could take as input the photo of a missing child and output an aged photo.

The process used to build and train the model is described in this Kaggle kernel:<br>
https://www.kaggle.com/vbookshelf/selfie-segmentation-with-keras-and-u-net

The dataset used for training can be found here:<br>
https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets

All javascript, html and css files used to create the web app are available in this repo.
