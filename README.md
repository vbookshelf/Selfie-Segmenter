## Selfie-Segmenter


Live Web App: http://selfieseg.test.woza.work/

<i>For best results please use the Chrome browser to access the app.<br>
In other browsers the app may freeze.</i>



<br>

<img src="http://selfieseg.test.woza.work/assets/selfieseg.png" width="500"></img>

<img src="http://selfieseg.test.woza.work/assets/seg_man.png" width="500"></img>

<br>

This is a prototype for an online tool that can take as input a portrait image of a person and output a segmented image that has separated the person from the background. The quality of the output segmentation is not very good, as shown above. One reason is because the model was trained using only 4000 of the 24,479 images in the dataset. My main purpose for building this app was to understand the workflow involved in building an end-to-end web based solution that ouputs an image. The same workflow can also be used for other fun machine learning applications like style transfer. Or on a more serious note, imagine having a freely available app like this that could take as input the photo of a missing child and output an aged photo.

The process used to build and train the model is described in this Kaggle kernel:<br>
https://www.kaggle.com/vbookshelf/selfie-segmenter-keras-and-u-net

The dataset used for training can be found here:<br>
https://www.kaggle.com/laurentmih/aisegmentcom-matting-human-datasets

All javascript, html and css files used to create the web app are available in this repo.
