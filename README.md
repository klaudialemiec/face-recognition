# Face recognition based on image.

The aim of project is to recognize if presented face is known or not.
 
To achive the goal we realized the following steps:
1. **Download labelled dataset "Labeled Faces in the Wild"** (http://vis-www.cs.umass.edu/lfw/), that cointains raw, unprocessed people photos (the photos can have low quality or diffrent sizes, be ill-lit or rotated). The dataset was extended by a few own photos,
2. **Image processing** (i.e. affine transformation, cropping, gamma correction, histogram equalization),
3. Application of **MTCNN model** (https://github.com/ipazc/mtcnn), what allows us to detect faces,
4. Usage of **siamse neural network** to recognize person on photo. The tripled loss function was applied to train the network.

For new input image the face was detected and preprocessed. 

Then it was compared with diffrent images included in dataset. If person was recognized, the app returns name of identified one.

