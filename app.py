# test.py

import os
import tensorflow as tf
import numpy as np
import cv2

####################
import urllib.request
from flask import Flask, flash, request, redirect, render_template
from werkzeug import secure_filename

UPLOAD_FOLDER = '/uploads'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__)
####################

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
##########################


# module-level variables ##############################################################################################
RETRAINED_LABELS_TXT_FILE_LOC = os.getcwd() + "/" + "retrained_labels.txt"
RETRAINED_GRAPH_PB_FILE_LOC = os.getcwd() + "/" + "retrained_graph.pb"

TEST_IMAGES_DIR = os.getcwd() + "/test_images"

SCALAR_RED = (0.0, 0.0, 255.0)
SCALAR_BLUE = (255.0, 0.0, 0.0)


def predict(file):
    a = 5
    result = ""
    classifications = []
    # for each line in the label file . . .
    for currentLine in tf.io.gfile.GFile(RETRAINED_LABELS_TXT_FILE_LOC):
        # remove the carriage return
        classification = currentLine.rstrip()
        # and append to the list
        classifications.append(classification)
    # end for

    # show the classifications to prove out that we were able to read the label file successfully
    print("classifications = " + str(classifications))

    # load the graph from file
    with tf.gfile.FastGFile(RETRAINED_GRAPH_PB_FILE_LOC, 'rb') as retrainedGraphFile:
        # instantiate a GraphDef object
        graphDef = tf.compat.v1.GraphDef()
        # read in retrained graph into the GraphDef object
        graphDef.ParseFromString(retrainedGraphFile.read())
        # import the graph into the current default Graph, note that we don't need to be concerned with the return value
        _ = tf.import_graph_def(graphDef, name='')
    # end with

    with tf.compat.v1.Session() as sess:
        file = file

        #openCVImage = cv2.imread(file)
        openCVImage = cv2.imdecode(np.fromstring(file, np.uint8), cv2.IMREAD_UNCHANGED)
        # if we were not able to successfully open the image, continue with the next iteration of the for loop
        if openCVImage is None:
            print("unable to open " + " as an OpenCV image")
        # end if

        # get the final tensor from the graph
        finalTensor = sess.graph.get_tensor_by_name('final_result:0')

        # convert the OpenCV image (numpy array) to a TensorFlow image
        tfImage = np.array(openCVImage)[:, :, 0:3]

        # run the network to get the predictions
        predictions = sess.run(finalTensor, {'DecodeJpeg:0': tfImage})

        # sort predictions from most confidence to least confidence
        sortedPredictions = predictions[0].argsort()[-len(predictions[0]):][::-1]

        print(sortedPredictions)
        a = sortedPredictions

        onMostLikelyPrediction = True
            # for each prediction . . .
        for prediction in sortedPredictions:
            strClassification = classifications[prediction]

            # if the classification (obtained from the directory name) ends with the letter "s", remove the "s" to change from plural to singular
            if strClassification.endswith("s"):
                strClassification = strClassification[:-1]
            # end if

            # get confidence, then get confidence rounded to 2 places after the decimal
            confidence = predictions[0][prediction]

            # if we're on the first (most likely) prediction, state what the object appears to be and show a % confidence to two decimal places
            if onMostLikelyPrediction:
                # get the score as a %
                scoreAsAPercent = confidence * 100.0
                # show the result to std out
                print("the object appears to be a " + strClassification + ", " + "{0:.2f}".format(scoreAsAPercent) + "% confidence")
                # write the result on the image
                # mark that we've show the most likely prediction at this point so the additional information in
                # this if statement does not show again for this image
                onMostLikelyPrediction = False
                result = strClassification + " " + "{0:.2f}".format(scoreAsAPercent)
            # end if

            # for any prediction, show the confidence as a ratio to five decimal places
            #print(strClassification + " (" +  "{0:.5f}".format(confidence) + ")")
        # end for

    # write the graph to file so we can view with TensorBoard
    #tfFileWriter = tf.compat.v1.summary.FileWriter(os.getcwd())
    #tfFileWriter.add_graph(sess.graph)
    #tfFileWriter.close()
    return result
    
@app.route('/uploader', methods=['GET', 'POST'])
def upload():

    f = request.files['file'].read()
    result = predict(f)
    #f.save(secure_filename(os.path.join(app.root_path, "uploads", f.filename)))
    return result
