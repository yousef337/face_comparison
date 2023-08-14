#!/usr/bin/env python3
'''Main faces-compare script'''
import cv2

import numpy as np
import rospy
from cv_bridge import CvBridge


from face_compare.images import get_face
from face_compare.model import facenet_model, img_to_encoding
from face_comparison.srv import faceSimilarity, faceSimilarityResponse
# load model
model = facenet_model(input_shape=(3, 96, 96))


def run(req):
    '''Kicks off script'''
    res = faceSimilarityResponse()
    cvBridge = CvBridge()
    img1 = cv2.cvtColor(cvBridge.imgmsg_to_cv2(req.img1), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cvBridge.imgmsg_to_cv2(req.img2), cv2.COLOR_BGR2RGB)
    # Load images
    try:
        face_one = get_face(img1)
        face_two = get_face(img2)
    except IndexError:
        res.error = False
        res.isSimilar = False
        return res
    
    res.error = False

    # Calculate embedding vectors
    embedding_one = img_to_encoding(face_one, model)
    embedding_two = img_to_encoding(face_two, model)

    dist = np.linalg.norm(embedding_one - embedding_two)

    if dist > 0.7:
        res.isSimilar = True
    else:
        res.isSimilar = False

    res.similar = dist
    return res

rospy.init_node("FaceSimilarityService")
rospy.Service('FaceSimilarity', faceSimilarity, run)
rospy.spin()


