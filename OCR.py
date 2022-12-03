# -*- coding: utf-8 -*-
"""
Created on Wed May 19 18:13:46 2021

@author: ion-m
"""
from easyocr import Reader
import argparse
import cv2

 # strip out non-ASCII text so we can draw the text on the image
def cleanup_text(text):
    return "".join([c if ord(c) < 128 else "" for c in text]).strip()

# break the input languages into a comma separated list
langs = ["ko"]
print("[INFO] OCR'ing with the following languages: {}".format(langs))

image = cv2.imread("014.jpg")
print("[INFO] OCR'ing input image...")
reader = Reader(langs)
results = reader.readtext(image)     # box, text, prob

for (bbox, text, prob) in results:
    print("[INFO] {:.4f}: {}".format(prob, text))
    (tl, tr, br, bl) = bbox

    tl = (int(tl[0]), int(tl[1]))
    tr = (int(tr[0]), int(tr[1]))
    br = (int(br[0]), int(br[1]))
    bl = (int(bl[0]), int(bl[1]))
    print(tl,tr,br,bl)

    text = cleanup_text(text)
    cv2.rectangle(image, tl, br, (0, 255, 0), 2)
    cv2.putText(image, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)