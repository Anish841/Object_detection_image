# import the necessary packages
from django.views.decorators.csrf import csrf_exempt
from django.shortcuts import HttpResponse
from django.http import JsonResponse
from django.shortcuts import render_to_response
from django.template import Context, loader
from django.conf import settings
import numpy as np
import urllib
import json
import cv2
import os
import requests
from PIL import Image
from StringIO import StringIO
from django.views.decorators.csrf import csrf_exempt
from django.http import JsonResponse
import numpy as np
import urllib
import json
import sys
import Image
from django.template import Context
from django.template.loader import get_template
from django.template import Context, Template

BLUE = [255,0,0]        # rectangle color
RED = [0,0,255]         # PR BG
GREEN = [0,255,0]       # PR FG
BLACK = [0,0,0]         # sure BG
WHITE = [255,255,255]   # sure FG

DRAW_BG = {'color' : BLACK, 'val' : 0}
DRAW_FG = {'color' : WHITE, 'val' : 1}
DRAW_PR_FG = {'color' : GREEN, 'val' : 3}
DRAW_PR_BG = {'color' : RED, 'val' : 2}


# setting up flags
rect = (0,0,1,1)
drawing = False         # flag for drawing curves
rectangle = False       # flag for drawing rect
rect_over = False       # flag to check if rect drawn
rect_or_mask = 100      # flag for selecting rect or mask mode
value = DRAW_FG         # drawing initialized to FG
thickness = 3           # brush thickness
image =None
img2=None
#import grabcut

# define the path to the face detector
FACE_DETECTOR_PATH = "{base_path}/cascades/haarcascade_frontalface_default.xml".format(
	base_path=os.path.abspath(os.path.dirname(__file__)))

@csrf_exempt
def detect(request):
	print "detect called"
	global image,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
	# initialize the data dictionary to be returned by the request
	data = {"success": False}
	imgName = request.POST.get("imgName", None)
	
	
	# check to see if this is a post request
	if request.method == "POST":
		# check to see if an image was uploaded
		if request.FILES.get("image", None) is not None:
			# grab the uploaded image
			image = _grab_image(stream=request.FILES["image"])

		# otherwise, assume that a URL was passed in
		else:
			# grab the URL from the request
			url = request.POST.get("url", None)

			# if the URL is None, then return an error
			if url is None:
				data["error"] = "No URL provided."
				return JsonResponse(data)

			# load the image and convert
			image = _grab_image(url=url)

		# convert the image to grayscale, load the face cascade detector,
		# and detect faces in the image
		
		###### START ########
			#os.system("grabcut.py "+"obama.jpg")
			#execfile('grabcut.py')
		#	grabcut.demo()
			
		###### END #########
		
		
		############# Changes by anish###########
# 		image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 		detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
# 		rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
# 			minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
		##### ENd#####################
		
		
		############ grabcut import START##############
	
	def onmouse(event,x,y,flags,param):
	    global image,img2,drawing,value,mask,rectangle,rect,rect_or_mask,ix,iy,rect_over
	
	    # Draw Rectangle
	    #print "on mouse before click"
	   
	    #if event == cv2.EVENT_RBUTTONDOWN:
	    #    rectangle = True
	    #    ix,iy = x,y
	    #	print "R button down"
	    if event == cv2.EVENT_RBUTTONDOWN:
	    	if rectangle == True:
	        	rectangle = False
	        	rect_over = True
	        	cv2.rectangle(image,(ix,iy),(x,y),BLUE,2)
	        	rect = (ix,iy,abs(ix-x),abs(iy-y))
	        	rect_or_mask = 0
	        	print " Now press the key 'n' a few times until no further change \n"
	        else:
	        	rectangle = True
	        	ix,iy = x,y
	    		print "R button down"
	    elif event == cv2.EVENT_MOUSEMOVE:
 	      if rectangle == True:
 	          image = img2.copy()
 	          cv2.rectangle(image,(ix,iy),(x,y),BLUE,2)
 	          rect = (ix,iy,abs(ix-x),abs(iy-y))
 	          rect_or_mask = 0
 	          print "Mouse move rectlangle is true"
	
	    elif event == cv2.EVENT_RBUTTONUP:
	        rectangle = False
	        rect_over = True
	        cv2.rectangle(image,(ix,iy),(x,y),BLUE,2)
	        rect = (ix,iy,abs(ix-x),abs(iy-y))
	        rect_or_mask = 0
	        print " Now press the key 'n' a few times until no further change \n"
	
	    # draw touchup curves
	
	    if event == cv2.EVENT_LBUTTONDOWN:
	        if rect_over == False:
	            print "first draw rectangle \n"
	        else:
	            drawing = True
	            cv2.circle(image,(x,y),thickness,value['color'],-1)
	            cv2.circle(mask,(x,y),thickness,value['val'],-1)
	            print "rect over=true"
	
	    elif event == cv2.EVENT_MOUSEMOVE:
	        if drawing == True:
	            cv2.circle(image,(x,y),thickness,value['color'],-1)
	            cv2.circle(mask,(x,y),thickness,value['val'],-1)
	            print "mouse move drawing is true"
	
	    elif event == cv2.EVENT_LBUTTONUP:
	        if drawing == True:
	            drawing = False
	            cv2.circle(image,(x,y),thickness,value['color'],-1)
	            cv2.circle(mask,(x,y),thickness,value['val'],-1)
	            
	    
	    #print rect
	
	print "before img2 copy" 
	img2 = image.copy()                               # a copy of original image
	mask = np.zeros(image.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
	output = np.zeros(image.shape,np.uint8)           # output image to be shown
	
	# input and output windows
	#cv2.namedWindow('output')
	cv2.namedWindow('input')
	cv2.setMouseCallback('input',onmouse)
	cv2.moveWindow('input',image.shape[1]+10,90)
		
	print " Instructions : \n"
	print " Draw a rectangle around the object using right mouse button \n"
			
	while(1):
	
	    #cv2.imshow('output',output)
	    cv2.imshow('input',image)
	    k = 0xFF & cv2.waitKey(1)
	
	    # key bindings
	    if k == 27:         # esc to exit
	        break
	    elif k == ord('0'): # BG drawing
	        print " mark background regions with left mouse button \n"
	        value = DRAW_BG
	    elif k == ord('1'): # FG drawing
	        print " mark foreground regions with left mouse button \n"
	        value = DRAW_FG
	    elif k == ord('2'): # PR_BG drawing
	        value = DRAW_PR_BG
	    elif k == ord('3'): # PR_FG drawing
	        value = DRAW_PR_FG
	    elif k == ord('s'): # save image
	        bar = np.zeros((image.shape[0],5,3),np.uint8)
	        res = np.hstack((img2,bar,image,bar,output))
	        cv2.imwrite(imgName+'_all.png',res)
	        #outimg = np.hstack(output)
	        cv2.imwrite(imgName+'_rendered.png',output)
	        print " Result saved as image \n"
	    elif k == ord('r'): # reset everything
	        print "resetting \n"
	        rect = (0,0,1,1)
	        drawing = False
	        rectangle = False
	        rect_or_mask = 100
	        rect_over = False
	        value = DRAW_FG
	        image = img2.copy()
	        mask = np.zeros(image.shape[:2],dtype = np.uint8) # mask initialized to PR_BG
	        output = np.zeros(image.shape,np.uint8)           # output image to be shown
	    elif k == ord('n'): # segment the image
	        print """ For finer touchups, mark foreground and background after pressing keys 0-3
	        and again press 'n' \n--------->"""
	        print rect_or_mask
	        if (rect_or_mask == 0):         # grabcut with rect
	            bgdmodel = np.zeros((1,65),np.float64)
	            fgdmodel = np.zeros((1,65),np.float64)
	            print "first n"
	            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_RECT)
	            rect_or_mask = 1
	        elif rect_or_mask == 1:         # grabcut with mask
	            bgdmodel = np.zeros((1,65),np.float64)
	            fgdmodel = np.zeros((1,65),np.float64)
	            print "second n"
	            cv2.grabCut(img2,mask,rect,bgdmodel,fgdmodel,1,cv2.GC_INIT_WITH_MASK)
	    elif k==ord('q'):
	    	break
	    
	    mask2 = np.where((mask==1) + (mask==3),255,0).astype('uint8')
	    output = cv2.bitwise_and(img2,img2,mask=mask2)
		   
	cv2.destroyAllWindows()
		
		#####END###############

		

		# construct a list of bounding boxes from the detection
		#rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]

		# update the data dictionary with the faces detected
	cv2.imwrite(imgName+'_rendered_delete.png',output) 
	
	data.update({"imageurl": imgName+'_rendered_delete.png', "success": True})

	#data.update("dummy response")
	# return a JSON response
	print "returning json response"
	return JsonResponse(data)
#/usr/local/lib/python2.7/dist-packages/Django-1.8.5-py2.7.egg/django/contrib/admin/templates
imgpath="/home/suraj/IIITB/Term3/DJangoWorkplace/"  	
@csrf_exempt
def home(request):
	if(request.GET.get('homebutton')):
		print "Home->Button called"
		imurl=request.GET.get('imurl')
		print imurl
		url = "http://localhost:8000/face_detection/detect/"
		# use our face detection API to find faces in images via image URL
		#image = cv2.imread("obama.jpg")
		payload = {"url":imurl,"imgName":"Test1"}
		print "sending json request"
		r = requests.post(url, data=payload).json()
		imguras= r["imageurl"]
		html= r
		print "------------\nHTML:\n"
		print html
		print "--------------------\nHttpres: \n"
		print HttpResponse(html)
		print "--------------\nReturning\n"
		image_data = open(imgpath+imguras, "rb").read()
		return HttpResponse(image_data, content_type="image/png")
		
	elif(request.GET.get('mybtn')):
		print "Home->mybtn called"
		url = "http://localhost:8000/face_detection/detect/"
		# use our face detection API to find faces in images via image URL
		#image = cv2.imread("obama.jpg")
		payload = {"url":"http://www.pyimagesearch.com/wp-content/uploads/2015/05/obama.jpg","imgName":"Test1"}
		print "sending json request"
		r = requests.post(url, data=payload).json()
		imguras= r["imageurl"]
		html= r
		print "------------\nHTML:\n"
		print html
		print "--------------------\nHttpres: \n"
		print HttpResponse(html)
		print "--------------\nReturning\n"
		image_data = open(imgpath+imguras, "rb").read()
		return HttpResponse(image_data, content_type="image/png")
		#return HttpResponse("<html><body><img src=/"+imguras+" /></body></html>")
	else:
		# initialize the data dictionary to be returned by the request
		print "Home->else called"
		data = {"success": False}
	
		# check to see if this is a post request
		if request.method == "POST":
			# check to see if an image was uploaded
			if request.FILES.get("image", None) is not None:
				# grab the uploaded image
				image = _grab_image(stream=request.FILES["image"])
	
			# otherwise, assume that a URL was passed in
			else:
				# grab the URL from the request
				url = request.POST.get("url", None)
	
				# if the URL is None, then return an error
				if url is None:
					data["error"] = "No URL provided."
					return JsonResponse(data)
	
				# load the image and convert
				image = _grab_image(url=url)
	
			# convert the image to grayscale, load the face cascade detector,
			# and detect faces in the image
			image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			detector = cv2.CascadeClassifier(FACE_DETECTOR_PATH)
			rects = detector.detectMultiScale(image, scaleFactor=1.1, minNeighbors=5,
				minSize=(30, 30), flags=cv2.cv.CV_HAAR_SCALE_IMAGE)
	
			# construct a list of bounding boxes from the detection
			rects = [(int(x), int(y), int(x + w), int(y + h)) for (x, y, w, h) in rects]
	
			# update the data dictionary with the faces detected
			data.update({"num_faces": len(rects), "faces": rects, "success": True})
	
		# return a JSON response
		#return JsonResponse(data)
			
	    	return HttpResponse( get_template("index.html").render())
	    	
	     	#return render_to_response('index.html')
	    	    	
	    	#html = "<!DOCTYPE html><html><head><meta charset=\"UTF-8\"><title>Grab Cut</title></head><body><form action=\"demo_form.asp\">Username: <input type=\"text\" name=\"usrname\"><br><input type=\"submit\" value=\"Submit\"></form></body></html>"
	    	#return HttpResponse(html)
		
def _grab_image(path=None, stream=None, url=None):
	# if the path is not None, then load the image from disk
	if path is not None:
		image = cv2.imread(path)

	# otherwise, the image does not reside on disk
	else:	
		# if the URL is not None, then download the image
		if url is not None:
			resp = urllib.urlopen(url)
			data = resp.read()

		# if the stream is not None, then the image has been uploaded
		elif stream is not None:
			data = stream.read()

		# convert the image to a NumPy array and then read it into
		# OpenCV format
		image = np.asarray(bytearray(data), dtype="uint8")
		image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image