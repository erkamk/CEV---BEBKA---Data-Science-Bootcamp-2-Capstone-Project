import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np

data = pd.read_csv("file path here")
df = data.AGE.value_counts()
df = df.to_frame().reset_index()
df.rename(columns={"index": "Age", "AGE": "Frequency"}, inplace = True)
df = df[df['Age'] >= 18]
df = df.sort_values('Age')
abc = data.AGE.value_counts('Age')
plt.bar(df['Age'], df['Frequency'], color ='maroon',
        width = 0.4) 
#%%
"""
eye weights
"""
from tensorflow.keras.models import model_from_json

json_file = open(r"file path here", 'r')
loaded_model_json1 = json_file.read()
json_file.close()
eye_VGG16_tl = model_from_json(loaded_model_json1)
# load weights into new model
eye_VGG16_tl.load_weights(r"file path here")
eye_VGG16_tl.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#%%
"""
emotion weights
"""

json_file = open(r"file path here", 'r')
loaded_model_json2 = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json2)
# load weights into new model
emotion_model.load_weights(r"file path here")
emotion_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#%%
"""
age weights
"""
json_file = open(r"file path here", 'r')
loaded_model_json3 = json_file.read()
json_file.close()
age_model = model_from_json(loaded_model_json3)
# load weights into new model
age_model.load_weights(r"file path here")
age_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#%%
def pre_procces_for_eye(frame):
    image = cv2.resize(frame, (150,150)) 
    image = np.array(image).reshape(-1,150,150,3)
    image = image.astype("float32")
    image/= 255.0
    return image
#%%
def pre_procces_for_age(frame):
    image = cv2.resize(frame, (180,180)) 
    image = np.array(image).reshape(-1,180,180,3)
    image = image.astype("float32")
    image/= 255.0
    return image

#%%
def pre_procces_for_emotion(frame):
    image = cv2.resize(frame, (48,48)) 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = np.array(image).reshape(-1,48,48,1)
    image = image.astype("float32")
    image/= 255.0
    return image


#%%
import torch


PATH = r"yolov5\best3.pt"
model_face_yolo = torch.hub.load('ultralytics/yolov5', 'custom', PATH)
cap = cv2.VideoCapture(r"file path here")
eye_dictionary = {0:"Open", 1:"Close"}
age_dictionary = {0:["18-24",0.3], 1: ["25-45",0.2], 2: ["45+",0.1]}
emotion_dictionary = {0: ["Negative",0.3], 1:["Positive",0.2]}

width= int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height= int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

writer= cv2.VideoWriter(r"Desktop\demo.mp4", cv2.VideoWriter_fourcc(*'DIVX'), 20, (width,height))

i=0
temp_li=[]
FPS= int(cap.get(cv2.CAP_PROP_FPS))
drowsy_queue = []
sleeping_queue = []

while(True):
    ret, frame = cap.read() 
    if ret == False: break

    face_result = model_face_yolo(frame)
    try:
        boxes = face_result.xyxy[0].numpy()
        people_x2y2 = np.sum(boxes[:,2:4], axis=1)
        index_of_driver = np.argmax(people_x2y2)
        x0, y0, x1, y1, _, _ = boxes[index_of_driver].astype(int) 
        #x0, y0, x1, y1, _, _ = face_result.xyxy[0][0].numpy().astype(int)         
    except:
        pass
    else:
        cropped_img_yolo = frame[y0:y1, x0:x1]
        img_age = pre_procces_for_age(cropped_img_yolo)
        img_eye = pre_procces_for_eye(cropped_img_yolo)
        img_emotion = pre_procces_for_emotion(cropped_img_yolo)


        predicted_index_eye = np.argmax(eye_VGG16_tl.predict(img_eye))
        predicted_index_age = np.argmax(age_model.predict(img_age))
        predicted_emotion = emotion_model.predict(img_emotion)
        predicted_index_emotion = np.argmax(predicted_emotion)
        predicted_emotion_max = np.max(predicted_emotion)  
        
        crash_prob= 0.0
        crash_prob += age_dictionary[predicted_index_age][1]
        
        if len(sleeping_queue)>= FPS*7:
            sleeping_queue.pop(0)
        
        if len(drowsy_queue)>= FPS*5:
            drowsy_queue.pop(0)
        
        sleeping_queue.append(eye_dictionary[predicted_index_eye])
        drowsy_queue.append(eye_dictionary[predicted_index_eye]) 
        
        


    
        driver_x = int(x0 + ((x1-x0)/2))-50

        cv2.putText(frame, "DRIVER", (driver_x, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 3)
        cv2.putText(frame, f'Eyes          : {eye_dictionary[predicted_index_eye]}', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
        cv2.putText(frame, f'Age Pred.     : {age_dictionary[predicted_index_age][0]}', (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
        #cv2.putText(frame, f'Emotion Pred. : {emotion_dictionary[predicted_index_emotion]}', (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255),thickness = 2)

        if predicted_emotion_max >= 0.8:
            cv2.putText(frame, f'Emotion Pred. : {emotion_dictionary[predicted_index_emotion][0]}', (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
            crash_prob += emotion_dictionary[predicted_index_emotion][1]
        else:
           cv2.putText(frame, "Emotion Pred. : Neutral", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)

        if len(drowsy_queue) >= FPS*2:
            if drowsy_queue.count("Open") < drowsy_queue.count("Closed")*3:
                cv2.putText(frame, "Drowsy   : True", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
                crash_prob += 0.3
            else:
                cv2.putText(frame, "Drowsy   : False", (25, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)
                
        if len(sleeping_queue) >= (FPS*4):
            if drowsy_queue.count("Open") < drowsy_queue.count("Closed") * 3:
                crash_prob = 1.0

        cv2.putText(frame, f'Crash Prob.     : {crash_prob:.2f}', (25, 165), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255),thickness = 2)

        
        #cv2.putText(frame, f'{age_dictionary[predicted_index_age]}', (10,10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 500, 500))
        #cv2.putText(frame, f'{eye_dictionary[predicted_index_eye]}', (200, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 500, 500))
        start_point = (x0, y0)
        end_point = (x1, y1)
        cv2.rectangle(frame, start_point, end_point, (0,255,0), 2)
        writer.write(frame)
        
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('a'):
        break
  
# Close the window / Release webcam
cap.release()
writer.release()
# De-allocate any associated memory usage 
cv2.destroyAllWindows()


