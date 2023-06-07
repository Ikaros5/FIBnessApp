import toga
from toga.constants import COLUMN
from toga.style import Pack
from collections import namedtuple
import warnings

#ML imports
import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import time
import math

#DB imports
import requests
import json

warnings.filterwarnings('ignore', category=DeprecationWarning)

#button_background_color = "#5BA62C"
button_background_color = "#2ee662"
button_background_color_special = "#2b8a1e"
title_color = "#13bf44"
button_text_color = "#000000"
button_font_size = 12
button_font_size_big = 16

#Global Variables ML
EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

"""
starting_box = toga.Box(
    children=[
        vertical1, 
        vertical2, 
        toga.Box(
            children=[
                horizontal1, 
                horizontal2
            ]
        )
    ], 
    style=Pack(direction=COLUMN, padding=50)
)
"""

class FitnessApp(toga.App):
    def startup(self):
        # Create the main window
        self.main_window = toga.MainWindow(self.name)
       	
        # Global Variables
        self.logged_in_user = ""
        self.authToken = ""
        self.rep_count = 0
        self.exercise = ""
        self.exercise_repetitions = {}
        self.listExercises = ["Right","Left","Military","Lateral"]

        self.show_starting_box()
        self.main_window.show()


#==============================   STARTING   ===============================================================================================================#
    def show_starting_box(self):
        #Create a box for the logo, text, and buttons
        starting_box = toga.Box(style=Pack(direction=COLUMN, padding=50))
        
        logo = toga.Image("icons/UPCLogo.png")
        logo2 = toga.Image("icons/FibnessLogo.png")
        logo_view = toga.ImageView(logo, style=Pack(width=125, height=125))
        logo_view2 = toga.ImageView(logo2, style=Pack(width=135, height=135))

        #Add the app name text to the box
        app_name = toga.Label('     FIBnessApp    ', style=Pack(flex = 1, font_size=46, font_weight='bold', padding_bottom = 20, color=title_color))
        empty_label = toga.Label('      ', style=Pack(font_size=46, font_weight='bold', padding_bottom = 20, color=title_color))
        starting_login_button = toga.Button(
            "Login",
            on_press=self.login_button_handler,
            style=Pack(padding=15, color=button_text_color, background_color=button_background_color, font_size = 12)
        )
        starting_signup_button = toga.Button(
            "Signup",
            on_press=self.signup_button_handler,
            style=Pack(padding=15, color=button_text_color, background_color=button_background_color, font_size = 12)
        )
        starting_exit_button = toga.Button(
            "Exit",
            on_press=self.exit_button_handler,
            style=Pack(padding=15, color=button_text_color, background_color=button_background_color, font_size = 12)
        )

        starting_box = toga.Box(
            children=[
                toga.Box(
                    children=[
                        empty_label,
                        logo_view,
                        logo_view2,
                    ]
                ),                        
                app_name,
                starting_login_button, 
                starting_signup_button, 
                starting_exit_button
            ], 
            style=Pack(direction=COLUMN, padding=30)
        )
        self.main_window.content = starting_box
#================================================================================================================================================================#

#==============================   LOGIN   =======================================================================================================================#
    def login_button_handler(self, widget):
        #print("Login  button press")        
        self.login_username_input = toga.TextInput(style=Pack(flex=1, padding_bottom=15))
        self.login_password_input = toga.PasswordInput(style=Pack(flex=1, padding_bottom=25))
        login_button = toga.Button('Login', on_press=self.login, style=Pack(font_size = button_font_size, padding=10, width=100, color=button_text_color, background_color=button_background_color))
        login_back_button = toga.Button('Back', on_press=self.go_back_starting, style=Pack(font_size = button_font_size, padding=10, width=100, color=button_text_color, background_color=button_background_color))
        
        login_box = toga.Box(
                children=[toga.Label('Username:', style=Pack(padding_top = 25, padding_bottom=3, font_size=14)), 
                        self.login_username_input, 
                        toga.Label('Password:', style=Pack(padding_bottom=3, font_size=14)), 
                        self.login_password_input, 
                        toga.Box(children=[login_back_button, login_button])],
                style=Pack(direction=COLUMN, padding=50)
        )
        self.main_window.content = login_box
    
    def login(self, widget):
        data = {"username":self.login_username_input.value, "password":self.login_password_input.value}
        json_data = json.dumps(data)
        headers = {'Content-type':'application/json', 'Accept':'text/plain'}        
        url = "https://fitapp.garlicbread.fun/auth/loginUser"
        response = requests.post(url, data=json_data, headers=headers)
        
        if response.status_code == 200:      
            self.logged_in_user = self.login_username_input.value
            self.authToken = response.text
            self.authToken = self.authToken[16:-2]
            print(self.authToken)
            self.show_main_box()
        else:
            print(f"Error {response.status_code} with message {response.text}")
            self.main_window.info_dialog("Login Failed", "Incorrect Username or Password")
    
    def go_back_starting(self, widget):
        # Go back to starting window and show it
        self.logged_in_user = ""
        self.authToken = ""
        self.show_starting_box()
#================================================================================================================================================================#

#==============================   SIGNUP   ======================================================================================================================#
    def signup_button_handler(self, widget):
        #print("Signup  button press")
        self.signup_username_input = toga.TextInput(style=Pack(flex=1, padding_bottom=15))
        self.signup_email_input = toga.TextInput(style=Pack(flex=1, padding_bottom=15))
        self.signup_password_input = toga.PasswordInput(style=Pack(flex=1, padding_bottom=25))
        signup_button = toga.Button('Signup', on_press=self.signup, style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size))
        signup_back_button = toga.Button('Back', on_press=self.go_back_starting, style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size))
        
        signup_box = toga.Box(
                children=[toga.Label('Username:', style=Pack(padding_top = 25, padding_bottom=3, font_size=14)), 
                        self.signup_username_input, 
                        toga.Label('Email:', style=Pack(padding_bottom=3, font_size=14)), 
                        self.signup_email_input, 
                        toga.Label('Password:', style=Pack(padding_bottom=3, font_size=14)), 
                        self.signup_password_input, 
                        toga.Box(children=[signup_back_button, signup_button])],
                style=Pack(direction=COLUMN, padding=50)
        )

        self.main_window.content = signup_box
    
    def exit_button_handler(self, widget):
        self.main_window.close()

    def signup(self, widget):
        #UPDATE DATABASE - DONE
        #Signup
        data1 = {"username":self.signup_username_input.value, "password":self.signup_password_input.value, "email":self.signup_email_input.value}
        json_data1 = json.dumps(data1)
        headers1 = {'Content-type':'application/json', 'Accept':'text/plain'}        
        response1 = requests.post("https://fitapp.garlicbread.fun/auth/registerUser", data=json_data1, headers=headers1)
        print("Response1: " + str(response1.status_code))

        if response1.status_code == 200:
            #Login
            data2 = {"username":self.signup_username_input.value, "password":self.signup_password_input.value}
            json_data2 = json.dumps(data2)
            headers2 = {'Content-type':'application/json', 'Accept':'text/plain'}        
            response2 = requests.post("https://fitapp.garlicbread.fun/auth/loginUser", data=json_data2, headers=headers2)
            tmpToken = response2.text
            tmpToken = tmpToken[16:-2]

            #Create Exercises
            for e in self.listExercises:
                data3 = {'excerciseName': e, 'repetitions': 0, 'startTime': '2023-05-13T09:50:37.427018', 'duration': 'PT10M'}
                headers3 = {'Authorization': 'Bearer ' + tmpToken, 'Content-Type': 'application/json'}
                response3 = requests.post("https://fitapp.garlicbread.fun/exercise-registry/updateExercise", headers=headers3, json=data3)        
            
            print("Response2: " + str(response2.status_code))
            print("Response3: " + str(response3.status_code)) 
 
            self.main_window.info_dialog("Resgistration Completed", f"Username: {self.signup_username_input.value}\nEmail: {self.signup_email_input.value}")
            self.show_starting_box()
        else:
            if response1.text == "User with given username already exists":
                self.main_window.info_dialog("Error", response1.text)
            else:
                data = json.loads(response1.text)
                if 'email' in data['error_messages'][0]:
                    #print("##################################################   1   ############################################################")
                    self.main_window.info_dialog("Error", "Invalid email")
                elif 'password' in data['error_messages'][0]:
                    #print("##################################################   2   ############################################################")
                    self.main_window.info_dialog("Error", "Invalid password.\n\n The password must have at least one digit, one lower case and one upper case characters, and be 8 or more characters long")
                else:
                    #print("##################################################   3   ############################################################")
                    self.main_window.info_dialog("Error", response1.text)


                
#================================================================================================================================================================#

#==============================   MAIN BOX   ====================================================================================================================#
    def show_main_box(self):
        #UPDATE DATABASE - DONE?
        #Get Exercises
        response = requests.get("https://fitapp.garlicbread.fun/exercise-registry/getTotalRepetitions", headers={"Authorization":"Bearer " + self.authToken})
        
        data = json.loads(response.text)

        for exercise in data['exercises']:
            self.exercise_repetitions[exercise['excercisename']] = int(exercise['total_repetitions'])
        
        print(self.exercise_repetitions)

        left_content = toga.Table(
            headings=["Exercise", "Total"],
            data=[
                ("Right", self.exercise_repetitions["Right"]),
                ("Left", self.exercise_repetitions["Left"]), 
                ("Military", self.exercise_repetitions["Military"]),
                ("Lateral", self.exercise_repetitions["Lateral"]),
                ("Deadlift", 0),
                ("Pushup", 0),
                ("Pullup", 0),
            ], missing_value="0",
            style=Pack(padding=10, font_size=26, color=title_color)
        )

        right_content = toga.Box(
            children=[
                toga.Box(
                    children=[
                        #ROW
                        toga.Button(
                            "Profile",
                            on_press=self.show_profile,
                            style=Pack(padding=10, color=button_text_color, background_color=button_background_color_special, font_size = button_font_size_big, height=60, width=138)
                        ),
                        toga.Button(
                            "Sign Out",
                            on_press=self.go_back_starting,
                            style=Pack(padding=10, color=button_text_color, background_color=button_background_color_special, font_size = button_font_size_big, height=60, width=138)
                        )
                    ]
                ),
                #COLUMN
                toga.Button(
                    "Right",
                    on_press=self.right_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Left",
                    on_press=self.left_button_handler,
                    style=Pack(padding=10, color=button_text_color,background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Military",
                    on_press=self.military_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Lateral",
                    on_press=self.lateral_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Deadlift",
                    on_press=self.undone_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Pushups",
                    on_press=self.undone_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                ),
                toga.Button(
                    "Pullups",
                    on_press=self.undone_button_handler,
                    style=Pack(padding=10, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
                )
            ], 
            style=Pack(direction=COLUMN)
        )

        left_container = toga.ScrollContainer()
        left_container.content = left_content
        
        right_container = toga.ScrollContainer()
        right_container.content = right_content

        main_box = toga.SplitContainer()
        main_box.content = [left_container, right_container]
        self.main_window.content = main_box

    def show_profile(self, widget):
        profile_username_label = toga.Label("Username: "+self.logged_in_user, style=Pack(flex = 1, font_size=button_font_size_big, font_weight='bold', padding_bottom = 30, color=button_text_color))
        profile_email_label = toga.Label("Email: "+self.logged_in_user+"@gmail.com", style=Pack(flex = 1, font_size=button_font_size_big, font_weight='bold', padding_bottom = 30, color=button_text_color))
        profile_back_button = toga.Button(
            'Back', 
            on_press=self.show_main_box_handler, 
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        change_password_button = toga.Button(
            "Change Passowrd",
            on_press=self.undone_button_handler,
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        profile_box = toga.Box(children=[profile_username_label, profile_email_label, toga.Box(children=[profile_back_button, change_password_button])], style=Pack(direction=COLUMN, padding=50))
        self.main_window.content = profile_box

    def show_main_box_handler(self, widget):
        self.show_main_box()
#================================================================================================================================================================#

#==============================   EXERCISE   ====================================================================================================================#
    def undone_button_handler(self, widget):
        self.main_window.info_dialog("Error", f"Sorry {self.logged_in_user}, this features is under development")

    def pushup_button_handler(self, widget):
        #print("Pushup button press")
        self.exercise = "Pushup"
        self.show_exercise_box() 

    def pullup_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Pullup"
        self.show_exercise_box()   

    def right_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Right"
        self.show_exercise_box()   

    def left_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Left"
        self.show_exercise_box()   

    def deadlift_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Deadlift"
        self.show_exercise_box()   

    def military_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Military"
        self.show_exercise_box()  

    def lateral_button_handler(self, widget):
        #print("Pullup  button press")
        self.exercise = "Lateral"
        self.show_exercise_box()     

    def show_exercise_box(self):
        title_name = toga.Label(self.exercise, style=Pack(flex = 1, font_size=36, font_weight='bold', padding_bottom = 30, color=title_color))
        self.exercise_label = toga.Label(self.rep_count, style=Pack(font_size=100, color=title_color, padding_bottom = 50)) 
        rep_increase_button = toga.Button(
            "Increase",
            on_press=self.rep_increase_handler,
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        rep_decrease_button = toga.Button(
            "Decrease",
            on_press=self.rep_decrease_handler,
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        exercise_back_button = toga.Button(
            'Back', 
            on_press=self.go_back_main, 
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        exercise_ML_button = toga.Button(
            'Camera', 
            on_press=self.call_ML_header, 
            style=Pack(padding=10, width=100, color=button_text_color, background_color=button_background_color, font_size = button_font_size)
        )
        exercise_box = toga.Box(children=[title_name, self.exercise_label, toga.Box(children=[exercise_back_button, rep_increase_button, rep_decrease_button, exercise_ML_button])], style=Pack(direction=COLUMN, padding=50))
        self.main_window.content = exercise_box    
    
    def rep_increase_handler(self, widget):
        self.rep_count += 1
        self.exercise_label.text = str(self.rep_count)

    def rep_decrease_handler(self, widget):
        self.rep_count -= 1
        if self.rep_count < 0:
            self.rep_count = 0
        self.exercise_label.text = str(self.rep_count)

    def go_back_main(self, widget):
        #Go back to main window and show it
        #UPDATE DATABASE - DONE
        data = {'excerciseName': self.exercise, 'repetitions': int(self.rep_count), 'startTime': '2023-05-13T09:50:37.427018', 'duration': 'PT10M'}
        headers = {'Authorization': 'Bearer ' + self.authToken, 'Content-Type': 'application/json'}
        response = requests.post("https://fitapp.garlicbread.fun/exercise-registry/updateExercise", headers=headers, json=data)
        print("\nStatus code: " + str(response.status_code) + ", response text: " + response.text)
        print(f"Updated {self.exercise} with {self.rep_count} repetitions")

        self.rep_count = 0
        self.show_main_box()
        
#================================================================================================================================================================#

#==============================   ML   ==========================================================================================================================#
    #ML Functions
    def draw_connections(self,frame, keypoints, edges, confidence_threshold):
        y, x, c = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for edge, color in edges.items():
            p1, p2 = edge
            y1, x1, c1 = shaped[p1]
            y2, x2, c2 = shaped[p2]
            
            if (c1 > confidence_threshold) & (c2 > confidence_threshold):      
                cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 2)

    def draw_keypoints(self,frame, keypoints, confidence_threshold):
        y, x, z = frame.shape
        shaped = np.squeeze(np.multiply(keypoints, [y,x,1]))
        
        for kp in shaped:
            ky, kx, kp_conf = kp
            if kp_conf > confidence_threshold:
                cv2.circle(frame, (int(kx), int(ky)), 4, (0,255,0), -1)

    def calculate_angle(self,a,b,c):
        a = np.array(a) # First
        b = np.array(b) # Mid
        c = np.array(c) # End
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle >180.0:
            angle = 360-angle
            
        return angle
    
    def draw_angle(self,img, p1, p2, p3):
        if (p1[2] > 0.3 and p2[2] > 0.3 and p3[2] > 0.3):
            cv2.line(img, (int(p1[1]), int(p1[0])), (int(p2[1]), int(p2[0])), (0, 255, 0), 3)
            cv2.line(img, (int(p3[1]), int(p3[0])), (int(p2[1]), int(p2[0])), (0, 255, 0), 3)
            cv2.circle(img, (int(p1[1]), int(p1[0])), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(p1[1]), int(p1[0])), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(p2[1]), int(p2[0])), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(p2[1]), int(p2[0])), 15, (0, 0, 255), 2)
            cv2.circle(img, (int(p3[1]), int(p3[0])), 10, (0, 0, 255), cv2.FILLED)
            cv2.circle(img, (int(p3[1]), int(p3[0])), 15, (0, 0, 255), 2)

    def call_ML_header(self, widget):
        self.call_ML()

    def call_ML(self):
        # Initialize the TFLite interpreter
        interpreter = tf.lite.Interpreter(model_path="C:/Users/isaac/beeware/FIBnessApp/src/FIBnessApp/lite-model_movenet_singlepose_lightning_3.tflite")
        interpreter.allocate_tensors()

        cap = cv2.VideoCapture(0) #Nos permite conectar nuestra webcam

        #Creamos el contador de repeticiones
        dir = 0
        show = True

        while cap.isOpened(): #Mientras la camara este activa
            ret, frame = cap.read() #Lee un frame de la webcam

            # Reshape image 
            img = frame.copy()
            img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
            input_image = tf.cast(img, dtype = tf.float32)

            # Setup input and output
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()

            # Make predictions
            interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
            interpreter.invoke()
            keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
            
            # Render section
            if (show):
                self.draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
                self.draw_keypoints(frame, keypoints_with_scores, 0.4)
            
            if self.exercise == "Left":
                #Obtenemos las coordenadas de los puntos
                left_shoulder = np.array(keypoints_with_scores[0][0][5]*[480,640,1])
                left_elbow = np.array(keypoints_with_scores[0][0][7]*[480,640,1])
                left_wrist = np.array(keypoints_with_scores[0][0][9]*[480,640,1])

                self.draw_angle(frame, left_shoulder, left_elbow, left_wrist)
                
                #Calculamos el angulo
                left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)

                cv2.putText(frame, str(int(left_angle)), (int(left_elbow[1]) -20, int(left_elbow[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                perl = np.interp(left_angle, (0, 200), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)

                if perl >= 75:
                    if dir == 0: #if going up
                        self.rep_count += 0.5
                        dir = 1
                if perl <= 25:
                    if dir == 1: #if going down after up
                        self.rep_count += 0.5
                        dir = 0

            if self.exercise == "Right":
                #Obtenemos las coordenadas de los puntos
                right_shoulder = np.array(keypoints_with_scores[0][0][6]*[480,640,1])
                right_elbow = np.array(keypoints_with_scores[0][0][8]*[480,640,1])
                right_wrist = np.array(keypoints_with_scores[0][0][10]*[480,640,1])

                self.draw_angle(frame, right_shoulder, right_elbow, right_wrist)
                
                #Calculamos el angulo
                right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(frame, str(int(right_angle)), (int(right_elbow[1]) -20, int(right_elbow[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                perl = np.interp(right_angle, (0, 200), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)

                if perl >= 75:
                    if dir == 0: #if going up
                        self.rep_count += 0.5
                        dir = 1
                if perl <= 25:
                    if dir == 1: #if going down after up
                        self.rep_count += 0.5
                        dir = 0

            if self.exercise == "Lateral":
                #Obtenemos las coordenadas de los puntos
                left_shoulder = np.array(keypoints_with_scores[0][0][5]*[480,640,1])
                left_elbow = np.array(keypoints_with_scores[0][0][7]*[480,640,1])
                left_hip = np.array(keypoints_with_scores[0][0][11]*[480,640,1])
                
                right_shoulder = np.array(keypoints_with_scores[0][0][6]*[480,640,1])
                right_elbow = np.array(keypoints_with_scores[0][0][8]*[480,640,1])
                right_hip = np.array(keypoints_with_scores[0][0][12]*[480,640,1])

                self.draw_angle(frame, left_elbow, left_shoulder, left_hip)
                self.draw_angle(frame, right_elbow, right_shoulder, right_hip)
                
                #Calculamos el angulo
                left_angle = self.calculate_angle(left_hip, left_shoulder, left_elbow)
                right_angle = self.calculate_angle(right_hip, right_shoulder, right_elbow)

                cv2.putText(frame, str(int(left_angle)), (int(left_shoulder[1]) -20, int(left_shoulder[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.putText(frame, str(int(right_angle)), (int(right_shoulder[1]) -20, int(right_shoulder[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                perl = np.interp(left_angle, (20, 100), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)
                perr = np.interp(right_angle, (20, 100), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)
                if dir == 1:
                    cv2.putText(frame, text="Aguanta unos segundos", org=(20, 20), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 0, 255),thickness=3)

                if perl >= 95 and perr >= 95:
                    if dir == 0: #if going up
                        self.rep_count += 0.5
                        dir = 1
                if perl <= 10 and perr <= 10:
                    if dir == 1: #if going down after up
                        self.rep_count += 0.5
                        dir = 0

            if self.exercise == "Military":
                #Obtenemos las coordenadas de los puntos
                left_shoulder = np.array(keypoints_with_scores[0][0][5]*[480,640,1])
                left_elbow = np.array(keypoints_with_scores[0][0][7]*[480,640,1])
                left_wrist = np.array(keypoints_with_scores[0][0][9]*[480,640,1])
                
                right_shoulder = np.array(keypoints_with_scores[0][0][6]*[480,640,1])
                right_elbow = np.array(keypoints_with_scores[0][0][8]*[480,640,1])
                right_wrist = np.array(keypoints_with_scores[0][0][10]*[480,640,1])

                self.draw_angle(frame, left_shoulder, left_elbow, left_wrist)
                self.draw_angle(frame, right_shoulder, right_elbow, right_wrist)
                
                #Calculamos el angulo
                left_angle = self.calculate_angle(left_shoulder, left_elbow, left_wrist)
                right_angle = self.calculate_angle(right_shoulder, right_elbow, right_wrist)

                cv2.putText(frame, str(int(left_angle)), (int(left_elbow[1]) -20, int(left_elbow[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                cv2.putText(frame, str(int(right_angle)), (int(right_elbow[1]) -20, int(right_elbow[0])+50), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

                perl = np.interp(left_angle, (20, 130), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)
                perr = np.interp(right_angle, (20, 130), (0, 100)) #Transformamos el rango de angulo 0-200 a % (0-100)

                if perl >= 95 and perr >= 95:
                    if dir == 0: #if going up
                        self.rep_count += 0.5
                        dir = 1
                if perl <= 20 and perr <= 20:
                    if dir == 1: #if going down after up
                        self.rep_count += 0.5
                        dir = 0

            self.exercise_label.text = str(self.rep_count)
            cv2.imshow('MoveNet Lightning', frame)

            if cv2.waitKey(10) & 0xFF==ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
#================================================================================================================================================================#

def main():
    return FitnessApp("FIBness", "Web")
