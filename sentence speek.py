import tkinter as tk
import cv2
from PIL import Image, ImageTk
import mediapipe as mp
import numpy as np
from joblib import load
from sklearn.preprocessing import Normalizer
import pyttsx3
import enchant

class CharacterPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Character Prediction App")

        # Create frames
        self.camera_frame = tk.Frame(self.root)
        self.camera_frame.pack(side=tk.LEFT, padx=10, pady=10)
        self.prediction_frame = tk.Frame(self.root)
        self.prediction_frame.pack(side=tk.TOP, padx=10, pady=10)

        # Create camera label
        self.camera_label = tk.Label(self.camera_frame)
        self.camera_label.pack()

        # Create predicted character label
        self.predicted_character_label = tk.Label(self.prediction_frame, text="Predicted Character:")
        self.predicted_character_label.pack()

        # Create suggestion text boxes
        self.suggestion_textboxes = []
        for i in range(4):
            suggestion_textbox = tk.Text(self.prediction_frame, height=1, width=20)
            suggestion_textbox.pack()
            self.suggestion_textboxes.append(suggestion_textbox)

        # Create predicted sentence label
        self.predicted_sentence_label = tk.Label(self.prediction_frame, text="Predicted Sentence:")
        self.predicted_sentence_label.pack()

        # Create text box for predicted sentence
        self.predicted_sentence_textbox = tk.Text(self.prediction_frame, height=2, width=40)
        self.predicted_sentence_textbox.pack()

        # Create Speak and Clear buttons
        self.speak_button = tk.Button(self.root, text="Speak", command=self.speak)
        self.speak_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        # Initialize mediapipe hand detection function
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # Load trained model and initialize a normalizer 
        self.model = load("model.joblib")
        self.normalizer = Normalizer()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize enchant dictionary
        self.dictionary = enchant.Dict("en_US")

        # Open camera feed
        self.video_capture = cv2.VideoCapture(0)
        self.update_camera()

    def update_camera(self):
        ret, frame = self.video_capture.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (420, 340))  # Adjust size as needed
            img = Image.fromarray(frame)
            imgtk = ImageTk.PhotoImage(image=img)
            self.camera_label.imgtk = imgtk
            self.camera_label.configure(image=imgtk)

            # Process hand landmarks for prediction
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.hands.process(rgb_frame)
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                coords = np.array([(lm.x, lm.y) for lm in hand_landmarks.landmark]).flatten()
                coords = self.normalizer.transform([coords])
                predicted_letter = self.model.predict(coords)
                self.predicted_character_label.config(text="Predicted Character: " + str(predicted_letter[0]))

                # Suggestions based on the predicted character
                suggestions = self.get_suggestions(predicted_letter[0])
                for i, suggestion in enumerate(suggestions):
                    self.suggestion_textboxes[i].delete(1.0, tk.END)
                    self.suggestion_textboxes[i].insert(tk.END, suggestion)

        self.root.after(10, self.update_camera)

    def get_suggestions(self, character):
        suggestions = self.dictionary.suggest(character)
        # Ensure at most 4 suggestions are returned
        return suggestions[:4]

    def speak(self):
        # Speak the predicted character
        predicted_character = self.predicted_character_label.cget("text").split(":")[-1].strip()
        self.engine.say(predicted_character)
        self.engine.runAndWait()

    def clear(self):
        # Clear the predicted sentence textbox
        self.predicted_sentence_textbox.delete(1.0, tk.END)

    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    root = tk.Tk()
    app = CharacterPredictionApp(root)
    app.run()
