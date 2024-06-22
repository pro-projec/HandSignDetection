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
        self.predicted_character_label = tk.Label(self.prediction_frame, text="Predicted Character:", fg="red")
        self.predicted_character_label.pack()

        # Create suggestion text boxes
        self.suggestion_textboxes = []
        for i in range(4):
            suggestion_textbox = tk.Text(self.prediction_frame, height=1, width=20)
            suggestion_textbox.pack()
            self.suggestion_textboxes.append(suggestion_textbox)

        # Create predicted sentence label
        self.predicted_sentence_label = tk.Label(self.prediction_frame, text="Predicted Sentence:", fg="red")
        self.predicted_sentence_label.pack()

        # Create text box for predicted sentence
        self.predicted_sentence_textbox = tk.Text(self.prediction_frame, height=2, width=40)
        self.predicted_sentence_textbox.pack()

        # Create Speak and Clear buttons
        self.speak_button = tk.Button(self.root, text="Speach", command=self.speak)
        self.speak_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        self.clear_button = tk.Button(self.root, text="Clear", command=self.clear)
        self.clear_button.pack(side=tk.BOTTOM, padx=10, pady=10, anchor=tk.SE)

        # Initialize MediaPipe hand detection function
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(max_num_hands=1)
        self.mp_drawing = mp.solutions.drawing_utils

        # Load trained model and initialize a normalizer 
        self.model = load("model.joblib")
        self.normalizer = Normalizer()

        # Initialize text-to-speech engine
        self.engine = pyttsx3.init()

        # Initialize enchant dictionary for spell checking
        self.dictionary = enchant.Dict("en_US")

        # Open camera feed
        self.video_capture = cv2.VideoCapture(0)
        self.update_camera()

        # Bind number keys to suggestion adding function
        for i in range(4):
            self.root.bind(str(i + 1), lambda event, idx=i: self.add_suggestion_to_sentence(idx))

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

                # Draw hand landmarks and bones
                self.mp_drawing.draw_landmarks(
                    frame, hand_landmarks, self.mpHands.HAND_CONNECTIONS,
                    landmark_drawing_spec=self.mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
                    connection_drawing_spec=self.mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
                )

                # Extract hand coordinates for prediction
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
        # Get words starting with the predicted character (uppercase)
        character_upper = character.upper()
        if character_upper in alphabet_words:
            return alphabet_words[character_upper]
        else:
            return ['No suggestions']  # Return a default value if character not found

    def speak(self):
        # Get the predicted sentence from the text box
        predicted_sentence = self.predicted_sentence_textbox.get(1.0, tk.END).strip()
        # Speak the predicted sentence
        self.engine.say(predicted_sentence)
        self.engine.runAndWait()

    def clear(self):
        # Clear the predicted sentence textbox
        self.predicted_sentence_textbox.delete(1.0, tk.END)

    def add_suggestion_to_sentence(self, idx):
        # Get the selected suggestion
        suggestion = self.suggestion_textboxes[idx].get(1.0, tk.END).strip()
        # Add the suggestion to the predicted sentence
        self.predicted_sentence_textbox.insert(tk.END, suggestion + " ")

    def run(self):
        self.root.mainloop()

# Dictionary of words for each alphabet letter
alphabet_words = {
    'A': ['a', 'am', 'an', 'are'],
    'B': ['be', 'by', 'boy', 'b.tech'],
    'C': ['cry', 'can', 'cut', 'could'],
    'D': ['do', 'did', 'dad', 'dog'],
    'E': ['em', 'eat', 'even', 'each'],
    'F': ['fun', 'for', 'fast', 'from'],
    'G': ['go', 'get', 'gone', 'give'],
    'H': ['hi', 'he', 'how', 'help'],
    'I': ['I', 'it', 'ice', 'into'],
    'J': ['jo', 'job', 'just', 'join'],
    'K': ['kid', 'key', 'keep', 'kind'],
    'L': ['let', 'lot', 'like', 'look'],
    'M': ['my', 'me', 'may', 'much'],
    'N': ['no', 'not', 'new', 'now'],
    'O': ['or', 'of', 'out', 'only'],
    'P': ['pet', 'pen', 'part', 'post'],
    'Q': ['qi', 'qua', 'quiz', 'quit'],
    'R': ['run', 'red', 'read', 'rest'],
    'S': ['so', 'see', 'say', 'stay'],
    'T': ['to', 'the', 'try', 'talk'],
    'U': ['up', 'use', 'user', 'unit'],
    'V': ['ve', 'via', 'very', 'vote'],
    'W': ['we', 'why', 'what', 'where'],
    'X': ['xi', 'xen', 'xray', 'xbox'],
    'Y': ['yes', 'you', 'year', 'your'],
    'Z': ['zoo', 'zip', 'zero', 'zone']
}

if __name__ == "__main__":
    root = tk.Tk()
    app = CharacterPredictionApp(root)
    app.run()
