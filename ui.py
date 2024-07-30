import tkinter as tk
import numpy as np
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import pandas as pd

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # configure the root window
        self.title('Fake News Detection System')
        self.geometry('800x600')
        
        # set colors and fonts
        bg_color = '#f0f0f0'
        text_color = '#333333'
        button_color = '#4CAF50'
        button_text_color = 'white'

        # label for the title
        self.label = tk.Label(self, text='Fake News Detection System', font=('Arial', 20, 'bold'), fg=text_color, bg=bg_color)
        self.label.pack(pady=10)

        # input text area
        self.input_label = tk.Label(self, text="Enter the news text:", font=('Arial', 14), fg=text_color, bg=bg_color)
        self.input_label.pack()
        self.inputtxt = tk.Text(self, bg="white", fg=text_color, font=('Arial', 12), height=10)
        self.inputtxt.pack(padx=20, pady=(0, 10), fill=tk.BOTH, expand=True)

        # model selection
        self.model_label = tk.Label(self, text="Select Model:", font=('Arial', 14), fg=text_color, bg=bg_color)
        self.model_label.pack()
        self.model_selection = tk.StringVar()
        self.model_selection.set("Naive Bayes")
        models = ["Naive Bayes", "Decision Tree", "SVM"]

        for model in models:
            tk.Radiobutton(self, text=model, variable=self.model_selection, value=model, font=('Arial', 12), fg=text_color, bg=bg_color).pack()

        # detect button
        self.button_detect = tk.Button(self, text='Detect', command=self.detect_button_clicked, bg=button_color, fg=button_text_color, font=('Arial', 14, 'bold'))
        self.button_detect.pack(pady=10)
        # output label
        self.output_label = tk.Label(self, text="Prediction:", font=('Arial', 14), fg=text_color, bg=bg_color)
        self.output_label.pack()

        # output
        self.output_text = tk.Text(self, bg="white", fg=text_color, font=('Arial', 12), state=tk.DISABLED, height=5)
        self.output_text.pack(padx=20, fill=tk.BOTH, expand=True)
        # Load the pre-trained models and vectorizer
        self.vectorization = TfidfVectorizer()

        # Load the dataset
        data = pd.read_csv('spacesRemoved.csv')
        x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'], test_size=0.25)
        print("Dataset Loaded")
        # Initialize models
        self.model_names = ["Naive Bayes", "Decision Tree", "SVM"]
        self.models = {
            "Naive Bayes": MultinomialNB(),
            "Decision Tree": DecisionTreeClassifier(),
            # Replace with your SVM model loading
            "SVM": pickle.load(open('SVModelData01.sav', 'rb'))
        }
        vectorization = TfidfVectorizer() 
        x_train = vectorization.fit_transform(x_train) 
        x_test = vectorization.transform(x_test)
        
        print("Texts Have Been Vectorized")


        # Fit Naive Bayes and Decision Tree models with the training data
        chunk_size = 1000  
        for i in range(0, x_train.shape[0], chunk_size):
            x_chunk = x_train[i:i+chunk_size]
            y_chunk = y_train[i:i+chunk_size]
            self.models["Naive Bayes"].partial_fit(x_chunk, y_chunk, classes=np.unique(y_train))
        print("Naive Bayes is Fitted, Ready Now !")
        self.models["Decision Tree"].fit(x_train, y_train)
        print("Decision Tree is Fitted, Ready Now !")

    def detect_button_clicked(self):
        news_text = self.take_input()

        selected_model_name = self.model_selection.get()
        selected_model = self.models[selected_model_name]

        # Vectorize the input text
        print([news_text])
        vectorized_input = self.vectorization.transform(news_text)

        # Use the selected model for prediction
        prediction = selected_model.predict(vectorized_input)

        # Display the prediction in the output text area
        self.display_output(prediction[0])

    def display_output(self, prediction):
        self.output_text.config(state=tk.NORMAL)
        self.output_text.delete('1.0', tk.END)
        if prediction == 0:
            prediction_text = "Fake"
        else:
            prediction_text = "True"
        self.output_text.insert(tk.END, f"The news is {prediction_text}!")
        self.output_text.config(state=tk.DISABLED)

    def take_input(self):
        news_input = self.inputtxt.get("1.0", "end-1c")
        return news_input


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
