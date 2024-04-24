from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
import tkinter as tk
import spacy
import spacy.cli
import wikipedia

#spacy.cli.download("en_core_web_lg")

# Load the English language model
nlp = spacy.load("en_core_web_lg")

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Tkinter interface
root = tk.Tk()
root.title("ChatGPT")

chat_history = tk.Text(root, width=150, height=50)
chat_history.pack()

user_input = tk.Entry(root, width=50)
user_input.pack()

def process_text(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Perform any additional processing or analysis here
    return doc

# Use spaCy to process the user input before generating a response
def submit_input():
    user_text = user_input.get().strip()
    user_input.delete(0, tk.END)
    chat_history.insert(tk.END, f"You: {user_text}\n")
    
    # Process the user input with spaCy
    processed_input = process_text(user_text)
    
    # Tokenize the processed input
    inputs = tokenizer.encode(processed_input.text, return_tensors='pt')

    # Use the processed input to generate a response
    response = model.generate(inputs, max_length=100, pad_token_id=tokenizer.eos_token_id, num_return_sequences=1, return_dict_in_generate=False)
    
    # Decode the generated tokens to text
    generated_text = tokenizer.decode(response[0], skip_special_tokens=True)

    # Check if the response is a question
    if '?' in user_text:
        # Search Wikipedia for relevant information
        try:
            wikipedia_summary = wikipedia.summary(user_text, sentences=3)  # Fetch more sentences for context
            generated_text += f"\nAccording to Wikipedia:\n{wikipedia_summary}"
        except wikipedia.exceptions.DisambiguationError as e:
            generated_text += f"\nThere are multiple possible meanings for this query: {e.options}"
        except wikipedia.exceptions.PageError:
            generated_text += "\nNo information found on Wikipedia."

    chat_history.insert(tk.END, f"ChatGPT: {generated_text}\n")

    # Display the Wikipedia information in a pop-up window
    wiki_window = tk.Toplevel()
    wiki_window.title("Wikipedia Summary")
    wiki_label = tk.Label(wiki_window, text=wikipedia_summary)
    wiki_label.pack()

submit_button = tk.Button(root, text="Submit", command=submit_input)
submit_button.pack()

root.mainloop()
