from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling
from torch.utils.data import Dataset
import torch
import tkinter as tk
import spacy
import spacy.cli
import wikipedia
import json
import os

# Load the English language model
nlp = spacy.load("en_core_web_lg")

# Load pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Define your custom dataset class
class MyDataset(Dataset):
    def __init__(self, tokenizer, file_path, block_size=128):
        self.examples = []
        with open(file_path, encoding="utf-8") as f:
            text = f.read()
            tokenized_text = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))
            for i in range(0, len(tokenized_text) - block_size + 1, block_size):  # Truncate in block of block_size
                self.examples.append(tokenizer.build_inputs_with_special_tokens(tokenized_text[i : i + block_size]))
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        return torch.tensor(self.examples[idx], dtype=torch.long)

# Check if the training data file exists
training_file_path = 'dataset.txt'
if os.path.isfile(training_file_path):
    try:
        # Load the dataset
        dataset = TextDataset(
            tokenizer=tokenizer,
            file_path=training_file_path,
            block_size=128,
        )
    except Exception as e:
        print(f"Error loading dataset: {e}")
        dataset = None
else:
    print("Training data file not found. Skipping dataset loading.")
    dataset = None

# Check if the dataset was successfully loaded
if dataset is not None:
    # Create a data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    # Define the fine-tuning dataset and training arguments
    training_args = TrainingArguments(
        output_dir='./results',  # output directory
        num_train_epochs=3,  # number of training epochs
        per_device_train_batch_size=2,  # batch size per device during training
        save_steps=10_000,  # save checkpoint every X steps
        save_total_limit=2,  # limit the total amount of checkpoints
    )

    # Define the trainer for fine-tuning
    trainer = Trainer(
        model=model,  # the instantiated 🤗 Transformers model to be trained
        args=training_args,  # training arguments, defined above
        train_dataset=dataset,  # training dataset
        data_collator=data_collator,  # data collator
    )

    # Fine-tune the GPT-2 model
    trainer.train()

# Tkinter interface
root = tk.Tk()
root.title("MikeGPT - Local Version!")

chat_history = tk.Text(root, width=150, height=50)
chat_history.pack()

user_input = tk.Entry(root, width=50)
user_input.pack()

# Define conversation history file path
conversation_file_path = "conversation_history.json"

def process_text(text):
    # Process the text with spaCy
    doc = nlp(text)
    # Perform any additional processing or analysis here
    return doc

def save_conversation(user_text, generated_text):
    # Load existing conversation history
    try:
        with open(conversation_file_path, "r") as file:
            conversation_history = json.load(file)
    except FileNotFoundError:
        conversation_history = []

    # Append new user input and generated response to conversation history
    conversation_history.append({"user": user_text, "bot": generated_text})

    # Save updated conversation history
    with open(conversation_file_path, "w") as file:
        json.dump(conversation_history, file, indent=4)

def load_conversation():
    try:
        with open(conversation_file_path, "r") as file:
            conversation_history = json.load(file)
        return conversation_history
    except FileNotFoundError:
        return []

# Use spaCy to process the user input before generating a response
def submit_input(tokenizer):
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

    # Save the conversation to history
    save_conversation(user_text, generated_text)

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

# Load conversation history
conversation_history = load_conversation()

# Display conversation history in the chat history box
for entry in conversation_history:
    chat_history.insert(tk.END, f"You: {entry['user']}\n")
    chat_history.insert(tk.END, f"ChatGPT: {entry['bot']}\n")

submit_button = tk.Button(root, text="Submit", command=lambda: submit_input(tokenizer))
submit_button.pack()

root.mainloop()
