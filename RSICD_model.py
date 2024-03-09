#!/usr/bin/env python
# coding: utf-8

# In[14]:


pip install pandas numpy Pillow tensorflow torch


# In[15]:


import pandas as pd
# Read train.csv
train_df = pd.read_csv("D:/RSICD/train.csv")

# Read test.csv
test_df = pd.read_csv("D:/RSICD/test.csv")

# Read valid.csv
valid_df = pd.read_csv("D:/RSICD/valid.csv")


# In[16]:


# Display the first few rows of each DataFrame
print("Train Dataset:")
print(train_df.head())

print("\nTest Dataset:")
print(test_df.head())

print("\nValidation Dataset:")
print(valid_df.head())

# Get information about each DataFrame
print("\nTrain Dataset Info:")
print(train_df.info())

print("\nTest Dataset Info:")
print(test_df.info())

print("\nValidation Dataset Info:")
print(valid_df.info())

# Summary statistics
print("\nTrain Dataset Summary Statistics:")
print(train_df.describe())

print("\nTest Dataset Summary Statistics:")
print(test_df.describe())

print("\nValidation Dataset Summary Statistics:")
print(valid_df.describe())


# In[23]:


import pandas as pd
from PIL import Image
import io


# In[24]:


import ast

# Function to convert string representation of dictionary to dictionary
def string_to_dict(s):
    return ast.literal_eval(s)

def convert_byte_array_to_image(byte_array):
    # Convert byte array to bytes-like object
    image_data = io.BytesIO(byte_array)
    # Open image using PIL
    image = Image.open(image_data)
    return image

# Convert byte-arrays to images for train_df
train_images = []
for index, row in train_df.iterrows():
    byte_array = string_to_dict(row['image'])['bytes']
    image = convert_byte_array_to_image(byte_array)
    train_images.append(image)

# Convert byte-arrays to images for test_df
test_images = []
for index, row in test_df.iterrows():
    byte_array = string_to_dict(row['image'])['bytes']
    image = convert_byte_array_to_image(byte_array)
    test_images.append(image)

# Convert byte-arrays to images for valid_df
valid_images = []
for index, row in valid_df.iterrows():
    byte_array = string_to_dict(row['image'])['bytes']
    image = convert_byte_array_to_image(byte_array)
    valid_images.append(image)


# In[19]:


import os

# Define the directory to save images
save_dir = 'D:/RSICD/train_images'

# Create the directory if it doesn't exist
os.makedirs(save_dir, exist_ok=True)

# Save images to disk with filenames from the DataFrame
for index, row in train_df.iterrows():
    filename = row['filename']
    image = train_images[index]
    image_filename = os.path.join(save_dir, os.path.basename(filename))
    image.save(image_filename)


# In[20]:


# Define the directory to save images
save_dir_test = 'D:/RSICD/test_images'

# Create the directory if it doesn't exist
os.makedirs(save_dir_test, exist_ok=True)

# Save test images to disk with filenames from the DataFrame
for index, row in test_df.iterrows():
    filename = row['filename']
    image = test_images[index]
    image_filename = os.path.join(save_dir_test, os.path.basename(filename))
    image.save(image_filename)


# In[21]:


# Define the directory to save images
save_dir_valid = 'D:/RSICD/valid_images'

# Create the directory if it doesn't exist
os.makedirs(save_dir_valid, exist_ok=True)

# Save validation images to disk with filenames from the DataFrame
for index, row in valid_df.iterrows():
    filename = row['filename']
    image = valid_images[index]
    image_filename = os.path.join(save_dir_valid, os.path.basename(filename))
    image.save(image_filename)


# In[25]:


import nltk
from nltk.tokenize import word_tokenize

# Tokenize captions
def tokenize_captions(df):
    df['tokenized_captions'] = df['captions'].apply(lambda x: word_tokenize(x))
    return df

# Apply tokenization to train, test, and validation datasets
train_df = tokenize_captions(train_df)
test_df = tokenize_captions(test_df)
valid_df = tokenize_captions(valid_df)

# Display the first few rows of each DataFrame to verify the tokenization
print("Train Dataset:")
print(train_df.head())

print("\nTest Dataset:")
print(test_df.head())

print("\nValidation Dataset:")
print(valid_df.head())


# In[27]:


from collections import Counter

# Function to build vocabulary
def build_vocab(dataframe):
    captions = dataframe['tokenized_captions'].tolist()
    # Flatten list of tokenized captions
    all_words = [word for sublist in captions for word in sublist]
    # Count the occurrence of each word
    word_counts = Counter(all_words)
    # Sort words by frequency
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    # Create a vocabulary mapping each word to an index
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(sorted_words)}
    # Add special tokens for padding and unknown words
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = len(vocab)
    return vocab

# Build vocabulary for train_df
vocab = build_vocab(train_df)

# Display the first few words in the vocabulary
print("Vocabulary Size:", len(vocab))
print("First Few Words in Vocabulary:")
print({k: vocab[k] for k in list(vocab)[:10]})


# In[28]:


import numpy as np

# Function to convert tokenized captions to numerical sequences
def captions_to_sequences(dataframe, vocab, max_length):
    sequences = []
    for tokens in dataframe['tokenized_captions']:
        # Convert tokens to indices using the vocabulary, replace unknown words with <UNK> index
        seq = [vocab.get(token, vocab['<UNK>']) for token in tokens]
        # Pad or truncate sequences to max_length
        if len(seq) < max_length:
            seq += [vocab['<PAD>']] * (max_length - len(seq))
        else:
            seq = seq[:max_length]
        sequences.append(seq)
    return np.array(sequences)

# Define maximum sequence length
max_length = 20

# Convert tokenized captions to sequences for train, test, and validation datasets
train_sequences = captions_to_sequences(train_df, vocab, max_length)
test_sequences = captions_to_sequences(test_df, vocab, max_length)
valid_sequences = captions_to_sequences(valid_df, vocab, max_length)

# Display the shape of the resulting arrays
print("Train Sequences Shape:", train_sequences.shape)
print("Test Sequences Shape:", test_sequences.shape)
print("Validation Sequences Shape:", valid_sequences.shape)


# In[29]:


from torchvision import transforms

# Define image transformation
image_transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),           # Convert image to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values
])

# Apply transformation to train images
train_images = [image_transform(image) for image in train_images]

# Apply transformation to test images
test_images = [image_transform(image) for image in test_images]

# Apply transformation to validation images
valid_images = [image_transform(image) for image in valid_images]


# In[64]:


import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, image):
        return self.resnet(image)


class TextDecoder(nn.Module):
    def __init__(self, model_name):
        super(TextDecoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.model(input_ids)[0]


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, text_decoder):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder

    def forward(self, image, captions):
        image_features = self.image_encoder(image)
        decoded_captions = self.text_decoder(captions)
        return image_features, decoded_captions


# Define your dataset and dataloader here

# Initialize the model components
image_encoder = ImageEncoder()
text_decoder = TextDecoder("gpt2-medium")
model = ImageCaptioningModel(image_encoder, text_decoder)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Define your training loop here


# In[65]:


# Assuming train_images contains the transformed images
# Extract captions from train_df
train_captions = train_df['captions'].tolist()

# Define the ImageCaptionDataset class to pair images with their corresponding captions
class ImageCaptionDataset(Dataset):
    def __init__(self, images, captions):
        self.images = images
        self.captions = captions
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        image = self.images[index]
        caption = self.captions[index]
        return image, caption

# Create an instance of the ImageCaptionDataset class
train_dataset = ImageCaptionDataset(train_images, train_captions)

# Define batch size and create DataLoader
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# Assuming you have a validation dataset named valid_dataset
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# Iterate through the DataLoader to get batches of data during training
for images, captions in train_loader:
    # Use these batches to train your model
    pass


# In[66]:


# Initialize the model
model = ImageCaptioningModel(embed_size=512, hidden_size=512, vocab_size=len(vocab), attention_dim=256, encoder_dim=2048)


# In[67]:


# Define the loss function
criterion = nn.CrossEntropyLoss()


# In[68]:


# Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


# In[69]:


import nltk
nltk.download('punkt')

def tokenizer(text):
    return nltk.word_tokenize(text.lower())


# In[76]:


# Assuming train_loader is your DataLoader for training data
# Define number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    
    for images, captions,lengths in train_loader:
        # Forward pass
        outputs = model(images, captions,lengths)
        
        # Compute the loss
        loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    # Print average loss for this epoch
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader)}')
    
    # Optionally, evaluate the model on validation set here
    
# Optionally, save the trained model
torch.save(model.state_dict(), 'image_captioning_model.pth')


# In[11]:


pip install torchvision


# In[ ]:





# In[ ]:





# In[73]:


import pandas as pd
import numpy as np
import os
import nltk
import torch
from torch.utils.data import Dataset, DataLoader

# Read train.csv
train_df = pd.read_csv("D:/RSICD/train.csv")
# Read test.csv
test_df = pd.read_csv("D:/RSICD/test.csv")
# Read valid.csv
valid_df = pd.read_csv("D:/RSICD/valid.csv")

# Function to tokenize captions
def tokenize_captions(df):
    captions = df['captions'].tolist()
    tokens = []
    for caption_list in captions:
        # Split each caption into individual sentences
        for caption in caption_list.split('\n'):
            # Tokenize each sentence into words
            words = nltk.word_tokenize(caption.lower())
            tokens.extend(words)
    return tokens

# Tokenize captions from train, test, and valid DataFrames
train_tokens = tokenize_captions(train_df)
test_tokens = tokenize_captions(test_df)
valid_tokens = tokenize_captions(valid_df)

# Create word-to-index and index-to-word mappings
def create_vocab_mappings(tokens):
    word_counts = Counter(tokens)
    # Sort words by frequency (most frequent first)
    sorted_words = sorted(word_counts, key=word_counts.get, reverse=True)
    # Add special tokens
    vocab = ['<UNK>', '<START>', '<END>', '<PAD>'] + sorted_words
    word_to_index = {word: index for index, word in enumerate(vocab)}
    index_to_word = {index: word for word, index in word_to_index.items()}
    return word_to_index, index_to_word

# Create vocabulary mappings for train, test, and valid tokens
word_to_index, index_to_word = create_vocab_mappings(train_tokens + test_tokens + valid_tokens)

# Define maximum sequence length (including <START> and <END> tokens)
max_seq_length = max(max(len(nltk.word_tokenize(caption.lower())) for caption in train_df['captions'].tolist()),
                     max(len(nltk.word_tokenize(caption.lower())) for caption in test_df['captions'].tolist()),
                     max(len(nltk.word_tokenize(caption.lower())) for caption in valid_df['captions'].tolist())) + 2  # Add 2 for <START> and <END> tokens

# Define a custom PyTorch dataset
class ImageCaptionDataset(Dataset):
    def __init__(self, df, images_dir, word_to_index, max_seq_length):
        self.df = df
        self.images_dir = images_dir
        self.word_to_index = word_to_index
        self.max_seq_length = max_seq_length
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        row = self.df.iloc[index]
        image_filename = os.path.join(self.images_dir, row['filename'])
        image = Image.open(image_filename).convert('RGB')
        # Tokenize caption and convert to sequence of integers
        caption_tokens = nltk.word_tokenize(row['captions'].lower())
        caption_sequence = [self.word_to_index.get(word, self.word_to_index['<UNK>']) for word in caption_tokens]
        # Add <START> and <END> tokens and pad sequence
        caption_sequence = [self.word_to_index['<START>']] + caption_sequence + [self.word_to_index['<END>']]
        caption_sequence += [self.word_to_index['<PAD>']] * (self.max_seq_length - len(caption_sequence))
        return image, torch.tensor(caption_sequence)

# Define PyTorch datasets and DataLoaders for train, test, and validation sets
batch_size = 32

train_images_dir = 'D:/RSICD/train_images'
test_images_dir = 'D:/RSICD/test_images'
valid_images_dir = 'D:/RSICD/valid_images'

train_dataset = ImageCaptionDataset(train_df, train_images_dir, word_to_index, max_seq_length)
test_dataset = ImageCaptionDataset(test_df, test_images_dir, word_to_index, max_seq_length)
valid_dataset = ImageCaptionDataset(valid_df, valid_images_dir, word_to_index, max_seq_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)


# In[72]:


import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

# Define the Encoder
class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        # Use a pre-trained CNN model as the encoder
        self.cnn = models.resnet50(pretrained=True)
        # Remove the final fully connected layer
        modules = list(self.cnn.children())[:-2]  # Remove the avgpool and final fc layer
        self.cnn = nn.Sequential(*modules)
        # Add a linear layer to transform the extracted features into the desired embedding size
        self.linear = nn.Linear(self.cnn[-1][-1].conv3.out_channels, embed_size)
        # Add a batch normalization layer
        self.bn = nn.BatchNorm1d(embed_size)
        
    def forward(self, images):
        features = self.cnn(images)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))
        return features

# Define the Attention Mechanism
class Attention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)
        att2 = self.decoder_att(decoder_hidden)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)
        alpha = self.softmax(att)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)
        return attention_weighted_encoding, alpha

# Define the Decoder with Attention Mechanism
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, dropout=0.5):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.attention = Attention(encoder_dim, hidden_size, attention_dim)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size, bias=True)
        self.init_h = nn.Linear(encoder_dim, hidden_size)  
        self.init_c = nn.Linear(encoder_dim, hidden_size)  
        self.f_beta = nn.Linear(hidden_size, encoder_dim)  
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(hidden_size, vocab_size)  
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.fc.out_features

        embeddings = self.embed(encoded_captions)

        h, c = self.init_hidden_state(encoder_out)  

        decode_lengths = [length - 1 for length in caption_lengths]
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)

        alphas = torch.zeros(batch_size, max(decode_lengths), encoder_out.size(1)).to(device)

        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t])) 
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))
            preds = self.fc(self.dropout(h))  
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  
        c = self.init_c(mean_encoder_out)
        return h, c

# Define the image captioning model
class ImageCaptioningModel(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, attention_dim, encoder_dim, dropout=dropout)
        
    def forward(self, images, captions, lengths):
        features = self.encoder(images)
        outputs, _, decode_lengths, _ = self.decoder(features, captions, lengths)
        return outputs, decode_lengths

# Define hyperparameters
embed_size = 256
hidden_size = 512
attention_dim = 512
encoder_dim = 2048
vocab_size = len(word_to_index)  # Make sure to replace `word_to_index` with your vocabulary dictionary

# Initialize the model
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, attention_dim, encoder_dim)

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define function to calculate BLEU score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

def calculate_bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothing = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

# Define training function
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for images, captions, lengths in dataloader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs, decode_lengths = model(images, captions, lengths)
        targets = captions[:, 1:]
        outputs = pack_padded_sequence(outputs, decode_lengths, batch_first=True).data
        targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)

# Define evaluation function
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_bleu_score = 0.0
    with torch.no_grad():
        for images, captions, lengths in dataloader:
            images, captions = images.to(device), captions.to(device)
            outputs, decode_lengths = model(images, captions, lengths)
            predicted_captions = torch.argmax(outputs, dim=2)
            for i in range(len(predicted_captions)):
                predicted_caption = ' '.join([index_to_word[index.item()] for index in predicted_captions[i] if index.item() != word_to_index['<PAD>']])
                reference_caption = ' '.join([index_to_word[index.item()] for index in captions[i] if index.item() != word_to_index['<PAD>']])
                total_bleu_score += calculate_bleu_score(reference_caption, predicted_caption)
    return total_bleu_score / len(dataloader)

# Define number of epochs
num_epochs = 10

# Training loop
for epoch in range(num_epochs):
    train_loss = train(model, train_loader, criterion, optimizer, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")
    if (epoch + 1) % 5 == 0:
        bleu_score = evaluate(model, valid_loader, criterion, device)
        print(f"Epoch [{epoch+1}/{num_epochs}], BLEU Score: {bleu_score:.4f}")

# After training, you can evaluate the model on the test dataset
test_bleu_score = evaluate(model, test_loader, criterion, device)
print(f"Test BLEU Score: {test_bleu_score:.4f}")


# In[79]:


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch
import torch.nn as nn
from torchvision import models
import torchvision.transforms as transforms
from transformers import GPT2Tokenizer, GPT2LMHeadModel
# Define your ImageCaptioningModel, EncoderCNN, and DecoderRNN classes here
class ImageEncoder(nn.Module):
    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

    def forward(self, image):
        return self.resnet(image)


class TextDecoder(nn.Module):
    def __init__(self, model_name):
        super(TextDecoder, self).__init__()
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)

    def forward(self, input_ids):
        return self.model(input_ids)[0]


class ImageCaptioningModel(nn.Module):
    def __init__(self, image_encoder, text_decoder):
        super(ImageCaptioningModel, self).__init__()
        self.image_encoder = image_encoder
        self.text_decoder = text_decoder

    def forward(self, image, captions):
        image_features = self.image_encoder(image)
        decoded_captions = self.text_decoder(captions)
        return image_features, decoded_captions

# ...
# Initialize the image encoder and text decoder
image_encoder = ImageEncoder()
text_decoder = TextDecoder(ImageCaptioningModel)

# Initialize the ImageCaptioningModel with the image encoder and text decoder
model = ImageCaptioningModel(image_encoder, text_decoder).to(device)

# Define hyperparameters
embed_size = 256
hidden_size = 512
attention_dim = 512
encoder_dim = 2048
vocab_size = len(word_to_index)  # Update with your vocabulary size
learning_rate = 0.001
num_epochs = 10
batch_size = 64  # Adjust as needed

# Initialize your model, loss function, optimizer, and move model to device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ImageCaptioningModel(embed_size, hidden_size, vocab_size, attention_dim, encoder_dim).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Define DataLoader for your training, validation, and test datasets
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Define function to calculate BLEU score
def calculate_bleu_score(reference, candidate):
    reference = [reference.split()]
    candidate = candidate.split()
    smoothing = SmoothingFunction().method4
    return sentence_bleu(reference, candidate, smoothing_function=smoothing)

# Training loop
for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    for images, captions, lengths in train_loader:
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        outputs, _ = model(images, captions, lengths)
        targets = captions[:, 1:]
        outputs = outputs[:, :targets.shape[1], :]  # Trim outputs to match target length
        loss = criterion(outputs.contiguous().view(-1, vocab_size), targets.contiguous().view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)
    
    # Validation
    model.eval()
    total_bleu_score = 0.0
    with torch.no_grad():
        for images, captions, lengths in valid_loader:
            images, captions = images.to(device), captions.to(device)
            outputs, _ = model(images, captions, lengths)
            predicted_captions = torch.argmax(outputs, dim=2)
            for i in range(len(predicted_captions)):
                predicted_caption = ' '.join([index_to_word[index.item()] for index in predicted_captions[i] if index.item() != word_to_index['<PAD>']])
                reference_caption = ' '.join([index_to_word[index.item()] for index in captions[i] if index.item() != word_to_index['<PAD>']])
                total_bleu_score += calculate_bleu_score(reference_caption, predicted_caption)
    avg_bleu_score = total_bleu_score / len(valid_loader)
    
    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, BLEU Score: {avg_bleu_score:.4f}')

# Evaluate the model on the test dataset
total_bleu_score = 0.0
with torch.no_grad():
    for images, captions, lengths in test_loader:
        images, captions = images.to(device), captions.to(device)
        outputs, _ = model(images, captions, lengths)
        predicted_captions = torch.argmax(outputs, dim=2)
        for i in range(len(predicted_captions)):
            predicted_caption = ' '.join([index_to_word[index.item()] for index in predicted_captions[i] if index.item() != word_to_index['<PAD>']])
            reference_caption = ' '.join([index_to_word[index.item()] for index in captions[i] if index.item() != word_to_index['<PAD>']])
            total_bleu_score += calculate_bleu_score(reference_caption, predicted_caption)
test_bleu_score = total_bleu_score / len(test_loader)
print(f'Test BLEU Score: {test_bleu_score:.4f}')


# In[ ]:




