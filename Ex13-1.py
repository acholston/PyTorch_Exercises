# Lab 12 RNN
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

torch.manual_seed(777)  # reproducibility


#Short Spanish Test Case
x_data = [[0, 6, 2], [0, 4, 2], [0, 1, 3], [0, 4, 3], [0, 1, 5]]
y_data = [[0, 2, 6], [0, 2, 4], [0, 1, 5], [0, 1, 4], [0, 3, 5]]

idx = ["el", "perro", "azul", "rojo", "gato", "amarillo", "cabello"]
idy = ["the", "red", "blue", "yellow", "cat", "dog", "horse"]

#Test => 0, 1, 2 (The blue dog)
# As we have one batch of samples, we will change them to variables only once


num_cases = 5
input_size = 7 
hidden_size = 100 
embed_size = 100
batch_size = 1 
sequence_length = 3  
num_layers = 1  


#For generating initial sentence encoding
class EncoderRNN(nn.Module):

    def __init__(self, input_size, hidden_size, embed_size):
        super(EncoderRNN, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

	#Feed through RNN
	
	#Embedding
        self.emb = nn.Embedding(self.input_size, embed_size)

	#RNN Module - Bidirectional for runn sentence both ways
	self.gru = nn.GRU(embed_size, self.hidden_size, bidirectional=True)

	#Initialize the hidden state of the decoder here
	self.init_s = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, h):
        x = self.emb(x)
	x, h = self.gru(x, h)

	s = self.init_s(h[0][0]).view(-1, self.hidden_size)
	h = h.transpose(0, 1).contiguous()
	h = h.view(-1, self.hidden_size*2)

        return x, h, s


#Decoder
class DecoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, sequence_length, embed_size):
	super(DecoderRNN, self).__init__()
	self.hidden_size = hidden_size
	self.sequence_length = sequence_length
	self.input_size = input_size

	
	self.U = nn.Linear(hidden_size*2, hidden_size)
	self.W = nn.Linear(hidden_size, hidden_size)
	self.Uc = nn.Linear(hidden_size*2, hidden_size)
	self.Wy = nn.Linear(hidden_size, hidden_size)
	self.a = nn.Linear(self.hidden_size*2, self.hidden_size)

	self.emb = nn.Embedding(self.input_size, embed_size)
	self.gru = nn.GRU(self.hidden_size, hidden_size)
	self.fc = nn.Linear(self.hidden_size, self.input_size)

    def attn(self, s, h):
	c = Variable(torch.zeros(self.hidden_size*2))
	
	for i in range(self.sequence_length):
	    temp = F.tanh(self.U(h[i]) + self.W(s))
	    temp = F.softmax(temp)
	    temp = torch.mul(temp.view(1, self.hidden_size), h[i].view(-1, self.hidden_size))
	    c += temp.view(self.hidden_size*2)

	return c

    def forward(self, y, h, s):
	y = self.emb(y)

	c = self.attn(s, h).view(-1, self.hidden_size*2)
	
	y = self.Wy(y) + self.Uc(c)
	y = y.view(1, 1, -1)
	s = s.view(1, 1, -1)

	y, s = self.gru(y, s)

	y = y.squeeze(0)
	y = self.fc(y)

	y = F.log_softmax(y)

	return y, s
	


# Instantiate RNN model
enc = EncoderRNN(input_size, hidden_size, embed_size)
dec = DecoderRNN(input_size, hidden_size, sequence_length, embed_size)

# Set loss and optimizer function
# CrossEntropyLoss = LogSoftmax + NLLLoss
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(enc.parameters(), lr=0.001)

# Train the model
for epoch in range(10000):
    inputs = Variable(torch.LongTensor([x_data[epoch%num_cases]]))
    labels = Variable(torch.LongTensor(y_data[epoch%num_cases]))
    enc_h = Variable(torch.zeros(2, sequence_length, hidden_size))
    c = Variable(torch.zeros(1, hidden_size*2))
    outputs = []
    out, h, s = enc(inputs, enc_h)
    for i in range(sequence_length):
	if i > 0:
	    _, idx = output.max(1)
	    inp = Variable(torch.LongTensor(idx.data))
	else:
	    inp = Variable(torch.LongTensor([0]))
	output, s = dec(inp, h, s)
        outputs.append(output[0])

    outputs = torch.stack(outputs)
    optimizer.zero_grad()
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    _, idx = outputs.max(1)
    idx = idx.data.numpy()
    #result_str = [idx2char[c] for c in idx.squeeze()]
    if epoch % 10 == 0:
	inputs = Variable(torch.LongTensor([[0, 1, 2]]))
        enc_h = Variable(torch.zeros(2, sequence_length, hidden_size))
        cont = Variable(torch.zeros(1, hidden_size*2))
        outputs = []
        out, h, s = enc(inputs, enc_h)
        for i in range(sequence_length):
	    if i > 0:
	        _, idx = output.max(1)
	        inp = Variable(torch.LongTensor(idx.data))
	    else:
	        inp = Variable(torch.LongTensor([0]))
	    output, s = dec(inp, h, s)
            outputs.append(output[0])

        outputs = torch.stack(outputs)

        _, idx = outputs.max(1)
        idx = idx.data.numpy()
        result_str = [idy[c] for c in idx.squeeze()]
	print(result_str)
        #print("epoch: %d, loss: %1.3f" % (epoch, loss.data[0]))
        #print("Predicted string: ", ''.join(result_str))

