from parameters import *


class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.device = device

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class AttnDecoderRNN(nn.Module):

    def __init__(self, hidden_size, output_size, device, dropout_p=0.1, max_length=30):
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)
        self.device = device


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(
            self.attn(torch.cat((embedded[0], hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=self.device)


class NMTwithAttention():

    def __init__(self, max_length, input_size, output_size, hidden_size, device):

        self.max_length=max_length
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.SOS_token = 0
        self.EOS_token = 1

        self.encoder = EncoderRNN(input_size=self.input_size,hidden_size=self.hidden_size,device=self.device)
        self.decoder = AttnDecoderRNN(hidden_size=self.hidden_size, output_size=self.output_size, device=self.device,
                                      max_length= self.max_length)


    def Preprocess(self, data):
        return torch.tensor(data, dtype=torch.long, device=self.device).view(-1, 1)

    def _train(self, input_tensor, target_tensor):

        encoder_hidden = self.encoder.initHidden()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        input_length = input_tensor.size(0)
        target_length = target_tensor.size(0)

        encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size, device=self.device)

        loss = 0

        for ei in range(input_length):
            encoder_output, encoder_hidden = self.encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] = encoder_output[0, 0]

        decoder_input = torch.tensor([[self.SOS_token]], device=self.device)

        decoder_hidden = encoder_hidden

        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False

        if use_teacher_forcing:
            # Teacher forcing: Feed the target as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                loss += self.criterion(decoder_output, target_tensor[di])
                decoder_input = target_tensor[di]  # Teacher forcing

        else:
            # Without teacher forcing: use its own predictions as the next input
            for di in range(target_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(decoder_input, decoder_hidden, encoder_outputs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()  # detach from history as input

                loss += self.criterion(decoder_output, target_tensor[di])
                if decoder_input.item() == self.EOS_token:
                    break

        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        return loss.item() / target_length


    def Train(self, train_src, train_mt, n_epochs, learning_rate=0.001, teacher_forcing_ratio=0.5, print_every=1):

        self.train_src = train_src
        self.train_mt = train_mt

        self.encoder_optimizer = optim.SGD(self.encoder.parameters(), lr=learning_rate)
        self.decoder_optimizer = optim.SGD(self.decoder.parameters(), lr=learning_rate)
        self.criterion = nn.NLLLoss()
        self.teacher_forcing_ratio = teacher_forcing_ratio


        print_loss_total = 0  # Reset every print_every

        index_list = np.arange(0, len(self.train_src))
        for iter in range(1, n_epochs + 1):

            np.random.shuffle(index_list)
            for i in tqdm.tqdm(index_list, desc="EPOCH " + str(iter)):
                input_tensor = self.Preprocess(self.train_src[i])
                target_tensor = self.Preprocess(self.train_mt[i])
                loss = self._train(input_tensor, target_tensor)
                print_loss_total += loss

            if iter % print_every == 0:
                print_loss_avg = print_loss_total / (print_every*len(index_list))
                print_loss_total = 0
                print(iter, iter / n_epochs * 100, print_loss_avg)


    def Infer(self):
        pass

