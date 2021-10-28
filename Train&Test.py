import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from torchnet import meter
import tqdm

class Net(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim, num_layers=2, batch_first=False)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input, hidden=None):
        seq_len, batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0, c_0 = Variable(h_0), Variable(c_0)
        else:
            h_0, c_0 = hidden
        embeds = self.embeddings(input)
        output, hidden = self.lstm(embeds, (h_0, c_0))
        output = self.linear1(
            output.view(seq_len * batch_size, -1))
        return output, hidden

def train():
    modle = Net(len(word2ix), 128, 256)
    criterion = nn.CrossEntropyLoss()
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    modle = modle.to(device)
    optimizer = torch.optim.Adam(modle.parameters(), lr=1e-3)
    criterion = criterion.to(device)
    loss_meter = meter.AverageValueMeter()

    period = []
    loss2 = []
    for epoch in range(8):
        loss_meter.reset()
        for i, data in tqdm.tqdm(enumerate(dataloader)):
            data = data.long().transpose(0, 1).contiguous()
            data = data.to(device)
            optimizer.zero_grad()

            input, target = Variable(data[:-1, :]), Variable(data[1:, :])
            output, _ = modle(input)
            loss = criterion(output, target.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            period.append(i + epoch * len(dataloader))
            loss2.append(loss_meter.value()[0])

        torch.save(modle.state_dict(), '.../model_poet.pth')
        plt.plot(period, loss2)
        plt.show()

def generate(model, start_words, ix2word, word2ix):
    txt = []
    for word in start_words:
        txt.append(word)
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    input = input.to(device)
    hidden = None
    num = len(txt)
    for i in range(48):
        output, hidden = model(input, hidden)
        if i < num:
            w = txt[i]
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            top_index = output.data[0].topk(1)[1][0]
            w = ix2word[top_index.item()]
            txt.append(w)
            input = Variable(input.data.new([top_index])).view(1, 1)
        if w == '<EOP>':
            break
    return ''.join(txt)

def gen_acrostic(model, start_words, ix2word, word2ix):
    result = []
    txt = []
    for word in start_words:
        txt.append(word)
    input = Variable(torch.Tensor([word2ix['<START>']]).view(1, 1).long())
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    input = input.to(device)
    hidden = None

    num = len(txt)
    index = 0
    pre_word = '<START>'
    for i in range(48):
        output, hidden = model(input, hidden)
        top_index = output.data[0].topk(1)[1][0]
        w = ix2word[top_index.item()]

        if (pre_word in {'。', '!', '<START>'}):
            if index == num:
                break
            else:
                w = txt[index]
                index += 1
                input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        else:
            input = Variable(input.data.new([word2ix[w]])).view(1, 1)
        result.append(w)
        pre_word = w
    return ''.join(result)

def test():
    modle = Net(len(word2ix), 128, 256)
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')
    modle.to(device)
    modle.load_state_dict(torch.load('.../model_poet.pth'))
    modle.eval()
    txt = generate(modle, '春>春來無伴閑遊少', ix2word, word2ix)
    print(txt)
    txt = gen_acrostic(modle, '草風', ix2word, word2ix)
    print(txt)


if __name__ == '__main__':
    data_path = '.../new.npz'
    datas = np.load(data_path, allow_pickle=True)
    data = datas['data']
    word2ix = datas['word2ix'].item()
    ix2word = datas['ix2word'].item()
    data = torch.from_numpy(data)
    dataloader = torch.utils.data.DataLoader(data, batch_size=10, shuffle=True, num_workers=1)
    train()
    test()
