import torch
import d2l
import pandas as pd
import multiprocessing
from torch import nn
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def read_sORF_data(test_file):

    test_datas = pd.read_csv(test_file, usecols=['seqs'])
    test_seqs = test_datas['seqs'].tolist()


    return (test_seqs)


def tokenize_3mer(sORF_seqs):
    tokenuzed_sORF_seqs = []

    for line in sORF_seqs:
        temp = []
        for i in range(0, len(line), 3):
            temp.append(line[i:i + 3])

        tokenuzed_sORF_seqs.append(temp)
    return tokenuzed_sORF_seqs

class BERTClassifier_notextCNN(nn.Module):
    def __init__(self, lscppbert):
        super(BERTClassifier_notextCNN, self).__init__()
        self.encoder = lscppbert.encoder
        self.hidden = lscppbert.hidden
        self.output = nn.Linear(512, 2)
        self.dropout = nn.Dropout(0.3)

    def forward(self, inputs):
        tokens_X, segments_X, valid_len_x = inputs
        encoded_X = self.encoder(tokens_X, segments_X, valid_len_x)
        out = torch.sigmoid(self.output(self.hidden(encoded_X[:, 0, :])))
        return out


class sORFLSCPPBERTDataset3mer(torch.utils.data.Dataset):
    def __init__(self, dataset, max_len, vocab=None):
        sORF_seqs = dataset
        sORF_seqs = tokenize_3mer(sORF_seqs)
        self.vocab = vocab
        self.max_len = max_len
        (self.all_token_ids, self.all_segments,
         self.valid_lens) = self._preprocess(sORF_seqs)
        print('read ' + str(len(self.all_token_ids)) + ' examples')

    def _preprocess(self, all_tokens):
        pool = multiprocessing.Pool(4)
        out = pool.map(self._mp_worker, all_tokens)
        all_token_ids = [token_ids for token_ids, segments, valid_len in out]
        all_segments = [segments for token_ids, segments, valid_len in out]
        valid_lens = [valid_len for token_ids, segments, valid_len in out]
        return (torch.tensor(all_token_ids, dtype=torch.long),
                torch.tensor(all_segments, dtype=torch.long),
                torch.tensor(valid_lens))

    def _mp_worker(self, tokens):
        self._truncate_pair_of_tokens(tokens)
        tokens, segments = d2l.get_tokens_and_segments(tokens)
        token_ids = self.vocab[tokens] + [self.vocab['<pad>']] * (self.max_len - len(tokens))
        segments = segments + [0] * (self.max_len - len(segments))
        valid_len = len(tokens)
        return token_ids, segments, valid_len

    def _truncate_pair_of_tokens(self, tokens):
        while len(tokens) > self.max_len - 3:
            tokens.pop()

    def __getitem__(self, idx):
        return (self.all_token_ids[idx], self.all_segments[idx],
                self.valid_lens[idx])

    def __len__(self):
        return len(self.all_token_ids)


batch_size, max_len, num_workers = 32, 112, 0
test_file = 'test.csv'
device = d2l.try_gpu()
_, vocab = d2l.load_pretraining_data(batch_size, max_len)

net = torch.load('model/LSCPP_BERT.bin')
max_len = 112


test_set = read_sORF_data(test_file)
test_iter = sORFLSCPPBERTDataset3mer(test_set, max_len, vocab)
test_iters = torch.utils.data.DataLoader(test_iter, batch_size,
                                             num_workers=0, shuffle=False)

net.to(device)

prediction_results = []

if isinstance(net, nn.Module):
    net.eval()

    if not device:
        device = next(iter(net.parameters())).device
    with torch.no_grad():
        for X in test_iters:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y_hat = net(X)
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = y_hat.argmax(axis=1)
            prediction_results += y_hat.cpu().numpy().tolist()

print(prediction_results)











