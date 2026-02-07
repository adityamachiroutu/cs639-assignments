import torch
import torch.nn as nn
import zipfile
import numpy as np

class BaseModel(nn.Module):
    def __init__(self, args, vocab, tag_size):
        super(BaseModel, self).__init__()
        self.args = args
        self.vocab = vocab
        self.tag_size = tag_size

    def save(self, path):
        # Save model
        print(f'Saving model to {path}')
        ckpt = {
            'args': self.args,
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }
        torch.save(ckpt, path)

    def load(self, path):
        # Load model
        print(f'Loading model from {path}')
        ckpt = torch.load(path, weights_only=False)
        self.vocab = ckpt['vocab']
        self.args = ckpt['args']
        self.load_state_dict(ckpt['state_dict'])


def load_embedding(vocab, emb_file, emb_size):
    """
    Read embeddings for words in the vocabulary from the emb_file (e.g., GloVe, FastText).
    Args:
        vocab: (Vocab), a word vocabulary
        emb_file: (string), the path to the embdding file for loading
        emb_size: (int), the embedding size (e.g., 300, 100) depending on emb_file
    Return:
        emb: (np.array), embedding matrix of size (|vocab|, emb_size) 
    """
    vocab_size = len(vocab)
    init_range = 0.08
    emb = np.random.uniform(-init_range, init_range, size=(vocab_size, emb_size)).astype(np.float32)

    if getattr(vocab, "pad_id", None) is not None:
        emb[vocab.pad_id] = 0.0

    def _iter_lines_from_file(fh):
        for raw in fh:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8", errors="ignore")
            yield raw

    if emb_file.endswith(".zip"):
        with zipfile.ZipFile(emb_file) as zf:
            name = None
            for candidate in zf.namelist():
                if candidate.endswith(".txt") or candidate.endswith(".vec"):
                    name = candidate
                    break
            if name is None:
                name = zf.namelist()[0]
            with zf.open(name) as fh:
                lines = _iter_lines_from_file(fh)
                for line in lines:
                    parts = line.rstrip().split()
                    if len(parts) <= 2:
                        continue
                    if len(parts) != emb_size + 1:
                        continue
                    word = parts[0]
                    if word in vocab:
                        emb[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)
    else:
        with open(emb_file, "r", encoding="utf-8") as fh:
            for line in fh:
                parts = line.rstrip().split()
                if len(parts) <= 2:
                    continue
                if len(parts) != emb_size + 1:
                    continue
                word = parts[0]
                if word in vocab:
                    emb[vocab[word]] = np.asarray(parts[1:], dtype=np.float32)

    return emb


class DanModel(BaseModel):
    def __init__(self, args, vocab, tag_size):
        super(DanModel, self).__init__(args, vocab, tag_size)
        self.define_model_parameters()
        self.init_model_parameters()

        # Use pre-trained word embeddings if emb_file exists
        if args.emb_file is not None:
            self.copy_embedding_from_numpy()

    def define_model_parameters(self):
        """
        Define the model's parameters, e.g., embedding layer, feedforward layer.
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        self.pad_id = self.vocab.word2id.get("<pad>", None)
        self.unk_id = self.vocab.word2id.get("<unk>", None)

        self.embedding = nn.Embedding(len(self.vocab), self.args.emb_size, padding_idx=self.pad_id)
        self.emb_dropout = nn.Dropout(self.args.emb_drop)
        self.hid_dropout = nn.Dropout(self.args.hid_drop)
        self.activation = nn.ReLU()

        self.hid_layers = nn.ModuleList()
        if self.args.hid_layer > 0:
            for i in range(self.args.hid_layer):
                in_size = self.args.emb_size if i == 0 else self.args.hid_size
                self.hid_layers.append(nn.Linear(in_size, self.args.hid_size))
            self.output_layer = nn.Linear(self.args.hid_size, self.tag_size)
        else:
            self.output_layer = nn.Linear(self.args.emb_size, self.tag_size)

    def init_model_parameters(self):
        """
        Initialize the model's parameters by uniform sampling from a range [-v, v], e.g., v=0.08
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        init_range = 0.08
        for name, param in self.named_parameters():
            if param.dim() > 1:
                nn.init.uniform_(param, -init_range, init_range)
            else:
                nn.init.uniform_(param, -init_range, init_range)

        if self.pad_id is not None:
            with torch.no_grad():
                self.embedding.weight[self.pad_id].fill_(0.0)

    def copy_embedding_from_numpy(self):
        """
        Load pre-trained word embeddings from numpy.array to nn.embedding
        Pass hyperparameters explicitly or use self.args to access the hyperparameters.
        """
        emb = load_embedding(self.vocab, self.args.emb_file, self.args.emb_size)
        with torch.no_grad():
            self.embedding.weight.copy_(torch.from_numpy(emb))

    def forward(self, x):
        """
        Compute the unnormalized scores for P(Y|X) before the softmax function.
        E.g., feature: h = f(x)
              scores: scores = w * h + b
              P(Y|X) = softmax(scores)  
        Args:
            x: (torch.LongTensor), [batch_size, seq_length]
        Return:
            scores: (torch.FloatTensor), [batch_size, ntags]
        """
        if self.training and self.args.word_drop > 0 and self.unk_id is not None:
            drop_mask = (torch.rand_like(x.float()) < self.args.word_drop)
            if self.pad_id is not None:
                drop_mask = drop_mask & (x != self.pad_id)
            x = torch.where(drop_mask, torch.full_like(x, self.unk_id), x)

        emb = self.embedding(x)
        emb = self.emb_dropout(emb)

        if self.pad_id is not None:
            mask = (x != self.pad_id).float()
        else:
            mask = torch.ones_like(x, dtype=torch.float32)

        if self.args.pooling_method == "sum":
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1)
        elif self.args.pooling_method == "max":
            masked = emb.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
            pooled = masked.max(dim=1).values
        else:
            pooled = (emb * mask.unsqueeze(-1)).sum(dim=1)
            lengths = mask.sum(dim=1).clamp(min=1.0)
            pooled = pooled / lengths.unsqueeze(-1)

        h = pooled
        for layer in self.hid_layers:
            h = layer(h)
            h = self.activation(h)
            h = self.hid_dropout(h)

        scores = self.output_layer(h)
        return scores
