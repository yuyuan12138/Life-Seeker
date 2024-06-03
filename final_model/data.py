import torch
import torch.utils.data as Data
import pandas as pd
from config import Config
config = Config()

def get_dataloader(train_path='train.tsv', test_path='test.tsv') -> list:
    train_data = pd.read_csv(train_path, sep='\t')
    train_dataset = MyDataset(train_data['text'], train_data['label'])
    train_loader = Data.DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)

    test_data = pd.read_csv(test_path, sep='\t')
    test_dataset = MyDataset(test_data['text'], test_data['label'])
    test_loader = Data.DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    return (train_loader, test_loader)

class MyDataset(Data.Dataset):
    def process_features(self, features):
        dna_map = {'A': 0, 'T': 1, 'C': 2, 'G': 3}
        encoding = torch.zeros((len(features), len(features[0]), 4), dtype=torch.float, device=config.device)
        for (i, seq) in enumerate(features):
            for (j, char) in enumerate(seq):
                encoding[i, j, dna_map[char]] = 1
        return encoding

    def process_data(self, features, labels):
        features = self.process_features(features)
        labels = torch.tensor(labels, device=config.device, dtype=torch.long)
        return features, labels

    def __init__(self, features:list, labels:list) -> None:
        super().__init__()
        self.features, self.labels = self.process_data(features, labels)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, index):
        return (self.features[index], self.labels[index])