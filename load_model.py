import torch
import pandas as pd
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import LogsDataset
from transformers import (set_seed,
                          GPT2Config,
                          GPT2Tokenizer,
                          GPT2ForSequenceClassification)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='log')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--maxlen', type=int, default=60)
    parser.add_argument('--batch', type=int, default=32)
    args = parser.parse_args()
    return args

args = parse_args()
set_seed(args.seed)
max_length = args.maxlen
batch_size = args.batch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model_path = 'model/gpt2'


if args.dataset=='log':
   labels_ids = {'Normal': 0, 'Anomalous': 1}
elif args.dataset=='big_log':
   labels_ids = {'Normal': 0, 'Anomalous': 1}

n_labels = len(labels_ids)

class Gpt2ClassificationCollator(object):


    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):
        texts = [sequence['log'] for sequence in sequences]

        if 'label' not in sequences[0].keys():
            labels = ['Normal' for sequence in sequences]
        else:
            labels = [sequence['label'] for sequence in sequences]
        labels = [self.labels_encoder[label] for label in labels]
        inputs = self.use_tokenizer(text=texts, return_tensors="pt", padding=True, truncation=True,  max_length=self.max_sequence_len)
        inputs.update({'labels':torch.tensor(labels)})

        return inputs


print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(pretrained_model_name_or_path=model_path, num_labels=n_labels)

print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(pretrained_model_name_or_path=model_path)
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token


print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(pretrained_model_name_or_path=model_path, config=model_config)
model.resize_token_embeddings(len(tokenizer))
model.config.pad_token_id = model.config.eos_token_id

model.to(device)
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer, 
                                                          labels_encoder=labels_ids, 
                                                          max_sequence_len=max_length)
# 加载保存的模型参数
model.load_state_dict(torch.load('best_one.pth'))

test_dataset = LogsDataset(path='dataset/data-Copy1-processed.json', 
                               use_tokenizer=tokenizer)
# dataset = LogsDataset(path='dataset/logfile.json', 
#                                use_tokenizer=tokenizer)
# test_size = (int)(0.005*len(dataset))
# _,test_dataset = torch.utils.data.random_split(
#     dataset, [len(dataset)-test_size,test_size])

test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
print('Created `test_dataset` with %d examples'%len(test_dataset))
print('Created `test_dataloader` with %d batches'%len(test_dataloader))


def test(model, dataloader,device_='cuda'):
    model.eval()

    predictions_labels = []

    for batch in tqdm(dataloader, total=len(dataloader)):

        batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
        with torch.no_grad():        
            outputs = model(**batch)
            loss, logits = outputs[:2]
            logits = logits.detach().cpu().numpy()
            predict_content = logits.argmax(axis=-1).flatten().tolist()
            predictions_labels += predict_content

    return predictions_labels

# 使用test函数进行预测
labels=test(model, test_dataloader)
df = pd.DataFrame(labels, columns=["label"])
# df["index"] = df.reset_index().index
# df = df[["index", "label"]]

# 替换标签为 "Normal" 和 "Anomalous"
df["label"] = df["label"].replace({0: "Normal", 1: "Anomalous"})
print(df)
df.to_csv("predictions1.csv", index=True)
