import torch
import argparse
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataset import LogsDataset
from sklearn.metrics import classification_report, accuracy_score
from transformers import (set_seed,
                          TrainingArguments,
                          Trainer,
                          GPT2Config,
                          GPT2Tokenizer,
                          AdamW,
                          get_linear_schedule_with_warmup,
                          GPT2ForSequenceClassification)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='log')
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--maxlen', type=int, default=60)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--eps', type=float, default=1e-8)

    args = parser.parse_args()
    return args


args = parse_args()
set_seed(args.seed)
epochs = args.epoch
batch_size = args.batch
max_length = args.maxlen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_path = 'model/gpt2'

if args.dataset == 'log':
    labels_ids = {'Normal': 0, 'Anomalous': 1}
else:
    labels_ids = {'Normal': 0, 'Anomalous': 1}

n_labels = 2


class Gpt2ClassificationCollator(object):

    def __init__(self, use_tokenizer, labels_encoder, max_sequence_len=None):

        self.use_tokenizer = use_tokenizer
        self.max_sequence_len = use_tokenizer.model_max_length if max_sequence_len is None else max_sequence_len
        self.labels_encoder = labels_encoder

        return

    def __call__(self, sequences):

        texts = [sequence['log'] for sequence in sequences]
        labels = [sequence['label'] for sequence in sequences]

        # 核心：将文本的label处理为0与1
        labels = [self.labels_encoder[label] for label in labels]
        # 核心：使用tokenizer将原来的文本编码成数字
        inputs = self.use_tokenizer(text=texts, return_tensors="pt",
                                    padding=True, truncation=True,  max_length=self.max_sequence_len)
        # Update the inputs with the associated encoded labels as tensor.
        inputs.update({'labels': torch.tensor(labels)})

        return inputs


def train(dataloader, optimizer_, scheduler_, device_):
    global model

    predictions_labels = []
    true_labels = []

    total_loss = 0

    model.train()

    for batch in tqdm(dataloader, total=len(dataloader)):

        # Add original labels - use later for evaluation.
        true_labels += batch['labels'].numpy().flatten().tolist()

        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        model.zero_grad()

        outputs = model(**batch)

        loss, logits = outputs[:2]

        total_loss += loss.item()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        logits = logits.detach().cpu().numpy()

        predictions_labels += logits.argmax(axis=-1).flatten().tolist()

    avg_epoch_loss = total_loss / len(dataloader)
    return true_labels, predictions_labels, avg_epoch_loss


def validation(dataloader, device_):
   
    global model

    predictions_labels = []
    true_labels = []
    total_loss = 0

    model.eval()

    for batch in tqdm(dataloader, total=len(dataloader)):

        true_labels += batch['labels'].numpy().flatten().tolist()
        batch = {k: v.type(torch.long).to(device_) for k, v in batch.items()}

        with torch.no_grad():

            outputs = model(**batch)


            loss, logits = outputs[:2]

            logits = logits.detach().cpu().numpy()


            total_loss += loss.item()

            predict_content = logits.argmax(axis=-1).flatten().tolist()

            predictions_labels += predict_content

    avg_epoch_loss = total_loss / len(dataloader)

    return true_labels, predictions_labels, avg_epoch_loss


print('Loading configuraiton...')
model_config = GPT2Config.from_pretrained(
    pretrained_model_name_or_path=model_path, num_labels=n_labels)
print('Loading tokenizer...')
tokenizer = GPT2Tokenizer.from_pretrained(
    pretrained_model_name_or_path=model_path)
# default to left padding
tokenizer.padding_side = "left"
# Define PAD Token = EOS Token = 50256
tokenizer.pad_token = tokenizer.eos_token



print('Loading model...')
model = GPT2ForSequenceClassification.from_pretrained(
    pretrained_model_name_or_path=model_path, config=model_config)
# resize model embedding to match new tokenizer
model.resize_token_embeddings(len(tokenizer))
# fix model padding token id
model.config.pad_token_id = model.config.eos_token_id

model.to(device)
print('Model loaded to `%s`' % device)
# Create data collator to encode text and labels into numbers.
gpt2_classificaiton_collator = Gpt2ClassificationCollator(use_tokenizer=tokenizer,
                                                          labels_encoder=labels_ids,
                                                          max_sequence_len=max_length)


if args.dataset == 'log':
    dataset = LogsDataset(path='dataset/bgl_logs_train.json',
                               use_tokenizer=tokenizer)
    # train_ratio = 0.8
    # valid_ratio = 0.2
    # # test_ratio = 0.2

    # # 因为int会舍去小数，有时会导致划分后的样本数少掉，手动加1可以解决
    # train_size = int(train_ratio * len(dataset))
    # valid_size = int(valid_ratio * len(dataset))
    # # test_size = int(test_ratio * len(dataset))+1

    # train_dataset, valid_dataset = torch.utils.data.random_split(
    #     dataset, [train_size, valid_size])
    train_dataset = dataset

    print('Created `train_dataset` with %d examples' % len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches' % len(train_dataloader))

    print()

    # print('Created `valid_dataset` with %d examples' % len(valid_dataset))
    # valid_dataloader = DataLoader(
    #     valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    # print('Created `eval_dataloader` with %d batches' % len(valid_dataloader))


elif args.dataset == '2log':

    train_dataset = LogsDataset(path='dataset/train.json',
                                use_tokenizer=tokenizer)
    print('Created `train_dataset` with %d examples' % len(train_dataset))
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches' % len(train_dataloader))

    print()

    valid_dataset = LogsDataset(path='dataset/test.json',
                                use_tokenizer=tokenizer)
    print('Created `valid_dataset` with %d examples' % len(valid_dataset))
    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches' % len(valid_dataloader))
else:
    dataset = LogsDataset(path='dataset/logfile.json',
                               use_tokenizer=tokenizer)
    train_ratio = 0.8
    valid_ratio = 0.2

    train_size = int(train_ratio * len(dataset))+1
    valid_size = int(valid_ratio * len(dataset))

    train_dataset, valid_dataset = torch.utils.data.random_split(
        dataset, [train_size, valid_size])

    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=gpt2_classificaiton_collator)
    print('Created `train_dataloader` with %d batches' % len(train_dataloader))

    print()

    print('Created `valid_dataset` with %d examples' % len(valid_dataset))

    valid_dataloader = DataLoader(
        valid_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `eval_dataloader` with %d batches' % len(valid_dataloader))

    test_dataset = LogsDataset(path='dataset/bgl_logs_train.json',
                               use_tokenizer=tokenizer)
    print('Created `test_dataset` with %d examples' % len(test_dataset))
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size,
                                 shuffle=False, collate_fn=gpt2_classificaiton_collator)
    print('Created `test_dataloader` with %d batches' % len(valid_dataloader))

optimizer = AdamW(model.parameters(),
                  lr=args.lr,
                  eps=args.eps
                  )


total_steps = len(train_dataloader) * epochs
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps=0,
                                            num_training_steps=total_steps)

all_loss = {'train_loss': [], 'val_loss': []}
all_acc = {'train_acc': [], 'val_acc': []}

# Loop through each epoch.
print('Epoch')
for epoch in tqdm(range(epochs)):
    print()
    print('Training on batches...')

    train_labels, train_predict, train_loss = train(
        train_dataloader, optimizer, scheduler, device)
    train_acc = accuracy_score(train_labels, train_predict)

    # print('Validation on batches...')
    # valid_labels, valid_predict, val_loss = validation(
    #     valid_dataloader, device)
    # val_acc = accuracy_score(valid_labels, valid_predict)

    # print("  train_loss: %.5f - val_loss: %.5f - train_acc: %.5f - valid_acc: %.5f" %
    #       (train_loss, val_loss, train_acc, val_acc))
    # print()
    print("  train_loss: %.5f - train_acc: %.5f" %
          (train_loss, train_acc))
    print()
    all_loss['train_loss'].append(train_loss)
    # all_loss['val_loss'].append(val_loss)
    all_acc['train_acc'].append(train_acc)
    # all_acc['val_acc'].append(val_acc)

torch.save(model.state_dict(), 'best_small_one.pth')
print("saved")

train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
true_labels, predictions_labels, avg_epoch_loss = validation(
    train_dataloader, device)


evaluation_report = classification_report(true_labels, predictions_labels, labels=list(
    labels_ids.values()), target_names=list(labels_ids.keys()))
print(evaluation_report)

# true_labels, predictions_labels, avg_epoch_loss = validation(
#     valid_dataloader, device)


# evaluation_report = classification_report(true_labels, predictions_labels, labels=list(
#     labels_ids.values()), target_names=list(labels_ids.keys()))

# print(evaluation_report)

# def test(model, dataloader,device_='cuda'):
#     model.eval()

#     predictions_labels = []

#     for batch in tqdm(dataloader, total=len(dataloader)):

#         batch = {k:v.type(torch.long).to(device_) for k,v in batch.items()}
#         with torch.no_grad():        
#             outputs = model(**batch)
#             loss, logits = outputs[:2]
#             logits = logits.detach().cpu().numpy()
#             predict_content = logits.argmax(axis=-1).flatten().tolist()
#             predictions_labels += predict_content

#     return predictions_labels

# # 使用test函数进行预测
# test_dataset = LogsDataset(path='dataset/bgl_logs_train.json', 
#                                use_tokenizer=tokenizer)

# test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=gpt2_classificaiton_collator)
# true_labels, predictions_labels, avg_epoch_loss = validation(
#     test_dataloader, device)


# evaluation_report = classification_report(true_labels, predictions_labels, labels=list(
#     labels_ids.values()), target_names=list(labels_ids.keys()))

# print(evaluation_report)

# batchsize不能随便动
# labels=test(model, test_dataloader)
# df = pd.DataFrame(labels, columns=["label"])

# # 替换标签为 "Normal" 和 "Anomalous"
# df["label"] = df["label"].replace({0: "Normal", 1: "Anomalous"})
# df.to_csv("predictions.csv", index=True)
