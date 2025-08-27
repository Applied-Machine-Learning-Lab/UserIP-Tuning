import os
import math
import torch
import argparse
from transformers import LlamaTokenizer, AdamW
from module4 import ContinuousPromptLearning # 这个module3加上codebook代码
from utils3 import rouge_score, bleu_score, DataLoader0, Batchify, now_time, ids2tokens, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity # 这里utils3适用yelp数据集
# DataLoader0适用于amazon-MoviesAndTV




parser = argparse.ArgumentParser(description='Prompt Tuning as User Inherent Profile Inference Machine (UserIP-Tuning)')
parser.add_argument('--data_path', type=str, default='./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/reviews.pickle',
                    help='path for loading the pickle data') # 只用reviews.pickle文件的内容。没有用user.json; item.json

parser.add_argument('--data_path_price_brand', type=str, default='./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/item.json',
                    help='path for loading the json data for price and brand') # 只用item.json文件的内容。没有用user.json; reviews.pickle
# parser.add_argument('--data_path_user', type=str, default='./CIKM20-NETE-Datasets/Yelp/user.json',
#                     help='path for loading the json data for user information') # 只用item.json文件的内容。没有用user.json; reviews.pickle

parser.add_argument('--index_dir', type=str, default='./CIKM20-NETE-Datasets/Amazon/MoviesAndTV/3', # 3 index可以改变
                    help='load indexes')
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')

parser.add_argument('--codebook_hobby_k', type=int, default=4,
                    help='codebook_hobby_k')
parser.add_argument('--codebook_background_k', type=int, default=3,
                    help='codebook_background_k')

parser.add_argument('--l_w_hobby', type=float, default=0.5,
                    help='the weight in the total loss for hobby codebook embedding')
parser.add_argument('--l_w_background', type=float, default=0.5,
                    help='the weight in the total loss for background codebook embedding')

parser.add_argument('--epochs', type=int, default=1,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch size')
parser.add_argument('--cuda', action='store_true', default='True',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=15,  # 表示每条用户review句子，取的最大长度，loaddata中会用到;这个words增多会使损失loss变小很多
                    help='number of words to generate for each sample')

args = parser.parse_args(args=[])


if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)

model_path = os.path.join(args.checkpoint, 'model_MoviesAndTV.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)


print(now_time() + 'Loading data')
bos = '<bos>' # 特殊字符串
eos = '<eos>'
pad = '<pad>'

tokenizer = LlamaTokenizer.from_pretrained('/data/ggghhh/luyusheng/huggingface/transformers/Llama-2-7b-hf/tokenizer.model',bos_token=bos, eos_token=eos, pad_token=pad,local_files_only=True)


corpus = DataLoader0(args.data_path, args.data_path_price_brand, args.index_dir, tokenizer, args.words) # 读取数据，加载划分训练集，验证集，测试集，用户特征user2feature，产品特征item2feature


nuser = len(corpus.user_dict) # 自定义的用户index集合 ，用户个数
nitem = len(corpus.item_dict) # 产品个数
print(nuser)
print(nitem)


feature_set = corpus.feature_set # 所有的用户，产品的特征集合，这个在训练的时候没有用上，在评估时用
train_data = Batchify(corpus.train, tokenizer, bos, eos, args.batch_size, shuffle=True) # 将句子的开头结尾特殊字符token加入，bos = '<bos>'；eos = '<eos>'
val_data = Batchify(corpus.valid, tokenizer, bos, eos, args.batch_size) #
test_data = Batchify(corpus.test, tokenizer, bos, eos, args.batch_size)


###############################################################################
# Build the model
###############################################################################

nuser = len(corpus.user_dict) # 自定义的用户index集合 ，用户个数
nitem = len(corpus.item_dict) # 产品个数
ntoken = len(tokenizer) # 加上bos，eos，pad的token : 50257 +1+1+1 = 50260
model = ContinuousPromptLearning.from_pretrained('/data/ggghhh/luyusheng/huggingface/transformers/Llama-2-7b-hf/',args.codebook_hobby_k, args.codebook_background_k, nuser, nitem) # ！！！自定义class继承了GPT2LMHeadModel加上user item 的soft prompt，这里默认freeze住llm的所有参数;
# ContinuousPromptLearning中且定义了模型标签labels
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table: 应该说的<bos>  <eos>  pad这三个在LLM中
model.to(device) # model传入gpu
optimizer = AdamW(model.parameters(), lr=args.lr)


###############################################################################
# Training code
###############################################################################


def train(data, l_w_hobby=0.5, l_w_background=0.5):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    mse_loss = torch.nn.MSELoss()
    while True:
        user, item, _, seq, mask, seq_price, mask_price, seq_brand, mask_brand = data.next_batch()  # data.step += 1
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)

        seq_price = seq_price.to(device)  # (batch_size, seq_len)
        mask_price = mask_price.to(device)
        seq_brand = seq_brand.to(device)  # (batch_size, seq_len)
        mask_brand = mask_brand.to(device)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        # model()输出的最后两个变量_,_是需要传入DCN推荐模型 两个新特征的codebook最近邻index： nearest_neighbor_hobby, nearest_neighbor_background
        optimizer.zero_grad()
        outputs, zq_hobby, embeded_hobby, zq_background, embeded_background, _, _ = model(user, item, seq, mask,
                                                                                          seq_price, mask_price,
                                                                                          seq_brand, mask_brand)  # 模型输入
        loss = outputs.loss + l_w_hobby * mse_loss(embeded_hobby.detach(), zq_hobby) + l_w_background * mse_loss(
            embeded_background.detach(), zq_background)
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | {:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step,
                                                                               data.total_step))
            text_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break


def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, _, seq, mask, seq_price, mask_price, seq_brand, mask_brand = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)

            seq_price = seq_price.to(device)
            mask_price = mask_price.to(device)
            seq_brand = seq_brand.to(device)
            mask_brand = mask_brand.to(device)

            outputs, _, _, _, _, _, _ = model(user, item, seq, mask, seq_price, mask_price, seq_brand, mask_brand)
            loss = outputs.loss

            batch_size = user.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq, _, seq_price, _, seq_brand, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq_price = seq_price.to(device)  # 上面生成的新的seq_price传入gpu，不然报错
            seq_brand = seq_brand.to(device)  # 新的seq_brand传入gpu，不然报错
            text = seq[:, :1].to(device)  # bos, (batch_size, 1) 以下还没改
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs, _, _, _, _, _, _ = model(user, item, text, None, seq_price, None, seq_brand,
                                                  None)  # 这里是模型.forward输出
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1,
                                     keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict




print(now_time() + 'Tuning Prompt Only')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data, args.l_w_hobby, args.l_w_background)
    val_loss = evaluate(val_data) # 这里就有问题！evaluate函数
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss)) # math.exp(val_loss) loss比较大
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


# print(now_time() + 'Tuning both Prompt and LM')
# for param in model.parameters():
#     param.requires_grad = True # 这里让LLM的参数可以更新训练，不再frozen
# optimizer = AdamW(model.parameters(), lr=args.lr)

# # Loop over epochs.
# best_val_loss = float('inf')
# endure_count = 0
# for epoch in range(1, args.epochs + 1):
#     print(now_time() + 'epoch {}'.format(epoch)) # 下面的train（）函数就要报错！！！Prompt和LM一起训练memory有问题
#     train(train_data)
#     val_loss = evaluate(val_data)
#     print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
#     # Save the model if the validation loss is the best we've seen so far.
#     if val_loss < best_val_loss:
#         best_val_loss = val_loss
#         with open(model_path, 'wb') as f:
#             torch.save(model, f)
#     else:
#         endure_count += 1
#         print(now_time() + 'Endured {} time(s)'.format(endure_count))
#         if endure_count == args.endure_times:
#             print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
#             break

# # Load the best saved model.
# with open(model_path, 'rb') as f:
#     model = torch.load(f).to(device)


