from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch
import copy


class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, freezeLM=True, **kwargs): # 什么高级写法，cls是什么，super（）是什么
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem)
        return model

    def init_prompt(self, nuser, nitem):
        self.src_len = 2 # 两个look up 表, 两个位置u i 放在开头的 对应论文fig.4 
        emsize = self.transformer.wte.weight.size(1)  # 768维度，soft embedding 的维度,相当于LLM中要用的固定维度大小. 用model.transformer.wte可以看出
        self.user_embeddings = nn.Embedding(nuser, emsize) # 输出nuser个embedding，每个embedding维度为768 ; 相当于定义查找表
        self.item_embeddings = nn.Embedding(nitem, emsize) 

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange) # 初始化权重矩阵W取值-0.1到0.1的均匀分布
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, ignore_index=-100):
        device = user.device
        batch_size = user.size(0) # batch size,一批输入多少个samples

        # embeddings 输入的{'user': 0, 'item': 0, 'rating': 5, 'text': 'or high quality skirt', 'feature': 'skirt'}
        u_src = self.user_embeddings(user)  # 输出(batch_size, emsize); 相当于这里带入user id 查表,找对应embedding向量
        i_src = self.item_embeddings(item)  # (batch_size, emsize); wpe 是 word positional embedding
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize) wte相当于词向量的东西输入词id进去得到词向量 word token embedding;
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  
        # 首先对u和i增加一个维度unsqueeze(1),才能输出(batch_size, total_len, emsize) 对total_len这个维度拼接; 不增加一个维度,无法拼接total_len

        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device) #ones,每个sample 增加两列 ; 矩阵传入gpu;self.src_len = 2
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len) #在之前句子长度上增加u和i的mask遮掩

            # prediction for training; pred_left就是创建值全为-100的128*2的矩阵, 这里左右拼接,左边为u和i两列,右边为text id,
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device) #(batch_size, src_len); ignore_index=-100
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device)) # replace <pad> with ignore_index, mask取1的位置就是text句子的位置;#转变为tensor([50257,   393,  1029,  3081, 23967,   220, 50258,  -100,  -100,  -100,-100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,-100,  -100,  -100,  -100])
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction) # 标签就是这个prediction拼接的;
            #这里forward调用GPT2LMHeadModel的参数attention_mask ,inputs_embeds ,labels这些!


class ContinuousPromptLearning(UIPrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class FeaturePrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        return super().from_pretrained(pretrained_model_name_or_path, **kwargs)

    def forward(self, context, explanation, exp_mask, ignore_index=-100):
        device = context.device
        text = torch.cat([context, explanation], 1)  # (batch_size, total_len)
        src = self.transformer.wte(text)  # (batch_size, total_len, emsize)

        if exp_mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src)
        else:
            # training
            # input padding
            pad_left = torch.ones_like(context, dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, exp_mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full_like(context, ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(exp_mask == 1, explanation, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction)


class DiscretePromptLearning(FeaturePrompt, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)


class MF(nn.Module):
    def __init__(self):
        super(MF, self).__init__()

    def forward(self, user, item):  # (batch_size, emsize)
        rating = torch.sum(user * item, 1)  # (batch_size,)
        return rating


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class MLP(nn.Module):
    def __init__(self, emsize, hidden_size=400, num_layers=2):
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(emsize * 2, hidden_size)
        self.last_layer = nn.Linear(hidden_size, 1)
        layer = nn.Linear(hidden_size, hidden_size)
        self.layers = _get_clones(layer, num_layers)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def init_weights(self):
        initrange = 0.1
        self.first_layer.weight.data.uniform_(-initrange, initrange)
        self.first_layer.bias.data.zero_()
        self.last_layer.weight.data.uniform_(-initrange, initrange)
        self.last_layer.bias.data.zero_()
        for layer in self.layers:
            layer.weight.data.uniform_(-initrange, initrange)
            layer.bias.data.zero_()

    def forward(self, user, item):  # (batch_size, emsize)
        ui_cat = torch.cat([user, item], 1)  # (batch_size, emsize * 2)
        hidden = self.sigmoid(self.first_layer(ui_cat))  # (batch_size, hidden_size)
        for layer in self.layers:
            hidden = self.sigmoid(layer(hidden))  # (batch_size, hidden_size)
        rating = torch.squeeze(self.last_layer(hidden))  # (batch_size,)
        return rating


class UIPromptWithReg:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, nuser, nitem, use_mf=True, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        model.init_prompt(nuser, nitem, use_mf)
        return model

    def init_prompt(self, nuser, nitem, use_mf):
        self.src_len = 2
        emsize = self.transformer.wte.weight.size(1)  # 768
        self.user_embeddings = nn.Embedding(nuser, emsize)
        self.item_embeddings = nn.Embedding(nitem, emsize)
        if use_mf:
            self.rec = MF()
        else:
            self.rec = MLP(emsize)

        initrange = 0.1
        self.user_embeddings.weight.data.uniform_(-initrange, initrange)
        self.item_embeddings.weight.data.uniform_(-initrange, initrange)

    def forward(self, user, item, text, mask, rating_prediction=True, ignore_index=-100):
        device = user.device
        batch_size = user.size(0)

        # embeddings
        u_src = self.user_embeddings(user)  # (batch_size, emsize)
        i_src = self.item_embeddings(item)  # (batch_size, emsize)
        w_src = self.transformer.wte(text)  # (batch_size, tgt_len, emsize)
        src = torch.cat([u_src.unsqueeze(1), i_src.unsqueeze(1), w_src], 1)  # (batch_size, total_len, emsize)

        if rating_prediction:
            rating = self.rec(u_src, i_src)  # (batch_size,)
        else:
            rating = None
        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src), rating
        else:
            # training
            # input padding
            pad_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            pad_input = torch.cat([pad_left, mask], 1)  # (batch_size, total_len)

            # prediction for training
            pred_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)  # (batch_size, src_len)
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device))  # replace <pad> with ignore_index
            prediction = torch.cat([pred_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction), rating


class RecReg(UIPromptWithReg, GPT2LMHeadModel):
    def __init__(self, config):
        super().__init__(config)
