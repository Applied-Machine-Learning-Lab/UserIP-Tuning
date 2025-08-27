from transformers import LlamaForCausalLM
#from transformers import GPT2LMHeadModel
import torch.nn as nn
import torch
import copy


class UIPrompt:
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, codebook_hobby_k, codebook_background_k, nuser, nitem, freezeLM=True, **kwargs): # 什么高级写法，cls是什么，super（）是什么
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)

        # freeze pretrained model parameters
        if freezeLM:
            for param in model.parameters():
                param.requires_grad = False

        model.init_prompt(nuser, nitem, codebook_hobby_k, codebook_background_k) # 调用下面函数
        return model

    def init_prompt(self, nuser, nitem, codebook_hobby_k, codebook_background_k):
        self.src_len = 1 # 换成一个，下面forward用两个left_pad拼接；两个look up 表, 两个位置u i 放在开头的 对应论文fig.4 
        emsize = self.transformer.wte.weight.size(1)  # 768维度，soft embedding 的维度,相当于LLM中要用的固定维度大小. 用model.transformer.wte可以看出
        #self.user_embeddings = nn.Embedding(nuser, emsize) # 输出nuser个embedding，每个embedding维度为768 ; 相当于定义查找表
        #self.item_embeddings = nn.Embedding(nitem, emsize) 
        self.user_embeddings_hobby = nn.Embedding(nuser, emsize) # 根据hobby即用户爱好品牌所学习的latent concept variable , θ1
        self.user_embeddings_background = nn.Embedding(nuser, emsize) # 根据price即 用户背景——消费价格所学习的latent concept variable, θ2
        
        initrange = 0.1
        self.user_embeddings_hobby.weight.data.uniform_(-initrange, initrange) # 初始化权重矩阵W取值-0.1到0.1的均匀分布
        self.user_embeddings_background.weight.data.uniform_(-initrange, initrange)

        # codebook layer
        self.vq_hobby_embedding = nn.Embedding(codebook_hobby_k, emsize) # codebook的向量维度768与LLM对应
        self.vq_background_embedding = nn.Embedding(codebook_background_k, emsize) # codebook的向量维度768与LLM对应
        self.vq_hobby_embedding.weight.data.uniform_(-1.0/codebook_hobby_k, 1.0/codebook_hobby_k)
        self.vq_background_embedding.weight.data.uniform_(-1.0/codebook_background_k, 1.0/codebook_background_k)
        
        self.codebook_hobby_k = codebook_hobby_k
        self.codebook_background_k = codebook_background_k

        
    def forward(self, user, item, text, mask, seq_price, mask_price, seq_brand, mask_brand, ignore_index=-100):
        device = user.device
        batch_size = user.size(0) # batch size,一批输入多少个samples

        # embeddings 输入的{'user': 0, 'item': 0, 'rating': 5, 'text': 'or high quality skirt', 'feature': 'skirt'}
        u_hb_src = self.user_embeddings_hobby(user)  # brand 输出(batch_size, emsize); 相当于这里带入user id 查表,找对应embedding向量
        u_bg_src = self.user_embeddings_background(user)  # price (batch_size, emsize); wpe 是 word positional embedding
        w_src = self.transformer.wte(text).to(device)  # (batch_size, tgt_len, emsize) wte相当于词向量的东西输入词id进去得到词向量 word token embedding;
        w_hb_src = self.transformer.wte(seq_brand).to(device)
        w_bg_src = self.transformer.wte(seq_price).to(device)
        src = torch.cat([w_hb_src, u_hb_src.unsqueeze(1),w_bg_src, u_bg_src.unsqueeze(1), w_src], 1)  
        # 首先对u和i增加一个维度unsqueeze(1),才能输出(batch_size, total_len, emsize) 对total_len这个维度拼接; 不增加一个维度,无法拼接total_len

        # codebook and construct loss
        hobby_embedding = self.vq_hobby_embedding.weight.data # [5,768]，5为超参数 codebook_hobby_k
        background_embedding = self.vq_background_embedding.weight.data # [3, 768]
        
        embeded_hobby = self.user_embeddings_hobby(user) # brand 输出(batch_size, emsize)
        embeded_background = self.user_embeddings_background(user)
        
        N, LLM_dim = embeded_hobby.shape
        
        embeded_hobby_broadcast = embeded_hobby.reshape(N, 1, LLM_dim)
        embeded_background_broadcast = embeded_background.reshape(N, 1, LLM_dim)
        hobby_embedding_broadcast = hobby_embedding.reshape(1, self.codebook_hobby_k, LLM_dim) # codebook的
        background_embedding_broadcast = background_embedding.reshape(1, self.codebook_background_k, LLM_dim) # codebook的
        
        distance_hobby = torch.sum((hobby_embedding_broadcast - embeded_hobby_broadcast)**2, 2).to(device)
        distance_background = torch.sum((background_embedding_broadcast - embeded_background_broadcast)**2, 2).to(device)
        
        nearest_neighbor_hobby = torch.argmin(distance_hobby, 1).to(device) # [batch_size]取出codebook里面的索引k，最近邻的索引。 [N] ,就是一个列向量，batch_size
        nearest_neighbor_background = torch.argmin(distance_background, 1).to(device) # [batch_size] shape就是[0,0, 2,....,0] 有batch_size个数字构成list
        
        zq_hobby = self.vq_hobby_embedding(nearest_neighbor_hobby).to(device) # [batch_size,LLM_dim]找出k索引对应的codebook的向量，和之前的θ1的向量做mseloss
        zq_background = self.vq_background_embedding(nearest_neighbor_background).to(device) #[32, 768]
        
        if mask is None:
            # auto-regressive generation
            return super().forward(inputs_embeds=src), zq_hobby, embeded_hobby, zq_background, embeded_background, nearest_neighbor_hobby, nearest_neighbor_background
        else:
            # training
            # input padding 
            pad_hb_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device) #ones,每个sample增加两列;矩阵传入gpu;self.src_len = 2
            pad_bg_left = torch.ones((batch_size, self.src_len), dtype=torch.int64).to(device)
            
            pad_input = torch.cat([mask_price, pad_hb_left, mask_brand, pad_bg_left, mask], 1)  # (batch_size, total_len) #在之前句子长度上增加u和i的mask遮掩

            # prediction for training; pred_left就是创建值全为-100的128*2的矩阵, 这里左右拼接,左边为u和i两列,右边为text id,
            pred_hb_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device)
            pred_bg_left = torch.full((batch_size, self.src_len), ignore_index, dtype=torch.int64).to(device) #(batch_size, src_len); ignore_index=-100
            #pred_hb_right = torch.where(mask_price == 1, seq_price, torch.tensor(ignore_index).to(device))
            #pred_bg_right = torch.where(mask_brand == 1, seq_brand, torch.tensor(ignore_index).to(device))
            pred_hb_right = torch.where(mask_price == 1, torch.tensor(ignore_index).to(device), torch.tensor(ignore_index).to(device))# 不需要预测hb和gb，只预测评论
            pred_bg_right = torch.where(mask_brand == 1, torch.tensor(ignore_index).to(device), torch.tensor(ignore_index).to(device)) # 全用-100填充，表示不进行预测，只对评论预测
            
            pred_right = torch.where(mask == 1, text, torch.tensor(ignore_index).to(device)) # replace <pad> with ignore_index, mask取1的位置就是text句子的位置;#转变为tensor([50257,   393,  1029,  3081, 23967,   220, 50258,  -100,  -100,  -100,-100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,-100,  -100,  -100,  -100])
            prediction = torch.cat([pred_hb_right, pred_hb_left, pred_bg_right, pred_bg_left, pred_right], 1)  # (batch_size, total_len)

            return super().forward(attention_mask=pad_input, inputs_embeds=src, labels=prediction), zq_hobby,embeded_hobby, zq_background, embeded_background, nearest_neighbor_hobby, nearest_neighbor_background # 标签就是这个prediction拼接的;
            #这里forward调用GPT2LMHeadModel的参数attention_mask ,inputs_embeds ,labels这些!


class ContinuousPromptLearning(UIPrompt, LlamaForCausalLM):
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
