# 加上llm输出的特征做成csv表格，拉回本地跑
import torch
import numpy as np
import pickle

model_path = '/data/ggghhh/luyusheng/codeMC/pepler/model_MoviesAndTV.pt'
device = torch.device('cuda' )

with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)

nuser = 7506 #!!!这个值一定要改，数据集中用户数量
user = torch.tensor(np.arange(nuser)).to(device) # nuser 从0index到nuser index的表

hobby_embedding = model.vq_hobby_embedding.weight.data
background_embedding = model.vq_background_embedding.weight.data

embeded_hobby = model.user_embeddings_hobby(user) # 改变这个就是所有人user从0到27146
embeded_background = model.user_embeddings_background(user)

N, LLM_dim = embeded_hobby.shape

embeded_hobby_broadcast = embeded_hobby.reshape(N, 1, LLM_dim)
embeded_background_broadcast = embeded_background.reshape(N, 1, LLM_dim)
hobby_embedding_broadcast = hobby_embedding.reshape(1, model.codebook_hobby_k, LLM_dim) # codebook的
background_embedding_broadcast = background_embedding.reshape(1, model.codebook_background_k, LLM_dim) # codebook的

distance_hobby = torch.sum((hobby_embedding_broadcast - embeded_hobby_broadcast)**2, 2).to(device)
distance_background = torch.sum((background_embedding_broadcast - embeded_background_broadcast)**2, 2).to(device)

nearest_neighbor_hobby = torch.argmin(distance_hobby, 1).to(device) # [batch_size]取出codebook里面的索引k，最近邻的索引。 [N] ,就是一个列向量，batch_size
nearest_neighbor_background = torch.argmin(distance_background, 1).to(device) # [batch_size] shape就是[0,0, 2,....,0] 有batch_size个数字构成list

nearest_neighbor_hobby_file = open('MoviesAndTV_nearest_neighbor_hobby.pkl','wb')
pickle.dump(nearest_neighbor_hobby.tolist(),nearest_neighbor_hobby_file)
nearest_neighbor_hobby_file.close()

nearest_neighbor_background_file = open('MoviesAndTV_nearest_neighbor_background.pkl','wb')
pickle.dump(nearest_neighbor_background.tolist(),nearest_neighbor_background_file)
nearest_neighbor_background_file.close()


# 读取存的最近邻index
with open('MoviesAndTV_nearest_neighbor_hobby.pkl', 'rb') as f:
    nearest_neighbor_hobby = pickle.load(f)

with open('MoviesAndTV_nearest_neighbor_background.pkl', 'rb') as f:
    nearest_neighbor_background = pickle.load(f)

data = pd.read_csv('./movies_LLM_4reviews.csv') # !!用这个index，要用dataloader重新做的index，参考之前做csv


nearest_neighbor_background_list = []
nearest_neighbor_hobby_list = []

user_list = list(data['user_id'])

for user in user_list:
    nearest_neighbor_background_list.append(nearest_neighbor_background[user])
    nearest_neighbor_hobby_list.append(nearest_neighbor_hobby[user])

data = data.assign(nearest_neighbor_background=nearest_neighbor_background_list,
                   nearest_neighbor_hobby=nearest_neighbor_hobby_list)


data.to_csv('movies_LLM_4reviews_codebook.csv',index=False)