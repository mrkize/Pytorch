
import timm

import matplotlib.pyplot as plt # for showing handwritten digits

from umap import UMAP

model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)

color = ['b', 'coral', 'peachpuff', 'sandybrown', 'linen', 'tan', 'orange', 'gold', 'darkkhaki', 'yellow', 'chartreuse', 'green', 'turquoise', 'skyblue']
# Configure UMAP hyperparameters
# model = model.load_VIT('./Network/VIT_Model_cifar10/VIT_PE.pth')

PE = model.pos_embed[0].detach().numpy()
# PE = model.pos_embed[0,:,:].detach().numpy()
# print(PE.shape)
reducer = UMAP(n_neighbors=100, # default 15, The size of local neighborhood (in terms of number of neighboring sample points) used for manifold approximation.
               n_components=2, # default 2, The dimension of the space to embed into.
               n_epochs=1000, # default None, The number of training epochs to be used in optimizing the low dimensional embedding. Larger values result in more accurate embeddings.
              )

# Fit and transform the data
X_trans = reducer.fit_transform(PE)

# Check the shape of the new data
# print('Shape of X_trans: ', X_trans.shape)
fig = plt.figure( figsize=(20,12), dpi=160 )
plt.scatter(X_trans[0,0],X_trans[0,1], c='r')
for co in range(14):
    plt.scatter(X_trans[14*co+1:14*(co+1)+1,0],X_trans[14*co+1:14*(co+1)+1,1], c=color[co])


for i in range(PE.shape[0]):
    plt.annotate(str(i), xy = (X_trans[i,0], X_trans[i,1]), xytext = (X_trans[i,0]+0.05, X_trans[i,1]+0.05))

#3 设置刻度及步长
# z = range(40)
# x_label = ['11:{}'.format(i) for i in x]
# plt.xticks( x[::5], x_label[::5])
# plt.yticks(z[::5])  #5是步长

#4 添加网格信息
plt.grid(True, linestyle='--', alpha=0.5) #默认是True，风格设置为虚线，alpha为透明度

#5 添加标题（中文在plt中默认乱码，不乱码的方法在本文最后说明）
plt.xlabel('Time')
plt.ylabel('Temperature')
plt.title('Curve of Temperature Change with Time')

#6 保存图片，并展示
plt.savefig('all.png')
plt.close(fig)
