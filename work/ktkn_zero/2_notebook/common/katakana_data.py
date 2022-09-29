import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator

# カタカナデータの表示
def plot_katakana(data, label, ind):
    img = data[ind]
    label = label[ind]    
    print(label)
    plt.imshow(img[0,:,:], cmap='gray')
    plt.show()
    
# データ拡張の関数定義
def data_augmentation(data, labels, params, num):
    # data: N, Channel(チャネル数), H, W
    # labels: N, Class(分類数)
    # num: 水増しするデータ数
    generator = ImageDataGenerator(**params)
    data_tmp = data.transpose(0,2,3,1)
    train_iter = generator.flow(x=data_tmp, y=labels, batch_size=1)#画像を保存しない場合
    print(f'{len(data_tmp)}個のデータから{num}個の水増しデータを作成します')
    for i in range(num):
        batches = train_iter.next()
        data_tmp = np.vstack([data_tmp, batches[0]])
        labels = np.vstack([labels, batches[1]])
        if np.mod(i, 200)==0: print(f'{i}個目の水増しデータを作成中【進捗率：{i/num:.1%}】')

    data = data_tmp.transpose(0,3,1,2)
    return data, labels