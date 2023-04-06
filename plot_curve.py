import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.log')
epoch = data['epoch']
acc = data['accuracy']
loss = data['loss']
test_acc = data['val_accuracy']
test_loss = data['val_loss']

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(epoch, loss, label='train')
plt.plot(epoch, test_loss, label='val')
plt.legend()
plt.title('loss')
plt.xlabel('epoch')

plt.subplot(1, 2, 2)
plt.plot(epoch, acc, label='train')
plt.plot(epoch, test_acc, label='val')
plt.legend()
plt.title('acc')
plt.xlabel('epoch')

plt.tight_layout()
plt.savefig('curve.png')
plt.show()