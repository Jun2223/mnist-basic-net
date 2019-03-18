import numpy as np
from neural_net import BasicNet
from mnist import load_mnist

(x_train, y_train), (x_test, y_test) = load_mnist()

train_size = 60000
batch_size = 100
batchs_per_epoch = int(train_size / batch_size)
until_epoch = 30
learning_rate = 0.01

loss_record = []
acc_train_record = []
acc_test_record = []

net = BasicNet()
keys = ('W1', 'B1', 'W2', 'B2')

epoch = 0
while epoch < until_epoch:
    epoch += 1
    for i in range(batchs_per_epoch):
        instance_indexes = np.random.choice(train_size, batch_size)
        batch_x = x_train[instance_indexes]
        batch_y = y_train[instance_indexes]

        grad = net.backward(batch_x, batch_y)

        for key in keys:
            net.params[key] -= learning_rate * grad[key]

        forward = net.forward(batch_x, batch_y)
        loss = forward['loss']
        loss_record.append(loss)
        if i == batchs_per_epoch - 1:
            print('loss: ', loss)

for p in range(60, 80):
    a = np.array([x_test[p]])
    prediction = net.forward(a, y_test[p])['probs']

    predicted_digit = np.argmax(prediction)
    label_digit = np.argmax(y_test[p])

    print(p+1)
    print('pred: ', predicted_digit)
    print('labe: ', label_digit, '\n')