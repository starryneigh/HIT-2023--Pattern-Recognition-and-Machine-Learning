import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from LeNet import LeNet
from torch import nn

batch_size = 30
lr = 0.002
epoch = 5
activate_list = ['relu', 'sigmoid', 'tanh', 'leaky_relu']
activate_test = False
dropout_list = np.arange(0.1, 0.6, 0.1)
dropout_test = False
lr_list = [0.0001, 0.0005, 0.001, 0.002, 0.003]
lr_test = False
bs_list = np.arange(10, 100, 10)
bs_test = False
num_list = np.arange(10000, 60001, 10000)
num_test = False


def test_net():
    net.eval()
    correct = 0
    confusion_matrix = torch.zeros(10, 10, dtype=torch.int32)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = net(inputs)
            _, predicts = torch.max(outputs.detach(), 1)
            correct += (predicts == labels).sum().item()
            for p, l in zip(predicts, labels):
                confusion_matrix[p, l] += 1
    return confusion_matrix


def train_net():
    global run_loss, loss_list
    net.train()
    print(f'epoch: {e}')
    for i, data in enumerate(train_loader):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        run_loss += loss.item()
        if i % print_num == print_num - 1:
            run_loss /= print_num
            loss_list.append(run_loss)
            # print(f'[{e + 1}, {(i + 1) * batch_size}:{train_num}] \tloss: {round(run_loss, 5)}')
            run_loss = 0.0


def plot_loss(loss_lists, name):
    plt.title("loss")
    for loss_list, label in zip(loss_lists, name):
        num = len(loss_list)
        x = np.arange(num)
        plt.plot(x, loss_list, label=label)
    plt.legend()
    plt.show()


def cal_conmat(con_mat):
    p_num = torch.sum(con_mat, 0)
    l_num = torch.sum(con_mat, 1)
    tp = torch.diagonal(con_mat)

    precision = tp / p_num
    recall = tp / l_num
    accuracy = (torch.sum(tp) / torch.sum(con_mat)).item()
    f1 = 2 / (1 / precision + 1 / recall)

    esti_dic = {"accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1": f1}
    return esti_dic


def plot_conmat(con_mat):
    fig = plt.figure()
    num = con_mat.size(0)
    plt.imshow(con_mat, interpolation='nearest', cmap=plt.cm.Oranges)
    plt.title('confusion_matrix')
    plt.colorbar()
    tick_marks = np.arange(num)
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    for i in range(num):
        for j in range(num):
            plt.text(i, j, format(con_mat[i, j]), va='center', ha='center')
    fig.show()


def print_esti(estimate: dict):
    accuracy = estimate['accuracy']
    precision = torch.mean(estimate['precision']).item()
    recall = torch.mean(estimate['recall']).item()
    f1 = torch.mean(estimate['f1']).item()
    print(f'accuracy: {round(accuracy, 3)}, precision: {round(precision, 3)}, '
          f'recall: {round(recall, 3)}, f1: {round(f1, 3)}')


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(0.5, 0.5)])
    train_data = torchvision.datasets.MNIST('../data/mnist', train=True, transform=transform)
    test_data = torchvision.datasets.MNIST('../data/mnist', train=False, transform=transform)
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=True, num_workers=2)
    # print(type(train_data[0][1]))
    # print(train_data)
    # print(test_data.data.size())
    train_num = len(train_data)
    test_num = len(test_data)
    print_num = int(train_num / (batch_size * 10))

    if activate_test:
        loss_lists = []
        for act in activate_list:
            net = LeNet(activate=act)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            criterion = nn.CrossEntropyLoss()
            run_loss = 0.0
            loss_list = []
            for e in range(epoch):
                start = time.time()
                train_net()
                end = time.time()
                print(f'训练时间为：{round(end - start, 3)}s')
                con_matrix = test_net()
            loss_lists.append(loss_list)
            estimate = cal_conmat(con_matrix)
            print_esti(estimate)
            plot_conmat(con_matrix)
        plot_loss(loss_lists, activate_list)

    elif dropout_test:
        loss_lists = []
        for dropout in dropout_list:
            net = LeNet(p=dropout)
            optimizer = optim.Adam(net.parameters(), lr=lr)

            criterion = nn.CrossEntropyLoss()
            run_loss = 0.0
            loss_list = []
            for e in range(epoch):
                start = time.time()
                train_net()
                end = time.time()
                print(f'训练时间为：{round(end - start, 3)}s')
                con_matrix = test_net()
            loss_lists.append(loss_list)
            estimate = cal_conmat(con_matrix)
            print_esti(estimate)
            plot_conmat(con_matrix)
        plot_loss(loss_lists, dropout_list)

    elif lr_test:
        loss_lists = []
        for lr in lr_list:
            net = LeNet()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            criterion = nn.CrossEntropyLoss()
            run_loss = 0.0
            loss_list = []
            for e in range(epoch):
                start = time.time()
                train_net()
                end = time.time()
                print(f'训练时间为：{round(end - start, 3)}s')
                con_matrix = test_net()
            loss_lists.append(loss_list)
            estimate = cal_conmat(con_matrix)
            print_esti(estimate)
            plot_conmat(con_matrix)
        plot_loss(loss_lists, lr_list)

    elif num_test:
        loss_lists = []
        for num in num_list:
            train_data = torchvision.datasets.MNIST('../data/mnist', train=True, transform=transform)
            train_data.data = train_data.data[:num]
            train_data.targets = train_data.targets[:num]
            train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=2)
            train_num = len(train_data)
            test_num = len(test_data)
            print_num = int(train_num / (batch_size * 10))
            net = LeNet()
            optimizer = optim.Adam(net.parameters(), lr=lr)

            criterion = nn.CrossEntropyLoss()
            run_loss = 0.0
            loss_list = []
            for e in range(epoch):
                start = time.time()
                train_net()
                end = time.time()
                print(f'训练时间为：{round(end - start, 3)}s')
                con_matrix = test_net()
            loss_lists.append(loss_list)
            estimate = cal_conmat(con_matrix)
            print_esti(estimate)
            plot_conmat(con_matrix)
        plot_loss(loss_lists, num_list)

    else:
        loss_lists = []
        net = LeNet()
        optimizer = optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        run_loss = 0.0
        for e in range(epoch):
            loss_list = []
            start = time.time()
            train_net()
            end = time.time()
            print(f'训练时间为：{round(end - start, 3)}s')
            con_matrix = test_net()
            loss_lists.append(loss_list)
            estimate = cal_conmat(con_matrix)
            print_esti(estimate)
        plot_conmat(con_matrix)
        plot_loss(loss_lists, np.arange(1, epoch+1))


