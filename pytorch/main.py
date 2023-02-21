from datasetpytorch import ProductReviewDataset
import torch
from model_pytorch import MFBlock
from torch import nn
import torch.optim as optim


def train():
    train_dataset = ProductReviewDataset(phase='train')
    test_dataset = ProductReviewDataset(phase='test')
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024,
                                               shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    network = MFBlock(max_users=train_dataset.customer_index.shape[0], max_items=train_dataset.product_index.shape[0],
                      num_emb=64)

    network.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    for epoch in range(200):  # loop over the dataset multiple times

        running_loss = 0.0
        train_loss = []
        network.train()
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            users, items, star_rating = data['user'], data['item'], data['star_rating']
            users = users.to(device)
            items = items.to(device)
            star_rating = star_rating.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(users, items)
            loss = criterion(outputs, star_rating.float())
            train_loss.append(loss.item())
            loss.backward()
            optimizer.step()

            # print statistics
            # if i % 10 == 0:  # print every 2000 mini-batches
            # print(f'[{epoch + 1}, {i + 1:5d}] train_loss: {sum(train_loss) / len(train_loss):.3f}')
        print(f'[{epoch + 1}] train_loss: {sum(train_loss) / len(train_loss):.3f}')
        test_loss = []
        for i, data in enumerate(test_loader):
            users, items, star_rating = data['user'], data['item'], data['star_rating']
            users = users.to(device)
            items = items.to(device)
            star_rating = star_rating.to(device)
            network.eval()
            with torch.no_grad():
                outputs = network(users, items)
                loss = criterion(outputs, star_rating.float())
                test_loss.append(loss)
        print(f'[{epoch + 1}] test_loss: {sum(test_loss) / len(test_loss):.3f}')

    print('Finished Training')


if __name__ == '__main__':
    train()
