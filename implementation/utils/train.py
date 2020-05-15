import torch

MODELS_PATH = './models'


def train(model, criterion, optimizer, trainloader, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs.requires_grad = True
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 300 == 299 or i == len(trainloader) - 1:
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)

                accuracy = (predicted == labels).sum().item() / total
                loss = running_loss / total

                print('Epoch: %d\tBatch: %d\tLoss: %.3f\tAccuracy: %.3f' %
                      (epoch + 1, i + 1, loss, accuracy))
                running_loss = 0.0

    print('Finished Training')


def train_separable_cnn(model, criterion, optimizer, trainloader, epochs=5):
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs.requires_grad = True
            optimizer.zero_grad()

            outputs, _, _ = model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 300 == 299 or i == len(trainloader) - 1:
                _, predicted = torch.max(outputs.data, 1)
                total = labels.size(0)

                accuracy = (predicted == labels).sum().item() / total
                loss = running_loss / total

                print('Epoch: %d\tBatch: %d\tLoss: %.3f\tAccuracy: %.3f' %
                      (epoch + 1, i + 1, loss, accuracy))
                running_loss = 0.0

    print('Finished Training')


def validate(model, criterion, testloader):
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    print('Test loss: %.3f' % (running_loss / len(testloader)))


def validate_separable_cnn(model, criterion, testloader):
    running_loss = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data

        outputs, _, _ = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item()

    print('Test loss: %.3f' % (running_loss / len(testloader)))


def save(model, model_name):
    torch.save(model.state_dict(), MODELS_PATH + '/' + model_name)


def load(model, model_name):
    model.load_state_dict(torch.load(MODELS_PATH + '/' + model_name))
    model.eval()

    return model
