import torch
from torch_geometric.nn import GCNConv
from torch.nn import Linear
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool
from torch_geometric.nn import global_max_pool
from torch_geometric.nn import global_add_pool
import torch.nn.functional as F
from torch.nn import CosineSimilarity

def train_base(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    
    for data in loader:
        # print('disini')
        # print(data.y)
        # break
        out, z = model(data.x, data.edge_index, data.batch)
        # print(out.size())
        # break
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    return out, loss

@torch.no_grad()
def test_base(model, loader):
    model.eval()
    correct = 0
    for data in loader:
        out, z = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return correct/len(loader.dataset), z

def train_base_epoch(model, train_loader, test_loader, epoch):
    list_loss = []
    list_train_acc = []
    list_test_acc = []

    for epoch in range(0, epoch):
        out, loss = train_base(model, train_loader)
        train_acc, z = test_base(model, train_loader)
        test_acc, z = test_base(model, test_loader)
        
        list_train_acc.append(round(train_acc, 4))
        list_test_acc.append(round(test_acc, 4))
        list_loss.append(round(loss.item(), 4))

        print(f"epoch: {epoch+1} train_acc: {train_acc:.4f} loss: {loss:.4f} test_acc: {test_acc:.4f}")
    
    return list_loss, list_train_acc, list_test_acc


# PREVIOUS PAPER
def train_cl(model, g_loader, gc_loader, classification = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    
    for _, (g_o, g_c) in enumerate(zip(g_loader, gc_loader)):
        h = model(g_o.x, g_c.x, g_o.edge_index, g_c.edge_index, g_o.batch)
        loss = criterion(h, g_o.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    
    return h, loss

@torch.no_grad()
def test_cl(model, g_loader, gc_loader):
    model.eval()
    correct = 0
    for _, (g_o, g_c) in enumerate(zip(g_loader, gc_loader)):
        z = model(g_o.x, g_c.x, g_o.edge_index, g_c.edge_index, g_o.batch)
        pred = z.argmax(dim=1)
        correct += int((pred == g_o.y).sum())

    return correct/len(g_loader.dataset)

def train_cl_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch):
    list_loss = []
    list_train_acc = []
    list_test_acc = []

    
    for epoch in range(0, epoch):
        out, loss = train_cl(model, g_train_loader, gc_train_loader)
        train_acc = test_cl(model, g_train_loader, gc_train_loader)
        test_acc = test_cl(model, g_test_loader, gc_test_loader)
        print(f"epoch: {epoch+1} train_acc: {train_acc:.4f} loss: {loss:.4f} accuracy: {test_acc:.4f}")
        
    return 0,0,0





# PROPOSED METHOD
def train_supcon(model, g_loader, gc_loader, classification = False):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    model.train()
    classification = classification
    criterion = torch.nn.CrossEntropyLoss() if classification else None
    
    for _, (g_o, g_c) in enumerate(zip(g_loader, gc_loader)):
        if classification: # classification
            # last parameter is classification mode
            h, y = model(g_o.x, g_c.x, g_o.edge_index, g_c.edge_index, g_o.batch, True)
            # print(y.size(), g_o.y)
            loss = criterion(y, g_o.y)
            loss.backward()
        else: # pretrain model
            h, x_o, x_c = model(g_o.x, g_c.x, g_o.edge_index, g_c.edge_index, g_o.batch)
            loss, _ = supervisedContrastiveLoss(h, g_o.y, 0.2)
            loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if classification:
        return h, loss, y 
    else:
        return h, loss

@torch.no_grad()
def test_supcon(model, g_loader, gc_loader):
    model.eval()
    correct = 0
    for _, (g_o, g_c) in enumerate(zip(g_loader, gc_loader)):
        z, y = model(g_o.x, g_c.x, g_o.edge_index, g_c.edge_index, g_o.batch, True)
        pred = y.argmax(dim=1)
        correct += int((pred == g_o.y).sum())

    return correct/len(g_loader.dataset), z





def train_supcon_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch):
    pretrain_losses = []
    z = None
    
    print('=== pretrain ===')
    for epoch in range(0, epoch):
        h, loss = train_supcon(model, g_train_loader, gc_train_loader)
        pretrain_losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"epoch: {epoch+1} training loss: {loss:.4f}")
    
    losses = []
    train_accs = []
    test_accs = []
    
    print('=== training classifier ===')
    for epoch in range(0, epoch):
        h, loss, y = train_supcon(model, g_train_loader, gc_train_loader, classification=True)
        train_acc, z = test_supcon(model, g_train_loader, gc_train_loader)
        test_acc, z = test_supcon(model, g_test_loader, gc_test_loader)
        
        losses.append(round(loss.item(), 4))
        train_accs.append(round(train_acc, 4))
        test_accs.append(round(test_acc, 4))
        
        # print(f"epoch: {epoch+1} training loss: {loss:.4f}")
        if epoch % 5 == 0:
            print(f"epoch: {epoch+1} loss: {loss:.4f}; train_acc: {train_acc:.4}; test_acc: {test_acc:.4}")
        # break
    return losses, train_accs, test_accs





# LOSS
# Based on Khosla 2020 - Supervised contrastive learning
def supervisedContrastiveLoss(embeddings, labels, tau):
    loss = 0
    outer = 0
    inner = 0
    denom = 0
    cos = CosineSimilarity(dim=0)

    # outer
    # z_i = embeddings of graph i
    # z_i = label of graph i
    for out_index, (z_i, y_i) in enumerate(zip(embeddings, labels)):
        Pi = torch.sum(labels == y_i)
        # loss = z_i
        # loop to all positive pair with z_i and skip z_i
        for in_index, (zp_i, lp_i) in enumerate(zip(embeddings, labels)):
            if lp_i != y_i or out_index == in_index:
                continue
            # print(z_i, zp_i)
            # print(cos(z_i, zp_i))
            num = torch.exp(cos(z_i, zp_i)/tau)
            # calculate denumerator
            for _, za_i in enumerate(embeddings):
                # only take zi != za_i
                if out_index == in_index:
                    continue
                denom = denom + torch.exp(cos(z_i, za_i)/tau)
                
            inner = inner + torch.log(num/denom)
        
        outer = outer + (-1 / Pi * inner)

        # reset inner denom and inner
        denom, inner = 0, 0
        
    
    loss = outer            
    # print(loss)
    return outer, embeddings