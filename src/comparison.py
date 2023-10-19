from data_preprocessing import *
from models import *
from training import *
from torch_geometric.nn import GCNConv
import pickle

benchmark_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG']
# benchmark_datasets = ['MUTAG']
ratio = 0.9
batch_size = 32
hidden_channels = 64
seed = 12345
epoch = 50
encoders = [GCNConv]
# dataset = loadDataset(benchmark_datasets[0]) 
# dataset_o, dataset_c = createComplementPairGraph(dataset)
# g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)

# for data in g_test_loader:
#     print(data.y)

# New Model
filename = 'proposed_'
print('Proposed Model')
for bd in benchmark_datasets:
    dataset = loadDataset(bd)
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    
    dataset_o, dataset_c = createComplementPairGraph(dataset)
    g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
    # create model
    model = ComplementarySupCon(dataset, hidden_channels, encoders[0], 's')
    
    # train
    print('Dataset: ', bd)
    filename = filename + bd + ".sav"
    list_loss, list_train_acc, list_test_acc = train_supcon_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch)
    print('')
    pickle.dump(model, open(filename, 'wb'))

# Base model
print('Model: Base Model')
filename = 'base_'
for bd in benchmark_datasets:
    dataset = loadDataset(bd) 
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    
    dataset_o, dataset_c = createComplementPairGraph(dataset)
    g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
    # create model
    model = BaseModel(dataset, hidden_channels, encoders[0])
    
    # train
    print('Dataset: ', bd)
    filename = filename + bd + ".sav"
    list_loss, list_train_acc, list_test_acc = train_base_epoch(model, g_train_loader, g_test_loader, epoch)
    print('')
    
    pickle.dump(model, open(filename, 'wb'))
    


# Old Model
print('Model: Previous Model')
filename = 'previous_'
for bd in benchmark_datasets:
    dataset = loadDataset(bd)
    torch.manual_seed(seed)
    dataset = dataset.shuffle()
    
    dataset_o, dataset_c = createComplementPairGraph(dataset)
    g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
    # create model
    model = CLComplement(dataset, hidden_channels, encoders[0], 's')
    
    # train
    print('Dataset: ', bd)
    filename = filename + bd + ".sav"
    list_loss, list_train_acc, list_test_acc = train_cl_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch)
    print('')
    pickle.dump(model, open(filename, 'wb'))
    
    
    
# # load
# print('Model: Previous Model')
# filename = 'previous_'
# for bd in benchmark_datasets:
#     dataset = loadDataset(bd)
#     torch.manual_seed(seed)
#     dataset = dataset.shuffle()
    
#     dataset_o, dataset_c = createComplementPairGraph(dataset)
#     g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
#     # train
#     print('Dataset: ', bd)
#     # load
#     filename = filename + bd + ".sav"
#     model = pickle.load(open(filename, 'rb'))
#     list_loss, list_train_acc, list_test_acc = train_cl_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch)
#     print('')
#     # pickle.dump(model, open(filename, 'wb'))