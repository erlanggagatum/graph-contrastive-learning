from data_preprocessing import *
from models import *
from training import *
from torch_geometric.nn import GCNConv


benchmark_datasets = ['IMDB-BINARY', 'IMDB-MULTI', 'MUTAG']
# benchmark_datasets = ['IMDB-BINARY']
ratio = 0.9
batch_size = 32
hidden_channels = 64
seed = 12345
epoch = 25
encoders = [GCNConv]
# dataset = loadDataset(benchmark_datasets[0]) 
# dataset_o, dataset_c = createComplementPairGraph(dataset)
# g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)

# for data in g_test_loader:
#     print(data.y)

# # Base model
# print('Model: Base Model')
# for bd in benchmark_datasets:
#     dataset = loadDataset(bd) 
#     torch.manual_seed(seed)
#     dataset = dataset.shuffle()
    
#     dataset_o, dataset_c = createComplementPairGraph(dataset)
#     g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
#     # create model
#     model = BaseModel(dataset, hidden_channels, encoders[0])
    
#     # train
#     print('Dataset: ', bd)
#     list_loss, list_train_acc, list_test_acc = train_base_epoch(model, g_train_loader, g_test_loader, epoch)
#     print('')


# # Old Model
# print('Model: Previous Model')
# for bd in benchmark_datasets:
#     dataset = loadDataset(bd) 
#     torch.manual_seed(seed)
#     dataset = dataset.shuffle()
    
#     dataset_o, dataset_c = createComplementPairGraph(dataset)
#     g_train_loader, g_test_loader, gc_train_loader, gc_test_loader = train_test_split(dataset_o, dataset_c, ratio, batch_size)
    
#     # create model
#     model = CLComplement(dataset, hidden_channels, encoders[0], 's')
    
#     # train
#     print('Dataset: ', bd)
#     list_loss, list_train_acc, list_test_acc = train_cl_epoch(model, g_train_loader, g_test_loader, gc_train_loader, gc_test_loader, epoch)
#     print('')