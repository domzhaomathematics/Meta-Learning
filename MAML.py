#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install torchmeta


# In[37]:


get_ipython().run_line_magic('env', 'CUDA_DEVICE_ORDER=PCI_BUS_ID')
get_ipython().run_line_magic('env', 'CUDA_VISIBLE_DEVICES=0')


# In[1]:


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F 
import numpy as np
from torchmeta.modules import (MetaModule, MetaConv2d, MetaBatchNorm2d,
                               MetaSequential, MetaLinear)
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict
import gc


# Meta learner, will handle the training of the outer loop 
#     Parameters
#     ----------
#     model : `torchmeta.modules.MetaModule` instance
#         The model.
#     optimizer : `torch.optim.Optimizer` instance, optional
#         The optimizer for the outer-loop optimization procedure. This argument
#         is optional for evaluation.
#     step_size : float (default: 0.1)
#         The step size of the gradient descent update for fast adaptation
#         (inner-loop update).
#     first_order : bool (default: False)
#         If `True`, then the first-order approximation of MAML is used.
#     learn_step_size : bool (default: False)
#         If `True`, then the step size is a learnable (meta-trained) additional
#         argument [2].
#     per_param_step_size : bool (default: False)
#         If `True`, then the step size parameter is different for each parameter
#         of the model. Has no impact unless `learn_step_size=True`.
#     num_adaptation_steps : int (default: 1)
#         The number of gradient descent updates on the loss function (over the
#         training dataset) to be used for the fast adaptation on a new task.
#     scheduler : object in `torch.optim.lr_scheduler`, optional
#         Scheduler for the outer-loop optimization [3].
#     loss_function : callable (default: `torch.nn.functional.cross_entropy`)
#         The loss function for both the inner and outer-loop optimization.
#         Usually `torch.nn.functional.cross_entropy` for a classification
#         problem, of `torch.nn.functional.mse_loss` for a regression problem.
#     device : `torch.device` instance, optional
#         The device on which the model is defined.
#     References
#     ----------
#     .. [1] Finn C., Abbeel P., and Levine, S. (2017). Model-Agnostic Meta-Learning
#            for Fast Adaptation of Deep Networks. International Conference on
#            Machine Learning (ICML) (https://arxiv.org/abs/1703.03400)
#     .. [2] Li Z., Zhou F., Chen F., Li H. (2017). Meta-SGD: Learning to Learn
#            Quickly for Few-Shot Learning. (https://arxiv.org/abs/1707.09835)
#     .. [3] Antoniou A., Edwards H., Storkey A. (2018). How to train your MAML.
#            International Conference on Learning Representations (ICLR).
#            (https://arxiv.org/abs/1810.09502)

# Intuition of MAML:
# I want to optimize theta, such that one I perform one gradient step on theta (becomes theta i), It minimizes the loss for that task's querry set.
# So theta i is the corrected theta when we perform one gradient step on theta from the S set, then we update theta such that it minimises the prediction from theta i. At the end we get theta optimzed such that when I calculate theta i (gradient step) it minimizes the loss for the querry set i when predicting with theta i. We do that for each different tasks.
# 
# so we can have a for loop that does the first step (computing theta i), alpha step size,
# then update theta such that it minimizes the loss of fthetai(Qi) (prediction using theta i) on the querry set, beta step size.

# 

# In[2]:


class MAML(object):
  def __init__(self,model,optimizer=None,step_size=0.1,loss_function=F.cross_entropy):
    #metamodel, the neural net for the tasks
    self.model=model

    self.optimizer=optimizer
    #the step size could be meta-learnable, but for now we put it fixed
    self.step_size=torch.tensor(step_size,dtype=torch.float32,requires_grad=False)
    self.loss_function=loss_function
  def accuracy(self,logits,targets):
    with torch.no_grad():
      _,predictions=torch.max(logits,dim=1)
      accuracy=torch.mean(predictions.eq(targets).float())

    return accuracy.item()

  def step(self,batch):
    outer_loss=0
    #average of accuracy accross tasks for query set
    outer_accuracy=0
    for task_id,task in enumerate(zip(*batch["train"],*batch["test"])):
      # the zip is now a array of 25 (one for each task) with 4 columns
      # train_inputs, train_target, test_inputs, test_target
      #each task in this zip is a batch for a specific task
      train_inputs,train_targets=task[0].cuda(),task[1].cuda() #support set
      test_inputs,test_targets=task[2].cuda(),task[3].cuda() #querry set

      train_logits=self.model(train_inputs)
      inner_loss=self.loss_function(train_logits,train_targets)
      self.model.zero_grad()
      #the model will have parameters called meta_params
      grads=torch.autograd.grad(inner_loss,self.model.parameters())
      
      #Updating the parameters for that tast
      #this becomes a for loop if we do many training steps inside, default is 1
      params=OrderedDict()
      for (name,param), grad in zip(self.model.named_parameters(),grads):
        params[name]=param-self.step_size*grad
      
      #this step in the paper is outside the inner loop, we evaluate on query set
      #the query set of that task, using the newly learned params (theta i), and updtate the real theta with it
      #we can caluclate the loss for each i during each step, so we don't have to remember the theta i
      #assign theta i (params) to the model temporarly to evaluate
      test_logit=self.model(test_inputs,params=params)

      #do we really take the average of accuracy for each task in the batch?
      
      #!!!! We could add a dictionary to collect the task loss for a specific id.

      outer_loss+=self.loss_function(test_logit,test_targets)
      outer_accuracy+=self.accuracy(test_logit,test_targets)
    
    outer_accuracy=float(outer_accuracy)/float(len(batch["train"][0])) #len of a torch tensor?
    #computes gradient
    outer_loss.backward()
    #the optimizer should already be "loaded" with the model's params
    self.optimizer.step()

    return outer_loss.detach(),outer_accuracy

  def train(self,dataloader,max_batches=500):
    num_batches=0
    for batch in dataloader:
      if num_batches>=max_batches:
        break
      l,a=self.step(batch)
      print(l,a)
      num_batches+=1
  def step_evaluate(self,batch):
    outer_loss=0
    for task in batch:
      train_inputs,train_targets=task["support"]
      test_inputs,test_targets=task["query"]

      train_logits=self.model(train_inputs)
      inner_loss=self.loss_function(train_logits,train_targets)
      self.model.zero_grad()
      #the model will have parameters called meta_params
      grads=torch.autograd.grad(inner_loss,self.model.meta_params())
      params=OrderedDict()
      
      #Updating the parameters for that tast
      for (name,param), grad in zip(model.meta_named_pars(),grads):
        params[name]=param-step_size*grad
      
      #this step in the paper is outside the inner loop, we evaluate on query set
      #the query set of that task, using the newly learned params (theta i), and updtate the real theta with it
      #we can caluclate the loss for each i during each step, so we don't have to remember the theta i
      #assign theta i (params) to the model temporarly to evaluate
      test_logits=model(test_inputs,params=params) 

      outer_accuracy+=self.accuracy(np.argmax(test_logits),test_targets)
      outer_loss+=self.loss_function(test_logits,test_targets)
    
    outer_accuracy=float(outer_accuracy)/float(len(batch)) #len of torch tensor?
    #we don't update the meta_params when evaluating
    return outer_loss,outer_accuracy
  def evaluate(self,dataloader,max_batches=500):
    mean_outer_loss,mean_accuracy,count= 0., 0., 0

    for batch in dataloader:
      if num_batches>=max_batches:
        break
      outer_loss,outer_accuracy=self.step_evaluate(batch)

      mean_outer_loss+=outer_loss
      mean_accuracy+=outer_accuracy
      count+=1
    
    return float(mean_outer_loss)/float(count) , float(mean_accuracy)/float(count)


# To do:
# modify model so that they have the needed methods (or use MetaConvModel)

# # **First Model: MAML Paper**

# In[3]:


#The Metamodule is the same as normal pytorch module, but allows to pass parameters in the forward argument
#This way we can evaluate with different parameters then the meta-parameters (theta i)
#As usual the forward method takes in a batch as input of data points as input


# In[4]:


def conv_block(in_channels, out_channels, **kwargs):
  return MetaSequential(OrderedDict([
      ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
      ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
          track_running_stats=False)),
      ('relu', nn.ReLU()),
      ('pool', nn.MaxPool2d(2))
  ]))


# In[5]:


class MetaConvModel(MetaModule):
  def __init__(self,in_channels,out_features,hidden_size=64,feature_size=64):
    super(MetaConvModel,self).__init__()
    self.in_channels=in_channels
    self.out_features=out_features
    self.hidden_size=hidden_size
    self.feature_size=feature_size

    self.features = MetaSequential(OrderedDict([                                         
    ('layer1', conv_block(in_channels, hidden_size, kernel_size=3,
                          stride=1, padding=1, bias=True)),
    ('layer2', conv_block(hidden_size, hidden_size, kernel_size=3,
                          stride=1, padding=1, bias=True)),
    ('layer3', conv_block(hidden_size, hidden_size, kernel_size=3,
                          stride=1, padding=1, bias=True)),
    ('layer4', conv_block(hidden_size, hidden_size, kernel_size=3,
                          stride=1, padding=1, bias=True))
    ]))
    self.classifier = MetaLinear(feature_size, out_features, bias=True)
  def forward(self, inputs, params=None):
    features = self.features(inputs, params=get_subdict(params, 'features'))
    features = features.view((features.size(0), -1))
    logits = self.classifier(features, params=get_subdict(params, 'classifier'))
    return logits
    


# In[6]:


torch.cuda.empty_cache()
gc.collect()


# # **Getting the data**

# In[7]:


from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchmeta.utils.data import BatchMetaDataLoader


# In[8]:


#-------------HyperParameters----------------------
num_shots=5
num_ways=5
num_shots_test=5
batch_size=25
num_workers=2


# In[9]:


dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
transform = Compose([Resize(84), ToTensor()])

meta_train_dataset = MiniImagenet("data",
                                  transform=transform,
                                  target_transform=Categorical(num_ways),
                                  num_classes_per_task=num_ways,
                                  meta_train=True,
                                  dataset_transform=dataset_transform,
                                  download=True)
meta_val_dataset = MiniImagenet("data",
                                transform=transform,
                                target_transform=Categorical(num_ways),
                                num_classes_per_task=num_ways,
                                meta_val=True,
                                dataset_transform=dataset_transform)
meta_test_dataset = MiniImagenet("data",
                                  transform=transform,
                                  target_transform=Categorical(num_ways),
                                  num_classes_per_task=num_ways,
                                  meta_test=True,
                                  dataset_transform=dataset_transform)


# In[10]:


meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)


# In[11]:


#exploring dataset
#train is the support set, test is the querry set
num_batches=0
for batch in meta_train_dataloader:
  if num_batches>=1:
    break
  num_batches+=1
  print(batch.keys())
  print(len(batch["train"]))
  print(len(batch["test"]))
  print(len(batch["train"][0]))
  print(len(batch["train"][0][0]))
  b=batch

  # one data point
  print(len(batch["train"][0][0][0]))
  #-------------
  
  print(len(batch["train"][0][0][0][0]))

  # batch["train"] contains (inputs,targets)
  # for batch["train"][0], it's a batch of 25 tasks, each containing a batch 
  #of data for the specific tast


# The structure of the data is as followed:
# a batch from the meta_detaloader gives us a train and test dictionary.
# Each train array is tuple with input and target. each input is a batch of task-specific batch.
# 
# Each task-specific batch for one specific task is the classic type of batch you would get normally, meaning (input,targets) form

# In[12]:


import gc
torch.cuda.empty_cache()
gc.collect()


# In[13]:


out_features=5
hidden_size=64


# In[14]:


ModelConvMiniImagenet=MetaConvModel(3,out_features=out_features,hidden_size=64,feature_size=5*5*hidden_size).cuda()
loss_function=torch.nn.CrossEntropyLoss().cuda()


# In[15]:


optimizer=torch.optim.Adam(ModelConvMiniImagenet.parameters(), lr=0.001)
#ModelConvMiniImagenet(b["train"][0][0].cuda())


# testing the Metalearner

# In[16]:


metalearner=MAML(ModelConvMiniImagenet,optimizer,loss_function=loss_function)


# In[17]:


epochs=50
for epoch in range(epochs):
  metalearner.train(meta_train_dataloader,100)


# In[18]:


def save_checkpoint(state, is_best, filename=‘checkpoint.pth.tar’):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, ‘model_best.pth.tar’)


# In[21]:



state={
  'epoch': epoch + 1,
  'state_dict': ModelConvMiniImagenet.state_dict(),
  'optimizer' : optimizer.state_dict(),
}
filename="epochs50_max_batch100_MAML_Paper.pth.tar"
torch.save(state,filename)


# In[ ]:


PATH = ''checkpoint.pth.tar"
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint[‘model_state_dict’])
optimizer.load_state_dict(checkpoint[‘optimizer_state_dict’])
epoch = checkpoint[‘epoch’]
loss = checkpoint[‘loss’]

