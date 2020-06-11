#!/usr/bin/env python
# coding: utf-8

# In[38]:


get_ipython().system('pip install torchmeta')


# In[39]:


import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np
from torchmeta.modules import (MetaModule,MetaConv2d,MetaBatchNorm2d,MetaSequential,MetaLinear)
from torchmeta.modules.utils import get_subdict
from collections import OrderedDict
import gc


# In[40]:


get_ipython().system('nvidia-smi')


# In[51]:


torch.cuda.set_device(4)


# In[52]:


def conv_block(in_channels, out_channels, **kwargs):
  return MetaSequential(OrderedDict([
      ('conv', MetaConv2d(in_channels, out_channels, **kwargs)),
      ('norm', nn.BatchNorm2d(out_channels, momentum=1.,
          track_running_stats=False)),
      ('relu', nn.ReLU()),
      ('pool', nn.MaxPool2d(2))
  ]))


# In[53]:


class TemplateBank(nn.Module):
  def __init__(self,num_templates,input_channels,output_channels, kernel_size):
    super(TemplateBank,self).__init__()
    self.coefficients_shape=(num_templates,1,1,1,1)
    #the templates are convolutions windows, we n_templates of the same size in the bank
    templates=[torch.Tensor(input_channels,output_channels,kernel_size,kernel_size) for i in range(num_templates)]
    #stack the tensors, same form but now usable for pytorch
    self.templates=nn.Parameter(torch.stack(templates))
  def forward(self,coefficients):
    #print("Linear combination of the templates",(self.templates*coefficients).sum(0))
    #linear combination
    return (self.templates*coefficients).sum(0)


# In[54]:


class SConv2d(MetaModule):
  def __init__(self,bank,stride=1,padding=1):
    super(SConv2d,self).__init__()
    self.stride , self.padding, self.bank= stride, padding, bank
    #soft parameter in front of the templates, determine by the shape of the bank
    self.coefficients=nn.Parameter(torch.zeros(bank.coefficients_shape))
  def forward(self,input,params=None):
    # these are the convolution parameters, we multiplied the linear coef to the templates
    #it's one tensor, create by the forward method of bank
    coeffs=OrderedDict(params)["coefficients"]
    
    parameters=self.bank(coeffs)
    #Performs a normal convolutions with the linear combination of the templates
    return F.conv2d(input,parameters,stride=self.stride,padding=self.padding)


# In[55]:


class conv_block_soft(MetaModule):
  def __init__(self,input_channels,output_channels,bank=None):
    super(conv_block_soft,self).__init__()
    self.bank=bank
    self.conv1=SConv2d(self.bank)
    self.bn1=nn.BatchNorm2d(output_channels)
    self.relu=nn.ReLU(inplace=True)
    self.maxpool=nn.MaxPool2d(2)
  def forward(self,x,params=None):
    x=self.conv1(x,params=get_subdict(params,"conv1"))
    x=self.bn1(x)
    x=self.relu(x)
    x=self.maxpool(x)
    return x


# In[69]:


class SConvNet(MetaModule):
  def __init__(self,num_templates,num_classes,in_channels,hidden_size=64,feature_size=64):
    super(SConvNet,self).__init__()
    print("SConvNet, Templates:",num_templates)
    layers_per_bank=2*(4-1) #find out why
  
    self.conv_3x3=MetaConv2d(in_channels,hidden_size,kernel_size=3,stride=1,padding=1,bias=False)
    self.bank=TemplateBank(num_templates,hidden_size,hidden_size,3)
    self.block1=conv_block_soft(hidden_size,hidden_size,self.bank)
    self.block2=conv_block_soft(hidden_size,hidden_size,self.bank)
    self.block3=conv_block_soft(hidden_size,hidden_size,self.bank)
    self.block4=conv_block_soft(hidden_size,hidden_size,self.bank)
    self.classifier=MetaLinear(feature_size,num_classes,bias=True)
    
    #initialisations
    coefficient_inits = torch.zeros((int(layers_per_bank),int(num_templates),1,1,1,1))
    nn.init.orthogonal_(coefficient_inits) # very important
    sconv_group=[]
    for name,module in self.named_modules():
        if isinstance(module,SConv2d):
          sconv_group.append((name,module))
    for j,(name,module) in enumerate(sconv_group):
        module.coefficients.data=coefficient_inits[j]
    
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        init.kaiming_normal_(m.weight)
        m.bias.data.zero_()

  def forward(self,x,params=None):
    x=self.conv_3x3(x,params=get_subdict(params,"conv_3x3"))
    x=self.block1(x,params=get_subdict(params,"block1"))
    x=self.block2(x,params=get_subdict(params,"block2"))
    x=self.block3(x,params=get_subdict(params,"block3"))
    x=self.block4(x,params=get_subdict(params,"block4"))
    
    x=x.view((x.size(0), -1))

    x=self.classifier(x,params=get_subdict(params,"classifier"))

    return x


# In[70]:


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


# # **Training the model**

# In[71]:


class MAML(object):
  def __init__(self,model,optimizer=None,step_size=0.1,loss_function=F.cross_entropy):
    #metamodel, the neural net for the tasks
    self.model=model

    self.optimizer=optimizer
    #the step size could be meta-learnable, but for now we put it fixed
    self.step_size=torch.tensor(step_size,dtype=torch.float32,requires_grad=False)
    self.loss_function=loss_function
    self.params_dict=OrderedDict(self.model.named_parameters())
  def accuracy(self,logits,targets):
    with torch.no_grad():
      _,predictions=torch.max(logits,dim=1)
      accuracy=torch.mean(predictions.eq(targets).float())

    return accuracy.item()

  def step(self,batch):
    outer_loss=0
    #average of accuracy accross tasks for query set
    outer_accuracy=0
    counter=0
    for task_id,task in enumerate(zip(*batch["train"],*batch["test"])):
      if counter>5:
            break
      counter+=1
      # the zip is now a array of 25 (one for each task) with 4 columns
      # train_inputs, train_target, test_inputs, test_target
      #each task in this zip is a batch for a specific task
      train_inputs,train_targets=task[0].cuda(),task[1].cuda() #support set
      test_inputs,test_targets=task[2].cuda(),task[3].cuda() #querry set
      
      #don't forget to pass it named_parameters, and shouldn't be an iterator
      train_logits=self.model(train_inputs,self.params_dict)#OrderedDict(self.model.named_parameters())
      inner_loss=self.loss_function(train_logits,train_targets)
      self.model.zero_grad()
      #the model will have parameters called meta_params
      grads=torch.autograd.grad(inner_loss,self.model.parameters())
      
      #Updating the parameters for that tast
      #this becomes a for loop if we do many training steps inside, default is 1
      params=OrderedDict()
      i=0
      '''for (name,param), grad in zip(self.model.named_parameters(),grads):
        #if name in ...:
        #find better way to do this
        if "coefficients" in name:
          params[name]=param-self.step_size*grad
        else:
          params[name]=param'''
    
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
    
    outer_accuracy=float(outer_accuracy)/counter  #float(len(batch["train"][0])) #len of a torch tensor?
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

      train_logits=self.model(train_inputs,params=model.named_parameters())
      #don't forget to pass in parameters
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


# In[72]:


def adjust_learning_rate(optimizer, epoch, gammas, schedule, loss):
  lr = args.learning_rate
  assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
  for (gamma, step) in zip(gammas, schedule):
    if (epoch >= step): lr = lr * gamma
    else: break
  for param_group in optimizer.param_groups: param_group['lr'] = lr
  return lr
def group_weight_decay(net, weight_decay, skip_list=()):
  decay, no_decay = [], []
  for name, param in net.named_parameters():
    if not param.requires_grad: continue
    if sum([pattern in name for pattern in skip_list]) > 0: no_decay.append(param)
    else: decay.append(param)
  return [{'params': no_decay, 'weight_decay': 0.}, {'params': decay, 'weight_decay': weight_decay}]

def accuracy(output, target, topk=(1,)):
  if len(target.shape) > 1: return torch.tensor(1), torch.tensor(1)
  
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
  return res


# In[ ]:





# # **Training the model**

# In[73]:


from torchmeta.datasets import Omniglot, MiniImagenet
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchmeta.utils.data import BatchMetaDataLoader


# In[74]:


#-------------HyperParameters----------------------
num_shots=5
num_ways=5
num_shots_test=5
batch_size=128
num_workers=1

#optimization
learning_rate=0.001
momentum=0.9
schedule=[60,120,160]
gammas=[0.2,0.2,0.2]
#regularization
decay=0.0005


# In[75]:


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


# In[15]:


dataset_transform = ClassSplitter(shuffle=True,
                                      num_train_per_class=num_shots,
                                      num_test_per_class=num_shots_test)
class_augmentations = [Rotation([90, 180, 270])]
transform = Compose([Resize(28), ToTensor()])

meta_train_dataset = Omniglot("data",
                              transform=transform,
                              target_transform=Categorical(num_ways),
                              num_classes_per_task=num_ways,
                              meta_train=True,
                              class_augmentations=class_augmentations,
                              dataset_transform=dataset_transform,
                              download=True)
meta_val_dataset = Omniglot("data",
                            transform=transform,
                            target_transform=Categorical(num_ways),
                            num_classes_per_task=num_ways,
                            meta_val=True,
                            class_augmentations=class_augmentations,
                            dataset_transform=dataset_transform)
meta_test_dataset = Omniglot("data",
                             transform=transform,
                             target_transform=Categorical(num_ways),
                             num_classes_per_task=num_ways,
                             meta_test=True,
                             dataset_transform=dataset_transform)


# In[76]:


meta_train_dataloader = BatchMetaDataLoader(meta_train_dataset,
                                                batch_size=batch_size,
                                                shuffle=True,
                                                num_workers=num_workers,
                                                pin_memory=True)


# In[77]:


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


# In[78]:


out_features=5
hidden_size=64
loss_function=torch.nn.CrossEntropyLoss().cuda()
in_channels=3 #1 for omniglot
bank_size=4


# In[79]:


ModelConvMiniImagenet=SConvNet(bank_size,out_features,in_channels,feature_size=5*5*hidden_size).cuda()
params = group_weight_decay(ModelConvMiniImagenet, decay, ['coefficients'])
optimizer=torch.optim.SGD(params, learning_rate, momentum=momentum, nesterov=(momentum > 0.0))


# In[33]:


no_temp_model=MetaConvModel(3,5,feature_size=5*5*hidden_size).cuda()
#params = group_weight_decay(no_temp_model, decay, ['coefficients'])
optimizer=torch.optim.Adam(no_temp_model.parameters(), lr=0.001)


# In[34]:


no_temp_model(b["train"][0][0].cuda())


# In[80]:


p=OrderedDict(ModelConvMiniImagenet.named_parameters())
ModelConvMiniImagenet(b["train"][0][0].cuda(),p)


# In[81]:


metalearner=MAML(ModelConvMiniImagenet,optimizer,loss_function=loss_function)


# In[82]:


epochs=100
for epoch in range(epochs):
  metalearner.train(meta_train_dataloader,100)


# In[37]:


state={
  'epoch': epoch + 1,
  'state_dict': ModelConvMiniImagenet.state_dict(),
  'optimizer' : optimizer.state_dict(),
}
filename="maml_paper_implementation_templates.pth.tar"
torch.save(state,filename)


# In[ ]:




