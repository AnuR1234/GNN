
# coding: utf-8

# In[1]:

import numpy as np
from math import isnan
import threading as trd
import math
import random as rnd
import copy
    

#computes information-gain measure
def info_gain(data,cov,data_l,cov_l,data_r,cov_r):
    """Returns information gain
       Parameters
       ----------
       data :  data of root node
       cov : covariance of root node
       data_l: data of left child
       data_r: data of right child
       data_l: covariance of left child
       data_r: covariancr of right child
       Returns
       -------
       Information gain
       """

    #entropy root-node
    a=np.linalg.slogdet(cov)[1]
    if(isnan(a)==True or a == -np.inf):
        a=np.log(1e-9)
    
    #entropy left node
    b=np.linalg.slogdet(cov_l)[1]
    if np.isnan(b)==True or b == -np.inf:
        b=np.log(1e-9)
    b=(data_l.shape[0]/data.shape[0]) * b
    
    #entropy right node
    c=np.linalg.slogdet(cov_r)[1]
    if np.isnan(c)==True or c == -np.inf:
        c=np.log(1e-9)
    c=(data_r.shape[0]/data.shape[0]) * c
    #print(a-b-c)
    return a-b-c

#implementation of axis-aligned splitting
def split_data_axis(data,bbox,split,direction):
    """Returns information gain
        Parameters
        ----------
        data :  data of root node
        split: split value
        direction: direction or dimension of split
        
        Returns
        -------
        Data of the left child and data of the right child
        """    
    idx = data[:, direction] <= split
    left_data = data[idx]
    right_data = data[~idx]
    left_bbox, right_bbox = bbox.copy(), bbox.copy()
    left_bbox[1, direction] = split
    right_bbox[0, direction] = split

    return left_data, right_data, left_bbox, right_bbox

#implementation of linear splitting
def split_data_lin(data,bbox,start,direction):
    """Similar to above function but with non-axis aligned linear splits
       """
    left_data, right_data = [], []
    idx = np.dot(data - start, direction) < 0
    left_data = data[idx]
    right_data = data[~idx]

    # this is an approximation, exact bounding boxes are not possible for non-axis aligned splits
    left_bbox = np.stack((left_data.min(0), left_data.max(0)), axis=0)
    right_bbox = np.stack((right_data.min(0), right_data.max(0)), axis=0)

    return left_data, right_data, left_bbox, right_bbox


class RandomDensityTree:

    def __init__(self,max_depth=10,num_splits=50,min_infogain=2,rand=True,splittype='axis'):
        self.rand=rand
        self.splittype=splittype
        self.max_depth = max_depth
        tree=[]
        for i in range(30):
            tree.append(0)
        self.tree=tree
        self.min_infogain=min_infogain
        self.num_splits=num_splits
        
    def fit(self,data,axis=0):
        '''fits the tree to the training data'''
        if(axis==1):
            data=np.transpose(data)
        bbox = np.stack((data.min(0), data.max(0)), axis=0)
        self.size=data.shape[0]
        self.root=data
        self.mean = np.mean(data,axis=0)
        self.cov=np.cov(np.transpose(data))
        self.rootnode=Node(data,bbox,self.cov,[],self.tree,num_splits=self.num_splits,min_infogain=self.min_infogain,max_depth=self.max_depth,pointer=0,rand=self.rand,splittype=self.splittype)
        self.tree[0]=self.rootnode

    def predict(self,points):
        '''for an input point returns the associated cluster'''
        new=[]
        for p in points:
            new.append(self.rootnode.predict(p))
        new=np.array(new)
        return new

    def sample(self):
        return self.rootnode.sample()
    
    def max_prob():
        '''returns maximum probability'''
        leafs=self.leaf_nodes()
        probs=[]
        for l in leafs:
            probs.append(np.det(cov))*math.sqrt(2*math.pi)
        return max(probs)

    def get_means(self):
        '''returns means of data'''
        means=[]
        self.rootnode.get_means(means)
        return means
    
    def get_split_info(self):
        '''return histories containing split informations of leaf nodes'''
        histories=[]
        self.rootnode.get_split_info(means)
        return histories
  
    def leaf_nodes(self):
        '''returns all leaf nodes'''
        leafs=[]
        self.rootnode.leaf_nodes(leafs)
        return np.array(leafs)
    
class Node:
    
    def __init__(self,data,bbox, cov,history,tree,num_splits,min_infogain,max_depth,pointer,rand=True,splittype='axis'):
        '''
        the init function automatically creates and trains the node
        '''
        self.maxdepth=max_depth
        self.min_infogain=min_infogain
        self.pointer=pointer
        self.size=data.shape[0]
        self.bbox = bbox
        self.tree=tree
        self.num_splits=num_splits

        self.split=float('nan')
        self.split_dim=float('nan')
        self.history=copy.deepcopy(history)
        self.splittype=splittype
        self.root=data
        self.mean=np.mean(data,axis=0)
        self.cov=cov
        
        self.isLeaf=False
        self.left_child=float('nan')
        self.right_child=float('nan')
        #if there is too little data or the maximum depth is reached, do not try to split, otherwise do
        if(max_depth==0 or data.shape[0]==1):
            self.isLeaf=True
           
        else:
            #generate random splits
            if(self.splittype=='axis'):
                rnd_splits=[]
                if(rand==True):
                    for dim in range(int(num_splits)):
                        direction=rnd.choice(range(data.shape[1]))

                        rnd_split=rnd.uniform(min(data[:,direction]),max(data[:,direction]))


                        rnd_splits.append({'split':rnd_split,'direction':direction})
                #generating non-random evenly spaced splits is also possible but only if the split is axis-aligned
                else:
                    rnd_splits=np.concatenate([np.linspace(min(data[:,0]),max(data[:,0])),np.linspace(min(data[:,1]),max(data[:,1]))],axis=0)
           
            #random splits are generated differently
            elif(self.splittype=='linear'):
                rnd_splits=[]
                for n in range(num_splits):
                    start=np.random.uniform(data.min(0), data.max(0))
                    direction=np.random.dirichlet(np.ones((data.shape[1],)), 1).squeeze()
                    direction *= np.random.choice([-1, 1], size=data.shape[1])
                    rnd_splits.append({'split': start,'direction': direction})
                    
            #create lists of left data, right data sets and information gains
            left_datas=[]
            info_gains=np.zeros(num_splits)
            right_datas=[]

            covs_left=[]
            covs_right=[]

            bboxs_left = []
            bboxs_right = []

            for s in range(num_splits):
                
                
                if (self.splittype=='linear'):
                    left_data,right_data,bbox_left,bbox_right=split_data_lin(data,self.bbox,rnd_splits[s]['split'],rnd_splits[s]['direction'])
                else:
                    left_data,right_data,bbox_left,bbox_right=split_data_axis(data,self.bbox,rnd_splits[s]['split'],rnd_splits[s]['direction'])
              
                if(left_data.shape[0]>left_data.shape[1] and right_data.shape[0]>right_data.shape[1]):
                    right_datas.append(right_data)
                    left_datas.append(left_data)

                    bboxs_left.append(bbox_left)
                    bboxs_right.append(bbox_right)

                    cov_l=np.cov(np.transpose(left_data))
                    cov_r=np.cov(np.transpose(right_data))

                    covs_left.append(cov_l)
                    covs_right.append(cov_r)
                    #   print(left_data)
                    info_gains[s]=(info_gain(data,self.cov,left_data,cov_l,right_data,cov_r)) 
                    
                else:
                    #add data nonetheless to not cause problems later
                    right_datas.append(float('nan'))
                    left_datas.append(float('nan'))
                    covs_left.append(float('nan'))
                    covs_right.append(float('nan'))
                    bboxs_left.append(np.nan)
                    bboxs_right.append(np.nan)
                    #information gain if this split is used
            if len(info_gains)==0:
                self.isLeaf=True
                   
            else:
                # choose best info_gain
                best=np.argmax(info_gains)

                if info_gains[best] >= min_infogain:
                    
                    self.split=rnd_splits[best]['split']   #best split
                    self.split_dim=rnd_splits[best]['direction'] 
                    self.history.append(rnd_splits[best])
                    if(2*pointer+2>=len(tree)):
                        for i in range((2*pointer+2)-len(tree)+1):
                            tree.append(0)
                    #append this split to history, this is different for both children
                    self.history[len(self.history)-1]['child']='left'
                    #the left child is generated and trained, the right child follows soon

                    leftnode=Node(left_datas[best],bboxs_left[best],covs_left[best],self.history,tree,self.num_splits,self.min_infogain,self.maxdepth-1,2*pointer+1,rand=rand,splittype=self.splittype)
                    tree[2*pointer+1]=leftnode
                    self.left_child=leftnode
              
                    self.history[len(self.history)-1]['child']='right'
                    rightnode=Node(right_datas[best],bboxs_right[best],covs_right[best],self.history,tree,self.num_splits,self.min_infogain,self.maxdepth-1,2*pointer+2,rand=rand,splittype=self.splittype)
                    tree[2*pointer+2]=rightnode
                    self.right_child=rightnode
                    
                else:
                    #this node is a leaf if no splits occured
                    self.isLeaf=True
       
    
    def predict(self,point):
        '''recursive function, returns cluster for input point'''
        if self.isLeaf==True:
            return self.pointer
        else:
            if(self.splittype=='axis'):
                if point[self.split_dim]<=self.split:
                    return self.left_child.predict(point)
                else:
                    return self.right_child.predict(point)
            else:
                #(b−a)×(c−a)
                 if np.cross((self.split+self.split_dim)-self.split,point-self.split)<0:
                    return self.left_child.predict(point)
                 else:
                    return self.right_child.predict(point)
                
    def sample(self):
        if not self.isLeaf:
            child = np.random.uniform(0, 1) < self.left_child.size / (self.left_child.size + self.right_child.size)
            return self.left_child.sample() if child else self.right_child.sample()
        else:
            return np.random.uniform(self.bbox[0], self.bbox[1])

    def get_split_info(self,histories):
        '''recursively returns split info (histories)'''
        if(self.isLeaf==True):
            histories.append(self.history)
        else:
            self.left_child.get_histories(histories)
            self.right_child.get_histories(histories)

    def get_means(self,means):
        '''recursively returns means'''
        if(self.isLeaf==True):
            means.append(self.mean)
        else:
            self.left_child.get_means(means)
            self.right_child.get_means(means)
                
    def leaf_nodes(self,leafs):
        '''recursively return leaf nodes'''
        if(self.isLeaf==True):
            leafs.append(self.pointer)
        else:
            self.left_child.leaf_nodes(leafs)
            self.right_child.leaf_nodes(leafs)
        
    #def isnan(self):
     #   return False
    
def partition_function(tree, x):
    # generate a lot of samples in the bounds of the data and the size of the bounded shape
    samples, b_size = generate_monte_carlo_sample(x)
    # add gaussian probability dimension for those samples
    g_probs_samples = np.random.random(len(samples))*tree.max_prob
    b_size = b_size*tree.max_prob
    # predict the target leb af nodes for all samples
    leaf_node_ids = tree.predict(samples)
    # compute the distribution integral over each leaf node
    g_ints = np.zeros((len(tree.leaf_nodes),))
    for ln_id in range(len(tree.leaf_nodes)):
        leaf_node = tree.leaf_nodes[ln_id]
        mean_vec = leaf_node.mean
        cov_mat = leaf_node.cov
        mnd = stats.multivariate_normal(mean_vec, cov_mat)
        sample_id_mask = leaf_node_ids==ln_id
        g_probs = mnd(samples[sample_id_mask])
        g_cnt = np.sum(g_probs_samples<=g_probs)
        g_ints[ln_id] = g_cnt/len(samples)*b_size
    


def generate_monte_carlo_sample(X, num_samples=1000000):
    """
    Generate more sample points
    """
    samples = np.random.rand(num_samples,len(X[0]))
    d_mins = np.min(X,axis=0)
    d_maxs = np.max(X,axis=0)
    samples = np.add(np.multiply(samples,d_maxs-d_mins),d_mins)
    b_size = np.prod(d_maxs-d_mins)
    return samples, b_size    




# In[ ]:



