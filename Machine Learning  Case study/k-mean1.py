import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.DataFrame({
    'x':[12,20,28,18,29,33,24,45,45,52,51,52,55,53,55,61,64,69,72],
    'y':[39,36,30,52,54,46,55,59,63,70,66,63,58,23,14,8,19,7,24]
     })

print("step 1: Initialisation - K initial ""(centroids) are generated at random")

print("------------------------------------")
print("Dataset for training");
print("-------------------------------------")
print(df)
print("-------------------------------------------")
np.random.seed(200)
k=3
#centroids[i]=[x,y]

centroids={
    i+1:[np.random.randint(0,80),np.random.randint(0,80)]
    for i in range(k)
    
}
print("--------------------------------")
print("Random centroid generated");
print(centroids);
print("------------------------------")

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color='k')

colmap={1:'r',2:'g',3:'b'}
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
    
plt.title(" Dataset with random centroid");

plt.xlim(0,80)
plt.ylim(0,80)
plt.show()

#-----------------------------------------------
#assignment - K clusters are created by associating each observaton with the nearest centroid 
def assignment(df,centroids):
    
    for i in centroids.keys():
        #sqrt((x1,x2)^2)-(y1-y2)^2
        df['distance_from_{}'.format(i)]=(np.sqrt(df['x']-centroids[i][0])**2 +(df['y']-centroids[i][1]**2))
        
        centroids_distance_cols=['distance_from_{}'.format(i) for i in centroids.keys()]
        
    df['closest']=df.loc[:,centroids_distance_cols].idxmin(axis=1)
    df['closest'] = df['closest'].fillna('distance_from_1')
    #df['closet']=df['closet'].map(lambda x: int(x.lstrip('distance_from_')))
    df['closest'] = df['closest'].map(lambda x: int(x.lstrip('distance_from_'))) 
    df['color']=df['closest'].map(lambda x:colmap[x])
    
    return df
print("step 2 : Assignment - K clusters are created by asscociating each observation with the nearest centroid");

print("Before assignment dataset")
print(df)
df=assignment(df,centroids)

print("first centroid: Red");
print("second centroid :Green");
print("Third centroid : Blue");

print("After assignment dataset");
print(df)

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Dataset with clustering & random centroid");
plt.show()

#--------------------------------------

old_centroids=copy.deepcopy(centroids)
print("step 3: updatae- The centroid of the clusters become the new mean Assignment and update are repeated iteratively until convergence");

def update(k):
    print("old values of centroids");
    print(k);
    
    for i in centroids.keys():
        centroids[i][0]=np.mean(df[df['closest']==i]['x'])
        centroids[i][0]=np.mean(df[df['closest']==i]['x'])
        
    print("New values of centroids")
    print(k)
    return k

centroids=update(centroids)

fig=plt.figure(figsize=(5,5))
ax=plt.axes()
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolors='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)

for i in old_centroids.keys():
    old_x=old_centroids[i][0]
    old_y=old_centroids[i][1]
    dx=(centroids[i][0]-old_centroids[i][0]*0.75)
    dy=(centroids[i][1]-old_centroids[i][1]*0.75)
    ax.arrow(old_x,old_y,dx,dy,head_width=2,head_length=3,fc=colmap[i],ec=colmap[i])
    
plt.title("Dataset with clustering and updated centroids")
plt.show()

##Repeat assignment stage
print("Before assignment dataset")
print(df)
df=assignment(df,centroids)
print("After assignment datset")
print(df)

#plot results
fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolors='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Dataset with clustering and updated centroids");
plt.show()

#continue until assigned categories dont change any more
while True:
    closest_centoids=df['closest'].copy(deep=True)
    centroids=update(centroids)
    print("Before assignment dataset")
    print(df)
    df=assignment(df,centroids)
    print("After assignment datset")
    print(df)
    if closest_centoids.equals(df['closest']):
        break
    
print("Final values of centroids");
print(centroids)

fig=plt.figure(figsize=(5,5))
plt.scatter(df['x'],df['y'],color=df['color'],alpha=0.5,edgecolor='k')
for i in centroids.keys():
    plt.scatter(*centroids[i],color=colmap[i])
plt.xlim(0,80)
plt.ylim(0,80)
plt.title("Final dataset with set centroids")
plt.show()















    

    
        