
import pandas as pd
import matplotlib.pyplot as plt

from textblob import TextBlob
    
dataset=pd.read_csv("Restaurant_Reviews.tsv",delimiter="\t")
x=pd.DataFrame(dataset["Review"])
y=pd.DataFrame(dataset["Liked"])
l=len(x)
li=[]
for i in range(0,l,1):
    analysis = TextBlob(x.iloc[i,0])
    if analysis.sentiment.polarity > 0:
        li.append(1)
    elif analysis.sentiment.polarity == 0:
        li.append(2)
    else:
        li.append(0)
y_pred=pd.DataFrame(li)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y["Liked"],y_pred)
x["liked"]=y_pred
pos,neg,neu=y_pred[0].value_counts()
labels = ["Positive","Negative","neutral"]
sizes = [pos,neg,neu]
colors = ['#ff9999','#66b3ff','#99ff99']
 
fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=50)
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)
ax1.axis('equal')  
plt.tight_layout()
plt.show()
