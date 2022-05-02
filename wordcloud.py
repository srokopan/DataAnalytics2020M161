import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt
from nltk.corpus import stopwords


df=pd.read_csv("train.csv")
gk = df.groupby('Label')
# group by Entertainment
print('Processing Entertainment')
entertainment = gk.get_group('Entertainment')
entertainmentContent = entertainment['Content']
entertainmentContent = [word for word in entertainmentContent if word not in stopwords.words('english')]

entertainmentWordCloud = WordCloud().generate(' '.join(entertainmentContent))
plt.imshow(entertainmentWordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# group by Business
print('Processing Business')
business = gk.get_group('Business')
businessContent = business['Content']
businessContent = [word for word in businessContent if word not in stopwords.words('english')]
businessWordCloud = WordCloud().generate(' '.join(businessContent))
plt.imshow(businessWordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# group by Health
print('Processing Health')
health = gk.get_group('Health')
healthContent = health['Content']
healthContent = [word for word in healthContent if word not in stopwords.words('english')]
healthWordCloud = WordCloud().generate(' '.join(healthContent))
plt.imshow(healthWordCloud, interpolation='bilinear')
plt.axis('off')
plt.show()

# group by Technology
print('Processing Technology')
technology = gk.get_group('Technology')
technologyContent = technology['Content']
technologyContent = [word for word in technologyContent if word not in stopwords.words('english')]
technologyWordCloud = WordCloud().generate(' '.join(technologyContent))
plt.imshow(technologyWordCloud, interpolation='bilinear')
plt.axis("off")
plt.show()