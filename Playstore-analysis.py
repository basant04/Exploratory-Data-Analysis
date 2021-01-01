import numpy as np
import pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
%matplotlib inline

dataset=pd.read_csv("playstore-analysis.csv")

dataset.head(10)

dataset.info()

dataset.isnull().sum()

                     """Tasks 1 """
#1. Data clean up – Missing value treatment
#a. Drop records where rating is missing since rating is our target/study variable
dataset.dropna(how='any', subset=['Rating'], axis=0, inplace = True)
dataset.Rating.isnull().sum()

#b. Check the null values for the Android Ver column.
#i. Are all 3 records having the same problem?
dataset.loc[dataset['Android Ver'].isnull()]

#ii. Drop the 3rd record i.e. record for “Life Made WIFI …”
dataset.drop([10472], inplace = True)
dataset.loc[dataset['Android Ver'].isnull()]

#iii. Replace remaining missing values with the mode
dataset['Android Ver'].fillna(dataset['Android Ver'].mode()[0], inplace=True)

#c. Current ver – replace with most common value
dataset['Current Ver'].fillna(dataset['Current Ver'].mode()[0], inplace=True)

                       """Tasks 2 """
#2. Data clean up – correcting the data types
#a. Which all variables need to be brought to numeric types?
np.dtype(dataset['Rating'])
""" 'Price','Reviews' and 'installs' need to be brought to numeric types."""

#b. Price variable – remove $ sign and convert to float
price = []
for i in dataset['Price']:
    if i[0]=='$':
        price.append(i[1:])
    else:
        price.append(i)

dataset.drop(labels=dataset[dataset['Price']=='Everyone'].index, inplace = True)
dataset['Price']= price
dataset['Price']= dataset['Price'].astype('float')

#c. Installs – remove ‘,’ and ‘+’ sign, convert to integer
install = []
for j in dataset['Installs']:
    install.append(j.replace(',','').replace('+','').strip())

dataset['Installs']= install
dataset['Installs']= dataset['Installs'].astype('int')

#d. Convert all other identified columns to numeric
dataset['Reviews']= dataset['Reviews'].astype('int')

                      """Tasks 3"""
#3. Sanity checks – check for the following and handle accordingly
#a. Avg. rating should be between 1 and 5, as only these values are allowed on the play
#store.
#i. Are there any such records? Drop if so.
dataset.loc[dataset.Rating < 1] & dataset.loc[dataset.Rating > 5]

#b. Reviews should not be more than installs as only those who installed can review the
#app.
#i. Are there any such records? Drop if so.
dataset.loc[dataset['Reviews'] > dataset['Installs']]

temp = dataset[dataset['Reviews']>dataset['Installs']].index
dataset.drop(labels=temp, inplace=True)

dataset.loc[dataset['Reviews'] > dataset['Installs']]

                         """Tasks 4"""
#4. Identify and handle outliers –
#a. Price column
#i. Make suitable plot to identify outliers in price
plt.boxplot(dataset['Price'])
plt.show()

#ii. Do you expect apps on the play store to cost $200? Check out these cases
print('Yes we can expect apps on the play store to cost $200')
dataset.loc[dataset['Price'] > 200]


#iii. After dropping the useless records, make the suitable plot again to identify
#outliers
plt.boxplot(dataset['Price'])
plt.show()

#iv. Limit data to records with price < $30
gt_30 = dataset[dataset['Price'] > 30].index
dataset.drop(labels=gt_30, inplace=True)

count = dataset.loc[dataset['Price'] > 30].index
count.value_counts().sum()

#b. Reviews column
#i. Make suitable plot
sns.distplot(dataset['Reviews'])
plt.show()

#ii. Limit data to apps with < 1 Million reviews
gt_1m = dataset[dataset['Reviews'] > 1000000 ].index
dataset.drop(labels = gt_1m, inplace=True)
print(gt_1m.value_counts().sum(),'cols dropped')

#c. Installs
#i. What is the 95th percentile of the installs?
percentile = dataset.Installs.quantile(0.95) #95th Percentile of Installs
print(percentile,"is 95th percentile of Installs")

#ii. Drop records having a value more than the 95th percentile
for i in range(0,101,1):
    print(' the {} percentile of installs is {} '.format(i,np.percentile(dataset['Installs'],i)))

           #Data analysis to answer business questions
                         """Task 5 """
#5. What is the distribution of ratings like? (use Seaborn) More skewed towards higher/lower
#values?
#a. How do you explain this?
sns.distplot(dataset['Rating'])
plt.show()
print('The skewness of this distribution is',dataset['Rating'].skew())
print('The Median of this distribution {} is greater than mean {} of this distribution'.format(dataset.Rating.median(),dataset.Rating.mean()))

#b. What is the implication of this on your analysis?
dataset['Rating'].mode()
'''Since mode>= median > mean, the distribution of Rating is Negatively 
Skewed.Therefore distribution of Rating is more Skewed towards lower values.'''

#6. What are the top Content Rating values?
#a. Are there any values with very few records?
dataset['Content Rating'].value_counts()

#b. If yes, drop those as they won’t help in the analysis
cr = []
for k in dataset['Content Rating']:
    cr.append(k.replace('Adults only 18+','NaN').replace('Unrated','NaN'))

dataset['Content Rating']=cr

temp2 = dataset[dataset["Content Rating"] == 'NaN'].index
dataset.drop(labels=temp2, inplace=True)
print('droped cols',temp2)

dataset['Content Rating'].value_counts()

#7. Effect of size on rating
#a. Make a joinplot to understand the effect of size on rating
sns.jointplot(y ='Size', x ='Rating', data = dataset, kind ='hex')
plt.show()

#b. Do you see any patterns?
"""Yes, patterns can be observed between Size and Rating ie.
 their is correlation between Size and Rating."""

#c. How do you explain the pattern?
"""Generally on increasing Rating, Size of App also increases.
But this is not always true ie. for higher Rating, their is
 constant Size. Thus we can conclude that their is positive
 correlation between Size and Rating."""
 
                    """ Task 8"""
#8. Effect of price on rating
#a. Make a jointplot (with regression line)
sns.jointplot(x='Price', y='Rating', data=dataset, kind='reg')
plt.show()

#b. What pattern do you see?
"""Generally on increasing the Price, Rating remains
almost constant greater than 4."""

#c. How do you explain the pattern?
"""Since on increasing the Price, Rating remains almost
constant greater than 4. Thus it can be concluded that their
is very weak Positive correlation between Rating and Price."""
dataset.corr()

#d. Replot the data, this time with only records with price > 0
ps1=dataset.loc[dataset.Price>0]
sns.jointplot(x='Price', y='Rating', data=ps1, kind='reg')
plt.show()

#e. Does the pattern change?
"""Yes, On limiting the record with Price > 0, the overall 
pattern changed a slight ie their is very weakly Negative
 Correlation between Price and Rating."""
 ps1.corr()

#f. What is your overall inference on the effect of price on the rating
"""Generally increasing the Prices, doesn't have signifcant
 effect on Higher Rating. For Higher Price, Rating is High
 and almost constant ie greater than 4"""
 
                           """Task 9"""
                           
# 9. Look at all the numeric interactions together –
#a. Make a pairplort with the colulmns - 
#'Reviews', 'Size', 'Rating', 'Price'
sns.pairplot(dataset, vars=['Reviews', 'Size', 'Rating', 'Price'], kind='reg')
plt.show()
                         """Task 10"""
                         
#10. Rating vs. content rating
#a. Make a bar plot displaying the rating for each content rating
dataset.groupby(['Content Rating'])['Rating'].count().plot.bar(color="darkgreen")
plt.show()

#b. Which metric would you use? Mean? Median? Some other quantile?
"""We must use Median in this case as we are having Outliers in Rating.
 Because in case of Outliers , median is the best measure of central 
 tendency."""
plt.boxplot(dataset['Rating'])
plt.show()

#c. Choose the right metric and plot
dataset.groupby(['Content Rating'])['Rating'].median().plot.barh(color="darkgreen")
plt.show()
 
                              """Task 11"""
#11. Content rating vs. size vs. rating – 3 variables at a time
#a. Create 5 buckets (20% records in each) based on Size
bins=[0, 20000, 40000, 60000, 80000, 100000]
dataset['Bucket Size'] = pd.cut(dataset['Size'], bins, labels=['0-20k','20k-40k','40k-60k','60k-80k','80k-100k'])
pd.pivot_table(dataset, values='Rating', index='Bucket Size', columns='Content Rating')

#b. By Content Rating vs. Size buckets, get the rating (20th percentile)
# for each combination
temp3=pd.pivot_table(dataset, values='Rating', index='Bucket Size', 
                     columns='Content Rating', aggfunc=lambda x:np.quantile(x,0.2))
temp3
#c. Make a heatmap of this
#i. Annotated
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(temp3, annot=True, linewidths=.5, fmt='.1f',ax=ax)
plt.show()

#ii. Greens color map
f,ax = plt.subplots(figsize=(5, 5))
sns.heatmap(temp3, annot=True, linewidths=.5, cmap='Greens',fmt='.1f',ax=ax)
plt.show()

#d. What’s your inference? Are lighter apps preferred in all categories?
# Heavier? Some?
"""Based on analysis, its not true that lighter apps are preferred in all 
categories. Because apps with size 40k-60k and 80k-100k have got the
 highest rating in all cateegories. So, in general we can conclude that
 heavier apps are preferred in all categories."""


