#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os


# In[38]:


os.chdir('/Users/@Copng-py/Downloads')


# In[39]:


df = pd.read_csv("Facebook_amazon.csv")
df


# In[40]:


# Check for missing values
import seaborn as sns

# Check for missing values
missing_values_count = df.isnull().sum()
missing_values_count


# In[41]:


# Plot the missing values as a heatmap
sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False, cbar=False)


# In[42]:


#data cleasing
df.dropna(inplace=True)


# In[43]:



# Plot the missing values as a heatmap
sns.heatmap(df.isnull(), cmap='viridis', yticklabels=False, cbar=False)


# In[44]:


import pandas as pd
import matplotlib.pyplot as plt

# Display basic statistical information for the numerical columns
print(df.describe())


# In[45]:


df = df.drop(columns=['num_special'])
print(df.describe())


# In[30]:


#Data Pre-processing for sentiment analysis
df.loc[df['category'] == 7.0, 'category'] = 0
df = df[df.category != 0]
df = df.replace({
    "category": {
        1: "New product announcement",
        2: "Sweepstakes and contest",
        3: "Sales",
        4: "Consumer Feedback",
        5: "Infotainment",
        6: "Organization Branding"
    }
})
category_dict = {
    "New product announcement": 1,
    "Sweepstakes and contest": 2,
    "Sales": 3,
    "Consumer Feedback": 4,
    "Infotainment": 5,
    "Organization Branding": 6
}
df["category_number"] = df["category"].map(category_dict)
df


# In[52]:


# Data Pre-processing
#Translate from Portuguese to English
get_ipython().system(' pip install googletrans==4.0.0-rc1')


# In[64]:


import googletrans
from googletrans import Translator

# create a translator object
translator = Translator()

# loop over each row in the dataframe and translate the status_message attribute
for index, row in df.iterrows():
    # translate the status_message to English
    translation = translator.translate(row['status_message'], dest='en')
    # replace the original Portuguese text with the translated English text
    df.at[index, 'status_message'] = translation.text


# In[250]:


df


# In[ ]:


#Data Pre-processing for sentiment analysis
get_ipython().system(' pip install emoji')


# In[ ]:


#Remove emojis
import re
import emoji

# define function to remove emojis from text
def remove_emoji(text):
    # convert emojis to text representation
    text = emoji.demojize(text)
    # remove emojis using regular expressions
    text = re.sub(r':[a-zA-Z_]+:', ' ', text)
    return text

# apply remove_emoji function to each row of value attribute in tidy_df
df['status_message'] = df['status_message'].apply(remove_emoji)


# In[ ]:


df['status_message'] = df['status_message'].str.lower()


# In[280]:


#Data Pre-processing for sentiment analysis
#a tidy format
# Use the melt function to convert status_message to a tidy format
tidy_df = df.melt(id_vars=['status_id'], value_vars=['status_message'], var_name='attribute', value_name='value')

# Print the tidy DataFrame
print(tidy_df)


# In[261]:


tidy_df['value'][5]


# In[272]:


#Data Pre-processing for sentiment analysis
import nltk
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer


# In[281]:


#Data Pre-processing for sentiment analysis
#Removing punctuation
import string

# define a function to remove punctuation
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# apply the function to your dataframe
tidy_df['value'] = tidy_df['value'].apply(remove_punctuation)
tidy_df


# In[282]:


import re

# remove 's suffix and apostrophes from words
tidy_df['value'] = tidy_df['value'].apply(lambda x: re.sub(r"'s\b|\W+'", "", x))
tidy_df


# In[ ]:


#Data Pre-processing
#removing_stopwords
import nltk
nltk.download('stopwords')


# In[ ]:



from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

#Data Pre-processing
#removing_stopwords
tidy_df['value'] = tidy_df['value'].apply(remove_stopwords)
tidy_df


# In[285]:


tidy_df


# In[286]:


import nltk
from nltk.tokenize import word_tokenize

# tokenize the text in the 'processed_value' column
tidy_df['tokenized_value'] = tidy_df['value'].apply(word_tokenize)
tidy_df


# In[290]:


# EDA
#Which word has the highest frequency

import re

# Remove certain words and numbers
stop_words = ['day', 'today', 'de']
tidy_df['value'] = tidy_df['value'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words and not re.match(r'^\d+$', word) and len(word) > 1]))

# create a new dataframe with the count of each word in the 'processed_value' column
word_counts = pd.Series(' '.join(tidy_df['value']).split()).value_counts()

# print the 10 most common words
print(word_counts.head(10))


# In[291]:


# plot a histogram for high frequency words
plt.figure(figsize=(12,6))
word_counts[:20].plot(kind='bar')
plt.xlabel('Words')
plt.ylabel('Frequency')
plt.title('Most Common Words')
plt.show()


# In[305]:


import pandas as pd
from collections import Counter


# Define a function to count word frequencies
def count_words(text):
    return Counter(text.split())

# Apply the count_words function to each row in the dataframe and create a new column with the results
tidy_df['word_counts'] = tidy_df['value'].apply(count_words)

# Use pandas' groupby function to group the data by word and sum the word counts
word_counts = pd.DataFrame(tidy_df['word_counts'].tolist()).stack().reset_index()
word_counts.columns = ['index', 'word', 'count']
word_counts = word_counts.groupby('word').sum().sort_values('count', ascending=False)

# Print the top 10 words by frequency
print(word_counts.head(10))


# In[403]:


print(word_counts)


# In[297]:


get_ipython().system(' pip install wordcloud')


# In[326]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a string of all words in the dataframe
all_words = ' '.join(tidy_df['value'])

# Generate the word frequencies
word_frequencies = Counter(all_words.split())

# Remove stopwords
stopwords = set(STOPWORDS)
stopwords.update(['said', 'one', 'will', 'now', 'say', 'said', 'new', 'year'])

# Generate the word cloud
wordcloud = WordCloud(width=1000, height=1000, background_color='white', stopwords=stopwords, min_font_size=10).generate_from_frequencies(word_frequencies)

# Display the generated image:
plt.figure(figsize=(8, 8), facecolor=None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad=0) 

# Save the image
plt.savefig("wordcloud.png")


# In[370]:


# Define weights for each variable
weights = {
    "num_reactions": 0.1,
    "num_shares": 0.05,
    "num_likes": 0.3,
    "num_loves": 0.4,
    "num_wows": 0.05,
    "num_hahas": 0.05,
    "num_sads": -0.2,
    "num_angrys": -0.3,
    "num_special": 0.05
}

# Compute sentiment score for each row
merged_df["sentiment_scores"] = (
    weights["num_reactions"] * df["num_reactions"] +
    weights["num_shares"] * df["num_shares"] +
    weights["num_likes"] * df["num_likes"] +
    weights["num_loves"] * df["num_loves"] +
    weights["num_wows"] * df["num_wows"] +
    weights["num_hahas"] * df["num_hahas"] +
    weights["num_sads"] * df["num_sads"] +
    weights["num_angrys"] * df["num_angrys"] +
    weights["num_special"] * df["num_special"]
)


# In[371]:


merged_df


# In[378]:


get_ipython().system(' pip install textblob')


# In[382]:


import pandas as pd
from textblob import TextBlob

# create a function to compute the sentiment score for each status_message
def get_sentiment_score(message):
    return TextBlob(message).sentiment.polarity

# apply the function to each row of the dataframe and store the result in a new column
tidy_df['sentiment_score'] = tidy_df['value'].apply(get_sentiment_score)

tidy_df


# In[386]:


def get_sentiment_nature(score, threshold=0.5):
    if score > threshold:
        return 'Positive'
    elif score <= 0 :
        return 'Negative'
    else:
        return 'Neutral'  
    
    
# apply the function to the sentiment_score column to get the sentiment class
tidy_df['sentiment_nature'] = tidy_df['sentiment_score'].apply(get_sentiment_nature)
tidy_df


# In[387]:


tidy_df["sentiment_nature"].value_counts()


# In[388]:


tidy_df.columns


# In[404]:


df.columns


# In[532]:


merged_df = pd.merge(tidy_df[['status_id', 'attribute', 'value', 'tokenized_value', 'word_counts',
       'sentiment_score', 'sentiment_nature']],
                     df[['status_id', 'num_likes', 'num_loves', 'num_wows', 'num_hahas',
       'num_sads', 'num_angrys', 'num_special', 'category']],
                     on='status_id', how='inner')


# In[533]:


merged_df


# In[534]:


# Define weights for each variable
weights = {
    "num_likes": 0.5,
    "num_loves": 0.8,
    "num_wows": 0.09,
    "num_hahas": 0.7,
    "num_sads": -0.4,
    "num_angrys": -0.7,
    "num_special": 0.05
}

# Replace NaN values with 0
merged_df.fillna(0, inplace=True)

# Compute sentiment score for each row
merged_df["engagement"] = (
    weights["num_likes"] * merged_df["num_likes"] +
    weights["num_loves"] * merged_df["num_loves"] +
    weights["num_wows"] * merged_df["num_wows"] +
    weights["num_hahas"] * merged_df["num_hahas"] +
    weights["num_sads"] * merged_df["num_sads"] +
    weights["num_angrys"] * merged_df["num_angrys"] +
    weights["num_special"] * merged_df["num_special"]
)


# In[535]:



merged_df


# In[536]:


category_dict = {
    "New product announcement": 1,
    "Sweepstakes and contest": 2,
    "Sales": 3,
    "Consumer Feedback": 4,
    "Infotainment": 5,
    "Organization Branding": 6
}

merged_df["category_number"] = merged_df["category"].map(category_dict)


# In[537]:





# In[538]:


merged_df = merged_df.drop(columns=['num_special'])


# In[539]:


print(merged_df.describe())


# In[540]:





# In[541]:


# group by category and sum the engagement values
grouped_df = merged_df.groupby('category')['engagement'].sum().reset_index()

# pivot the data to prepare for plotting
pivot_df = grouped_df.pivot(index='category', columns='engagement', values='engagement')

# plot a stacked bar chart
pivot_df.plot(kind='bar', stacked=True, figsize=(10, 8))
plt.title('Engagement by Category and Sentiment')
plt.xlabel('Category')
plt.ylabel('Engagement')
plt.xticks(rotation=25)
plt.show()


# In[542]:


import pandas as pd
import seaborn as sns


# create the correlation matrix
corr_matrix = merged_df.corr()

# visualize the correlation matrix using a heatmap
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')


# In[543]:


merged_df.columns


# In[496]:


# Group by category and sentiment
grouped_df = merged_df.groupby(['category', 'sentiment_nature'])['value'].count()


# Unstack the sentiment column to create a separate column for each sentiment
grouped_df = grouped_df.unstack()

# Fill NaN values with 0
grouped_df = grouped_df.fillna(0)

# Rename the columns
grouped_df.columns = ['Negative', 'Neutral', 'Positive']

# Print the result
print(grouped_df)


# In[401]:


# count the number of occurrences of each unique value in 'category' column
category_counts = merged_df['category'].value_counts()

# print the result
print(category_counts)


# In[414]:


import matplotlib.pyplot as plt

# create a new dataframe with the counts of positive, neutral, and negative sentiments by category
counts_df = pd.DataFrame(merged_df.groupby(['category', 'sentiment_nature'])['value'].count())
counts_df = counts_df.reset_index()
counts_df.columns = ['Category', 'Sentiment_nature', 'Count']

# create a pivot table to reshape the data for plotting
pivot_df = counts_df.pivot(index='Category', columns='Sentiment_nature', values='Count')
pivot_df


# In[420]:


# plot a stacked bar chart
pivot_df.plot(kind='bar', stacked=True, figsize=(10, 8))
plt.title('Sentiment Counts by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.xticks(rotation=25)
plt.legend(title='Sentiment')
plt.show()


# In[499]:


from wordcloud import WordCloud
import matplotlib.pyplot as plt

# Create a string of all words in the dataframe
all_words = ' '.join(tidy_df['value'])

# Generate the word frequencies
word_frequencies = Counter(all_words.split())

# Remove stopwords
stopwords = set(STOPWORDS)
stopwords.update(['said', 'one', 'will', 'now', 'say', 'said', 'new', 'year', ''r''])

# Generate the word cloud
wordcloud = WordCloud(width=1000, height=1000, background_color='white', stopwords=stopwords, min_font_size=10).generate_from_frequencies(word_frequencies)

# Display the generated image:
plt.figure(figsize=(8, 8), facecolor=None) 
plt.imshow(wordcloud) 
plt.axis("off") 
plt.tight_layout(pad=0) 

# Save the image
plt.savefig("wordcloud.png")


# In[487]:


import nltk
from nltk.corpus import stopwords

# download the stopwords if you haven't already
nltk.download('stopwords')

# define the stopwords to be removed
stop_words = set(stopwords.words('english'))
stopwords.update(['said', 'one', 'will', 'now', 'say', 'said', 'new', 'year', '00h'])

# create a new list for each sentiment class without stopwords
negative_words = [(word, count) for (word, count) in negative_counts.most_common() if word not in stop_words and not word.isdigit() and word not in ['want', 'de', 'day', 'today']]
neutral_words = [(word, count) for (word, count) in neutral_counts.most_common() if word not in stop_words and not word.isdigit() and word not in ['want', 'de', 'day', 'today']]
positive_words = [(word, count) for (word, count) in positive_counts.most_common() if word not in stop_words and not word.isdigit() and word not in ['want', 'de', 'day', 'today']]

# print the 10 most common words for each sentiment class without stopwords
print("Most common Negative words:", negative_words[:10])
print("Most common Neutral words:", neutral_words[:10])
print("Most common Positive words:", positive_words[:10])


import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Create a function to generate a word cloud
def create_wordcloud(words, title):
    wordcloud = WordCloud(width = 800, height = 400, background_color ='white', 
                stopwords = stop_words, 
                min_font_size = 10).generate_from_frequencies(dict(words))
    plt.figure(figsize = (8, 8), facecolor = None) 
    plt.imshow(wordcloud) 
    plt.axis("off") 
    plt.title(title, fontsize=16)
    plt.tight_layout(pad = 0) 
    plt.show() 


# generate wordclouds for each sentiment class
generate_wordcloud(dict(negative_words), "Most Common Negative Words")
generate_wordcloud(dict(neutral_words), "Most Common Neutral Words")
generate_wordcloud(dict(positive_words), "Most Common Positive Words")


import matplotlib.pyplot as plt

# get the top 10 most common words for each sentiment category
negative_words = dict(negative_words[:10])
neutral_words = dict(neutral_words[:10])
positive_words = dict(positive_words[:10])

# create a figure with subplots for each sentiment category
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# plot the frequency of each word in a horizontal bar chart for each sentiment category
axes[0].barh(list(negative_words.keys()), negative_words.values())
axes[1].barh(list(neutral_words.keys()), neutral_words.values())
axes[2].barh(list(positive_words.keys()), positive_words.values())

# set the titles and labels for each subplot
axes[0].set_title('Negative')
axes[1].set_title('Neutral')
axes[2].set_title('Positive')

fig.suptitle('Top 10 most common words by sentiment category', fontsize=14)

plt.tight_layout()
plt.show()


# In[470]:


get_ipython().system(' pip install prettytable')


# In[471]:


from prettytable import PrettyTable

# create a table with three columns
table = PrettyTable(['Sentiment', 'Word', 'Count'])

# add the rows to the table
for word, count in negative_words[:10]:
    table.add_row(['Negative', word, count])
for word, count in neutral_words[:10]:
    table.add_row(['Neutral', word, count])
for word, count in positive_words[:10]:
    table.add_row(['Positive', word, count])

# print the table
print(table)


# In[472]:


from prettytable import PrettyTable

# create PrettyTables for each sentiment class
negative_table = PrettyTable(['Word', 'Count'])
neutral_table = PrettyTable(['Word', 'Count'])
positive_table = PrettyTable(['Word', 'Count'])

# fill the tables with the 10 most common words for each sentiment class
for word, count in negative_words[:10]:
    negative_table.add_row([word, count])
    
for word, count in neutral_words[:10]:
    neutral_table.add_row([word, count])
    
for word, count in positive_words[:10]:
    positive_table.add_row([word, count])

# print the tables
print("Most common Negative words:")
print(negative_table)

print("Most common Neutral words:")
print(neutral_table)

print("Most common Positive words:")
print(positive_table)


# In[478]:


merged_df.columns


# In[ ]:





# In[483]:


import matplotlib.pyplot as plt

# get the top 10 most common words for each sentiment category
negative_words = dict(negative_words[:10])
neutral_words = dict(neutral_words[:10])
positive_words = dict(positive_words[:10])

# create a figure with subplots for each sentiment category
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))

# plot the frequency of each word in a horizontal bar chart for each sentiment category
axes[0].barh(list(negative_words.keys()), negative_words.values())
axes[1].barh(list(neutral_words.keys()), neutral_words.values())
axes[2].barh(list(positive_words.keys()), positive_words.values())

# set the titles and labels for each subplot
axes[0].set_title('Negative')
axes[1].set_title('Neutral')
axes[2].set_title('Positive')

fig.suptitle('Top 10 most common words by sentiment category', fontsize=14)

plt.tight_layout()
plt.show()


# In[ ]:





# In[ ]:




