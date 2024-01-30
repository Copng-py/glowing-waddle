# Sentiment analysis

> **Note**
> `Normalization`, `WordCloud`, `Opinion Mining`
 
> The dataset available on [Kaggle Pages](https://www.kaggle.com/datasets/thedevastator/facebook-posts-of-amazon-tourism).


| Steps | Library Descriptions |
| :---          | :---         |
| Word Normalization (-Remove) | `emoji` :  Emojis from text, `string` : Punctuation, `re` : Suffix and Apostrophes , `nltk` : Stopwords |
| Wordcloud | `wordcloud` |
| Opinion Mining | `TextBlob` : calculating sentiment |


## Word Normalization
<details>

<summary> Remove emojis from text</summary>


````
import re
import emoji

# 
def remove_emoji(text):
    # convert emojis to text representation
    text = emoji.demojize(text)
    # remove emojis using regular expressions
    text = re.sub(r':[a-zA-Z_]+:', ' ', text)
    return text

# apply remove_emoji function to each row of value attribute in tidy_df
df['status_message'] = df['status_message'].apply(remove_emoji)
````
</details>


<details>

<summary> Remove punctuation</summary>

```
import string

# define a function to remove punctuation
def remove_punctuation(text):
    for punctuation in string.punctuation:
        text = text.replace(punctuation, '')
    return text

# apply the function to your dataframe
tidy_df['value'] = tidy_df['value'].apply(remove_punctuation)
tidy_df
```
</details>


<details>

<summary> Remove suffix and apostrophes from words </summary>

```
import re

tidy_df['value'] = tidy_df['value'].apply(lambda x: re.sub(r"'s\b|\W+'", "", x))
tidy_df
```
</details>



<details>

<summary> Removing stopwords </summary>

```
import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop_words = stopwords.words('english')

def remove_stopwords(text):
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token.lower() not in stop_words]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

tidy_df['value'] = tidy_df['value'].apply(remove_stopwords)
tidy_df
```
</details>



## Wordcloud

<details>

<summary> Create wordcloud </summary>

```
! pip install wordcloud

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


```

![download](https://github.com/Copng-py/glowing-waddle/assets/146678457/9508fb9c-869c-41e7-9cff-4bc6861ac53c)

</details>




## Opinion Mining

<details>

<summary> Compute the sentiment score </summary> 

```
import pandas as pd
from textblob import TextBlob

# create a function to compute the sentiment score for each status_message
def get_sentiment_score(message):
    return TextBlob(message).sentiment.polarity

# apply the function to each row of the dataframe and store the result in a new column
tidy_df['sentiment_score'] = tidy_df['value'].apply(get_sentiment_score)

# Create function to the sentiment_score column to get the sentiment class
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
```

</details>
