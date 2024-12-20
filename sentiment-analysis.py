# %%
# Importing necessary libraries
import pandas as pd  # For handling and analyzing structured data
import numpy as np  # For numerical computations
import matplotlib.pyplot as plt  # For data visualization
import seaborn as sns  # For enhanced visualization with statistical graphics

# Setting a plot style for consistent visualization aesthetics
plt.style.use('ggplot')

# Importing NLTK for text processing
import nltk


# %%
# Loading the dataset into DataFrame
df = pd.read_csv('/kaggle/input/amazon-fine-food-reviews/Reviews.csv')


# %%
# Accessing the first review text in the 'Text' column of the dataset
df['Text'].values[0]

# Printing the shape of the DataFrame (rows, columns)
print(df.shape)

# Reducing the DataFrame to the first 500 rows for easier handling
df = df.head(500)


# %% [markdown]
# # Assessing Bias and Calculating Sentiment Scores

# %%
# Plot the distribution of review scores (star ratings) to observe potential bias in ratings
axis = df['Score'].value_counts().sort_index().plot(
    kind='bar',                      # Create a bar chart
    title='Review Count',            # Set the chart title
    figsize=(10, 5)                  # Define the figure size
)
axis.set_xlabel('Review Stars')      # Label the x-axis with 'Review Stars'
plt.show()                           # Display the plot


# %% [markdown]
# # Text Preprocessing with NLTK
# 

# %%
# Selects the 50th text sample from the 'Text' column in the DataFrame
ex = df['Text'][50]
print(ex)


# %%
# Tokenizes the selected text (splitting it into individual words)
tokens = nltk.word_tokenize(ex)

# %%
# Part-of-Speech tagging: labels each token with its corresponding part of speech
tagged = nltk.pos_tag(tokens)
print(tagged)

# %%
# Named Entity Recognition (NER): Identifies named entities (e.g., person names, organizations)
entities = nltk.chunk.ne_chunk(tagged)
# entities.pprint()  [if need to check, remove comment]

# %% [markdown]
# # Sentiment Analysis with VADER

# %%
# Importing necessary libraries for Sentiment Analysis and progress tracking
from nltk.sentiment import SentimentIntensityAnalyzer  # VADER sentiment analysis tool
from tqdm.notebook import tqdm  # For showing progress bar in Jupyter notebooks

# Initialize the SentimentIntensityAnalyzer from NLTK
sia = SentimentIntensityAnalyzer()


# %%
# Analyzing the sentiment of a sample sentence using VADER
sia.polarity_scores('I am so happy!')


# %%
# Extracting the 50th review text from the dataset
ex = df['Text'][50]

# Analyzing the sentiment of the extracted review text using VADER
sia.polarity_scores(ex)

# %%
# Create an empty dictionary to store the sentiment scores for each review
res = {}

# Loop through each row in the dataset to analyze the sentiment of the review text
# tqdm is used to display a progress bar as we iterate through the DataFrame
for i, row in tqdm(df.iterrows(), total=len(df)):
    # Extract the review text and review ID for each row
    text = row['Text']
    myid = row['Id']
    
    # Store the sentiment analysis result in the dictionary, using the review ID as the key
    res[myid] = sia.polarity_scores(text)


# %%
# Convert the sentiment analysis results dictionary to a DataFrame
# .T transposes the dictionary, making the review IDs as rows and sentiment scores as columns
vaders = pd.DataFrame(res).T

# Reset the index of the DataFrame and rename the 'index' column to 'Id' to match the original DataFrame
vaders = vaders.reset_index().rename(columns={'index': 'Id'})

# Merge the VADER sentiment DataFrame with the original DataFrame (df) on the 'Id' column
# This adds the sentiment scores as additional columns to the original DataFrame
vaders = vaders.merge(df, how='left')


# %%
# Show the first few rows of sentiment scores and review data
vaders.head()


# %% [markdown]
# ## Plot VADER results

# %%
# Plotting VADER Sentiment Analysis Results
# Using a bar plot to visualize the relationship between review scores and their corresponding compound sentiment score

ax = sns.barplot(data=vaders, x="Score", y="compound")
ax.set_title('Compound Sentiment Score by Amazon Review Rating')

# Show the plot
plt.show()


# %%
# Plotting VADER Sentiment Scores (Positive, Neutral, Negative) for each Review Score
# Creating a set of bar plots to visualize the distribution of positive, neutral, and negative sentiment scores

fig, axs = plt.subplots(1, 3, figsize=(12, 3))

# Plot Positive sentiment score
sns.barplot(data=vaders, x="Score", y='pos', ax=axs[0])
axs[0].set_title('Positive Sentiment')

# Plot Neutral sentiment score
sns.barplot(data=vaders, x="Score", y='neu', ax=axs[1])
axs[1].set_title('Neutral Sentiment')

# Plot Negative sentiment score
sns.barplot(data=vaders, x="Score", y='neg', ax=axs[2])
axs[2].set_title('Negative Sentiment')

# Adjust layout to avoid overlap and improve readability
plt.tight_layout()

# Show the plot
plt.show()


# %% [markdown]
# # Sentiment Analysis with Pretrained RoBERTa Model

# %%
# Importing necessary libraries for using the pretrained RoBERTa model
from transformers import AutoTokenizer  # For tokenizing text
from transformers import AutoModelForSequenceClassification  # For loading the pretrained RoBERTa model
from scipy.special import softmax  # To apply the softmax function to get probabilities


# %%
# Load the pretrained RoBERTa model and tokenizer from CardiffNLP for sentiment analysis
MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"  # Model designed for sentiment analysis on Twitter data

# Initialize the tokenizer, which will convert text into token IDs for the model
tokenizer = AutoTokenizer.from_pretrained(MODEL)

# Load the pre-trained RoBERTa model for sequence classification (sentiment analysis in this case)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# %%
# Vader results for reference
print(ex)
sia.polarity_scores(ex)

# %%
# Run sentiment analysis using the pretrained RoBERTa model
enc_text = tokenizer(ex, return_tensors='pt')  # Tokenize the example text for input to RoBERTa

output = model(**enc_text)  # Get the model's raw output (logits)
scores = output[0][0].detach().numpy()  # Detach the output tensor and convert to a NumPy array

# Apply softmax to get probabilities (since raw scores are logits)
scores = softmax(scores)

# Organize the scores into a dictionary for easy interpretation
scores_dict = {
    'roberta_neg': scores[0],  # Probability of negative sentiment
    'roberta_neu': scores[1],  # Probability of neutral sentiment
    'roberta_pos': scores[2],  # Probability of positive sentiment
}

# Print the sentiment scores
print(scores_dict)


# %%
# Defining a function to make sentiment analysis with RoBERTa more accessible
def polarity_scores_roberta(ex):
    """
    This function takes a text input `ex`, tokenizes it, 
    runs sentiment analysis using a pretrained RoBERTa model, 
    and returns a dictionary with sentiment scores.
    
    Parameters:
    ex (str): The input text to analyze.
    
    Returns:
    dict: A dictionary with sentiment probabilities for negative, neutral, and positive classes.
    """
    # Tokenizing the input text
    enc_text = tokenizer(ex, return_tensors='pt')
    
    # Getting the model's output
    output = model(**enc_text)
    
    # Detaching the output tensor and applying softmax to get probabilities
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    
    # Organizing the scores into a dictionary
    scores_dict = {
        'roberta_neg': scores[0],  # Probability of negative sentiment
        'roberta_neu': scores[1],  # Probability of neutral sentiment
        'roberta_pos': scores[2],  # Probability of positive sentiment
    }
    
    return scores_dict


# %%
res = {}
for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        text = row['Text']
        myid = row['Id']

        # Skip if text is empty or not a valid string
        if not isinstance(text, str) or not text.strip():
            continue

        # VADER Sentiment
        vader_result = sia.polarity_scores(text)
        vader_result_rename = {f"vader_{key}": value for key, value in vader_result.items()}

        # RoBERTa Sentiment
        roberta_result = polarity_scores_roberta(text)

        # Merging both results
        both = {**vader_result_rename, **roberta_result}
        res[myid] = both

    except Exception as e:
        print(f'Error processing id {myid}: {e}')

# %%
# Convert the results into a DataFrame
results_df = pd.DataFrame(res).T
results_df = results_df.reset_index().rename(columns={'index': 'Id'})

# Merge with the original DataFrame (df) on 'Id'
results_df = results_df.merge(df, how='left', on='Id')

# %%
# Displaying the final results
results_df.head()

# %% [markdown]
# ## Comparison of Sentiment Scores

# %% [markdown]
# ## VADER vs RoBERTa Sentiment Analysis

# %%
# Pairplot to compare VADER and RoBERTa sentiment scores
sns.pairplot(data=results_df,
             vars=['vader_neg', 'vader_neu', 'vader_pos',
                   'roberta_neg', 'roberta_neu', 'roberta_pos'],
             hue='Score',  # Group by the review score (hue)
             palette='tab10')  # Color palette for different categories
plt.show()  # Display the plot

# %% [markdown]
# # Hugging Face Transformers Pipeline (Additional)

# %%
from transformers import pipeline

sent_pipeline = pipeline("sentiment-analysis")

# %%
sent_pipeline('i love sentiment analysis')

# %%
sent_pipeline('too bad, expected better')


