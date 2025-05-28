# Amazon Food Reviews Sentiment Analysis

This project performs sentiment analysis on Amazon Fine Food Reviews using two popular NLP techniques: **VADER** (a lexicon and rule-based sentiment analysis tool) and **RoBERTa**, a pretrained transformer-based model from the Hugging Face Transformers library.

---

## Project Overview

The goal of this project is to analyze the sentiment of customer reviews on Amazon food products and compare the performance and outputs of:

* **VADER Sentiment Analyzer** — Effective for social media and general text sentiment analysis.
* **RoBERTa-based Sentiment Classifier** — A transformer model fine-tuned on Twitter data for sentiment classification.

---

## Features

* Loads and preprocesses the Amazon Fine Food Reviews dataset (subset of 500 reviews for quick experimentation).
* Visualizes review score distribution to check for rating bias.
* Performs text preprocessing with NLTK including tokenization, POS tagging, and Named Entity Recognition.
* Runs sentiment analysis using both VADER and RoBERTa.
* Merges sentiment scores with original review data for comparison.
* Provides data visualizations comparing sentiment scores across review ratings.
* Demonstrates Hugging Face transformers pipeline for quick sentiment predictions.

---

## Dataset

* Source: [Amazon Fine Food Reviews Dataset on Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
* The dataset contains reviews, scores, and metadata related to Amazon food products.
* For performance reasons, the project uses the first 500 reviews subset.

---

## Installation

Make sure you have Python 3.x installed. Then install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn nltk tqdm transformers scipy
```

Additionally, download NLTK data for tokenization and tagging:

```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('vader_lexicon')
```

---

## Usage

1. Clone the repository:

```bash
git clone https://github.com/your-username/amazon-food-reviews-sentiment.git
cd amazon-food-reviews-sentiment
```

2. Place the `Reviews.csv` dataset in the appropriate path or modify the path in the script.

3. Run the notebook or script to:

* Load and preprocess the data.
* Perform sentiment analysis using VADER and RoBERTa.
* Visualize the results.

---

## Code Highlights

* Text tokenization and part-of-speech tagging using NLTK.
* Sentiment scoring using VADER.
* Sentiment classification using pretrained RoBERTa model via Hugging Face.
* Visualization of sentiment scores vs. review ratings using seaborn and matplotlib.
* Comparison plots highlighting differences between lexicon-based and transformer-based sentiment scores.

---

## Results

* Bar plots showing distribution of star ratings.
* Sentiment scores distributions across ratings (positive, neutral, negative).
* Pairplots comparing VADER and RoBERTa sentiment scores.

---

## Future Improvements

* Scale analysis to full dataset.
* Experiment with other transformer models or fine-tune RoBERTa on the dataset.
* Implement more advanced preprocessing like lemmatization and stopword removal.
* Build a web app for interactive sentiment prediction.

---

## License

This project is open-source and available under the MIT License.

---

## Contact

For questions or suggestions, feel free to reach out:

**chauhanikunj05@gmail.com**
