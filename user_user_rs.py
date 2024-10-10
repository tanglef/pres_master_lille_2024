# %%
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

ratings = pd.read_csv(
    "https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_ratings_final.csv",
    encoding="latin-1",
)
display(ratings.sample(n=5, random_state=42))

# %%


def apply_pivot(df, fillby=None):
    if fillby is not None:
        return df.pivot_table(
            index="userId", columns="prod_name", values="rating"
        ).fillna(fillby)
    return df.pivot_table(index="userId", columns="prod_name", values="rating")


# 3.1 Dividing the dataset into train and test
train, test = train_test_split(ratings, test_size=0.30, random_state=42)
test = test[test.userId.isin(train.userId)]
# 3.2 Apply pivot operation and fillna used to replace NaN values with 0 i.e. where user didn't made any rating
df_train_pivot = apply_pivot(df=train, fillby=0)
df_test_pivot = apply_pivot(df=test, fillby=0)
# 3.3 dummy dataset (train and test)
## Train
dummy_train = train.copy()
dummy_train["rating"] = dummy_train["rating"].apply(lambda x: 0 if x >= 1 else 1)
dummy_train = apply_pivot(df=dummy_train, fillby=1)
## Test
dummy_test = test.copy()
dummy_test["rating"] = dummy_test["rating"].apply(lambda x: 1 if x >= 1 else 0)
dummy_test = apply_pivot(df=dummy_test, fillby=0)

# %%
df_train_pivot[
    (
        df_train_pivot["0.6 Cu. Ft. Letter A4 Size Waterproof 30 Min. Fire File Chest"]
        != 0
    )
    | (df_train_pivot["4C Grated Parmesan Cheese 100% Natural 8oz Shaker"] != 0)
]

# %%
# to calculate mean, use only ratings given by user instead of fillna by 0 as it increase denominator in mean
mean = np.nanmean(apply_pivot(df=train), axis=1)
df_train_subtracted = (apply_pivot(df=train).T - mean).T
# Make rating=0 where user hasn't given any rating
df_train_subtracted.fillna(0, inplace=True)
# Creating the User Similarity Matrix using pairwise_distance function. shape of user_correlation is userXuser i.e. 18025X18025
user_correlation = 1 - pairwise_distances(df_train_subtracted, metric="cosine")
user_correlation[np.isnan(user_correlation)] = 0
# user_correlation[user_correlation<0] = 0
# Convert the user_correlation matrix into dataframe
user_correlation_df = pd.DataFrame(user_correlation)
user_correlation_df["userId"] = df_train_subtracted.index
user_correlation_df.set_index("userId", inplace=True)
user_correlation_df.columns = df_train_subtracted.index.tolist()

# %%
user_predicted_ratings = np.dot(user_correlation, df_train_pivot)

# To find only product not rated by the user, ignore the product rated by the user by making it zero.
user_final_rating = np.multiply(user_predicted_ratings, dummy_train)

pickle.dump(user_final_rating, open("./model/user_final_rating.pkl", "wb"))
# %%


def find_top_recommendations(pred_rating_df, userid, topn):
    recommendation = pred_rating_df.loc[userid].sort_values(ascending=False)[0:topn]
    recommendation = (
        pd.DataFrame(recommendation)
        .reset_index()
        .rename(columns={userid: "predicted_ratings"})
    )
    return recommendation


user_input = str(input("Enter your user id"))
recommendation_user_user = find_top_recommendations(user_final_rating, user_input, 5)
recommendation_user_user["userId"] = user_input
print("Recommended products for user id:{} as below".format(user_input))
display(recommendation_user_user)
print("Earlier rated products by user id:{} as below".format(user_input))
display(train[train["userId"] == user_input].sort_values(["rating"], ascending=False))
# %%
################################
## STEP 01: Import Libraries  ##
################################
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from imblearn import over_sampling
from IPython.display import display

#############################
## STEP 02: Read Data    ####
#############################
# Reading product review sentiment file
df_prod_review = pd.read_csv(
    "https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_review_sentiment.csv",
    encoding="latin-1",
)
display(df_prod_review.sample(n=5, random_state=42))

#################################
## STEP 03: Data Preparation ####
#################################
x = df_prod_review["Review"]
y = df_prod_review["user_sentiment"]
print(
    "Checking distribution of +ve and -ve review sentiment: \n{}".format(
        y.value_counts(normalize=True)
    )
)
# Split the dataset into test and train
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=50
)

# As we saw above that data is imbalanced, balance training data using over sampling

ros = over_sampling.RandomOverSampler(random_state=0)
X_train, y_train = ros.fit_resample(pd.DataFrame(X_train), pd.Series(y_train))
print(
    "Checking distribution of +ve and -ve review sentiment after oversampling: \n{}".format(
        y_train.value_counts(normalize=True)
    )
)
# convert into list of string
X_train = X_train["Review"].tolist()


################################################################
## STEP 04: Feature Engineering (Convert text into numbers) ####
################################################################
word_vectorizer = TfidfVectorizer(
    strip_accents="unicode",
    token_pattern=r"\w{1,}",
    ngram_range=(1, 3),
    stop_words="english",
    sublinear_tf=True,
    max_df=0.80,
    min_df=0.01,
)

# Fiting it on Train
word_vectorizer.fit(X_train)
# transforming the train and test datasets
X_train_transformed = word_vectorizer.transform(X_train)
X_test_transformed = word_vectorizer.transform(X_test.tolist())
# print(list(word_vectorizer.get_feature_names()))

###############################################
## STEP 05: ML Model (Logistic Regression) ####
###############################################


def evaluate_model(y_pred, y_actual):
    print(classification_report(y_true=y_actual, y_pred=y_pred))
    # confusion matrix
    cm = confusion_matrix(y_true=y_actual, y_pred=y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    # Calculating the Sensitivity
    sensitivity = round(TP / float(FN + TP), 2)
    print("sensitivity: {}".format(sensitivity))
    # Calculating the Specificity
    specificity = round(TN / float(TN + FP), 2)
    print("specificity: {}".format(specificity))


# 4.1 Model Training
logit = LogisticRegression()
logit.fit(X_train_transformed, y_train)
# 4.2 Prediction on Train Data
y_pred_train = logit.predict(X_train_transformed)
# 4.3 Prediction on Test Data
y_pred_test = logit.predict(X_test_transformed)
# 4.4 Evaluation on Train
print("Evaluation on Train dataset ..")
evaluate_model(y_pred=y_pred_train, y_actual=y_train)
print("Evaluation on Test dataset ..")
# 4.5 Evaluation on Test
evaluate_model(y_pred=y_pred_test, y_actual=y_test)

############################
## STEP 06: Save Model  ####
############################
pickle.dump(logit, open("./model/logit_model.pkl", "wb"))
pickle.dump(word_vectorizer, open("./model/word_vectorizer.pkl", "wb"))

# %%
# Reading product review data
df_prod_review = pd.read_csv(
    "https://raw.githubusercontent.com/aakashgoel12/blogs/master/input/product_review.csv",
    encoding="latin-1",
)
display(df_prod_review.sample(n=5, random_state=42))


# %%
model = pickle.load(open("./model/logit_model.pkl", "rb"))
word_vectorizer = pickle.load(open("./model/word_vectorizer.pkl", "rb"))
user_final_rating = pickle.load(open("./model/user_final_rating.pkl", "rb"))


# %%
def find_top_recommendations(pred_rating_df, userid, topn):
    recommendation = pred_rating_df.loc[userid].sort_values(ascending=False)[0:topn]
    recommendation = (
        pd.DataFrame(recommendation)
        .reset_index()
        .rename(columns={userid: "predicted_ratings"})
    )
    return recommendation


def get_sentiment_product(x):
    ## Get review list for given product
    product_name_review_list = df_prod_review[df_prod_review["prod_name"] == x][
        "Review"
    ].tolist()
    ## Transform review list into DTM (Document/review Term Matrix)
    features = word_vectorizer.transform(product_name_review_list)
    ## Predict sentiment
    return model.predict(features).mean()


def find_top_pos_recommendation(
    user_final_rating,
    user_input,
    df_prod_review,
    word_vectorizer,
    model,
    no_recommendation,
):
    ## 10 is manually coded, need to change
    ## Generate top recommenddations using user-user based recommendation system w/o using sentiment analysis
    recommendation_user_user = find_top_recommendations(
        user_final_rating, user_input, 10
    )
    recommendation_user_user["userId"] = user_input
    ## filter out recommendations where predicted rating is zero
    recommendation_user_user = recommendation_user_user[
        recommendation_user_user["predicted_ratings"] != 0
    ]
    print(
        "Recommended products for user id:{} without using sentiment".format(user_input)
    )
    display(recommendation_user_user)
    ## Get overall sentiment score for each recommended product
    recommendation_user_user["sentiment_score"] = recommendation_user_user[
        "prod_name"
    ].apply(get_sentiment_product)
    ## Transform scale of sentiment so that it can be manipulated with predicted rating score
    scaler = MinMaxScaler(feature_range=(1, 5))
    scaler.fit(recommendation_user_user[["sentiment_score"]])
    recommendation_user_user["sentiment_score"] = scaler.transform(
        recommendation_user_user[["sentiment_score"]]
    )
    ## Get final product ranking score using 1*Predicted rating of recommended product + 2*normalized sentiment score on scale of 1–5 of recommended product
    recommendation_user_user["product_ranking_score"] = (
        1 * recommendation_user_user["predicted_ratings"]
        + 2 * recommendation_user_user["sentiment_score"]
    )
    print(
        "Recommended products for user id:{} after using sentiment".format(user_input)
    )
    ## Sort product ranking score in descending order and show only top `no_recommendation`
    display(
        recommendation_user_user.sort_values(
            by=["product_ranking_score"], ascending=False
        ).head(no_recommendation)
    )


# %%
user_input = str(input("Enter your user id"))
find_top_pos_recommendation(
    user_final_rating,
    user_input,
    df_prod_review,
    word_vectorizer,
    model,
    no_recommendation=5,
)

# %%
adj = np.array([[5, 1, 4, 2], [5, 1, np.nan, np.nan], [np.nan, 1, 5, np.nan]])
mean = np.nanmean(adj, axis=1)
adj[np.isnan(adj)] = 0
adj_sub = adj - mean.reshape(-1, 1)
corr = 1 - pairwise_distances(adj_sub, metric="cosine")
print(corr)
ratings = corr @ adj
print(ratings)
dummy_adj = np.where(adj > 0, 0, 1)
exp = np.multiply(ratings, dummy_adj)
print(exp)

# %%
