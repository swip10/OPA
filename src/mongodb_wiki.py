import mwclient
import time
from tqdm import tqdm
from transformers import pipeline

from config import config
from src.db.mongodb import MongoOPA
# from src.db.mongodb import Collection

# config pour selectioner le serveur et le port
client = MongoOPA(
    host=config.mongodb_host,
    port=config.mongodb_port
)

# client.opa_db.drop_collection(str(Collection.WIKI))
endid = client.get_wiki_last_revision() + 1

site = mwclient.Site("en.wikipedia.org")
page = site.pages["Cryptocurrency"]
revs = list(page.revisions(endid=endid))
print(revs)

# sort the revisions by timestamp
revs = sorted(revs, key=lambda revision: revision["timestamp"])

# use sentiment analysis to know if POSITIVE or NEGATIVE is a phrase
sentiment_pipeline = pipeline("sentiment-analysis")
# print(sentiment_pipeline(["I love you"]))
# print(sentiment_pipeline(["Even if that burger was looking good it was very unpleasant to eat"]))


def find_sentiment(text, pipline_sent):
    """
    :return float between [-1, 1] with -1 for negative sentence and 1 for positive
    """
    sent = pipline_sent([text[:250]])[0]
    score = sent["score"]
    if sent["label"] == "NEGATIVE":
        score *= -1.
    return score


# create a dictionary of editions
edits = dict()
for rev in tqdm(revs):
    date = time.strftime('%Y-%m-%d', rev["timestamp"])  # group sentiments by day

    if "comment" not in rev or "revid" not in rev:
        continue

    if date not in edits:
        edits[date] = dict(sentiments=list(), edit_count=0, revid=list())

    edits[date]["edit_count"] += 1
    comment = rev["comment"]
    edits[date]["sentiments"].append(find_sentiment(comment, sentiment_pipeline))
    edits[date]["revid"].append(rev["revid"])
    edits[date]["timestamp"] = date

documents = [elem_dict for elem_dict in edits.values()]
client.initialize_with_wiki_revisions(editions=documents, reset=False)
