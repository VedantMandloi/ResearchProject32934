import nltk
import pickle
import uvicorn
import logging

from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from starlette.applications import Starlette
from starlette.responses import UJSONResponse, PlainTextResponse

# from bert_serving.client import BertClient
# bc = BertClient()

log = logging.getLogger('')
log.setLevel(logging.WARNING)

# PATH_TO_BERT_CLASSIFIER = './prototype_classifier/bert_svm_model.sav'
# bert_clf = pickle.load(open(PATH_TO_BERT_CLASSIFIER, 'rb'))

PATH_TO_CLASSIFIER = './prototype_classifier/d2v_svm_model.sav'
PATH_TO_MODEL = './saved_models/doc2vec/combined_d2v_model'

app = Starlette()

d2v_svm_model = Doc2Vec.load(PATH_TO_MODEL)
d2v_clf = pickle.load(open(PATH_TO_CLASSIFIER, 'rb'))


def tokenize(text: str) -> list:
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens


def infer_vec(model: Doc2Vec, tagged_doc: TaggedDocument) -> list:
    vector = model.infer_vector(tagged_doc.words, steps=30)
    return vector


def get_doc2vec_prediction(text: str) -> str:
    tagged_text = TaggedDocument(words=tokenize(text), tags=[1])
    vector = infer_vec(d2v_svm_model, tagged_text)
    prediction = d2v_clf.predict([vector])
    return prediction[0]


# def get_bert_prediction(text: str) -> str:
#     embedding = bc.encode([text])
#     prediction = bert_clf(embedding)
#     return prediction[0]


@app.route('/prioritize/doc2vec', methods=['POST'])
async def predict_priority(request):
    text = await request.json()
    d2v_label = get_doc2vec_prediction(text['text'])
    return UJSONResponse({'Doc2Vec prediction': d2v_label})


# @app.route('/prioritize/bert', methods=['POST'])
# async def predict_bert_priority(request):
#     text = await request.json()
#     bert_label = get_bert_prediction(text['text'])
#     return UJSONResponse({'BERT prediction': bert_label})

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000, debug=False, log_level='info')