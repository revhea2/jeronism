from flask import Flask, render_template, request
from nltk import pos_tag, RegexpParser, word_tokenize, tag
import nltk

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        sentence = request.form['sentence']
        result = tag_noun_phrase(sentence.split('\n'))
        return render_template('index.html', sentence=sentence, result=result)
    return render_template('index.html', )




def _tag_np(sentence):
    tk_sentence = word_tokenize(sentence)
    tag_tk_sentence = pos_tag(tk_sentence)
    noun_pattern = """NP: {<PDT>?<DT>?<JJ>*(<NNP>+|(<NN>|<NNS>)+)(<IN><DT>?<JJ>*(<NN>|<NNS>)+)*}"""
    chunk = RegexpParser(noun_pattern)
    output = chunk.parse(tag_tk_sentence)
    res = []
    for chunk in output:
        if type(chunk) == nltk.tree.Tree:
            res.append(["np", ' '.join([x[0] for x in list(chunk)])])

        else:
            tagged = chunk[1]
            if "NN" in chunk[1] or "PRP" in chunk[1]:
                tagged = "np"
            res.append([tagged, chunk[0]])

    return res


def _tag_vp(np):
    vp = []
    i = 0
    while i < len(np):
        tagged, phrase = np[i]
        if i < len(np) - 1:
            if "VB" in tagged and np[i + 1][0] == "np":
                vp.append(["vp", phrase + " " + np[i + 1][1]])
                i += 2
                continue
        if "VB" in tagged:
            tagged = "vp"
        vp.append([tagged, phrase])
        i += 1
    return vp


def tag_noun_phrase(sentences):
    if not sentences:
        return []
    result = []
    for sentence in sentences:
        np = _tag_np(sentence)
        vp = _tag_vp(np)
        result.append(vp)

    return result


if __name__ == '__main__':
    app.run(debug=True)
    # app.run(debug=True, host='0.0.0.0')
