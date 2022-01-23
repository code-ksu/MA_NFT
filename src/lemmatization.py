import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk import pos_tag

lemmatizer = WordNetLemmatizer()
stop_words = stopwords.words('english') #filter out stop words
tag_dict = {
    "J": wordnet.ADJ,
    "N": wordnet.NOUN,
    "V": wordnet.VERB,
    "R": wordnet.ADV
}
def lemma_text(text):
    new_text_array = []
    sentences = sent_tokenize(text.lower(), language = 'english')
    for sent in sentences:
        tokens = word_tokenize(sent, language = 'english')
        tok_pos = nltk.pos_tag(tokens)
        for tok, pos in tok_pos:
            if tok in stop_words:
                continue
            tok_type = pos[0].upper() # gets N for noun
            if tok_type in ['.', ',', ':']: # ignore punctuation
                continue
                
            mapped_type = tag_dict.get(tok_type, wordnet.NOUN) # default NOUN
            if tok.startswith('.'):
                tok = tok[1:] 
            if tok.endswith('.'):
                tok = tok[:-1]   
            new_text_array.append(lemmatizer.lemmatize(tok, mapped_type))
    return new_text_array #' '.join(new_text_array)