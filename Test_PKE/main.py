import pke
import utils

if __name__ == "__main__":
    # text = "Keyphrase extraction is the task of automatically selecting a small set of phrases that best describe a given free text document. Supervised keyphrase extraction requires large amounts of labeled training data and generalizes very poorly outside the domain of the training data. At the same time, unsupervised systems have poor accuracy, and often do not generalize well, as they require the input document to belong to a larger corpus also given as input. Addressing these drawbacks, in this paper, we tackle keyphrase extraction from single documents with EmbedRank: a novel unsupervised method, that leverages sentence embeddings. EmbedRank achieves higher F-scores than graph-based state of the art systems on standard datasets and is suitable for real-time processing of large amounts of Web data. With EmbedRank, we also explicitly increase coverage and diversity among the selected keyphrases by introducing an embedding-based maximal marginal relevance (MMR) for new phrases. A user study including over 200 votes showed that, although reducing the phrases’ semantic overlap leads to no gains in F-score, our high diversity selection is preferred by humans."

    text = utils.load_str("./Data/temp.txt")

    # initialize a TopicRank extractor
    extractor = pke.unsupervised.MultipartiteRank()

    # load the content of the document and perform French stemming

    extractor.load_document(input=text,
                            language='en',
                            normalization="stemming")

    # keyphrase candidate selection, here sequences of nouns and adjectives
    # defined by the Universal PoS tagset
    extractor.candidate_selection(pos={"NOUN", "PROPN" "ADJ"})

    # candidate weighting, here using a random walk algorithm
    extractor.candidate_weighting(threshold=0.74, method='average')

    # N-best selection, keyphrases contains the 10 highest scored candidates as
    # (keyphrase, score) tuples
    keyphrases = extractor.get_n_best(n=10)

    print(keyphrases)
