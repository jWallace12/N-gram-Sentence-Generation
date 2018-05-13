import nltk
import random
import math
import operator
from random import randint
from nltk import FreqDist
from nltk.util import ngrams
from collections import Counter

#Preprocesses the text.  Replaces singletons with "<UNK>" and appends
#<s> to the start of each sentence and </s> to the end
def processText(text):
    print("Processing...")
    text = '<s> ' + text
    text = text.replace('\r\n', ' </s>\r\n<s> ')
    text = ' </s>\r\n'.join(text.rsplit(' </s>\r\n<s> ', 1))
    textTkns = nltk.word_tokenize(text)
    textFD = FreqDist(textTkns)
    textSingles = textFD.hapaxes()
    for word in textSingles:
        text = text.replace(" " + word + " ", " <UNK> ")
    print("Done")
    return text

#Calculates the perplexity of the unigram language model
def calcUniPerplex(uniModel, numWords, testWords):
    total = 0
    for word in testWords:
        prob = 0.0
        word = (word,)
       	if (word in uniModel):
            prob = float(uniModel[word])/float(numWords)
        else:        
            prob = float(uniModel[("<UNK>",)])/float(numWords)
        total = total + (math.log(prob, 2))
    multiplier = (-1/float(len(testWords)))
    total = (total * multiplier)
    perplexity = math.pow(2, total)
    print("Unigram Perplexity: " + str(int(perplexity)))

#Calculates the perplexity of the bigram language model
def calcBiPerplex(uniModel, biModel, vocabSize, testBigrams, testLength, trainLength):
    total = 0
    prob = float(uniModel[testBigrams[0][0],]) / float(trainLength)
    total = total + (math.log(prob, 2))
    for bigram in testBigrams:
        if ((bigram[1],) not in uniModel):
		bigram = (bigram[0], "<UNK>")
	if ((bigram[0],) not in uniModel):
		bigram = ("<UNK>", bigram[1])
	if (bigram in biModel):
		prob = float(biModel[bigram] + 1) / float(uniModel[(bigram[0],)] + vocabSize)
	else:
		prob = float(1) / float(uniModel[(bigram[0],)] + vocabSize)
	total = total + (math.log(prob, 2))
    multiplier = (-1/float(testLength))
    total = (total * multiplier)
    perplexity = math.pow(2, total)
    print("Bigram Perplexity: " + str(int(perplexity)))
	
#Calculates the perplexity of the trigram language model
def calcTriPerplex(uniModel, biModel, triModel, vocabSize, testTrigrams, testLength, trainLength):
    total = 0
    prob = float(uniModel[testTrigrams[0][0],]) / float(trainLength)
    total = total + (math.log(prob, 2))
    prob = float(biModel[testTrigrams[0][0], testTrigrams[0][1]]) / float(uniModel[testTrigrams[0][0],])
    total = total + (math.log(prob, 2))
    for trigram in testTrigrams:
        prob = 0.0
	if ((trigram[2],) not in uniModel):
		trigram = (trigram[0], trigram[1], "<UNK>")
	if ((trigram[1],) not in uniModel):
		trigram = (trigram[0], "<UNK>", trigram[2])
	if ((trigram[0],) not in uniModel):
		trigram = ("<UNK>", trigram[1], trigram[2])
	if (trigram in triModel):
		prob = float(triModel[trigram] + 1) / float(biModel[trigram[0], trigram[1]] + vocabSize)
	else:
		if ((trigram[0], trigram[1]) in biModel):		
			prob = float(1) / float(biModel[(trigram[0], trigram[1])] + vocabSize)
		else:
			prob = float(1) / float(vocabSize + 1)		
	total = total + (math.log(prob, 2))
    multiplier = (-1/float(len(testTrigrams)))
    total = (total * multiplier)
    perplexity = math.pow(2, total)
    print("Trigram Perplexity: " + str(int(perplexity)))

#Generate random sentences by choosing probable unigrams, and calculates
#the probability of the corresponding sentence
def genUniSentence(uniModel, totalWords, numSent, minLen, maxLen):
    uniCandidates = []
    for word in uniModel:
        if (uniModel[word] > 500):
                uniCandidates.append(word[0])
    for i in range(0, numSent):
        sent = ""
        sentProb = 0.0
        sentLen = randint(maxLen-5, maxLen)
        currUni = uniCandidates[randint(0, len(uniCandidates)-1)]
        numWords = 0
        while currUni == "</s>":
            currUni = uniCandidates[randint(0, len(uniCandidates)-1)]
        sent = sent + currUni + " "
        sentProb = sentProb + math.log(float(uniModel[(currUni,)])/float(totalWords), 2)
        numWords = numWords + 1
        while currUni != "</s>" and numWords < sentLen:
            currUni = uniCandidates[randint(0, len(uniCandidates)-1)]
            while currUni == "</s>" and numWords < minLen:
                currUni = uniCandidates[randint(0, len(uniCandidates)-1)]
            while currUni == "<s>":
                currUni = uniCandidates[randint(0, len(uniCandidates)-1)]
            if currUni != "</s>":
                sent = sent + currUni + " "
                numWords = numWords + 1
                sentProb = sentProb + math.log(float(uniModel[(currUni,)])/float(totalWords), 2)
        print(sent)
        print("Sentence " + str(i+1) + " Log-Space Probability: " + str(sentProb) + "\n")
            
            
#Generates random sentences with probable bigrams, and calculates the
#probabilites of the corresponding sentences
def genBiSentence(uniModel, biModel, numSent, minLen, maxLen, vocab):
    vocabSize = len(vocab)
    sortedModel = sorted(biModel.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0, numSent):
        sent = ""
        numWords = 0
        sentProb = 0.0
        biCandidates = []
        j = 0
        while len(biCandidates) < 20 and j < len(sortedModel):
            if sortedModel[j][0][0] == "<s>":
                biCandidates.append(sortedModel[j])
            j = j + 1
        while len(biCandidates) < 20:
            randWord = vocab[randint(0, vocabSize-1)]
            while ("<s>", randWord) in biModel:
                randWord = vocab[randint(0, vocabSize-1)]
            biCandidates.append((("<s>", randWord), 1))
        currBi = biCandidates[randint(0, 19)]
        sent = sent + currBi[0][1] + " "
        numWords = numWords + 1
        if currBi[0] in biModel:
            sentProb = sentProb + math.log(float(biModel[currBi[0]] + 1)/float(uniModel[currBi[0][0],] + vocabSize), 2)
        else:
            sentProb = sentProb + math.log(float(1)/float(uniModel[currBi[0][0],] + vocabSize), 2)
        while currBi[0][1] != "</s>" and numWords < maxLen:
            biCandidates = []
            j = 0
            while len(biCandidates) < 20 and j < len(sortedModel):
                if sortedModel[j][0][0] == currBi[0][1]:
                    biCandidates.append(sortedModel[j])
                j = j + 1
            while len(biCandidates) < 20:
                randWord = vocab[randint(0, vocabSize-1)]
                while (currBi[0][1], randWord) in biModel:
                    randWord = vocab[randint(0, vocabSize-1)]
                biCandidates.append(((currBi[0][1], randWord), 1))
            currBi = biCandidates[randint(0, 19)]
            while currBi[0][1] == "</s>" and numWords < minLen:
                currBi = biCandidates[randint(0, 19)]
            if currBi[0][1] != "</s>":
                sent = sent + currBi[0][1] + " "
                numWords = numWords + 1
                if currBi[0] in biModel:
                    sentProb = sentProb + math.log(float(biModel[currBi[0]] + 1)/float(uniModel[currBi[0][0],] + vocabSize), 2)
                else:
                    sentProb = sentProb + math.log(float(1)/float(uniModel[currBi[0][0],] + vocabSize), 2)
        print(sent)
        print("Sentence " + str(i+1) + " Log-Space Probability: " + str(sentProb) + "\n")

#Generates random sentences by choosing probable trigrams, and calculates
#the probabilites of the corresponding sentences
def genTriSentence(uniModel, biModel, triModel, numSent, minLen, maxLen, vocab):
    vocabSize = len(vocab)
    sortedModel = sorted(triModel.items(), key=operator.itemgetter(1), reverse=True)
    for i in range(0, numSent):
        sent = ""
        numWords = 0
        sentProb = 0.0
        triCandidates = []
        j = 0
        while len(triCandidates) < 20 and j < len(sortedModel):
            if sortedModel[j][0][0] == "<s>" and sortedModel[j][0][1] == "<s>":
                triCandidates.append(sortedModel[j])
            j = j + 1
        while len(triCandidates) < 20:
            randWord = vocab[randint(0, vocabSize-1)]
            while ("<s>", "<s>", randWord) in biModel:
                randWord = vocab[randint(0, vocabSize-1)]
            triCandidates.append((("<s>", "<s>", randWord), 1))
        currTri = triCandidates[randint(0, 19)]
        sent = sent + currTri[0][2] + " "
        numWords = numWords + 1
        if currTri[0] in triModel:
            sentProb = sentProb + math.log(float(triModel[currTri[0]] + 1)/float(uniModel[currTri[0][0],] + vocabSize), 2)
        else:
            sentProb = sentProb + math.log(float(1)/float(uniModel[currTri[0][0],] + vocabSize), 2)
        while currTri[0][2] != "</s>" and numWords < maxLen:
            triCandidates = []
            j = 0
            while len(triCandidates) < 20 and j < len(sortedModel):
                if sortedModel[j][0][0] == currTri[0][1] and sortedModel[j][0][1] == currTri[0][2]:
                    triCandidates.append(sortedModel[j])
                j = j + 1
            while len(triCandidates) < 20:
                randWord = vocab[randint(0, vocabSize-1)]
                while (currTri[0][1], currTri[0][2], randWord) in triModel:
                    randWord = vocab[randint(0, vocabSize-1)]
                triCandidates.append(((currTri[0][1], currTri[0][2], randWord), 1))
            currTri = triCandidates[randint(0, 19)]
            while currTri[0][2] == "</s>" and numWords < minLen:
                currTri = triCandidates[randint(0, 19)]
            if currTri[0][2] != "</s>":
                sent = sent + currTri[0][2] + " "
                numWords = numWords + 1
                if currTri[0] in triModel:
                    sentProb = sentProb + math.log(float(triModel[currTri[0]] + 1)/float(biModel[(currTri[0][0], currTri[0][1])] + vocabSize), 2)
                else:
                    if (currTri[0][0], currTri[0][1]) in biModel:
                        sentProb = sentProb + math.log(float(1)/float(biModel[(currTri[0][0], currTri[0][1])] + vocabSize), 2)
                    else:
                        sentProb = sentProb + math.log(float(1)/float(1 + vocabSize), 2)
        print(sent)
        print("Sentence " + str(i+1) + " Log-Space Probability: " + str(sentProb) + "\n")

#Generates the language model from N-gram dictionary by recording the
#frequency of each ngram and saving the values into a hashmap
def genModel(ngramDict):
    model = {}
    for ngram in ngramDict:
        model[ngram] = ngramDict[ngram]
    return model

#Main function
def main():

    #Preprocess the train file
    train = processText(open("data/train.txt").read())
    test = open("data/test.txt").read()
    trainWords = train.split()
    testWords = test.split()

    #Retrieve and display the vocabulary
    print("Task 1: Training Data Vocabulary Size")
    vocab = set(trainWords)
    vocabSize = len(vocab)
    print("vocabSize = " + str(vocabSize))
    
    #generate models
    uniModel = genModel(Counter(ngrams(trainWords, 1)))
    biModel = genModel(Counter(ngrams(trainWords, 2)))
    triModel = genModel(Counter(ngrams(train.replace("<s> ", "<s> <s> ").split(), 3)))

    #Generate random sentences from each model
    print("\nTask 2: Generate Sentences")
    print("Unigrams:")
    genUniSentence(uniModel, len(trainWords), 10, 5, 20)
    print("\nBigrams:")
    genBiSentence(uniModel, biModel, 10, 5, 20, list(vocab))
    print("\nTrigrams")
    genTriSentence(uniModel, biModel, triModel, 10, 5, 20, list(vocab))

    #Calculate the perplexity of each model
    print("\nTask 3: Calculate Perplexity")
    calcUniPerplex(uniModel, len(trainWords), testWords)
    calcBiPerplex(uniModel, biModel, vocabSize, list(ngrams(testWords, 2)), len(testWords), len(trainWords))
    calcTriPerplex(uniModel, biModel, triModel, vocabSize, list(ngrams(testWords, 3)), len(testWords), len(trainWords))

main()
