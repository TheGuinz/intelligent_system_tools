import os
import sys
import string
import re
from pathlib import Path
# numpy and pandas for data manipulation
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None # no warnings
# import nltk
import nltk
# import sklearn
from sklearn.metrics.pairwise import cosine_similarity # to calculate the similarities between keyword candidates and text
# import contractions
import contractions
# import spacy - library for advanced natural languages processing
import spacy
# os.system('python -m spacy download en') # used for keywords filling
nlp = spacy.load("en_core_web_sm")
# os.system('python -m spacy download xx_ent_wiki_sm') # used for locations filling
nlp_wk = spacy.load("xx_ent_wiki_sm")
# import spellchecker
from spellchecker import SpellChecker
# import queue and threading
import queue
from threading import Thread
# import tensforflow_hub
import tensorflow_hub as hub
embed = hub.load("https://tfhub.dev/google/tf2-preview/nnlm-en-dim128/1")
os.environ['TFHUB_CACHE_DIR'] = f'{str(Path.home())}/workspace/tf_cache' # path for downloading pre traind model
# import area code
import area_code_nanp # US area code
# import longitude latitude
from geopy.geocoders import Nominatim


def _similarity(text_a, text_b, nu_of_keywords=5):
    """
        Description: Find similarity of 2 given text using a pre-trained model

        :param text_a [string]: text 1
        :param text_b [list]: text 2
        :param nu_of_keywords:
    """

    # load pretraind model for similarity
    text_embedding = embed([text_a])
    results_embeddings = embed(text_b)

    # calculate the similarity between document and results embeddings
    distances = cosine_similarity(text_embedding, results_embeddings)

    # get the top similar keywords
    keywords = [text_b[index] for index in distances.argsort()[0][-nu_of_keywords:]]

    # get the indices of minimum distance in numpy array
    keyword = keywords[np.where(distances == np.amin(distances))[0].tolist()[0]]

    return keyword


def _format(text, correct_spelling=True, remove_emojis=True, remove_stop_words=True):
    """
        Description: Apply function to clean a given text.

        :param tweet:
        :param correct_spelling:
        :param remove_emojis:
        :param remove_stop_words:

        Example:
        input: 'Barbados #Bridgetown JAMAICA ÛÒ Two cars set ablaze: SANTA CRUZ ÛÓ Head of the St Elizabeth Police Superintende...  http://t.co/wDUEaj8Q4J'
        output: 'barbados bridgetown jamaica  two cars set ablaze santa cruz  head elizabeth police superintend'
    """

    def _spellings(text, spell=SpellChecker()):
        """
            Correct the missplled words of a given tweet
        """
        text = text.split()
        misspelled = spell.unknown(text)
        result = map(lambda word : spell.correction(word) if word in misspelled else word, text)

        return " ".join(result)

    text = text.lower().strip()

    # remove urls
    text = re.compile(r'https?://\S+|www\.\S+').sub(r'', text)

    # remove html tags
    text = re.compile(r'<.*?>').sub(r'', text)

    # using contractions.fix to expand the shotened words
    text = contractions.fix(text).lower().strip()

    # remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # check for spelling errors
    if correct_spelling:
        text = _spellings(text)

    # remove emojis if any
    if remove_emojis:
        text = text.encode('ascii', 'ignore').decode('utf8').strip()

    # remove numbers if any
    text = ''.join([i for i in text if not i.isdigit()])

    # remove spaces
    text = re.sub(' +', ' ', text)

    # remove stop words (Examples of stop words in English are “a”, “the”, “is”, “are” and etc)
    if remove_stop_words:
        text = ' '.join([word for word in text.split(' ') if word not in nlp.Defaults.stop_words])

    return text


def _keyword(text, keywords):
    """
        Description: Apply function to extract a keyword from a given text

        :param text [string]: a text to search a keyword
        :param keywords [list]: known keywords list

        Example:
        input 1: 'our deeds are the reason of this earthquake may allah forgive us all'
        input 2: ['earthquake', 'aftershock', 'accident', ..., 'annihilation']
        output: 'earthquake'
    """
    keyword = ""

    # first, check if any word from the given text is already defined as a keyword somewhere else
    for k in text.split(' '):
        if k in keywords:
            keyword = k

    if not keyword:
        text_ = nlp(text)

        # custom list of parts-of-speech (pos) tags we are interested in
        pos_tag = ['VERB', 'NOUN", "ADJ", "PROPN']
        result = []

        # if the token pos tag matches one of the pos_tag, then add the text form of the token to result list
        for token in text_:
            if (token.pos_ in pos_tag):
                result.append(token.text)

        # find similarity between each possible keywords to the tweet itself by using a pre-traind model
        keyword = _similarity(text, result) if result else "NaN"

    return keyword


def _location(text):
    """
        Description: Apply function to extract a location from a given text

        :param text:

        Example:
        [1] input: 'tha kicks antiblight loan effort memphis'
            output: 'memphis'
        [2] input: 'mourning notices ny stabbing arson victims stir politics grief posters shira bank'
            output: 'New York'
        [3] input: 'mourning notices ny stabbing arson victims stir politics grief posters shira bank israel'
            output: 'Israel'
        [4] input: '301'
            output: 'Texas'
    """

    # area code (e.g 301 -> Texas)
    if re.search("[2-9][0-9][0-9]", text) and text.isdigit():
        if int(text) >= 201:
            location = area_code_nanp.get_region(int(text))
    # longitude and latitude to address
    elif re.search(r"[-+]?\d*\.\d+|\d+", text) and len(re.findall(r"[-+]?\d*\.\d+|\d+", text)) == 2:
        result = re.findall(r"[-+]?\d*\.\d+|\d+", text)
        try:
            lat, lon = result[0], result[1]
            # calling the nominatim tool
            geoLoc = Nominatim(user_agent="GetLoc")
            # passing the coordinates
            locname = geoLoc.reverse(f"{lat}, {lon}")
            # get address/location name
            location = locname.address[locname.address.rfind('')+2:]
        except:
            location = "NaN"
            pass
    else:
        us_state_to_abbrev = {
            "Alabama": "AL",
            "Alaska": "AK",
            "Arizona": "AZ",
            "Arkansas": "AR",
            "California": "CA",
            "Colorado": "CO",
            "Connecticut": "CT",
            "Delaware": "DE",
            "Florida": "FL",
            "Georgia": "GA",
            "Hawaii": "HI",
            "Idaho": "ID",
            "Illinois": "IL",
            "Indiana": "IN",
            "Iowa": "IA",
            "Kansas": "KS",
            "Kentucky": "KY",
            "Louisiana": "LA",
            "Maine": "ME",
            "Maryland": "MD",
            "Massachusetts": "MA",
            "Michigan": "MI",
            "Minnesota": "MN",
            "Mississippi": "MS",
            "Missouri": "MO",
            "Montana": "MT",
            "Nebraska": "NE",
            "Nevada": "NV",
            "New Hampshire": "NH",
            "New Jersey": "NJ",
            "New Mexico": "NM",
            "New York": "NY",
            "North Carolina": "NC",
            "North Dakota": "ND",
            "Ohio": "OH",
            "Oklahoma": "OK",
            "Oregon": "OR",
            "Pennsylvania": "PA",
            "Rhode Island": "RI",
            "South Carolina": "SC",
            "South Dakota": "SD",
            "Tennessee": "TN",
            "Texas": "TX",
            "Utah": "UT",
            "Vermont": "VT",
            "Virginia": "VA",
            "Washington": "WA",
            "West Virginia": "WV",
            "Wisconsin": "WI",
            "Wyoming": "WY",
            "District of Columbia": "DC",
            "American Samoa": "AS",
            "Guam": "GU",
            "Northern Mariana Islands": "MP",
            "Puerto Rico": "PR",
            "United States Minor Outlying Islands": "UM",
            "U.S. Virgin Islands": "VI"}

        # text_ = nlp_wk(text)
        text_ = nlp(text)

        loc = []

        for ent in text_.ents:
            if(ent.label_ == "GPE"):
                loc.append(ent.text)

        if not loc:
            abbrev_to_us_state = dict(map(reversed, us_state_to_abbrev.items()))
            for item in text.upper().split(' '):
                if item in abbrev_to_us_state:
                    loc.append(abbrev_to_us_state.get(item))

        if len(loc) == 0:
            # no location was found
            location = ""
        elif len(loc) == 1:
            # a single location was found
            print("LOC:", loc)
            location = loc[0]
            print("HERE 2")
        elif len(loc) >= 2:
            # if more than 2 locations were found, then find similarity between each possible location to the text itself by using a pre-traind model
            location = _similarity(text, loc)

    return location.lower()


def text_processing(input_df):
    # Preprocess data:

    # Steps:
    # [1] Format tweet: correct spelling, remove emojis, and remove stop_words
    # [2] Format keyword: fill missing keywords for certain tweets following specific scenarios
    # [3] Format location: fill missing locations for certain tweets following specific scenarios

    # get all available keywords from the the data (unique values) for step [2], but first replace all '%20' with ' '
    for i in range(len(input_df["keyword"])):
        k = input_df['keyword'].iloc[i]
        try:
            if "%20" in k:
                input_df['keyword'].iloc[i] = k[:k.index("%20")] + ' ' + k[k.index("%20") + len("%20"):]
        except:
            pass
    train_keywords = input_df['keyword'].unique()


    def _thread_func(q, result):
        """
        Threaded function for queue processing
        """
        while not q.empty():
            # Fetch new work from the Queue
            work = q.get()
            try:
                # index, keyword, location, tweet, target
                i, k, l, tw, ta = work[0], work[1], work[2], work[3], work[4]

                # Step 1 - tweet text formatting
                tw = _format(tw)

                # Step 2 - keyword formatting
                if pd.isnull(k):
                    # keyword is empty, so search for a keyword within the tweet itself,
                    # if no keyword is found then fill with NaN
                    keyword = _keyword(tw, train_keywords)
                    k = keyword if keyword else "NaN"

                # Step 3 - location formatting
                if pd.isnull(l):
                    # location is empty, so search for a location within the tweet itself,
                    # if no location is found then fill with NaN
                    location = _location(tw)
                    l = location if location else "NaN"
                else:
                    # location is not empty, so first make sure there's no a location within the tweet itself
                    location_tweet = _location(tw)
                    location_legit = _location(l)
                    # scenarios:
                    # - location found within the tweet, so overwrite the value under location with it
                    # - location is not found within the tweet, so make sure the given location not some garbage text
                    # - location is not legit, replace it with NaN
                    l = location_tweet if location_tweet else location_legit if location_legit else "NaN"

                # Store data back at correct index
                result[i] = (k, l, tw, ta)
            except Exception as e:
                result[i] = {("ERROR!", e, work[0], work[1], work[2], work[3], work[4])}
                pass

            # Signal to the queue that the task has been processed
            q.task_done()

        return True

    all_raw_tweets = []

    for i in range(len(input_df)):
        all_raw_tweets.append(input_df.iloc[i])

    # Set up a queue to hold all the tweets
    q = queue.Queue(maxsize=0)

    # Use many threads (50 max, or one for each file)
    num_threads = min(50, len(all_raw_tweets))

    # Populating Queue with tasks
    tweets = [{} for x in all_raw_tweets]

    # Load up the queue with the raw text to get the format vesrion of each one
    for i in range(len(all_raw_tweets)):
        # extract keyword, location, tweet, and target, and put as a queue item with id
        k, l, tw, ta = all_raw_tweets[i][0], all_raw_tweets[i][1], all_raw_tweets[i][2], all_raw_tweets[i][3]
        q.put((i, k, l, tw, ta))

    # Starting worker threads on queue processing
    for i in range(num_threads):
        worker = Thread(target=_thread_func, args=(q, tweets))
        # Setting threads as "daemon" allows main program to exit eventually even if these dont finish correctly.
        worker.setDaemon(True)
        # Start worker thread
        worker.start()

    # Wait until the queue has been processed
    q.join()

    df = pd.DataFrame(tweets, columns =['keyword', 'location', 'tweet', 'target'])

    return df