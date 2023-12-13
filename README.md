# Training of SVM & LSTM Models for Automated SNOMED-CT Labeling
Repository of codes and methods for the automatic coding of pathology reports. 

## Data
283.501 pathology reports, extracted from the laboratory information system along with their corresponding SNOMED-3 codes, during a 8-year period (from January 2014 to December 2022) were used as training and validation set for a ML model (SVM) and a RNN (LSTM). Text and codes were then stored in a .csv file. The 50 most frequent diagnosis (D) or morphology (M) codes are retained with relative diagnosis text and appropriately translated to SNOMED-CT codes.  
        
        //Most frequent 50 codes
        code_df=df.groupby("Labels").count().reset_index()
        
        code_df_sorted=code_df.sort_values(by="Sentences", ascending=False)
        
        top_50 = code_df_sorted.head(50)
        
        top_50_labels=code_df_sorted["Labels"][:50]
        
        df=df[df["Labels"].isin(top_50_labels)].reset_index(drop=True)

### Preprocessing (LSTM)
1. Classic NPL preprocessing steps such as removing stopwords, special characters, punctuation marks.
        
        #Removing punctuation
        
        import string
        
        def remove_punctuation(sentences):
        
            translator=str.maketrans(","," ", string.punctuation)
            
            return sentences.translate(translator)
            
        df["Sentences"]=df["Sentences"].str.replace('\d=','')
        
        df["Sentences"]=df["Sentences"].apply(remove_punctuation)
        
        stop = set(stopwords.words('italian'))
        
        punctuation = list(string.punctuation)
        
        stop.update(punctuation)
        
        #Removing HTML
        
        def strip_html(text):
        
            soup = BeautifulSoup(text, "html.parser")
            
            return soup.get_text()
            
        #Removing the square brackets
        
        def remove_between_square_brackets(text):
        
            return re.sub('\[[^]]*\]', '', text)
            
        #Removing URL's
        
        def remove_between_square_brackets(text):
        
            return re.sub(r'http\S+', '', text)
            
        #Removing the stopwords from text
        
        def remove_stopwords(text):
        
            final_text = []
            
            for i in text.split():
            
                if i.strip().lower() not in stop and i.strip().lower().isalpha():
                
                    final_text.append(i.strip().lower())
                    
            return " ".join(final_text)
            
        def denoise_text(text):
        
            text = strip_html(text)
            
            text = remove_between_square_brackets(text)
            
            text = remove_stopwords(text)
            
            return text
            
        df['Sentences']=df['Sentences'].apply(denoise_text)

2. Lemmatization
        
        for index, row in df2.iterrows():
        
            df2.at[index, "Sentences"] = utils2.clean(row["Sentences"])
            
            df2.at[index, "Labels"] = row["Labels"].strip()

3. Ouliers removal and Tokenization
        
        #The maximum number of words to be used. (most frequent)
        
        MAX_NB_WORDS = 50000
        
        #Max number of words in each sentence.
        
        MAX_SEQUENCE_LENGTH = 750
        
        #This is fixed.
        
        EMBEDDING_DIM = 100
        
        #Tokenizer
        
        tokenizer = Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
        
        tokenizer.fit_on_texts(df['Sentences'].values)
        
        word_index = tokenizer.word_index

4. Class balancing, keeping a maximum of 1000 rows per label.

        #Set the maximum number of rows per label
        
        max_rows_per_label = 1000
        
        #Create a mask to track which rows to keep
        
        keep_mask = np.zeros(len(df), dtype=bool)
        
        #Loop through each unique label in the dataframe
        
        for label in df["Labels"].unique():
    
        #Get the indices of the rows that correspond to the current label
        
        label_indices = df.index[df["Labels"] == label]
        
        #If there are more than max_rows_per_label rows, randomly select a subset
        
        if len(label_indices) > max_rows_per_label:
        
            selected_indices = np.random.choice(label_indices, size=max_rows_per_label, replace=False)
            
        else:
        
            selected_indices = label_indices
            
        #Update the mask to mark the selected rows as True
        
        keep_mask[selected_indices] = True

5. Padding
    
        X = tokenizer.texts_to_sequences(df['Sentences'].values)
        
        X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
        
        print('Shape of data tensor:', X.shape)

7. Encoding
     
        Y = pd.get_dummies(df['Labels']).values
        
        print('Shape of label tensor:', Y.shape)

## Model (LSTM)
    
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)
        
        print(X_train.shape,Y_train.shape)
        
        print(X_test.shape,Y_test.shape)
        
        model = Sequential()
        
        model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
        
        model.add(SpatialDropout1D(0.2))
        
        model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
        
        model.add(AttentionLayer())  # Add attention layer as a custom layer
        
        model.add(Dense(70, activation='softmax'))
        
        model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])
        
        print(model.summary())
![Cattura](https://github.com/Gizmopath/SVM-LSTM-models-for-automated-SNOMED-CT-labeling/assets/119873860/4338745c-bb42-4ad7-bf87-a005834ef84f)

## Hardware
Python 3.8.5
Ubuntu 20.04
Intel Core i7-10700K CPU, 32 GB of RAM. 
NVIDIA GeForce GTX 1080 GPU.
