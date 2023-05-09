import json
import sys
import pandas as pd
import numpy as np
import os
from sklearn.cluster import KMeans
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def Weight_Loss(param_age, param_food_type, param_weight, param_height):
    # print(" Age: %s\n Veg-NonVeg:%s\n Weight%s\n Hight%s\n" % (param_age, param_food_type,param_weight, param_height))

    # In[44]:

    # Reading of the Dataet
    data = pd.read_csv('input.csv')

    # In[46]:

    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    print('BREAKFAST FOOD ITEMS')
    print(breakfastfoodseparated)
    print('LUNCH FOOD ITEMS')
    print(Lunchfoodseparated)
    print('DINNER FOOD ITEMS')
    print(Dinnerfoodseparated)

    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    print(LunchfoodseparatedIDdata)

    # retrieving rows by loc method
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    print(breakfastfoodseparatedIDdata)

    # retrieving rows by loc method
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    print(DinnerfoodseparatedIDdata)
    print(DinnerfoodseparatedIDdata.describe())
    # DinnerfoodseparatedIDdata.iloc[]

    # In[47]:
    age = int(param_age)
    veg = float(param_food_type)
    weight = float(param_weight)
    height = float(param_height)
    bmi = weight/(height**2)
    agewiseinp = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if (i == age):
                print('age is between', str(lp), str(lp+10))
                tr = round(lp/20)
                agecl = round(lp/20)
    # In[280]:

    # conditions
    print("Your body mass index is: ", bmi)
    if (bmi < 16):
        print("severely underweight")
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi = 1
    elif (bmi >= 30):
        print("severely overweight")
        clbmi = 0
    val1 = DinnerfoodseparatedIDdata.describe()
    valTog = val1.T
    print(valTog.shape)
    print(valTog)
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (clbmi+agecl)/2
    print(val)

    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)

    # In[132]:
    # K-Means Based  Dinner Food

    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    print(kmeans.predict([Datacalorie[0]]))
    XValu = np.arange(0, len(kmeans.labels_))

    dnrlbl = kmeans.labels_

    # In[49]:
    # K-Means Based  lunch Food

    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)

    XValu = np.arange(0, len(kmeans.labels_))
    lnchlbl = kmeans.labels_
    # In[128]:
    # K-Means Based  lunch Food

    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    XValu = np.arange(0, len(kmeans.labels_))

    brklbl = kmeans.labels_
    inp = []
    # Reading of the Dataet
    datafin = pd.read_csv('inputfin.csv')
    datafin.head(5)
    # train set
    arrayfin = [agecl, clbmi,]
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    print(weightgaincat)
    print(len(weightlosscat))
    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightlosscat), 6), dtype=np.float32)

    print('####################')
    # In[287]:
    # randomforest
    for jj in range(len(weightlosscat)):
        valloc = list(weightlosscat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti
    print(X_test)
    print(len(weightlosscat))
    print(len(X_test))
    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X_train = weightlossfin  # Features
    y_train = yt  # Labels

    # Split dataset into training set and test set
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X_train = weightlossfin  # Features
    y_train = yt  # Labels

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    print(X_test[1])
    X_test2 = X_test
    y_pred = clf.predict(X_test)
    findata = []
    print('SUGGESTED FOOD ITEMS ::')
    for ii in range(len(y_pred)):
        if y_pred[ii] == 2:  # weightloss
            print('#################')
            print(Food_itemsdata[ii])

            findata.append(Food_itemsdata[ii])
            if int(veg) == 1:
                datanv = ['Chicken Burger']
                for it in range(len(datanv)):
                    if findata == datanv[it]:
                        print('VegNovVeg')

    with open("./output.json", "w") as f:
        json.dump(findata, f)


def Weight_Gain(param_age, param_food_type, param_weight, param_height):

    data = pd.read_csv('input.csv')
    data.head(5)
    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    print('BREAKFAST FOOD ITEMS')
    print(breakfastfoodseparated)
    print('LUNCH FOOD ITEMS')
    print(Lunchfoodseparated)
    print('DINNER FOOD ITEMS')
    print(Dinnerfoodseparated)

    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    print(LunchfoodseparatedIDdata)

    # retrieving rows by loc method
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    print(breakfastfoodseparatedIDdata)

    # retrieving rows by loc method
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    print(DinnerfoodseparatedIDdata)
    print(DinnerfoodseparatedIDdata.describe())
    # DinnerfoodseparatedIDdata.iloc[]

    # In[47]:
    age = int(param_age)
    veg = float(param_food_type)
    weight = float(param_weight)
    height = float(param_height)
    bmi = weight/(height**2)
    agewiseinp = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if (i == age):
                print('age is between', str(lp), str(lp+10))
                tr = round(lp/20)
                agecl = round(lp/20)
    # In[280]:

    # conditions
    print("Your body mass index is: ", bmi)
    if (bmi < 16):
        print("severely underweight")
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi = 1
    elif (bmi >= 30):
        print("severely overweight")
        clbmi = 0
    val1 = DinnerfoodseparatedIDdata.describe()
    valTog = val1.T
    print(valTog.shape)
    print(valTog)
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi+agecl)/2
    print(val)
    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)

    # In[132]:
    # K-Means Based  Dinner Food

    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    print(kmeans.predict([Datacalorie[0]]))
    XValu = np.arange(0, len(kmeans.labels_))
    dnrlbl = kmeans.labels_
    # K-Means Based  lunch Food

    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    XValu = np.arange(0, len(kmeans.labels_))
    lnchlbl = kmeans.labels_

    # In[128]:
    # K-Means Based  lunch Food

    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    XValu = np.arange(0, len(kmeans.labels_))
    brklbl = kmeans.labels_

    inp = []
    # Reading of the Dataet
    datafin = pd.read_csv('inputfin.csv')
    datafin.head(5)
    # train set
    arrayfin = [agecl, clbmi,]
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    print(weightgaincat)
    print(len(weightlosscat))
    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            print(valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(weightgaincat), 10), dtype=np.float32)

    print('####################')
    # In[287]:
    for jj in range(len(weightgaincat)):
        valloc = list(weightgaincat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti
    print(X_test)
    print(len(weightlosscat))
    print(weightgainfin.shape)
    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    X_train = weightgainfin  # Features
    y_train = yr  # Labels

    # Split dataset into training set and test set
#    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3) #

    # Import train_test_split function
    from sklearn.model_selection import train_test_split

    # X_train= weightlossfin# Features
    # y_train=yt # Labels

    # Import Random Forest Model
    from sklearn.ensemble import RandomForestClassifier

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    print(X_test[1])
    X_test2 = X_test
    y_pred = clf.predict(X_test)
    print('ok')
    findata = []
    print('SUGGESTED FOOD ITEMS ::')
    for ii in range(len(y_pred)):
        if y_pred[ii] == 1:
            print('#################')
            print(Food_itemsdata[ii])
            findata.append(Food_itemsdata[ii])
            if int(veg) == 1:
                datanv = ['Chicken Burger']
                for it in range(len(datanv)):
                    if findata == datanv[it]:
                        print('VegNovVeg')
    with open("./output.json", "w") as f:
        json.dump(findata, f)


def Healthy(param_age, param_food_type, param_weight, param_height):

    # # New Section
    # Reading of the Dataet
    data = pd.read_csv('./input.csv')
    data.head(5)
    Breakfastdata = data['Breakfast']
    BreakfastdataNumpy = Breakfastdata.to_numpy()

    Lunchdata = data['Lunch']
    LunchdataNumpy = Lunchdata.to_numpy()

    Dinnerdata = data['Dinner']
    DinnerdataNumpy = Dinnerdata.to_numpy()

    Food_itemsdata = data['Food_items']
    breakfastfoodseparated = []
    Lunchfoodseparated = []
    Dinnerfoodseparated = []

    breakfastfoodseparatedID = []
    LunchfoodseparatedID = []
    DinnerfoodseparatedID = []

    for i in range(len(Breakfastdata)):
        if BreakfastdataNumpy[i] == 1:
            breakfastfoodseparated.append(Food_itemsdata[i])
            breakfastfoodseparatedID.append(i)
        if LunchdataNumpy[i] == 1:
            Lunchfoodseparated.append(Food_itemsdata[i])
            LunchfoodseparatedID.append(i)
        if DinnerdataNumpy[i] == 1:
            Dinnerfoodseparated.append(Food_itemsdata[i])
            DinnerfoodseparatedID.append(i)

    print('BREAKFAST FOOD ITEMS')
    print(breakfastfoodseparated)
    print('LUNCH FOOD ITEMS')
    print(Lunchfoodseparated)
    print('DINNER FOOD ITEMS')
    print(Dinnerfoodseparated)

    # retrieving rows by loc method |
    LunchfoodseparatedIDdata = data.iloc[LunchfoodseparatedID]
    print(LunchfoodseparatedID)
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.iloc[Valapnd]
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.T
    print(LunchfoodseparatedIDdata)

    # retrieving rows by loc method
    breakfastfoodseparatedIDdata = data.iloc[breakfastfoodseparatedID]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.iloc[Valapnd]
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.T
    print(breakfastfoodseparatedIDdata)

    # retrieving rows by loc method
    DinnerfoodseparatedIDdata = data.iloc[DinnerfoodseparatedID]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    val = list(np.arange(5, 15))
    Valapnd = [0]+val
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.iloc[Valapnd]
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.T
    print(DinnerfoodseparatedIDdata)
    print(DinnerfoodseparatedIDdata.describe())
    # DinnerfoodseparatedIDdata.iloc[]

    # In[47]:
    age = int(param_age)
    veg = float(param_food_type)
    weight = float(param_weight)
    height = float(param_height)
    bmi = weight/(height**2)
    agewiseinp = 0

    for lp in range(0, 80, 20):
        test_list = np.arange(lp, lp+20)
        for i in test_list:
            if (i == age):
                print('age is between', str(lp), str(lp+10))
                tr = round(lp/20)
                agecl = round(lp/20)
    # In[280]:

    # conditions
    print("Your body mass index is: ", bmi)
    if (bmi < 16):
        print("severely underweight")
        clbmi = 4
    elif (bmi >= 16 and bmi < 18.5):
        print("underweight")
        clbmi = 3
    elif (bmi >= 18.5 and bmi < 25):
        print("Healthy")
        clbmi = 2
    elif (bmi >= 25 and bmi < 30):
        print("overweight")
        clbmi = 1
    elif (bmi >= 30):
        print("severely overweight")
        clbmi = 0
    val1 = DinnerfoodseparatedIDdata.describe()
    valTog = val1.T
    print(valTog.shape)
    print(valTog)
    DinnerfoodseparatedIDdata = DinnerfoodseparatedIDdata.to_numpy()
    LunchfoodseparatedIDdata = LunchfoodseparatedIDdata.to_numpy()
    breakfastfoodseparatedIDdata = breakfastfoodseparatedIDdata.to_numpy()
    ti = (bmi+agecl)/2
    print(val)

    dt = np.delete(DinnerfoodseparatedIDdata, [1, 3], axis=1)
    print(dt)

    # In[132]:
    # K-Means Based  Dinner Food

    Datacalorie = DinnerfoodseparatedIDdata[1:, 1:len(
        DinnerfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    print(kmeans.predict([Datacalorie[0]]))
    XValu = np.arange(0, len(kmeans.labels_))

    dnrlbl = kmeans.labels_

    # In[49]:
    # K-Means Based  lunch Food

    Datacalorie = LunchfoodseparatedIDdata[1:, 1:len(LunchfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    XValu = np.arange(0, len(kmeans.labels_))

    lnchlbl = kmeans.labels_

    # In[128]:
    # K-Means Based  lunch Food

    Datacalorie = breakfastfoodseparatedIDdata[1:, 1:len(
        breakfastfoodseparatedIDdata)]
    print(Datacalorie)
    X = np.array(Datacalorie)
    kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
    print('## Prediction Result ##')
    print(kmeans.labels_)
    XValu = np.arange(0, len(kmeans.labels_))

    brklbl = kmeans.labels_
    print(len(brklbl))

    inp = []
    # Reading of the Dataet
    datafin = pd.read_csv('./inputfin.csv')
    datafin.head(5)
    # train set
    # arrayfin=[agecl,clbmi,]
    dataTog = datafin.T
    bmicls = [0, 1, 2, 3, 4]
    agecls = [0, 1, 2, 3, 4]
    weightlosscat = dataTog.iloc[[1, 2, 7, 8]]
    weightlosscat = weightlosscat.T
    weightgaincat = dataTog.iloc[[0, 1, 2, 3, 4, 7, 9, 10]]
    weightgaincat = weightgaincat.T
    healthycat = dataTog.iloc[[1, 2, 3, 4, 6, 7, 9]]
    healthycat = healthycat.T
    weightlosscatDdata = weightlosscat.to_numpy()
    weightgaincatDdata = weightgaincat.to_numpy()
    healthycatDdata = healthycat.to_numpy()
    weightlosscat = weightlosscatDdata[1:, 0:len(weightlosscatDdata)]
    weightgaincat = weightgaincatDdata[1:, 0:len(weightgaincatDdata)]
    healthycat = healthycatDdata[1:, 0:len(healthycatDdata)]

    print(weightgaincat)
    print(len(weightlosscat))
    weightlossfin = np.zeros((len(weightlosscat)*5, 6), dtype=np.float32)
    weightgainfin = np.zeros((len(weightgaincat)*5, 10), dtype=np.float32)
    healthycatfin = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    t = 0
    r = 0
    s = 0
    yt = []
    yr = []
    ys = []
    for zz in range(5):
        for jj in range(len(weightlosscat)):
            valloc = list(weightlosscat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightlossfin[t] = np.array(valloc)
            yt.append(brklbl[jj])
            t += 1
        for jj in range(len(weightgaincat)):
            valloc = list(weightgaincat[jj])
            print(valloc)
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            weightgainfin[r] = np.array(valloc)
            yr.append(lnchlbl[jj])
            r += 1
        for jj in range(len(healthycat)):
            valloc = list(healthycat[jj])
            valloc.append(bmicls[zz])
            valloc.append(agecls[zz])
            healthycatfin[s] = np.array(valloc)
            ys.append(dnrlbl[jj])
            s += 1

    X_test = np.zeros((len(healthycat)*5, 9), dtype=np.float32)
    print('####################')
    # In[287]:
    for jj in range(len(healthycat)):
        valloc = list(healthycat[jj])
        valloc.append(agecl)
        valloc.append(clbmi)
        X_test[jj] = np.array(valloc)*ti
    print(X_test)
    print(len(weightlosscat))
    print(weightgainfin.shape)

    X_train = healthycatfin  # Features
    y_train = ys  # Labels

    # Create a Gaussian Classifier
    clf = RandomForestClassifier(n_estimators=100)

    # Train the model using the training sets y_pred=clf.predict(X_test)
    clf.fit(X_train, y_train)

    print(X_test[1])
    X_test2 = X_test
    y_pred = clf.predict(X_test)
    print('ok')

    print('SUGGESTED FOOD ITEMS ::')
    findata = []
    for ii in range(len(y_pred)):
        if y_pred[ii] == 2:
            print('#################')
            try:
                print(Food_itemsdata[ii])
                findata.append(Food_itemsdata[ii])
            except KeyError:
                print(f"{ii} is not a valid index in Food_itemsdata")

            if int(veg) == 1:
                datanv = ['Chicken Burger']
                for it in range(len(datanv)):
                    if findata == datanv[it]:
                        print('VegNovVeg')
    with open("./output.json", "w") as f:
        json.dump(findata, f)


choice = sys.argv[1]
if (choice == '1'):
    Weight_Gain(sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
if (choice == '2'):
    Weight_Loss(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
if (choice == '3'):
    Healthy(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
