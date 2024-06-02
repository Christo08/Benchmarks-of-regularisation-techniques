# z-scaling 5-fold cross-validation
import os
import pickle
import shutil

import pandas as pd
from matplotlib import image
from pmlb import fetch_data
from scipy.stats import zscore


def printTypeAndShapeOfData(imageName):
    data = image.imread(imageName)
    print(data.dtype)
    print(data.shape)


if __name__ == '__main__':
    datasets = ["Agaricus Lepiota",
                "Balls",
                "Bean Leaf",
                "Bird Song",
                "Cifar-10",
                "Diabetes",
                "Gametes Epistasis",
                "Healthcare",
                "Liver Cirrhosis",
                "Magic",
                "mfeat_pixel",
                "MNSIT",
                "Mofn",
                "Shoes",
                "Solar flare",
                "Rain in Australia",
                "Wine quality white"]
    path = 1
    names = "0.\t exit\n"
    counter = 1
    for dataset in datasets:
        names += str(counter) + ".\t " + dataset + "\n"
        counter += 1
    while True:
        nameIndex = int(input("Please select a dataset's name by enter a number:\n" + names))
        if nameIndex == 0:
            break
        dataset = datasets[nameIndex - 1]
        if dataset == "Balls":
            labelAndPathDataframe = pd.read_csv('../Data/Images/Balls/balls.csv')
            nameCounter = 0
            for index, row in labelAndPathDataframe.iterrows():
                filepaths = row["filepaths"]
                finalPath = filepaths.replace("train/", "")
                finalPath = finalPath.replace("test/", "")
                finalPath = finalPath.replace("valid/", "")
                print("Copying Data/Images/Balls/" + filepaths + " to Data/Images/Balls/cleanedData/" + finalPath)
                shutil.copyfile('C:/Users/User/OneDrive/tuks/master/code/Data/Images/Balls/' + filepaths,
                                'C:/Users/User/OneDrive/tuks/master/code/Data/Images/Balls/cleanedData/' + finalPath)
        elif dataset == "Bird Song":
            testingBirdSong = pd.read_csv('../Data/Numeric/Bird Song/test.csv')
            trainBirdSong = pd.read_csv('../Data/Numeric/Bird Song/train.csv')
            birdSong = pd.concat([testingBirdSong, trainBirdSong])

            birdSong = birdSong.drop(['id'], axis=1)

            birdSong['species'] = birdSong['species'].astype('category')
            birdSong['species'] = birdSong['species'].cat.codes
            birdSong.rename(columns={'species': 'target'}, inplace=True)

            birdSong['genus'] = birdSong['genus'].astype('category')
            birdSong['genus'] = birdSong['genus'].cat.codes
            birdSong.to_csv('Data/Numeric/Bird Song/cleanedData.csv', sep=',', index=False, encoding='utf-8')
        elif dataset == "Agaricus Lepiota":
            agaricusLepiota = fetch_data('agaricus_lepiota')
            agaricusLepiota = agaricusLepiota.drop(['veil-type'], axis=1)
            for column in agaricusLepiota.columns:
                print(column)
                if (column == "cap-shape" or
                        column == "stalk-color-above-ring" or
                        column == "gill-color" or
                        column == "population" or
                        column == "odor" or
                        column == "ring-type" or
                        column == "cap-color" or
                        column == "habitat" or
                        column == "stalk-root"):
                    agaricusLepiota[column] = zscore(agaricusLepiota[column])
            agaricusLepiota.to_csv('Data/Numeric/Agaricus Lepiota/cleanedData.csv', sep=',', index=False,
                                   encoding='utf-8')
        elif dataset == "Bean Leaf":
            labelAndPathDataframe = pd.read_csv('../Data/Images/Bean Leaf Lesions Classification/train.csv')
            for index, row in labelAndPathDataframe.iterrows():
                filepaths = row["image:FILE"]
                filename = filepaths.replace("train/", "")
                print(
                    "Copying Data/Images/Bean Leaf Lesions Classification/" + filepaths + " to Data/Images/Bean Leaf Lesions Classification/cleandData/" + filename)
                shutil.copyfile('Data/Images/Bean Leaf Lesions Classification/' + filepaths,
                                'Data/Images/Bean Leaf Lesions Classification/cleandData/' + filename)
            labelAndPathDataframe = pd.read_csv('../Data/Images/Bean Leaf Lesions Classification/val.csv')
            for index, row in labelAndPathDataframe.iterrows():
                filepaths = row["image:FILE"]
                filename = filepaths.replace("val/", "")
                print(
                    "Copying Data/Images/Bean Leaf Lesions Classification/" + filepaths + " to Data/Images/Bean Leaf Lesions Classification/cleandData/" + filename)
                shutil.copyfile('Data/Images/Bean Leaf Lesions Classification/' + filepaths,
                                'Data/Images/Bean Leaf Lesions Classification/cleandData/' + filename)
            dataframe.to_csv("Data/Images/Bean Leaf Lesions Classification/cleandData/MapOfImagesAndLabels.csv",
                             sep=',', index=False, encoding='utf-8')
        elif dataset == "Cifar-10":
            with open("Data/Images/Cifar-10/train", 'rb') as fo:
                train_data = pickle.load(fo, encoding='latin1')
            for item in train_data:
                print(item, type(train_data[item]))
        elif dataset == "Gametes Epistasis":
            gametesDataset = fetch_data('GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1')
            gametesDataset.drop([0])
            gametesDataset.to_csv('Data/Numeric/Gametes/cleanedData.csv', sep=',', index=False, encoding='utf-8')
            print(gametesDataset)
        elif dataset == "Healthcare":
            healthcareDataSet = pd.read_csv(
                'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Healthcare\\healthcare_dataset.csv')

            for column in healthcareDataSet.columns:
                if column == "Age" or column == "Billing Amount" or column == "Room Number":
                    healthcareDataSet[column] = zscore(healthcareDataSet[column])
                else:
                    healthcareDataSet[column] = healthcareDataSet[column].astype('category')
                    healthcareDataSet[column] = healthcareDataSet[column].cat.codes
            healthcareDataSet.rename(columns={'Test Results': 'target'}, inplace=True)
            healthcareDataSet.to_csv(
                'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Healthcare\\cleanedData.csv',
                sep=',', index=False,
                encoding='utf-8')
        elif dataset == "Magic":
            magicDataSet = fetch_data('magic')
            magicDataSet = magicDataSet.drop(magicDataSet.columns[0], axis=1)
            counter = 0
            for column in magicDataSet.columns:
                if counter == len(magicDataSet.columns) - 1:
                    break
                magicDataSet[column] = zscore(magicDataSet[column])
                counter += 1
            magicDataSet = magicDataSet.sample(frac=1)
            magicDataSet.to_csv("Data/Numeric/Magic/cleanedData.csv", sep=',', index=False, encoding='utf-8')
            print(magicDataSet)
        elif dataset == "mfeat_pixel":
            pixelDataSet = fetch_data('mfeat_pixel')
            for column in pixelDataSet.columns:
                if column != "target":
                    pixelDataSet[column] = zscore(pixelDataSet[column])

            birdSong['target'] = birdSong['target'].astype('category')
            birdSong['target'] = birdSong['target'].cat.codes

            pixelDataSet.to_csv('Data/Numeric/Mfeat_pixel/cleanedData.csv', sep=',', index=False, encoding='utf-8')
        elif dataset == "Diabetes":
            diabetesDataSet = pd.read_csv(
                '../Data/Numeric/DiabetesHealthIndicators/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
            diabetesDataSet.rename(columns={'Diabetes_binary': 'target'}, inplace=True)
            diabetesDataSet.to_csv('Data/Numeric/DiabetesHealthIndicators/cleanedData.csv', sep=',', index=False,
                                   encoding='utf-8')
        elif dataset == "Liver Cirrhosis":
            liverCirrhosisDataSet = pd.read_csv('C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Liver '
                                                'Cirrhosis\\liver_cirrhosis.csv')
            for column in liverCirrhosisDataSet.columns:
                print(column, type(column))
                if (column == "Status" or column == "Drug" or column == "Sex" or column == "Stage" or
                        column == "Ascites" or column == "Hepatomegaly" or column == "Spiders" or column == "Edema"):
                    liverCirrhosisDataSet[column] = liverCirrhosisDataSet[column].astype('category')
                    liverCirrhosisDataSet[column] = liverCirrhosisDataSet[column].cat.codes
                else:
                    liverCirrhosisDataSet[column] = zscore(liverCirrhosisDataSet[column])

            liverCirrhosisDataSet.rename(columns={'Stage': 'target'}, inplace=True)
            liverCirrhosisDataSet.to_csv(
                'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Liver Cirrhosis\\cleanedData.csv',
                sep=',', index=False,
                encoding='utf-8')
        elif dataset == "MNSIT":
            print("")
        elif dataset == "Mofn":
            mofinDataset = fetch_data('mofn_3_7_10')
            mofinDataset.to_csv('Data/Numeric/Mofn/cleanedData.csv', sep=',', index=False, encoding='utf-8')
            print(mofinDataset)
        elif dataset == "Shoes":
            basePath = "Data/Images/Shoes"
            nameCounter = 0
            dataframe = pd.DataFrame()
            for path in os.listdir(basePath + "/test"):
                for imageName in os.listdir(basePath + "/test/" + path):
                    print(
                        "Copying " + basePath + "/test/" + path + "/" + imageName + " to Data/Images/Shoes/cleandData/" + path + "/" + imageName)
                    shutil.copyfile(basePath + "/test/" + path + "/" + imageName,
                                    'Data/Images/Shoes/cleandData/' + path + "/" + imageName)
            for path in os.listdir(basePath + "/train"):
                for imageName in os.listdir(basePath + "/train/" + path):
                    filename = str(nameCounter) + '.jpg'
                    print(
                        "Copying " + basePath + "/train/" + path + "/" + imageName + " to Data/Images/Shoes/cleandData/" + path + "/" + imageName)
                    shutil.copyfile(basePath + "/train/" + path + "/" + imageName,
                                    'Data/Images/Shoes/cleandData/' + path + "/" + imageName)
        elif dataset == "Solar flare":
            solarDataSet = fetch_data('solar_flare_2')
            counter = 0
            solarDataSet["largest_spot_size"] = zscore(solarDataSet["largest_spot_size"])
            solarDataSet['target'] = solarDataSet['target'].astype('category')
            solarDataSet['target'] = solarDataSet['target'].cat.codes
            solarDataSet.to_csv("Data/Numeric/Solar Flares/cleanedData.csv", sep=',', index=False, encoding='utf-8')
        elif dataset == "Rain in Australia":
            rainDataSet = pd.read_csv(
                'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Rain in Australia\\weatherAUS.csv')
            rainDataSet = rainDataSet.drop(columns=["Date"])

            rows_to_remove = rainDataSet[rainDataSet['RainTomorrow'].isna()]
            rainDataSet = rainDataSet.drop(rows_to_remove.index)

            columns_with_nan = rainDataSet.columns[rainDataSet.isna().any()].tolist()
            rainDataSet.fillna(-1, inplace=True)

            rainDataSet['Location'] = rainDataSet['Location'].astype('category')
            rainDataSet['Location'] = rainDataSet['Location'].cat.codes

            rainDataSet['WindGustDir'] = rainDataSet['WindGustDir'].astype('category')
            rainDataSet['WindGustDir'] = rainDataSet['WindGustDir'].cat.codes

            rainDataSet['WindDir9am'] = rainDataSet['WindDir9am'].astype('category')
            rainDataSet['WindDir9am'] = rainDataSet['WindDir9am'].cat.codes

            rainDataSet['WindDir3pm'] = rainDataSet['WindDir3pm'].astype('category')
            rainDataSet['WindDir3pm'] = rainDataSet['WindDir3pm'].cat.codes

            rainDataSet['RainToday'] = rainDataSet['RainToday'].astype('category')
            rainDataSet['RainToday'] = rainDataSet['RainToday'].cat.codes

            rainDataSet.rename(columns={'RainTomorrow': 'target'}, inplace=True)
            rainDataSet['target'] = rainDataSet['target'].astype('category')
            rainDataSet['target'] = rainDataSet['target'].cat.codes

            columns_with_nan = rainDataSet.columns[rainDataSet.isna().any()].tolist()
            print(columns_with_nan)

            for column in rainDataSet.columns:
                if (column != "Location" and
                        column != "WindGustDir" and
                        column != "WindDir9am" and
                        column != "WindDir3pm" and
                        column != "RainToday" and
                        column != "target"):
                    rainDataSet[column].fillna(rainDataSet[column].mean(), inplace=True)
                    rainDataSet[column] = zscore(rainDataSet[column])
            rainDataSet.to_csv(
                'C:\\Users\\User\\OneDrive\\tuks\\master\\code\\Data\\Numeric\\Rain in Australia\\cleanedData.csv',
                sep=',', index=False,
                encoding='utf-8')
        elif dataset == "Wine quality white":
            whiteWineDataSet = fetch_data('wine_quality_white')
            counter = 0
            for column in whiteWineDataSet.columns:
                if counter != len(whiteWineDataSet.columns) - 1:
                    whiteWineDataSet[column] = zscore(whiteWineDataSet[column])
                    counter += 1
            whiteWineDataSet['target'] = whiteWineDataSet['target'].astype('category')
            whiteWineDataSet['target'] = whiteWineDataSet['target'].cat.codes
            whiteWineDataSet.to_csv('Data/Numeric/White Wine Quality/cleanedData.csv', sep=',', index=False,
                                    encoding='utf-8')
