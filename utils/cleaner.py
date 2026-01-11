import os
import shutil

import pandas as pd
from pmlb import fetch_data
from scipy.stats import zscore

from utils.menus import show_dataset_menu, CLEANED_DATASETS


def clean():
    while True:
        dataset_names = show_dataset_menu()
        if len(dataset_names) == 0:
            return
        else:
            for dataset_name in dataset_names:
                print(f"Cleaning {dataset_name}")
                if dataset_name == CLEANED_DATASETS[0]:
                    agaricus_lepiota = fetch_data('agaricus_lepiota')
                    agaricus_lepiota = agaricus_lepiota.drop(['veil-type'], axis=1)
                    for column in agaricus_lepiota.columns:
                        if (column == "cap-shape" or
                            column == "stalk-color-above-ring" or
                            column == "gill-color" or
                            column == "population" or
                            column == "odor" or
                            column == "ring-type" or
                            column == "cap-color" or
                            column == "habitat" or
                            column == "stalk-root"):
                                agaricus_lepiota[column] = zscore(agaricus_lepiota[column])
                    agaricus_lepiota.to_csv('Data/Numeric/Agaricus Lepiota/cleanedData.csv',
                                           sep=',',
                                           index=False,
                                           encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[1]:
                    label_and_path_dataframe = pd.read_csv('Data/Images/Balls/balls.csv')
                    for index, row in label_and_path_dataframe.iterrows():
                        file_paths = row["filepaths"]
                        final_path = file_paths.replace("train/", "")
                        final_path = final_path.replace("test/", "")
                        final_path = final_path.replace("valid/", "")
                        shutil.copyfile(f"Data/Images/Balls/{file_paths}",
                                        f"Data/Images/Balls/cleanedData/{final_path}")
                elif dataset_name == CLEANED_DATASETS[2]:
                    label_and_path_dataframe = pd.read_csv('Data/Images/Bean Leaf Lesions Classification/train.csv')
                    for index, row in label_and_path_dataframe.iterrows():
                        file_paths = row["image:FILE"]
                        file_name = file_paths.replace("train/", "")
                        shutil.copyfile(f"Data/Images/Bean Leaf Lesions Classification/{file_paths}",
                                        f"Data/Images/Bean Leaf Lesions Classification/cleandData/{file_name}")
                    label_and_path_dataframe = pd.read_csv('Data/Images/Bean Leaf Lesions Classification/val.csv')
                    for index, row in label_and_path_dataframe.iterrows():
                        file_paths = row["image:FILE"]
                        file_name = file_paths.replace("val/", "")
                        shutil.copyfile(f"Data/Images/Bean Leaf Lesions Classification/{file_paths}",
                                        f"Data/Images/Bean Leaf Lesions Classification/cleandData/{file_name}")
                elif dataset_name == CLEANED_DATASETS[3]:
                    testing_bird_song = pd.read_csv('Data/Numeric/Bird Song/test.csv')
                    train_bird_song = pd.read_csv('Data/Numeric/Bird Song/train.csv')
                    bird_song = pd.concat([testing_bird_song, train_bird_song])

                    bird_song = bird_song.drop(['id'], axis=1)

                    bird_song['species'] = bird_song['species'].astype('category')
                    bird_song['species'] = bird_song['species'].cat.codes
                    bird_song.rename(columns={'species': 'target'}, inplace=True)

                    bird_song['genus'] = bird_song['genus'].astype('category')
                    bird_song['genus'] = bird_song['genus'].cat.codes
                    bird_song.to_csv('Data/Numeric/Bird Song/cleanedData.csv',
                                     sep=',',
                                     index=False,
                                     encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[4]:
                    diabetes_Dataset = pd.read_csv(
                        'Data/Numeric/DiabetesHealthIndicators/diabetes_binary_5050split_health_indicators_BRFSS2015.csv')
                    diabetes_Dataset.rename(columns={'Diabetes_binary': 'target'}, inplace=True)
                    diabetes_Dataset.to_csv('Data/Numeric/DiabetesHealthIndicators/cleanedData.csv',
                                           sep=',',
                                           index=False,
                                           encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[5]:
                    gametes_dataset = fetch_data('GAMETES_Epistasis_2_Way_1000atts_0.4H_EDM_1_EDM_1_1')
                    gametes_dataset.drop([0])
                    gametes_dataset.to_csv('Data/Numeric/Gametes/cleanedData.csv',
                                          sep=',',
                                          index=False,
                                          encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[6]:
                    healthcare_dataset = pd.read_csv('Data/Numeric/Healthcare/healthcare_dataset.csv')

                    for column in healthcare_dataset.columns:
                        if column == "Age" or column == "Billing Amount" or column == "Room Number":
                            healthcare_dataset[column] = zscore(healthcare_dataset[column])
                        else:
                            healthcare_dataset[column] = healthcare_dataset[column].astype('category')
                            healthcare_dataset[column] = healthcare_dataset[column].cat.codes
                    healthcare_dataset.rename(columns={'Test Results': 'target'}, inplace=True)
                    healthcare_dataset.to_csv('Data/Numeric/Healthcare/cleanedData.csv',
                                             sep=',',
                                             index=False,
                                             encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[7]:
                    liver_cirrhosis_dataset = pd.read_csv('Data/Numeric/Liver Cirrhosis/liver_cirrhosis.csv')
                    for column in liver_cirrhosis_dataset.columns:
                        if (column == "Status" or
                            column == "Drug" or
                            column == "Sex" or
                            column == "Stage" or
                            column == "Ascites" or
                            column == "Hepatomegaly" or
                            column == "Spiders" or
                            column == "Edema"):
                            liver_cirrhosis_dataset[column] = liver_cirrhosis_dataset[column].astype('category')
                            liver_cirrhosis_dataset[column] = liver_cirrhosis_dataset[column].cat.codes
                        else:
                            liver_cirrhosis_dataset[column] = zscore(liver_cirrhosis_dataset[column])

                    liver_cirrhosis_dataset.rename(columns={'Stage': 'target'}, inplace=True)
                    liver_cirrhosis_dataset.to_csv('Data/Numeric/Liver Cirrhosis/cleanedData.csv',
                                                 sep=',',
                                                 index=False,
                                                 encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[8]:
                    magic_dataset = fetch_data('magic')
                    magic_dataset = magic_dataset.drop(magic_dataset.columns[0], axis=1)
                    counter = 0
                    for column in magic_dataset.columns:
                        if counter == len(magic_dataset.columns) - 1:
                            break
                        magic_dataset[column] = zscore(magic_dataset[column])
                        counter += 1
                    magic_dataset = magic_dataset.sample(frac=1)
                    magic_dataset.to_csv("Data/Numeric/Magic/cleanedData.csv",
                                         sep=',',
                                         index=False,
                                         encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[9]:
                    pixel_dataset = fetch_data('mfeat_pixel')
                    for column in pixel_dataset.columns:
                        if column != "target":
                            pixel_dataset[column] = zscore(pixel_dataset[column])

                    pixel_dataset['target'] = pixel_dataset['target'].astype('category')
                    pixel_dataset['target'] = pixel_dataset['target'].cat.codes

                    pixel_dataset.to_csv('Data/Numeric/Mfeat_pixel/cleanedData.csv',
                                         sep=',',
                                         index=False,
                                         encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[10]:
                    mofin_dataset = fetch_data('mofn_3_7_10')
                    mofin_dataset.to_csv('Data/Numeric/Mofn/cleanedData.csv',
                                         sep=',',
                                         index=False,
                                         encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[11]:
                    base_path = "Data/Images/Shoes"
                    test_path = base_path + "/test"
                    training_path = base_path + "/train"
                    cleaned_path = 'Data/Images/Shoes/cleandData'
                    for path in os.listdir(test_path):
                        for imageName in os.listdir(f"{test_path}/{path}"):
                            shutil.copyfile(f"{test_path}/{path}/{imageName}",
                                            f"{cleaned_path}/{path}/{imageName}")
                    for path in os.listdir(training_path):
                        for imageName in os.listdir(f"{training_path}/{path}"):
                            shutil.copyfile(f"{training_path}/{path}/{imageName}",
                                            f"{cleaned_path}/{path}/{imageName}")
                elif dataset_name == CLEANED_DATASETS[12]:
                    rain_dataset = pd.read_csv('Data/Numeric/Rain in Australia/weatherAUS.csv')
                    rain_dataset = rain_dataset.drop(columns=["Date"])

                    rows_to_remove = rain_dataset[rain_dataset['RainTomorrow'].isna()]
                    rain_dataset = rain_dataset.drop(rows_to_remove.index)

                    rain_dataset.fillna(-1, inplace=True)

                    rain_dataset['Location'] = rain_dataset['Location'].astype('category')
                    rain_dataset['Location'] = rain_dataset['Location'].cat.codes

                    rain_dataset['WindGustDir'] = rain_dataset['WindGustDir'].astype('category')
                    rain_dataset['WindGustDir'] = rain_dataset['WindGustDir'].cat.codes

                    rain_dataset['WindDir9am'] = rain_dataset['WindDir9am'].astype('category')
                    rain_dataset['WindDir9am'] = rain_dataset['WindDir9am'].cat.codes

                    rain_dataset['WindDir3pm'] = rain_dataset['WindDir3pm'].astype('category')
                    rain_dataset['WindDir3pm'] = rain_dataset['WindDir3pm'].cat.codes

                    rain_dataset['RainToday'] = rain_dataset['RainToday'].astype('category')
                    rain_dataset['RainToday'] = rain_dataset['RainToday'].cat.codes

                    rain_dataset.rename(columns={'RainTomorrow': 'target'}, inplace=True)
                    rain_dataset['target'] = rain_dataset['target'].astype('category')
                    rain_dataset['target'] = rain_dataset['target'].cat.codes

                    for column in rain_dataset.columns:
                        if (column != "Location" and
                                column != "WindGustDir" and
                                column != "WindDir9am" and
                                column != "WindDir3pm" and
                                column != "RainToday" and
                                column != "target"):
                            rain_dataset[column].fillna(rain_dataset[column].mean(), inplace=True)
                            rain_dataset[column] = zscore(rain_dataset[column])
                    rain_dataset.to_csv('Data/Numeric/Rain in Australia/cleanedData.csv',
                                        sep=',',
                                        index=False,
                                        encoding='utf-8')
                elif dataset_name == CLEANED_DATASETS[13]:
                    white_wine_dataset = fetch_data('wine_quality_white')
                    counter = 0
                    for column in white_wine_dataset.columns:
                        if counter != len(white_wine_dataset.columns) - 1:
                            white_wine_dataset[column] = zscore(white_wine_dataset[column])
                            counter += 1
                    white_wine_dataset['target'] = white_wine_dataset['target'].astype('category')
                    white_wine_dataset['target'] = white_wine_dataset['target'].cat.codes
                    white_wine_dataset.to_csv('Data/Numeric/White Wine Quality/cleanedData.csv',
                                              sep=',',
                                              index=False,
                                              encoding='utf-8')