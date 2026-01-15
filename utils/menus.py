CLEANED_DATASETS = ["Agaricus Lepiota", #0 -r
                    "Balls", #1
                    "Bean Leaf", #2
                    "Bird Song", #3 -r
                    "Diabetes", #4
                    "Gametes Epistasis", #5 -r
                    "Healthcare", #6 -r
                    "Liver Cirrhosis", #7
                    "Magic", #8
                    "mfeat_pixel",#9
                    "Mofn", #10
                    "Shoes", #11
                    "Rain in Australia", #12 -r
                    "Wine quality white"] #13
IMAGE_DATASETS = ["Balls", #0
                  "BeanLeafs", #1
                  "FashionMNIST", #2
                  "Cifar10", #3
                  "Shoes"] #4
NUMERIC_DATASETS = ["Diabetes", #0
                    "LiverCirrhosis", #1
                    "Magic", #2
                    "MfeatPixel", #3
                    "WhiteWineQuality"] #4
MODULE_TYPES= ["CNN", "NN"]
IMAGE_REGULAR_TYPES = ["Baseline",
                       "Batch Normalisation",
                       "Dropout",
                       "Geometric Transformation",
                       "Layer Normalisation",
                       "Pruning",
                       "Regularisation Term",
                       "Weight Normalisation",
                       "Weight Perturbation"]
OPTIMISER_REGULAR_TYPES= ["Baseline",
                          "Dropout",
                          "Pruning",
                          "Weight Normalisation",
                           "Weight Perturbation"]
NUMERIC_REGULAR_TYPES = ["Baseline",
                         "Batch Normalisation",
                         "Dropout",
                         "Layer Normalisation",
                         "Pruning",
                         "Regularisation Term",
                         "SMOTE",
                         "Weight Normalisation",
                         "Weight Perturbation"]

def show_menu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for database_name in items:
            print(str((items.index(database_name)+1))+". "+database_name)
        selection = int(input())-1
    return items[selection]

def show_model_menu():
    module_names = ["All"]
    module_names.extend(MODULE_TYPES)
    module_names.append("Back")
    module_option = show_menu("Select module type by entering a number: ", module_names)
    if module_option == module_names[0]:
        names =  module_names[1:-1]
    elif module_option == module_names[len(module_names) - 1]:
        return []
    else:
        names = [module_option]
    return names

def show_dataset_menu(type = "CLEANED"):
    dataset_names = ["All"]
    if type == "CLEANED":
        dataset_names.extend(CLEANED_DATASETS)
    elif type == MODULE_TYPES[0]:
        dataset_names.extend(IMAGE_DATASETS)
    elif type == MODULE_TYPES[1]:
        dataset_names.extend(NUMERIC_DATASETS)
    else:
        dataset_names.extend(IMAGE_DATASETS)
        dataset_names.extend(NUMERIC_DATASETS)
    dataset_names.append("Custom")
    dataset_names.append("Back")
    datasets_option = show_menu("Select dataset by entering a number: ", dataset_names)
    if datasets_option == dataset_names[0]:
        names =  dataset_names[1:-2]
    elif datasets_option == dataset_names[len(dataset_names) - 2]:
        print("Enter the datasets' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        names = []
        for select_dataset_index in select_dataset_indexes:
            names.append(dataset_names[int(select_dataset_index) - 1])
    elif datasets_option == dataset_names[len(dataset_names) - 1]:
        return []
    else:
        names = [datasets_option]
    return names

def show_regularisation_menu(type):
    regular_types = ["All"]
    if type == MODULE_TYPES[0] or type == "BOTH":
        regular_types.extend(IMAGE_REGULAR_TYPES)
    if type == MODULE_TYPES[1] or type == "BOTH":
        regular_types.extend(NUMERIC_REGULAR_TYPES)
    regular_types.append("Custom")
    regular_types.append("Back")
    regular_option = show_menu("Select regularisation method by entering a number: ", regular_types)
    if regular_option == regular_types[0]:
        names = regular_types[1:-2]
    elif regular_option == regular_types[len(regular_types) - 2]:
        print("Enter the regularisation types' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        names = []
        for select_dataset_index in select_dataset_indexes:
            names.append(regular_types[int(select_dataset_index) - 1])
    elif regular_option == regular_types[len(regular_types) - 1]:
        return []
    else:
        names = [regular_option]
    return names

def show_optimiser_regularisation_menu():
    regular_types = ["All"]
    regular_types.extend(OPTIMISER_REGULAR_TYPES)
    regular_types.append("Custom")
    regular_types.append("Back")
    regular_option = show_menu("Select regularisation method by entering a number: ", regular_types)
    if regular_option == regular_types[0]:
        names = regular_types[1:-2]
    elif regular_option == regular_types[len(regular_types) - 2]:
        print("Enter the regularisation types' numbers separated by a comma:")
        select_dataset_indexes = input().replace(' ', '').split(",")
        names = []
        for select_dataset_index in select_dataset_indexes:
            names.append(regular_types[int(select_dataset_index) - 1])
    elif regular_option == regular_types[len(regular_types) - 1]:
        return []
    else:
        names = [regular_option]
    return names