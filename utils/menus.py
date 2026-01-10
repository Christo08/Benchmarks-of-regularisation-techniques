DATASETS = ["Agaricus Lepiota", #0
            "Balls", #1
            "Bean Leaf", #2
            "Bird Song", #3
            "Diabetes", #4
            "Gametes Epistasis", #5
            "Healthcare", #6
            "Liver Cirrhosis", #7
            "Magic", #8
            "mfeat_pixel",#9
            "Mofn", #10
            "Shoes", #11
            "Rain in Australia", #12
            "Wine quality white"] #13

def show_menu(prompt, items):
    selection = -1
    while selection > len(items) or selection < 0:
        print(prompt)
        for database_name in items:
            print(str((items.index(database_name)+1))+". "+database_name)
        selection = int(input())-1
    return items[selection]

def show_dataset_menu():
    dataset_names = ["All"]
    dataset_names.extend(DATASETS)
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