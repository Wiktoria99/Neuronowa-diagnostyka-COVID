from PIL import Image
from use_net_from_file import predict
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_confusion_matrix
import numpy as np

def calculate_matrix():
    main_directory = "./"
    file_test_split = open(main_directory + "test_split.txt", "r")
    file_test_split_lines = file_test_split.readlines()

    matrix = np.array([[0, 0],
                   [0, 0]])

    dir_to_images = main_directory + "data/test/"
    for line in file_test_split_lines:
        splited_line = line.split()

        file_name = splited_line[1]
        label = splited_line[2]     # positive/negative
        image = Image.open(dir_to_images + label + "/" + file_name)

        # TU COS ZROB, A WYKONA SIE DLA KAZDEGO OBRAZKA 
        res = predict(image)
        if label == 'positive':
            if res == 'positive':
                matrix[0][0] += 1
            elif res == 'negative':
                matrix[0][1] += 1
        elif label == 'negative':
            if res == 'negative':
                matrix[1][1] += 1
            elif res == 'positive':
                matrix[1][0] += 1
        
    TP = matrix[0][0]
    FN = matrix[0][1]
    FP = matrix[1][0]
    TN = matrix[1][1]

    print("Miary jakości klasyfikatora:\nMacierz pomyłek:")
    print(matrix[0])
    print(matrix[1])
    print("Precyzja - określa zdolność klasyfikatora do wykrywania klasy pozytywnej (stanu patologicznego)")
    print("Precyzja = " + str(TP/(TP+FN)))
    print("Specyficzność – określa zdolność klasyfikatora do wykrywania klasy negatywnej (stanu normalnego)")
    print("Specyficzność = " + str(TN/(TN+FP)))
    print("Accuracy – całkowita sprawność klasyfikatora, określa prawdopodobieństwo poprawnej klasyfikacji, czyli stosunek poprawnych klasyfikacji do wszystkich klasyfikacji.")
    print("Accuracy = " + str((TP+TN)/(TP+TN+FP+FN)))

    class_names = ['positive', 'negative']
    fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                show_absolute=True,
                                show_normed=True,
                                colorbar=True,
                                class_names=class_names)
    plt.show()

if __name__ == "__main__":
    calculate_matrix()
        