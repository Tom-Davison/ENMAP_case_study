import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv1D, MaxPooling1D
from keras.optimizers import SGD
import numpy as np
import matplotlib.colors as mcolors

def Patch(data,height_index,width_index, PATCH_SIZE):
    #transpose_array = data.transpose((2,0,1))
    #print transpose_array.shape
    height_slice = slice(height_index, height_index+PATCH_SIZE)
    width_slice = slice(width_index, width_index+PATCH_SIZE)
    patch = data[height_slice, width_slice, :]
    return patch

def train_test_CNN(X_train, y_train, n_classes):

    print("Making Model")
    model = Sequential() 
    model.add(Conv1D(filters=20, kernel_size=3, activation='relu', input_shape = (30,1), padding ='same'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(units=128, activation='relu')) 
    model.add(Dense(n_classes, activation='softmax'))

    sgd = SGD(learning_rate=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()

    model.fit(X_train, y_train, batch_size=64, epochs=1, verbose=1) #, validation_data=(X_test, y_test)
    """
    print('Plotting')
    plt.figure(figsize=(10, 10))
    ground_truth = plt.imshow(y, cmap=cmap, norm=norm)
    cbar = plt.colorbar(ground_truth, ticks=sorted(class_mapping.keys()))
    cbar.ax.set_yticklabels([class_mapping[key] for key in sorted(class_mapping.keys())])
    plt.show(block=True)
    """
    return model
    

def predict_CNN(model, X, y, unit_class_mapping, PATCH_SIZE):

    #convert y to consecutive labels
    y = (y / 10) - 1

    cmap = plt.cm.get_cmap('tab10', len(unit_class_mapping))
    norm = mcolors.BoundaryNorm(boundaries=[key - 0.5 for key in sorted(unit_class_mapping.keys())] + [max(original_to_new_index_mapping.keys()) + 0.5], ncolors=len(original_to_new_index_mapping))

    unique_labels = sorted(unit_class_mapping.keys())
    consecutive_to_label = {i: label for i, label in enumerate(unique_labels)}

    height = y.shape[0]
    width = y.shape[1]

    print(height, width)

    patches = []
    positions = []
    outputs = np.zeros((height, width))

    for i in range(height):
        for j in range(width):
            target = int(y[int(i + PATCH_SIZE / 2), int(j + PATCH_SIZE / 2)])
            if target == 0:
                continue
            else:
                image_patch = Patch(X, i, j, PATCH_SIZE)
                patches.append(image_patch.reshape(1, image_patch.shape[2], image_patch.shape[0]).astype('float32'))
                positions.append((int(i + PATCH_SIZE / 2), int(j + PATCH_SIZE / 2)))

    patches = np.concatenate(patches, axis=0)

    predictions = model.predict(patches)

    for prediction, position in zip(predictions, positions):
        outputs[position[0]][position[1]] = np.argmax(prediction) + 1

    # Convert predictions to class labels
    predicted_labels = np.argmax(predictions, axis=1)

    # Check unique labels and their counts
    uniqueLabels, labelCounts = np.unique(predicted_labels, return_counts=True)
    print("Unique Labels are: ", uniqueLabels)
    print("The number of predicted labels is: ", labelCounts + 1)


    # Initialize dictionaries to count true positives and total samples per class
    class_counts = {key: {'true_positive': 0, 'total': 0} for key in unit_class_mapping.keys()}

    # Iterate over the true labels and predictions
    for true_label, predicted_label in zip(y.flatten(), outputs.flatten()):

        true_label = int(true_label)
        predicted_label = int(predicted_label)  
        
        class_counts[true_label]['total'] += 1
        if true_label == predicted_label:
            print('True (match)')
            class_counts[true_label]['true_positive'] += 1

    # Calculate and print accuracy per class
    for class_id, counts in class_counts.items():
        if counts['total'] > 0:
            accuracy = counts['true_positive'] / counts['total']
        else:
            accuracy = 0
        print(f"Accuracy for {unit_class_mapping[class_id]}: {accuracy:.10f}")

    # Create a mask for correct (green) and incorrect (red) labels
    correct_mask = (y == outputs)
    incorrect_mask = ~correct_mask

    # Create an RGB image to store the mask
    mask_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Color correct labels green (0, 255, 0)
    mask_image[correct_mask] = [0, 255, 0]

    # Color incorrect labels red (255, 0, 0)
    mask_image[incorrect_mask] = [255, 0, 0]

    # Plot the predicted image and the mask image side by side
    plt.figure(figsize=(20, 10))

    # Plot predicted image
    plt.subplot(1, 2, 1)
    predict_image = plt.imshow(outputs, cmap=cmap, norm=norm)
    cbar = plt.colorbar(predict_image, ticks=sorted(unit_class_mapping.keys()))
    cbar.ax.set_yticklabels([unit_class_mapping[key] for key in sorted(unit_class_mapping.keys())])
    plt.title("Predicted Image")

    # Plot correct vs. incorrect mask
    plt.subplot(1, 2, 2)
    plt.imshow(mask_image)
    plt.title("Correct vs Incorrect Labels")

    plt.show()
    print('Plot done')