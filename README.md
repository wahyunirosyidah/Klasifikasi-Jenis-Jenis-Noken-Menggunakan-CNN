# Klasifikasi-Jenis-Jenis-Noken-Menggunakan-CNN

# Dataset
[Dataset](https://drive.google.com/drive/folders/19FPHauhooLXQnMlZcdCVmxkgDTHBe6M1?usp=sharing)
|          Name         | Count |
|:---------------------:|:----------:|
|  Noken Bitu Agia (confirmed)  |  250  |
|  Noken Junum Ese (confirmed)  |  250  |
|   Total    |  500  |

![image](https://github.com/user-attachments/assets/730b36c0-59e2-4347-83f7-7ceefc285442)

## Pembagian Data (80-20)

|          Name         | Train | Valid |
|:---------------------:|:----------:|:----------:|
|  Noken Bitu Agia (confirmed)  |  200  |  50  |
|  Noken Junum Ese (confirmed)  |  200  |  50  |
|   Total    |  400  |  100  |

# Model
## Scratch
```
    model=tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),
         tf.keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(512, (3,3), activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
```

### Model Performance
![image](https://github.com/user-attachments/assets/17bbd650-016b-44a5-97ca-6b0f54a039a4)
![image](https://github.com/user-attachments/assets/d0a5e5e6-f110-485a-83c0-ee5fc7786cef)
![image](https://github.com/user-attachments/assets/525c3814-36c6-491f-a16e-863a12c7d67d)

## MobileNetV2 (not use imagenet)

```
# Membangun Model
def create_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_output = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
```
### Model Performance
![image](https://github.com/user-attachments/assets/4dd3fa77-4a3f-4be1-864e-c4e6ad346503)

| Train Accuracy | Train Loss | Validation Accuracy | Validation Loss | 
|:--------------:|:----------:|:-------------------:|:----------------:|
|     100.00 %   |   0.61 %   |       98.00 %       |      6.19 %      |

![image](https://github.com/user-attachments/assets/7e374263-be90-4b85-a9fa-7253caf7e080)
![image](https://github.com/user-attachments/assets/1b4a8f33-47d6-403d-b517-cd85b6338f4a)

## MobileNetV2 (use imagenet)
```
# Membangun Model
def create_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_output = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
     
```

### Model Performance
![image](https://github.com/user-attachments/assets/d0508825-6b57-4449-a5fc-510c4a9a153b)
| Validation Accuracy | Validation Loss | 
|:---------------------:|:----------:|
|  95.83 %  |  11.53 %  |
![image](https://github.com/user-attachments/assets/eaeb2464-a071-442e-a42e-9171e29736ef)
![image](https://github.com/user-attachments/assets/a14b68e5-8b47-4833-a7bc-e3a00ede81e5)

## EfficientNetB0 (not use imagenet)
```
def create_model():
    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_output = base_model(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(base_output)
    x = tf.keras.layers.Dense(128, activation='relu')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(1, activation='sigmoid')(x)

    model = tf.keras.Model(inputs, outputs)
    return model
```

### Model Performance
![image](https://github.com/user-attachments/assets/c2b156f0-2108-4c71-86ea-1694e57f4872)

| Validation Accuracy | Validation Loss | 
|:---------------------:|:----------:|
|  98.96 %  |  6.19 %  |

![image](https://github.com/user-attachments/assets/e25fc23c-3b6a-443a-94f1-76e2dd315b58)
![image](https://github.com/user-attachments/assets/b1cacaab-d8ad-4ffb-92ed-e2d5c7242aa8)

Classification Report:
              precision    recall  f1-score   support

   Bitu Agia       0.98      1.00      0.99        50
   Junum Ese       1.00      0.98      0.99        50

    accuracy                           0.99       100
   macro avg       0.99      0.99      0.99       100
weighted avg       0.99      0.99      0.99       100

## ResNet50 (use imagenet)
```
#create model
def create_model():
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary Classification
    ])
     return model
```
### Model Performance
![image](https://github.com/user-attachments/assets/30330179-6ffc-4355-bf13-35550d9d7231)

| Validation Accuracy | Validation Loss | 
|:---------------------:|:----------:|
|  97.92 %  |  7.27 %  |
![image](https://github.com/user-attachments/assets/ed9aa328-a07c-4044-b70c-357b4f03b766)
![image](https://github.com/user-attachments/assets/da6ef98d-f689-470c-bccd-90d35c92a90f)

## VGG_16
```
#create model
def create_model():
    model = tf.keras.models.Sequential([
        base_model,
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')  # Binary Classification
    ])
     return model
```
### Model Performance
![image](https://github.com/user-attachments/assets/661afecf-2ac3-4567-ac81-955dd1823d19)


| Validation Accuracy | Validation Loss | 
|:---------------------:|:----------:|
|  95.83 %  |  51.04 %  |
![image](https://github.com/user-attachments/assets/e85d27df-42a9-4692-8a21-ae56968daec7)
![image](https://github.com/user-attachments/assets/0e7af1c5-3e85-40e4-a32c-3bfcd0cc36b8)
Classification Report:
              precision    recall  f1-score   support

   Bitu Agia       0.94      0.96      0.95        50
   Junum Ese       0.96      0.94      0.95        50

    accuracy                           0.95       100
   macro avg       0.95      0.95      0.95       100
weighted avg       0.95      0.95      0.95       100
