# Klasifikasi-Jenis-Jenis-Noken-Menggunakan-CNN


# Multi Class
## Dataset
[Multi Class Dataset](https://drive.google.com/drive/folders/1i1nanwvlf9E023Rx974PMSqwMzNTUHF8?usp=drive_link)
|          Name         | Count |
|:---------------------:|:----------:|
|  Noken Bitu Agia (confirmed)  |  250  |
|  Noken Junum Ese (confirmed)  |  250  |
|   Noken Pipih (unconfirmed)    |  250  |
|   Total    |  750  |

## Pembagian Data (80-20)

|          Name         | Train | Valid |
|:---------------------:|:----------:|:----------:|
|  Noken Bitu Agia (confirmed)  |  200  |  50  |
|  Noken Junum Ese (confirmed)  |  200  |  50  |
|   Noken Pipih (unconfirmed)    |  200  |  50  |
|   Total    |  600  |  150  |

## Model

# Binary
[Binary Dataset](https://drive.google.com/drive/folders/19FPHauhooLXQnMlZcdCVmxkgDTHBe6M1?usp=sharing)
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

## Model
### Scratch
     ```
     model=tf.keras.models.Sequential([
        data_augmentation,
        tf.keras.layers.Rescaling(1./255),

        tf.keras.layers.Conv2D(16,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(32,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),

        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(1, activation='sigmoid')
     ```

#### Model Performance
![image](https://github.com/user-attachments/assets/848fbb71-6402-4cec-b0c5-93f178f91b2f)
![image](https://github.com/user-attachments/assets/83d69559-6474-40a5-a869-f5372fd0a4bd)


### MobileNetV2
     ```
    model=tf.keras.models.Sequential([
        data_augmentation,
        base_model, #MobileNetV2
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
     ```

#### Model Performance
![image](https://github.com/user-attachments/assets/39cdcef3-4dd1-474f-a51f-5914c032989d)
![image](https://github.com/user-attachments/assets/c911fb1f-dba8-4491-8593-8d2d1958e3d8)


