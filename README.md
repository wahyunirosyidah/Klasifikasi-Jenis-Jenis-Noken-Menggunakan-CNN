# Klasifikasi-Jenis-Jenis-Noken-Menggunakan-CNN
# Warning!
Belum dilakukan riset lebih mendalam terkait jenis noken berdasarkan bahan dan suku. Pengelompokkan data dilakukan dengan perkiraan


# Berdasarkan Bahan 
## Notebook
[(TF)Klasifikasi Jenis-Jenis Noken Menggunakan CNN.ipynb](https://colab.research.google.com/drive/1b-iIZeRmW3bdcMFaJpxDtIwyGOYlEbPR#scrollTo=9maYNrVQf6pn)

## Dataset
[Dataset](https://drive.google.com/drive/folders/1yjOE3nyIO0S3TfLidPEaGRWUzBSHHCRQ?usp=sharing)
|          Name         | Count | 
|:---------------------:|:----------:|
|  Anggrek  |  250  | 
|  Kulit Kayu Non Rajut  |  250  | 
|  Rajut  |  250  | 

![image](https://github.com/user-attachments/assets/e846f1bf-b29b-4864-8d2c-b653f7eb9c2d)


### Model
     ```
     model = tf.keras.Sequential([
          based_model,  # MobileNet V2
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(3, activation='softmax') ])
     ```
#### Model Performance
The performance of our model was evaluated using test data
![image](https://github.com/user-attachments/assets/8722aeb7-de63-4313-bbba-39b265fd6bc4)
![image](https://github.com/user-attachments/assets/eca0b9ce-599d-4caf-98f3-776c7681d64a)
![image](https://github.com/user-attachments/assets/ff12567a-818e-48c6-9342-77dee177e048)


# Berdasarkan Suku
## Notebook


## Dataset
[Dataset per Suku](https://drive.google.com/drive/folders/171SlrTmbl-reJQehzPlFZoavZY6HWYRh?usp=drive_link)

|          Name         | Count | 
|:---------------------:|:----------:|
|  Mee  |  250  | 
|  Sentani  |  250  | 

### Model
     ```
     model = tf.keras.Sequential([
          based_model,  # MobileNet V2
          tf.keras.layers.Flatten(),
          tf.keras.layers.Dense(128, activation='relu'),
          tf.keras.layers.BatchNormalization(),
          tf.keras.layers.Dense(1, activation='sigmoid') ])
     ```
     
#### Model Performance
The performance of our model was evaluated using test data
