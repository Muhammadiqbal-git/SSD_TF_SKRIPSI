from firebase_admin import credentials, initialize_app, storage
import os



cred = credentials.Certificate(os.getenv("GOOGLE_APPLICATION_CREDENTIALS")) # add your Credentials keys path to sys environtment
initialize_app(credential=cred)

model_f_name = '12_2023-11-02_mobilenet_v2_Id-88.h5'

model_dir = os.path.join('D:\\', '1.Skripsi', 'SSD_VGG_TF_SKRIPSI', 'Workspace', 'trained_model')
model_path = os.path.join(model_dir, model_f_name)
model_size = os.path.getsize(model_path)
print(model_path)
print("model size = {} MB".format(model_size/1000000))

bucket = storage.bucket(name='cloud-mqtt-detection.appspot.com')
def upload_data(): 
    print('uploading ..')
    blob = bucket.blob(model_f_name)
    blob.upload_from_filename(model_path)
    print('done uploading')

def download_data():
    l_blobs = bucket.list_blobs(max_results=3)
    for blob in l_blobs:
        print(blob)

        blob_path = os.path.join(model_dir, blob.name)
        if os.path.exists(blob_path):
            print('{} is already exist'.format(blob.name))
        else:
            print('{} downloaded (dummy)'.format(blob.name))
        



if __name__ == '__main__':
    print('1. Upload the mentioned model')
    print('2. Download from the bucket')
    user_input = input()
    if (int(user_input) == 1):
        upload_data()
    elif (int(user_input) == 2):
        download_data()
    else:
        print("{} is not an option".format(user_input))



