from PIL import Image
import os
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

IMAGE_FOLDER = '/public_bme/data/jianght/datas/Pathology'
# img_dirs = ['yinxing','yangxing','5000yin','5043yang']
# for path1 in os.listdir(f'{IMAGE_FOLDER}/{img_dirs}'):
save_dir = '/public_bme/data/jianght/datas/Pathology_256'


def resize():
    error_list = []
    for root,dirs,files in tqdm(os.walk(f'{IMAGE_FOLDER}/yinxing')):
        # print(files,'sd')
        for file in files:
            if '.jpg' not in file or os.path.isfile(save_dir+root.split(IMAGE_FOLDER)[-1]+'/'+file):
                continue
            try:
                img_path = os.path.join(root,file)
                img = Image.open(img_path)
                img_resize = img.resize((256,256))
                save_path = save_dir+root.split(IMAGE_FOLDER)[-1]
                # print(save_path)
                if not os.path.exists(save_path):
                    print(f'making {save_path}')
                    os.makedirs(save_path)
                # print(save_path)
                img_resize.save(save_path+'/'+file)
            except KeyboardInterrupt:
                exit()
            except:
                error_list.append(root+'/'+file)
                print('error:  ',root+'/'+file)
                continue

    print(error_list)
            




IMAGE_FOLDER = '/public_bme/data/jianght/datas/Pathology'
save_dir = '/public_bme/data/jianght/datas/Pathology_256'

error_list = []

def process_file(root, file):
    if '.jpg' not in file or os.path.isfile(save_dir+root.split(IMAGE_FOLDER)[-1]+'/'+file):
        return
    try:
    # if True:
        img_path = os.path.join(root,file)
        img = Image.open(img_path)
    except KeyboardInterrupt:
        exit()
    except:
        error_list.append(root+'/'+file)
        print('error:  ',root+'/'+file)
        return
    
    img_resize = img.resize((256,256))
    save_path = save_dir+root.split(IMAGE_FOLDER)[-1]
    if not os.path.exists(save_path):
        print(f'making {save_path}')
        os.makedirs(save_path)
    img_resize.save(save_path+'/'+file)
    print('new:  ',root+'/'+file)


executor = ThreadPoolExecutor(max_workers=32)


for root,dirs,files in tqdm(os.walk(f'{IMAGE_FOLDER}/yinxing')):
    for file in files:
        executor.submit(process_file, root, file)


executor.shutdown(wait=True)

print(error_list)





