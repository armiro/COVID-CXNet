import os, shutil


# find normal CXR images from the metadata csv file
file = open(file='path/to/CheXpert-v1.0-small/train.csv', mode='r')
header = file.readline()
normal_cxr_cases = list()
records = file.readlines()
for record in records:
    this_record = record.split(',')
    if this_record[4] is not "":
        if this_record[5]:
            normal_cxr_cases.append(this_record[0])
print('number of total normal cases:', len(normal_cxr_cases))


# scrape all image folders, find normal CXR images and copy/rename into the target path
target_path = './chexpert_normal/'
path = 'path/to/CheXpert-v1.0-small/train/'
image_count = 0
for folder_name in os.listdir(path=path):
    subpath = path + folder_name + '/'
    for subfolder_name in os.listdir(path=subpath):
        img_folder = subpath + subfolder_name + '/'
        for img_name in os.listdir(path=img_folder):
            if img_name[0] is not '.':
                this_img = img_folder + img_name
                for case in normal_cxr_cases:
                    if this_img.find(case) is not -1:
                        case_number = case[case.rfind('patient')+7:case.rfind('study')-1]
                        img_name = this_img[this_img.rfind('/'):]
                        shutil.copy(src=this_img, dst=target_path)
                        os.rename(src=target_path+img_name, dst=target_path+str(image_count)+'.jpg')
                        image_count += 1
                        break
                # break
        # break
    # break

