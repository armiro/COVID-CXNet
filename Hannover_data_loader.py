import os, numpy as np, subprocess, shutil, pandas as pd


hannover_repo_link = "https://github.com/ml-workgroup/covid-19-image-repository.git"
hannover_repo_name = hannover_repo_link[hannover_repo_link.rfind("/")+1:hannover_repo_link.rfind(".")]

# create folder by cloning, or upgrade previously created folder by pulling
clone_cmd = "git clone %s" % hannover_repo_link
if not os.path.exists(path="./%s" % hannover_repo_name):
    subprocess.run(clone_cmd.split(" "))
else:
    cwd = os.getcwd()
    os.chdir(path=hannover_repo_name)
    subprocess.run("git pull origin master".split(" "))
    os.chdir(path=cwd)

# determine if a CXR is COVID-19 positive or negative, based on an issue in the source repo & our own understanding
# related issue: https://github.com/ml-workgroup/covid-19-image-repository/issues/4
metadata = pd.read_csv('./' + hannover_repo_name + '/data.csv')
img_names = metadata['image_id']
offsets = -pd.to_numeric(metadata['admission_offset'], errors='coerce')
is_covid = offsets.apply(lambda i: "n" if i < -14 else "y")

normal_cxr_images = list()
for idx, img_name in enumerate(img_names):
    if is_covid[idx] == 'n':
        normal_cxr_images.append(img_name)

print('normal cxr image names:', normal_cxr_images)

init_covid_idx = 510
init_normal_idx = 3

hannover_img_path = os.path.join(hannover_repo_name, "png")
normal_cxnet_path = "./chest_xray_images/normal/"
covid_cxnet_path = "./chest_xray_images/covid19/"
for idx, img in enumerate(os.listdir(path=hannover_img_path)):
    img_name = img.split(".")[0]
    if img_name in normal_cxr_images:
        shutil.copy(src=hannover_img_path+'/'+img, dst=normal_cxnet_path)
        os.rename(src=normal_cxnet_path + img, dst=normal_cxnet_path + "090+%s" % str(init_normal_idx) + '.png')
        init_normal_idx += 1
    else:
        shutil.copy(src=hannover_img_path+'/'+img, dst=covid_cxnet_path)
        os.rename(src=covid_cxnet_path + img, dst=covid_cxnet_path + "%s" % str(init_covid_idx) + '.png')
        init_covid_idx += 1

print('added Hannover images successfully!')
