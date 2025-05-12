import os 
import shutil
import json
import argparse

# copy all photos from 0/train/preprocessed, 1/train/preprocessed, etc for years 0 through 5 to a folder named up_to_5 but rename each photo 0_1.png, etc
def copy_photos(celeb):
    # make the up_to_5 folder if it doesn't exist
    if not os.path.exists(f'/playpen-nas-ssd/awang/data/{celeb}/up_to_5/train'):
        os.makedirs(f'/playpen-nas-ssd/awang/data/{celeb}/up_to_5/train')
    for year in range(0, 6):  # Modify the range to copy photos for years 0 to 5
        for i in range(0, 20):
            src = '/playpen-nas-ssd/awang/data/{}/{}/train/preprocessed/{}.png'.format(celeb, year, i)
            dst = '/playpen-nas-ssd/awang/data/{}/up_to_5/train/{}_{}.png'.format(celeb, year, i)
            shutil.copyfile(src, dst)

            # copy the latent codes too
            src = '/playpen-nas-ssd/awang/data/{}/{}/train/preprocessed/{}_latent.npy'.format(celeb, year, i)
            dst = '/playpen-nas-ssd/awang/data/{}/up_to_5/train/{}_{}_latent.npy'.format(celeb, year, i)
            shutil.copyfile(src, dst)

# same thing as copy_photos() but for the test folder and without latent codes
def copy_test_photos(celeb):
    # make the up_to_5 folder if it doesn't exist
    if not os.path.exists(f'/playpen-nas-ssd/awang/data/{celeb}/up_to_5/test'):
        os.makedirs(f'/playpen-nas-ssd/awang/data/{celeb}/up_to_5/test')
    for year in range(0, 6):  # Modify the range to copy photos for years 0 to 5
        for i in range(0, 20):
            src = '/playpen-nas-ssd/awang/data/{}/{}/test/preprocessed/{}.png'.format(celeb, year, i)
            dst = '/playpen-nas-ssd/awang/data/{}/up_to_5/test/{}_{}.png'.format(celeb, year, i)
            shutil.copyfile(src, dst)

# combine the dataset.json files from each year into one new dataset.json file in up_to_5, with the updated names of the photos from the above function
# but keep the original dataset.json files in each year's folder the same
def combine_dataset_json(celeb):
    # get the dataset.json file from each year
    for year in range(0, 6):  # Modify the range to combine dataset.json files for years 0 to 5
        src = '/playpen-nas-ssd/awang/data/{}/{}/train/preprocessed/dataset.json'.format(celeb, year)
        dst = '/playpen-nas-ssd/awang/data/{}/up_to_5/train/{}_dataset.json'.format(celeb, year)
        shutil.copyfile(src, dst)

    # create the dataset.json file in up_to_5
    dataset = {}
    labels = []
    # update the names of the photos in the dataset.json file
    for year in range(0, 6):  # Modify the range to update names for years 0 to 5
        
        for i in range(0, 20):
            # get dataset.json file from each year
            with open('/playpen-nas-ssd/awang/data/{}/up_to_5/train/{}_dataset.json'.format(celeb, year), 'r') as f:
                year_dataset = json.load(f)
                year_labels = year_dataset['labels']
                for label in year_labels:
                    img_name = label[0]
                    label[0] = '{}_{}'.format(year, img_name)
                labels += year_labels
        # delete the year_dataset from the directory
        os.remove('/playpen-nas-ssd/awang/data/{}/up_to_5/train/{}_dataset.json'.format(celeb, year))
    dataset['labels'] = labels

    # save the updated dataset.json file
    with open('/playpen-nas-ssd/awang/data/{}/up_to_5/train/dataset.json'.format(celeb), 'w') as f:
        json.dump(dataset, f)
    

# same thing as combine_dataset_json() but for the test folder
def combine_test_dataset_json(celeb):
    # get the dataset.json file from each year
    for year in range(0, 6):  # Modify the range to combine dataset.json files for years 0 to 5
        src = '/playpen-nas-ssd/awang/data/{}/{}/test/preprocessed/dataset.json'.format(celeb, year)
        dst = '/playpen-nas-ssd/awang/data/{}/up_to_5/test/{}_dataset.json'.format(celeb, year)
        shutil.copyfile(src, dst)

    # create the dataset.json file in up_to_5
    dataset = {}
    labels = []
    # update the names of the photos in the dataset.json file
    for year in range(0, 6):  # Modify the range to update names for years 0 to 5
        
        for i in range(0, 20):
            # get dataset.json file from each year
            with open('/playpen-nas-ssd/awang/data/{}/up_to_5/test/{}_dataset.json'.format(celeb, year), 'r') as f:
                year_dataset = json.load(f)
                year_labels = year_dataset['labels']
                for label in year_labels:
                    img_name = label[0]
                    label[0] = '{}_{}'.format(year, img_name)
                labels += year_labels
        # delete the year_dataset from the directory
        os.remove('/playpen-nas-ssd/awang/data/{}/up_to_5/test/{}_dataset.json'.format(celeb, year))
    dataset['labels'] = labels

    # save the updated dataset.json file
    with open('/playpen-nas-ssd/awang/data/{}/up_to_5/test/dataset.json'.format(celeb), 'w') as f:
        json.dump(dataset, f)


# get celeb from command line argument using argparse
parser = argparse.ArgumentParser()
parser.add_argument('--celeb', type=str, required=True)
args = parser.parse_args()
celeb = args.celeb

copy_photos(celeb)
copy_test_photos(celeb)
combine_dataset_json(celeb)
combine_test_dataset_json(celeb)