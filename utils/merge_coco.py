# merge two coco.json files into one
# useful when seperate labelling jobs or extend the existing data

# for merge imgae & label folders, check out rsync bash command
# e.g. $ rsync -ah backup_mussel_label0/label/ label/

import json
import os

# path to the coco files, notice the tail slashes
path = 'coco/'
entries = os.listdir(path)
entries.sort()

main = open(path + entries[0])
main = json.load(main)

# number of imgs in the first json file
main_image_number = len(main['images'])
# number of annotations in the first json file
main_annotation_number = len(main['annotations'])

# retrieve info of imgs & annotations
for entry in entries[1:]:
    file = open(path + entry)
    file = json.load(file)

    for i in file['images']:
        main['images'].append(i)

    for i in file['annotations']:
        main['annotations'].append(i)

# update image id
for i in range(len(main['images'])):
    main['images'][i]['id'] = i+1

# update annotations id & image id
for i in range(len(main['annotations'])):
    main['annotations'][i]['id'] = i+1
    if main['annotations'][i]['id'] > main_annotation_number:
        main['annotations'][i]['image_id'] = main['annotations'][i]['image_id'] + \
            main_image_number

# save merged coco.json in this dir
with open('updated_coco.json', 'w') as outfile:
    json.dump(main, outfile)
