
import os
file_list = [os.path.join('training/label_2/', f) for f in os.listdir('training/label_2/') if 'txt' in f]
vehicle_types = 'car van truck bus'

annotation_test = 'kitti_annotaion.txt'
annotation_write = open(annotation_test,'w')

for file in file_list:
    f1 = open(file,'r')
    is_annotated = False
    row = '../training/image_2/'+ file.split('/')[-1].split('.')[0]+'.png'
    lines = f1.readlines()
    for line in lines:
        line = line.split()
    
        if line[0].lower() in vehicle_types:
            print(line[0])
            is_annotated = True
            label = 0
            x_min = int(float(line[4]))
            y_min = int(float(line[5]))
            x_max = int(float(line[6]))
            y_max = int(float(line[7]))
            row = row + ' ' +  str(x_min) + ',' + str(y_min) + ',' + str(x_max) + ',' + str(y_max) + ',' + str(label)
    if is_annotated:
        annotation_write.write(row)
        annotation_write.write('\n')
annotation_write.close()
