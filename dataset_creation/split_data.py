import os

hr_dir = './data/celeba_hr/celeba_180x220'
lr_dir = './data/celeba_lr/celeba_55x45'
out_base_dir = './data/img_align_celeba'

hr_test_dir = '{}180x220_test'.format(out_base_dir)
lr_test_dir = '{}55x45_test'.format(out_base_dir)

hr_val_dir = '{}180x220_val'.format(out_base_dir)
lr_val_dir = '{}55x45_val'.format(out_base_dir)

hr_train_dir = '{}180x220_train'.format(out_base_dir)
lr_train_dir = '{}55x45_train'.format(out_base_dir)


def move(from_dir, to_dir, from_idx, to_idx, step):
    os.mkdir(to_dir)
    to_dir = to_dir + '/data'
    os.mkdir(to_dir)

    for i in range(from_idx, to_idx, step):
        str_idx = str(i)
        padded = ('0' * (6 - len(str_idx))) + str_idx
        try:
            os.rename('{}/{}.jpg'.format(from_dir, padded),
                      '{}/{}.jpg'.format(to_dir, padded))
        except:
            print(padded)


print('moving from {} to {} - idxs({}, {}), step {}'.format(hr_dir, hr_test_dir, 1, 200, 1))
move(hr_dir, hr_test_dir, 1, 200, 1)

print('moving from {} to {} - idxs({}, {}), step {}'.format(lr_dir, lr_test_dir, 1, 200, 1))
move(lr_dir, lr_test_dir, 1, 200, 1)

print('moving from {} to {} - idxs({}, {}), step {}'.format(hr_dir, hr_val_dir, 200, 202600, 2))
move(hr_dir, hr_val_dir, 200, 202600, 2)

print('moving from {} to {} - idxs({}, {}), step {}'.format(lr_dir, lr_val_dir, 200, 202600, 2))
move(lr_dir, lr_val_dir, 200, 202600, 2)

print('moving from {} to {} - idxs({}, {}), step {}'.format(hr_dir, hr_train_dir, 201, 202600, 2))
move(hr_dir, hr_train_dir, 201, 202600, 2)

print('moving from {} to {} - idxs({}, {}), step {}'.format(lr_dir, lr_train_dir, 201, 202600, 2))
move(lr_dir, lr_train_dir, 201, 202600, 2)
