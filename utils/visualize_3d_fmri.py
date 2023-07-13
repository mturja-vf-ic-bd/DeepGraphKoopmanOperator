import nibabel as nib
from matplotlib import pyplot as plt

FILENAME = "/Users/mturja/ABCD/sub-NDARINV3H54D8DK/ses-baselineYear1Arm1/func/sub-NDARINV3H54D8DK_ses-baselineYear1Arm1_task-nback_run-01_bold.nii.gz"


def plot_block(img, start=(0, 0, 0), end=(2, 2, 2)):
    P = end[0] - start[0]
    Q = end[1] - start[1]
    R = end[2] - start[2]
    plt.figure(figsize=(25, P * Q * R * 5))
    count = 1
    for i in range(start[0], end[0]):
        for j in range(start[1], end[1]):
            for k in range(start[2], end[2]):
                plt.subplot(P * Q * R, 1, count)
                plt.plot(img[i, j, k])
                count = count + 1
    plt.show()


if __name__ == '__main__':
    init_index = 10
    data = nib.load(FILENAME).get_data()[:, :, :, init_index:]
    print(data.shape)
    a = 50
    plot_block(data, (a, a, a), (a + 2, a + 2, a + 2))
