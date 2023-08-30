import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from stonesoup.denbridge.yifan import read_raw

def main():
    temporal_smooth = 8  # number of historical images are used for smoothing
    min_block_size = 8  # the minimum size of the detected block

    imgs = read_raw('src/fn2.raw')
    output_plot = 'Radar_image_process_plots/'
    output_csv = 'Radar_image_process_plots/detections.csv'
    # imgs = read_raw('E:/fn2.raw')

    fgbg = cv2.bgsegm.createBackgroundSubtractorMOG(history=20, nmixtures=4, backgroundRatio=0.8, noiseSigma=9.0)

    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])
    time_index = []
    detection_x = []
    detection_y = []
    d_size = []
    for i, raw_img in enumerate(imgs):
        # if i > 10:
        #     break

        if i < temporal_smooth:  # if i >= temporal_smooth:
            continue

        img_history = imgs[i - temporal_smooth:i, :, :]

        img = np.uint8(np.mean(img_history, axis=0))
        img_interest_blurred = cv2.GaussianBlur(img, (7, 7), cv2.BORDER_DEFAULT)

        fgmask = fgbg.apply(img_interest_blurred)
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, (7, 7))

        contours, hierarchy = cv2.findContours(fgmask, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        valid_blocks = []
        valid_detect_pnts = []
        for _i in range(len(contours)):
            if contours[_i].shape[0] >= min_block_size:
                valid_blocks.append(contours[_i])
                valid_detect_pnts.append((np.mean(contours[_i][:, 0, 0]), np.mean(contours[_i][:, 0, 1])))
                # store detections and time index
                time_index.append(i)
                detection_x.append(valid_detect_pnts[-1][0])
                detection_y.append(valid_detect_pnts[-1][1])
                d_size.append(contours[_i].shape[0])
                print()
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax1.imshow(raw_img)
        ax2.imshow(fgmask)
        ax3.imshow(raw_img)
        ax1.set_title(f'Raw input T={i}')
        ax2.set_title('Detections (blocks)')
        ax3.set_title('Detections (point) on image')
        ax4.set_title('Accumulated detections')
        ax4.set_xlim(0, img.shape[1])
        ax4.set_ylim(0, img.shape[0])

        for _i in range(len(valid_blocks)):
            ax3.plot(valid_detect_pnts[_i][0], valid_detect_pnts[_i][1], 'r.', markersize=1)
            # Remind: imshow() and plot() have reversed y-axis.
            ax4.plot(valid_detect_pnts[_i][0], img.shape[0] - valid_detect_pnts[_i][1], 'r.', markersize=1)
        plt.pause(0.01)
        plt.savefig(output_plot + '{:04d}.png'.format(i))

    df = pd.DataFrame.from_dict({'Time': time_index, 'X': detection_x, 'Y': detection_y, 'size': d_size})
    df.to_csv(output_csv, index=False)

if __name__ == "__main__":
    main()

