import copy
from typing import List, Optional

import cv2
import imageio
import numpy as np
import open3d as o3d
import torch
from PIL import Image, ImageOps


def plot_histogram(x, bins=10, x_min=None, x_max=None, save_path=None, figsize=None, title=None, xlabel=None,
                   ylabel='Frequency',
                   grid=True, tight_layout=False, mode='Agg'):
    assert mode in ['Agg', 'TkAgg']
    import matplotlib
    matplotlib.use(mode)
    import matplotlib.pyplot as plt

    if figsize is not None:
        plt.figure(figsize)
    else:
        plt.figure()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if grid:
        plt.grid()

    if x_min is not None and x_max is not None:
        plt.xlim(x_min, x_max)

    plt.hist(x, bins)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show()
    else:
        plt.close()


def plot_line(x=None, y=None, y_std=None, x_min=None, x_max=None, y_min=None, y_max=None, save_path=None, style='-b',
              figsize=None, title=None, xlabel=None, ylabel=None, grid=True, tight_layout=False, markersize=None,
              mode='Agg'):
    assert mode in ['Agg', 'TkAgg']
    import matplotlib
    matplotlib.use(mode)
    import matplotlib.pyplot as plt

    if x is not None and y is None:
        y = x
        x = [i for i in range(len(x))]

    if figsize is not None:
        plt.figure(figsize)
    else:
        plt.figure()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if grid:
        plt.grid()

    if y_std is None:
        if markersize is not None:
            plt.plot(x, y, style, markersize=markersize)
        else:
            plt.plot(x, y, style)
    else:
        plt.plot(x, y, style)
        if y_min is not None and y_max is None:
            plt.fill_between(x, np.maximum(np.array(y) - np.array(y_std), y_min), np.array(y) + np.array(y_std), alpha=0.5)
        elif y_min is None and y_max is not None:
            plt.fill_between(x, np.array(y) - np.array(y_std), np.minimum(np.array(y) + np.array(y_std), y_max),
                             alpha=0.5)
        else:
            plt.fill_between(x, np.maximum(np.array(y) - np.array(y_std), y_min),
                             np.minimum(np.array(y) + np.array(y_std), y_max), alpha=0.5)

    if y_min is not None and y_max is not None:
        plt.ylim(y_min, y_max)
    elif y_min is not None:
        plt.ylim(bottom=y_min)
    elif y_max is not None:
        plt.ylim(top=y_max)
    # plt.xlim(x_min, x_max)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show()
    else:
        plt.close()


def plot_images_in_grid(images: List, grid_size=None, figsize=(12, 6), save_path=None, transpose=False, title=None,
                        sub_fig_titles=None, tight_layout=False, axis_off=True, mode='Agg'):
    # Grid (num_rows, num_cols)
    assert mode in ['Agg', 'TkAgg']
    import matplotlib
    matplotlib.use(mode)
    import matplotlib.pyplot as plt

    if grid_size is None:
        grid_size = (1, len(images))

    if isinstance(grid_size, int):
        grid_size = (1, grid_size)

    fig, axs = plt.subplots(grid_size[0], grid_size[1], figsize=figsize)

    if title is not None:
        fig.subplots_adjust(top=0.8)
        fig.suptitle(title)

    if axis_off:
        for ax in axs.flat:
            ax.axis('off')

    for k, image in enumerate(images):

        col = k % grid_size[1]
        row = k // grid_size[1]

        if sub_fig_titles is not None and isinstance(sub_fig_titles, list):
            if grid_size[0] == 1:
                axs[col].set_title(sub_fig_titles[k])
            elif grid_size[1] == 1:
                axs[row].set_title(sub_fig_titles[k])
            else:
                axs[row, col].set_title(sub_fig_titles[k])

        if isinstance(image, torch.Tensor):
            image = image.detach().cpu().numpy()

        if transpose:
            image = image.transpose(1, 2, 0)

        if grid_size[0] == 1:
            axs[col].imshow(image)
        elif grid_size[1] == 1:
            axs[row].imshow(image)
        else:
            axs[row, col].imshow(image)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show(block=True)
    else:
        plt.close()


def plot_image(image, save_path=None, transpose=False, title=None, axis_off=True, x_label=None, y_label=None,
               tight_layout=True,
               mode='Agg'):
    assert mode in ['Agg', 'TkAgg']
    import matplotlib
    matplotlib.use(mode)
    import matplotlib.pyplot as plt

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu().numpy()

    if transpose:
        image = image.transpose(1, 2, 0)

    plt.figure()
    if title is not None:
        plt.title(title)

    if axis_off:
        plt.axis('off')
    else:
        if x_label is not None:
            plt.xlabel(x_label)
        if y_label is not None:
            plt.ylabel(y_label)

    plt.imshow(image)
    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show(block=True)
    else:
        plt.close()


def plot_torch_image(batch, save_path='/home/kamil/img.png', index=None):
    import matplotlib.pyplot as plt
    if save_path[-4:] != '.png':
        save_path = save_path + '.png'

    if len(batch.shape) == 4:
        index = np.random.randint(0, len(batch)) if index is None else index
        image = batch[index].permute(1, 2, 0).cpu().numpy()
    elif len(batch.shape) == 3:
        image = batch.permute(1, 2, 0).cpu().numpy()
    plt.figure()
    plt.axis('off')
    plt.imshow(image)
    plt.savefig(save_path)
    plt.tight_layout()
    plt.close()


def plot_heatmap_from_unordered_data(x, y, z, height=100, width=100, save_path=None, figsize=None, title=None,
                                     text_annotation=False, text_colour='red', cmap=None, xlabel=None,
                                     ylabel=None, zlabel=None, tight_layout=False, mode='Agg'):
    assert mode in ['Agg', 'TkAgg']
    import matplotlib
    matplotlib.use(mode)

    import matplotlib.pyplot as plt

    assert len(x) == len(y) == len(z)

    if cmap is None:
        cmap = plt.cm.Blues

    x_min = np.min(x)
    x_max = np.max(x)
    x_spacing = (x_max - x_min) / width

    y_min = np.min(y)
    y_max = np.max(y)
    y_spacing = (y_max - y_min) / height

    sum_heatmap = np.zeros((height, width))
    counter = np.zeros((height, width))

    for i in range(len(y)):
        x_val = x[i]
        x_idx = int(np.round((x_val - x_min) / x_spacing)) - 1

        y_val = y[i]
        y_idx = int(np.round((y_val - y_min) / y_spacing)) - 1

        counter[y_idx, x_idx] += 1
        sum_heatmap[y_idx, x_idx] += z[i]

    counter_temp = np.copy(counter)
    counter_temp[counter_temp == 0] = 1  # Set to 1 to avoid division by 0
    heatmap = sum_heatmap / counter_temp
    heatmap[counter == 0] = np.nan

    if text_annotation:
        var_heatmap = np.zeros((height, width))
        for i in range(len(y)):
            x_val = x[i]
            x_idx = int(np.round((x_val - x_min) / x_spacing)) - 1

            y_val = y[i]
            y_idx = int(np.round((y_val - y_min) / y_spacing)) - 1

            var_heatmap[y_idx, x_idx] += (z[i] - heatmap[y_idx, x_idx]) ** 2

        var_heatmap = var_heatmap / counter_temp
        std_heatmap = np.sqrt(var_heatmap)
        var_heatmap[counter == 0] = np.nan
        std_heatmap[counter == 0] = np.nan

    if figsize is not None:
        fix, ax = plt.subplots(figsize=figsize)
    else:
        fix, ax = plt.subplots()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    ax.set_xticks(np.arange(width + 1) - 0.5, [round(i, 1) for i in np.arange(width + 1) * x_spacing + x_min])
    ax.set_yticks(np.arange(height + 1) - 0.5, [round(i, 1) for i in np.arange(height + 1) * y_spacing + y_min])

    im = plt.imshow(heatmap, cmap=cmap)
    cbar = plt.colorbar(im)
    if zlabel is not None:
        cbar.set_label(zlabel, rotation=270, labelpad=15)

    if text_annotation:
        for i in range(height):
            for j in range(width):
                if not np.isnan(heatmap[i, j]):
                    if counter[i, j] > 1:
                        label = f'{round(heatmap[i, j], 1)}+-{round(std_heatmap[i, j], 1)}\n({int(counter[i, j])}\nsamples)'
                    else:
                        label = f'{round(heatmap[i, j], 1)}\n({int(counter[i, j])} sample)'
                    _ = ax.text(j, i, label, ha="center", va="center", color=text_colour)

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show()
    else:
        plt.close()


def plot_lines(x: Optional[List[List]] = None, y: Optional[List[List]] = None, y_std: Optional[List[List]] = None,
               labels=None, styles: Optional[List] = None, save_path=None, figsize=None, title=None, xlabel=None,
               ylabel=None, xscale='linear', yscale='linear', xticks=None, xticks_labels=None, xlim=None, ylim=None,
               grid=True, tight_layout=False, mode='Agg'):
    assert mode in ['Agg', 'TkAgg']
    assert xscale in ["linear", "log", "symlog", "logit"]
    assert yscale in ["linear", "log", "symlog", "logit"]

    import matplotlib
    matplotlib.use(mode)
    import matplotlib.pyplot as plt

    if x is not None and y is None:
        y = x
        x = [[i for i in range(len(y[j]))] for j in range(len(y))]

    if isinstance(y[0], list):
        assert labels is not None, 'When plotting more than a single line, you must provide labels'

    if y_std is not None:
        assert len(y) == len(y_std), 'y and y_std do not have the same length'

    if figsize is not None:
        plt.figure(figsize)
    else:
        plt.figure()

    if title is not None:
        plt.title(title)

    if xlabel is not None:
        plt.xlabel(xlabel)

    if ylabel is not None:
        plt.ylabel(ylabel)

    if grid:
        plt.grid()

    plt.yscale(yscale)
    plt.xscale(xscale)

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if xticks is not None and xticks_labels is not None:
        plt.xticks(xticks, xticks_labels)

    for idx in range(len(y)):
        if y_std is None:
            if styles is not None:
                if labels is not None:
                    plt.plot(x[idx], y[idx], styles[idx], label=labels[idx])
                else:
                    plt.plot(x[idx], y[idx], styles[idx])
            else:
                if labels is not None:
                    plt.plot(x[idx], y[idx], label=labels[idx])
                else:
                    plt.plot(x[idx], y[idx])
        else:
            if styles is not None:
                if labels is not None:
                    plt.errorbar(x[idx], y[idx], y_std[idx], styles[idx], label=labels[idx])
                else:
                    plt.errorbar(x[idx], y[idx], y_std[idx], styles[idx])
            else:
                if labels is not None:
                    plt.errorbar(x[idx], y[idx], y_std[idx], label=labels[idx])
                else:
                    plt.errorbar(x[idx], y[idx], y_std[idx])

    if labels is not None:
        plt.legend()

    if tight_layout:
        plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    if mode == 'TkAgg':
        plt.show()
    else:
        plt.close()


def plot_matches(rgb_0, kpts_0, rgb_1, kpts_1, num_points_to_plot=-1, shuffle_matches=False,
                 match_flag=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS):
    size = 1
    distance = 1
    # Create keypoints
    keypoints_0_cv = []
    keypoints_1_cv = []
    for kpt_0, kpt_1 in zip(kpts_0, kpts_1):
        if not (kpt_1 == np.array([-1, -1])).all():
            # if socket.gethostname() == 'omen' or socket.gethostname() == 'slifer':
            keypoints_0_cv.append(cv2.KeyPoint(x=float(kpt_0[0]), y=float(kpt_0[1]), size=size))
            keypoints_1_cv.append(cv2.KeyPoint(x=float(kpt_1[0]), y=float(kpt_1[1]), size=size))
            # else:
            #     keypoints_0_cv.append(cv2.KeyPoint(x=float(kpt_0[0]), y=float(kpt_0[1]), _size=size))
            #     keypoints_1_cv.append(cv2.KeyPoint(x=float(kpt_1[0]), y=float(kpt_1[1]), _size=size))
    keypoints_0_cv = tuple(keypoints_0_cv)
    keypoints_1_cv = tuple(keypoints_1_cv)

    # Create a list of matches
    matches = []
    for idx in range(len(keypoints_0_cv)):
        match = cv2.DMatch()
        match.trainIdx = idx
        match.queryIdx = idx
        match.trainIdx = idx
        match.distance = distance
        matches.append(match)

    if shuffle_matches:
        # Shuffle all matches
        matches = list(np.array(matches)[np.random.permutation(len(matches))])

    img = cv2.drawMatches(rgb_0, keypoints_0_cv, rgb_1, keypoints_1_cv, matches[:num_points_to_plot], None,
                          flags=match_flag)
    return img


def visualise_flow(rgb1, segmap1, rgb2, segmap2, flow_1_to_2, save_path=None):
    from thousand_tasks.core.utils.img_to_pcd import get_pixel_indices_from_segmap

    segmented_rgb1 = rgb1 * segmap1[..., None]
    segmented_rgb2 = rgb2 * segmap2[..., None]

    # Get indices of all object pixels in first image
    obj_indices_img1 = get_pixel_indices_from_segmap(segmap1)

    # Calculate indices in second image based on flow
    flow_x = flow_1_to_2[obj_indices_img1[:, 1], obj_indices_img1[:, 0], 0]
    flow_y = flow_1_to_2[obj_indices_img1[:, 1], obj_indices_img1[:, 0], 1]

    obj_indices_img2_pred = obj_indices_img1.copy().astype(np.float64)
    obj_indices_img2_pred[:, 0] += flow_x
    obj_indices_img2_pred[:, 1] += flow_y

    if not (rgb1.shape == rgb2.shape):
        obj_indices_img2_pred[:, 0] *= (rgb2.shape[1] / rgb1.shape[1])
        obj_indices_img2_pred[:, 1] *= (rgb2.shape[0] / rgb1.shape[0])

    obj_indices_img2_pred = np.round(obj_indices_img2_pred).astype(int)

    # Check if all pixels are in the image and remove invalid ones
    valid = np.ones(len(obj_indices_img2_pred), dtype=bool)
    valid = np.logical_and(valid, 0 <= obj_indices_img2_pred[:, 0])
    valid = np.logical_and(valid, obj_indices_img2_pred[:, 0] < rgb2.shape[1])
    valid = np.logical_and(valid, 0 <= obj_indices_img2_pred[:, 1])
    valid = np.logical_and(valid, obj_indices_img2_pred[:, 1] < rgb2.shape[0])

    obj_indices_img1 = obj_indices_img1[valid]
    obj_indices_img2_pred = obj_indices_img2_pred[valid]

    # Filter pixels that land outside the second segmentation map
    obj_indices_img2 = get_pixel_indices_from_segmap(segmap2)
    valid2 = np.zeros(len(obj_indices_img2_pred), dtype=bool)

    for i, kpt in enumerate(obj_indices_img2_pred):
        if (kpt == obj_indices_img2).all(axis=1).any():
            valid2[i] = True

    obj_indices_img1 = obj_indices_img1[valid2]
    obj_indices_img2_pred = obj_indices_img2_pred[valid2]

    # move object by flow
    rgb2_pred = np.zeros_like(segmented_rgb2)
    rgb2_pred[obj_indices_img2_pred[:, 1], obj_indices_img2_pred[:, 0]] = segmented_rgb1[
        obj_indices_img1[:, 1], obj_indices_img1[:, 0]]

    flow_img = flow_1_to_2.copy()
    flow_img -= flow_img.min(axis=0).min(axis=0)
    flow_img /= flow_img.max(axis=0).max(axis=0)
    flow_img = np.concatenate((flow_img, np.zeros_like(segmap1)[..., None]), axis=-1)

    if save_path is not None:
        plot_images_in_grid(images=[rgb1, rgb2, segmented_rgb1, segmented_rgb2, flow_img, rgb2_pred],
                            sub_fig_titles=['RGB 1', 'RGB 2', 'Segmented RGB 1', 'Segmented RGB 2', 'Flow',
                                            'RGB 1 warped with flow'],
                            grid_size=(3, 2), tight_layout=True,
                            mode='Agg', save_path=save_path)
    else:
        plot_images_in_grid(images=[rgb1, rgb2, segmented_rgb1, segmented_rgb2, flow_img, rgb2_pred],
                            sub_fig_titles=['RGB 1', 'RGB 2', 'Segmented RGB 1', 'Segmented RGB 2', 'Flow',
                                            'RGB 1 warped with flow'],
                            grid_size=(3, 2), tight_layout=True,
                            mode='TkAgg')


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])  # Orange for demo (source)
    target_temp.paint_uniform_color([0, 0.651, 0.929])  # Blue for live (target)
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name="Registration Result: Demo (Orange) vs Live (Blue)",
        width=800,
        height=600
    )


def create_gif_from_rgb_images(rgb_frames, output_file_path, target_res=None, fps=30):
    assert output_file_path.split('.')[-1] == 'gif', 'Output file path extension should be \'.gif\''
    if target_res is not None:
        assert isinstance(target_res, tuple) and len(target_res) == 2
    if target_res is not None:
        rgb_frames = np.asarray(
            [np.asarray(ImageOps.contain(Image.fromarray(frame), size=target_res)) for frame in rgb_frames])

    imageio.mimsave(output_file_path, rgb_frames, fps=fps)
