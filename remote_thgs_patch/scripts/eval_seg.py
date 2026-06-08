import os 
import torch
import numpy as np
import cv2
import json

logging = False


def binary_mask_metrics(pred_mask, gt_mask):

    tp = torch.sum((pred_mask == 1) & (gt_mask == 1)).float()
    tn = torch.sum((pred_mask == 0) & (gt_mask == 0)).float()
    fp = torch.sum((pred_mask == 1) & (gt_mask == 0)).float()
    fn = torch.sum((pred_mask == 0) & (gt_mask == 1)).float()

    iou = tp / (tp + fp + fn + 1e-6)
    dice = 2 * tp / (2 * tp + fp + fn + 1e-6)
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    acc = (tp + tn) / (tp + tn + fp + fn + 1e-6)

    return {
        "IoU": iou.item(),
        "Dice": dice.item(),
        "Precision": precision.item(),
        "Recall": recall.item(),
        "F1": f1.item(),
        "Acc": acc.item(),
    }


def polygon_to_mask(img_shape, points_list):
    points = np.asarray(points_list, dtype=np.int32)
    mask = np.zeros(img_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points], 1)
    return mask


def lerf(scene_name, path_gt, path_pred):
    """
        Evaluate LERF masks against dataset annotations.

        Prediction folders may contain *_gt.png files for visualization, but
        metrics should not depend on GT files copied into path_pred.
    """
    gt_root = f'{path_gt}/{scene_name}/'
    gt_imgs = [f for f in os.listdir(gt_root) if f.endswith('.jpg')]

    scene_met = []
    for img in gt_imgs:
        img_name = img.split('.')[0]
        with open(f"{path_gt}/{scene_name}/{img_name}.json") as f:
            data = json.load(f)

        image = cv2.imread(os.path.join(gt_root, img), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(os.path.join(gt_root, img))
        h, w = image.shape[:2]

        prompt_list = [f['category'] for f in data['objects']]
        prompt_list = list(set(prompt_list))  # remove duplicates
        img_met = []
        for i in range(len(prompt_list)):
            raw_prompt = prompt_list[i]
            prompt = raw_prompt.replace(' ', '_')
            rendered_mask_p = os.path.join(path_pred, scene_name, img_name, prompt + '.png')

            gt_mask = np.zeros((h, w), dtype=np.uint8)
            for obj in data['objects']:
                if obj['category'] == raw_prompt:
                    gt_mask = np.maximum(gt_mask, polygon_to_mask((h, w), obj['segmentation']))

            if not os.path.exists(rendered_mask_p):
                print(rendered_mask_p)
                rendered_mask = np.zeros_like(gt_mask)
            else:
                rendered_mask = cv2.imread(rendered_mask_p, cv2.IMREAD_GRAYSCALE)
                if rendered_mask is None:
                    rendered_mask = np.zeros_like(gt_mask)
                elif rendered_mask.shape != gt_mask.shape:
                    rendered_mask = cv2.resize(rendered_mask, (w, h), interpolation=cv2.INTER_NEAREST)
            rendered_mask = rendered_mask > 128
            mask = gt_mask > 0
            metrics = binary_mask_metrics(torch.tensor(rendered_mask), torch.tensor(mask))
            img_met.append([metrics[x] for x in ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'Acc']])
            if logging: print(f"{img_name} {prompt} IoU, P, R: {metrics['IoU']}, {metrics['Precision']}, {metrics['Recall']}")
        scene_met.append(np.mean(img_met, axis=0))
    mean_metric = np.mean(scene_met, axis=0)
    print(f"{scene_name} mIoU, mAcc: {mean_metric[0]}, {mean_metric[5]}")
    return mean_metric

def main_lerf(scene_list, path_gt, path_pred):
    all_metrics = []
    for scene in scene_list:
        metric = lerf(scene, path_gt, path_pred)
        all_metrics.append(metric)
    mean_metrics = np.mean(all_metrics, axis=0)
    print(f'Overall metrics, (mIoU, mAcc, mP, mR, F1): {mean_metrics[0]}, {mean_metrics[5]}, {mean_metrics[2]}, {mean_metrics[3]}, {mean_metrics[4]}')


def _3dovs(scene_name, path_gt, path_pred):
    """
        Using the top 7 prompt list to evaluate the mask of replica.
    """
    gt_root = f'{path_gt}/{scene_name}/segmentations'
    gt_imgs = [f for f in os.listdir(gt_root) if not f.endswith('.txt')]

    scene_met = []
    for img in gt_imgs:
        img_name = img

        prompt_list = [f.split('.')[0] for f in os.listdir(os.path.join(gt_root, img_name))]
        img_met = []
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]
            gt_mask_p = os.path.join(gt_root, img_name, prompt + '.png')
            if prompt == 'wood wall': # as shown in the appendix of LangSplat
                prompt = 'wood'
            rendered_mask_p = os.path.join(path_pred, scene_name, img_name, prompt + '.png')
            if not os.path.exists(rendered_mask_p):
                print(rendered_mask_p)
                continue
            rendered_mask = cv2.imread(rendered_mask_p, cv2.IMREAD_GRAYSCALE)
            gt_mask = cv2.imread(gt_mask_p, cv2.IMREAD_GRAYSCALE)
            rendered_mask = cv2.resize(rendered_mask, (gt_mask.shape[1], gt_mask.shape[0]))
            rendered_mask = rendered_mask > 0.5
            mask = gt_mask > 0
            metrics = binary_mask_metrics(torch.tensor(rendered_mask), torch.tensor(mask))
            img_met.append([metrics[x] for x in ['IoU', 'Dice', 'Precision', 'Recall', 'F1', 'Acc']])
            if logging: print(f"{img_name} {prompt} IoU, P, R: {metrics['IoU']}, {metrics['Precision']}, {metrics['Recall']}")
        scene_met.append(np.mean(img_met, axis=0))
    mean_metric = np.mean(scene_met, axis=0)
    print(f"{scene_name} mIoU, mAcc: {mean_metric[0]}, {mean_metric[5]}")
    return mean_metric

def main_3dovs(scene_list, path_gt, path_pred):
    all_metrics = []
    for scene in scene_list:
        metric = _3dovs(scene, path_gt, path_pred)
        all_metrics.append(metric)
    mean_metrics = np.mean(all_metrics, axis=0)
    print(f'Overall metrics, (mIoU, mAcc, mP, mR, F1): {mean_metrics[0]}, {mean_metrics[5]}, {mean_metrics[2]}, {mean_metrics[3]}, {mean_metrics[4]}')

            
if __name__ == "__main__":
    from argparse import ArgumentParser
    parser = ArgumentParser('Evaluate the segmentation mask.')
    parser.add_argument('--path_gt', '-g', type=str)
    parser.add_argument('--path_pred', '-p', type=str)
    parser.add_argument('--scene_list', nargs='+', default=['room'])
    parser.add_argument('--dataset', '-d', type=str, default='lerf')
    parser.add_argument('--logging', action='store_true')
    args = parser.parse_args()
    logging = args.logging
    if args.dataset == 'lerf':
        main_lerf(args.scene_list, args.path_gt, args.path_pred)
    elif args.dataset == '3dovs':
        main_3dovs(args.scene_list, args.path_gt, args.path_pred)
    else:
        raise ValueError('Unknown dataset')
